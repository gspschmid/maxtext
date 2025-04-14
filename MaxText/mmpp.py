"""MmppTransformer.
Derived from layers/models.py while dropping support for various configs."""

from functools import lru_cache, partial
from typing import Callable, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
import common_types
from layers import embeddings
from layers import linears
from layers import models
from layers import quantizations

Config = common_types.Config
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Quant = quantizations.AqtQuantization


class MmppTransformer(nn.Module):
  """Transformer, specialized for mmpp."""
  config: Config
  mesh: Mesh
  quant: Quant

  def setup(self):
    cfg = self.config
    assert cfg.using_pipeline_parallelism and cfg.use_mmpp
    assert not cfg.use_untrainable_positional_embedding  # straightforward embedding
    assert not cfg.trainable_position_size > 0
    assert not cfg.logits_via_embedding                  # no shared embedding
    assert not cfg.set_remat_policy_on_layers_per_stage  # we control remat manually
    assert cfg.scan_layers

    decoder_layers = models.Decoder.get_decoder_layers(cfg)
    assert len(decoder_layers), f"unsupported decoder block: {cfg.decoder_block}"
    self.decoder_layer = decoder_layers[0]

  @property
  def num_logical_stages(self):
    cfg = self.config
    return cfg.ici_pipeline_parallelism * cfg.num_pipeline_repeats

  def scan_decoder_layers(self, cfg, mesh, decoder_layer, length, metdata_axis_name):
    initializing = self.is_mutable_collection("params")
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    scan_fn = nn.scan(
        decoder_layer,
        variable_axes={
            "params": params_spec,
            "cache": cache_spec,
            "intermediates": 0,
            "aqt": 0,
            "_overwrite_with_gradient": 0,
        },
        split_rngs={
            "params": True,
            "dropout": cfg.enable_dropout,
        },
        in_axes=(
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
        ),
        length=length,
        metadata_params={nn.PARTITION_NAME: metdata_axis_name},
    )
    return scan_fn(config=cfg, mesh=mesh, name=metdata_axis_name, quant=self.quant)

  def get_pipeline_stage_module(self, stage_index, base_stage, num_layers_in_stage):
    cfg = self.config
    mesh = get_context_or_fallback(self.mesh).get_stage_mesh(stage_index)
    name = f"stage{stage_index}_layers"
    if num_layers_in_stage == 1:
      stage_module = base_stage(config=cfg, mesh=mesh, quant=self.quant, name=name)
    else:
      stage_module = self.scan_decoder_layers(
          cfg, mesh, base_stage, num_layers_in_stage, name
      )
    return stage_module

  @nn.compact
  def _embedding(
      self,
      decoder_input_tokens,
      deterministic,
  ):
    cfg = self.config

    # [batch, length] -> [batch, length, emb_dim]
    y = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
    )(decoder_input_tokens.astype("int32"))
    y = nn.Dropout(
        rate=cfg.dropout_rate,
        broadcast_dims=(-2,),
        name="embedding_dropout",
    )(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    return y

  @nn.compact
  def _logits(self, y, deterministic):
    cfg = self.config

    y = models.Decoder.get_norm_layer(cfg)(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate,
        broadcast_dims=(-2,),
        name="logits_dropout",
    )(y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    logits = linears.DenseGeneral(
        cfg.vocab_size,
        weight_dtype=cfg.weight_dtype,
        dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        kernel_axes=("embed", "vocab"),
        name="logits_dense",
        matmul_precision=cfg.matmul_precision,
    )(y)  # We do not quantize the logits matmul.
    logits = nn.with_logical_constraint(
        logits, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
    )
    if cfg.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  @nn.compact
  def _stage(self, stage_index, y, decoder_segment_ids, decoder_positions):
    cfg = self.config

    # NOTE: Fixed to avoid the need for static args:
    deterministic = True
    model_mode = common_types.MODEL_MODE_TRAIN

    ## If first stage: embedding
    if stage_index == 0:
      decoder_input_tokens = y
      y = self._embedding(decoder_input_tokens, deterministic)

    ## Layers
    num_layers_per_stage, rem = divmod(cfg.num_decoder_layers, self.num_logical_stages)
    assert 0 <= stage_index < self.num_logical_stages

    num_layers_per_stage += 1 if stage_index < rem else 0
    layer_module = self.decoder_layer
    if stage_index != self.num_logical_stages - 1:
      # Remat all but the last stage
      layer_module = nn.remat(
          layer_module,
          prevent_cse=not cfg.scan_layers,
          policy=models.Decoder.get_remat_policy(cfg),
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      )
    stage_module = self.get_pipeline_stage_module(
        stage_index,
        layer_module,
        num_layers_per_stage,
    )
    y = stage_module(
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
    )
    y = y[0] if cfg.scan_layers else y

    ## If last stage: logits
    if stage_index == self.num_logical_stages - 1:
      y = self._logits(y, deterministic)

    return y

  # NOTE: This path is used to initialize the model state.
  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      enable_dropout=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: Optional[int] = None,
      slot: Optional[int] = None,
  ):
    """The transformer implemented in the usual flax way (unusable for mmpp)."""
    assert enable_dropout == False  # ~> deterministic = True
    assert model_mode == common_types.MODEL_MODE_TRAIN
    del previous_chunk
    del true_length
    del slot
    del enable_dropout
    del model_mode

    y = decoder_input_tokens
    for stage_index in range(self.num_logical_stages):
      y = self._stage(
          stage_index,
          y,
          decoder_segment_ids,
          decoder_positions,
      )
    return y


# TODO: CLEANUP
def _make_forward_section(model, stage_index):
  def _stage(
      rngs,
      params,
      y,
      decoder_positions,
      decoder_segment_ids,
  ):
    return model.apply(
      params,
      stage_index,
      y,
      decoder_positions,
      decoder_segment_ids,
      rngs=rngs,
      method=model._stage,
    )
  return jax.jit(_stage)


# NOTE: This path is only used with mmpp.pipelined (for training steps).
def apply_model(
    model: MmppTransformer,
    rngs,
    params,
    # The usual __call__ arguments:
    decoder_input_tokens,
    decoder_positions,
    decoder_segment_ids=None,
    enable_dropout=False,
    model_mode=common_types.MODEL_MODE_TRAIN,
    previous_chunk=None,
    true_length: Optional[int] = None,
    slot: Optional[int] = None,
):
  """The transformed implemented using mini_mpmd."""
  assert enable_dropout == False  # ~> deterministic = True
  assert model_mode == common_types.MODEL_MODE_TRAIN
  del previous_chunk
  del true_length
  del slot
  del enable_dropout
  del model_mode

  y = decoder_input_tokens
  ctx = get_context()
  for stage_index in range(model.num_logical_stages):
    y = ctx.section(
        f"forward{stage_index}",
        partial(_make_forward_section, model, stage_index),
    )(
        rngs,
        params,
        y,
        decoder_segment_ids,
        decoder_positions,
    )
  return y


###

import contextlib
import dataclasses
import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.debug import inspect_array_sharding
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def slice_mesh(mesh, axis_name, slice_index):
  axis = mesh.axis_names.index(axis_name)
  devices = mesh.devices.take(indices=slice_index, axis=axis)
  return Mesh(devices, mesh.axis_names[:axis] + mesh.axis_names[axis+1:])


# MmppContext provides the state for correctly transforming mmpp stages.
# We effectively execute the MmppTransformer in three variants:
#  1. The usual Flax way, entering via __call__. We only use this for model.init.
#  2. A first pass in mmpp.pipelined to infer shardings and other metadata.
#  3. A second pass in mmpp.pipelined to compile the stages separately.
# As part of these steps, MmppContext modifies which mesh is used and whether
# the model is broken into separate jax.jits. In particular, 1. and 2. use only
# stage 0's mesh, but compile everything in a single jax.jit. Step 3. compiles
# wrap's each stage in separate jax.jit and uses the appropriate meshes.
@dataclasses.dataclass(frozen=True)
class MmppContext:
  mesh: Mesh
  use_stage0_mesh_only: bool
  section_decorator: Optional[Callable[[str, Callable], Callable]]
  section_cache: dict[str, Callable] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    assert "stage" in self.mesh.axis_names

  def get_stage_mesh(self, stage_index: int) -> Mesh:
    slice_index = 0 if self.use_stage0_mesh_only else stage_index
    num_physical_stages = self.mesh.shape["stage"]
    return slice_mesh(self.mesh, "stage", slice_index % num_physical_stages)

  def section(
      self,
      name: str,
      make_section_fn: Callable[..., Callable],
      **kwargs,
  ) -> Callable:
    """Annotates a section and caches the resulting function."""
    assert self.section_decorator
    if name not in self.section_cache:
      section_fn = make_section_fn()
      self.section_cache[name] = self.section_decorator(name, section_fn, **kwargs)
    return self.section_cache[name]

  def section_names(self) -> list[str]:
    return list(self.section_cache.keys())



_mmpp_context: Optional[MmppContext] = None

def get_context() -> MmppContext:
  assert _mmpp_context is not None, \
    'MmppContext unavailable. Are you calling from outside mmpp.pipelined?'
  return _mmpp_context

def get_context_or_fallback(mesh: Mesh) -> MmppContext:
  if _mmpp_context is None:
    return MmppContext(mesh, use_stage0_mesh_only=True, section_decorator=None)
  return _mmpp_context

@contextlib.contextmanager
def set_context(ctx: MmppContext):
  global _mmpp_context
  old_ctx = _mmpp_context
  _mmpp_context = ctx
  try:
    yield
  finally:
    _mmpp_context = old_ctx


def sharding_extractor():
  def store(shardings, index, sharding):
    shardings[index] = sharding

  def register_store_callbacks(xs):
    xs_flat, xs_tree = jax.tree.flatten(xs)
    shardings = [None] * len(xs_flat)
    for index, x in enumerate(xs_flat):
      inspect_array_sharding(x, callback=functools.partial(store, shardings, index))
    return shardings, xs_tree

  # TODO: Add support for static args (e.g. via linear_util.WrappedFun)
  def dump_shardings(fun):
    in_shardings, in_tree = None, None
    out_shardings, out_tree = None, None
    @functools.wraps(fun)
    def wrapper(*args):
      nonlocal in_shardings, in_tree
      nonlocal out_shardings, out_tree
      assert out_shardings is None, 'called more than once'
      in_shardings, in_tree = register_store_callbacks(args)
      res = fun(*args)
      out_shardings, out_tree = register_store_callbacks(res)
      return res
    return (
      wrapper,
      lambda: jax.tree.unflatten(in_tree, in_shardings),
      lambda: jax.tree.unflatten(out_tree, out_shardings),
    )

  return dump_shardings


def test_sharding_extractor():
  mesh = Mesh(np.array(jax.devices()).reshape((4,2)), ('a', 'b'))
  s0 = NamedSharding(mesh, P())
  s1 = NamedSharding(mesh, P('a'))
  s2 = NamedSharding(mesh, P(None, 'b'))
  s3 = NamedSharding(mesh, P('a', 'b'))

  def foo(x, y):
    z = jax.lax.with_sharding_constraint(x[0] * x[1], s1)
    return z, z + y

  dump_shardings = sharding_extractor()
  foo, ins_thunk, outs_thunk = dump_shardings(foo)

  foo = jax.jit(foo, in_shardings=((s0, s1), s2))
  arr = jnp.ones((16, 16))
  foo((arr, arr), arr)[0].block_until_ready()

  specs = lambda ss: jax.tree.map(lambda s: s.spec, ss)
  in_specs = specs(ins_thunk())
  out_specs = specs(outs_thunk())
  assert in_specs == specs(((s0, s1), s2)), f'unexpected in specs {in_specs=}'
  assert out_specs == specs((s1, s3)), f'unexpected out specs {out_specs=}'


def pipelined(mesh, step_fn, example_inputs):
  # Phase 1: Infer shardings and other metadata
  print('PHASE1')
  dump_shardings = sharding_extractor()
  in_shardings_thunk = {}
  out_shardings_thunk = {}

  def dump_section_shardings(section_name, section_fn, **kwargs):
    wrapped, in_shardings_thunk[section_name], out_shardings_thunk[section_name] = \
      dump_shardings(section_fn)
    return wrapped

  ctx = MmppContext(
    mesh=mesh,
    use_stage0_mesh_only=True,
    section_decorator=dump_section_shardings,
  )
  with set_context(ctx):
    res = jax.jit(step_fn)(*example_inputs)
    jax.tree.map(lambda x: x.block_until_ready(), res)

  assert all(
    section_name in in_shardings_thunk and section_name in out_shardings_thunk
    for section_name in ctx.section_names()
  )
  in_shardings = {
    section_name: thunk()
    for section_name, thunk in in_shardings_thunk.items()
  }
  out_shardings = {
    section_name: thunk()
    for section_name, thunk in out_shardings_thunk.items()
  }
  for section_name in ctx.section_names():
    ins = jax.tree.map(lambda x: x.spec, in_shardings[section_name])
    outs = jax.tree.map(lambda x: x.spec, out_shardings[section_name])
    print(f'  {section_name=}:\n\t{ins=}\n\t{outs=}\n')

  # Phase 2: Produce final jitted sections
  print('PHASE2')
  def jit_with_shardings(section_name, section_fn, *, static_argnums=()):
    # return section_fn
    # TODO: donate_argnums?
    section_fn.__name__ = f"section_{section_name}"
    return jax.jit(
        section_fn,
        in_shardings=in_shardings[section_name],
        out_shardings=out_shardings[section_name],
        static_argnums=static_argnums,
    )

  ctx = MmppContext(
    mesh=mesh,
    use_stage0_mesh_only=False,
    section_decorator=jit_with_shardings,
  )
  @functools.wraps(step_fn)
  def wrapper(*args):
    with set_context(ctx):
      return step_fn(*args)

  return wrapper
