"""Model definition for mmpp. Derived from train.py and layers/models.py while
dropping support for various configs."""

import dataclasses
from functools import lru_cache, partial, wraps
from typing import Any, Callable, Optional, Sequence

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from MaxText import common_types
from MaxText.layers import embeddings
from MaxText.layers import linears
from MaxText.layers import models
from MaxText.layers import quantizations

from MaxText import max_utils
from MaxText import mmpp

# Profiling imports
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')
import nvtx

# vjp-related utils imports
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info
from jax._src.tree_util import Partial
from jax._src.util import tuple_update


Config = common_types.Config
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Quant = quantizations.AqtQuantization

EPS = 1e-8


### The flax model definition

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
    assert not cfg.logits_via_embedding                  # no weight sharing
    assert not cfg.set_remat_policy_on_layers_per_stage  # we control remat manually
    assert cfg.scan_layers

    decoder_layers = models.Decoder.get_decoder_layers(cfg)
    assert len(decoder_layers) == 1, f"unsupported decoder block: {cfg.decoder_block}"
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
    stage_mesh = mmpp.get_context_or_fallback(self.mesh).get_stage_mesh(stage_index)
    name = f"stage{stage_index}_layers"
    if num_layers_in_stage == 1:
      stage_module = base_stage(config=cfg, mesh=stage_mesh, quant=self.quant, name=name)
    else:
      stage_module = self.scan_decoder_layers(
          cfg, stage_mesh, base_stage, num_layers_in_stage, name
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
      encoder_images=None,
      enable_dropout=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: Optional[int] = None,
      slot: Optional[int] = None,
  ):
    """The transformer implemented in the usual flax way (unusable for mmpp)."""
    assert enable_dropout == False  # ~> deterministic = True
    assert model_mode == common_types.MODEL_MODE_TRAIN
    del encoder_images
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


def loss_and_aux_from_logits(config, data, logits):
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)
  assert config.num_experts == 1
  aux = {
      "intermediate_outputs": None,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": jnp.array(0.0),
  }
  return loss, aux


### Define each stage's forward and backward as separate jittable functions

def forward(
    model,
    stage_index,
    # Actual params
    params,
    input_activations,
    data,
    rng,
):
  cfg = model.config

  rng1, aqt_rng = jax.random.split(rng)
  rngs = {"dropout": rng1, "params": aqt_rng}

  for k, v in data.items():
    data[k] = v[:cfg.micro_batch_size_to_train_on, :]

  output_activations = model.apply(
    params,
    stage_index,
    data["inputs"] if stage_index == 0 else input_activations,
    data["inputs_segmentation"],
    data["inputs_position"],
    rngs=rngs,
    method=model._stage,
  )

  if stage_index == model.num_logical_stages - 1:
    return loss_and_aux_from_logits(cfg, data, output_activations)
  else:
    return output_activations


def fwd_and_bwd(
    fun: Callable, argnums: Sequence[int], caller_saved_among_argnums: Sequence[bool],
    has_aux: bool = False, jitted: bool = True,
) -> tuple[Callable, Callable]:
  def fwd(*args, **kwargs):
    dbg = debug_info('fwd_and_bwd', fun, args, kwargs)
    f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
    f_partial, dyn_args = argnums_partial(
        f, argnums, args, require_static_args_hashable=False)
    return jax._src.api._saved_input_vjp(
        f_partial, caller_saved_among_argnums, *dyn_args, has_aux=has_aux)
  def bwd(f_vjp, *outgrad_and_saved):
    assert len(outgrad_and_saved) == sum(caller_saved_among_argnums) + 1
    return f_vjp(*outgrad_and_saved)
  if jitted:
    fwd = jit(fwd)
    bwd = jit(bwd)
  return fwd, bwd


def vjp_unpack(f_vjp):
  assert isinstance(f_vjp, Partial)
  flat_data, tree = jax.tree.flatten(f_vjp)
  # NB: Don't use None as the dummy value!
  dataless_vjp = jax.tree.unflatten(tree, [123] * len(flat_data))
  return (flat_data, dataless_vjp)

def vjp_pack(f_vjp_unpacked):
  assert isinstance(f_vjp_unpacked, tuple)
  flat_data, dataless_vjp = f_vjp_unpacked
  dummy_flat_data, tree = jax.tree.flatten(dataless_vjp)
  assert len(dummy_flat_data) == len(flat_data)
  f_vjp = jax.tree.unflatten(tree, flat_data)
  return f_vjp

def with_vjp_unpack(fwd):
  @wraps(fwd)
  def wrapper(*args, **kwargs):
    out = fwd(*args, **kwargs)
    assert 2 <= len(out) <= 3
    return tuple_update(out, 1, vjp_unpack(out[1]))
  return wrapper

def with_vjp_pack(bwd):
  @wraps(bwd)
  def wrapper(*args, **kwargs):
    assert len(args) == 3
    args = tuple_update(args, 0, vjp_pack(args[0]))
    return bwd(*args, **kwargs)
  return wrapper


def model_fwd_and_bwd(model, stage_index):
  num_stages = model.num_logical_stages
  fwd, bwd = fwd_and_bwd(
    partial(forward, model, stage_index),
    # Take vjp wrt params and input activations
    argnums=(0, 1),
    # Caller saves params (to avoid duplicating in vjp residuals)
    caller_saved_among_argnums=(True, False,),
    has_aux=(stage_index == num_stages - 1),
    jitted=False,
  )
  fwd.__name__ = f"forward{stage_index}"
  bwd.__name__ = f"backward{stage_index}"
  fwd = with_vjp_unpack(fwd)
  bwd = with_vjp_pack(bwd)
  return fwd, bwd


@dataclasses.dataclass(frozen=True)
class ParamInfo:
  shape: Any
  dtype: Any
  sharding: Any


def init_stage_grads(param_infos, stage_index):
  stage_mesh = mmpp.get_context().get_stage_mesh(stage_index)
  def zeros_like_param(pi):
    zeros = jnp.zeros(pi.shape, dtype=pi.dtype)
    sharding = mmpp.sharding_with_mesh(pi.sharding, stage_mesh)
    return jax.lax.with_sharding_constraint(zeros, sharding)
  return jax.tree.map(zeros_like_param, param_infos)


def fwd_stage(fwd, params, input_activations, data, rng):
  res = fwd(params, input_activations, data, rng)
  return params, *res


def bwd_stage(bwd, params, stashed, out_cot, grads_acc):
  grads, in_cot = bwd(stashed, out_cot, params)
  grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
  return params, grads_acc, in_cot


def update_stage_state(tx, params, opt_state, grads):
  # No OWG: https://github.com/google/flax/blob/240a5107c02d60c171098fbc3f2738d8b6f5ba75/flax/training/train_state.py#L108-L110
  assert nn.fp8_ops.OVERWRITE_WITH_GRADIENT not in grads
  updates, new_opt_state = tx.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state


def get_section_fns(model, state_by_stage) -> dict[mmpp.SectionName, Callable]:
  section_fns = {}
  for stage_index, state in enumerate(state_by_stage):
    fwd, bwd = model_fwd_and_bwd(model, stage_index)
    param_infos = jax.tree.map(lambda x: ParamInfo(x.shape, x.dtype, x.sharding), state.params)
    section_fns[(mmpp.SectionKind.Prologue, stage_index)] = partial(init_stage_grads, param_infos, stage_index)
    section_fns[(mmpp.SectionKind.Forward, stage_index)] = partial(fwd_stage, fwd)
    section_fns[(mmpp.SectionKind.Backward, stage_index)] = partial(bwd_stage, bwd)
    section_fns[(mmpp.SectionKind.Epilogue, stage_index)] = partial(update_stage_state, state.tx)
  return section_fns


### Managing flax and optax state

def split_params_by_stage(num_stages, all_params):
  # Assumption: no params are shared between stages; we specialize to MmppTransformer.
  params_by_stage = []
  _all_params = all_params["params"]
  for stage_index in range(num_stages):
    _params = {}
    layers_name = f"stage{stage_index}_layers"
    _params[layers_name] = _all_params[layers_name]
    if stage_index == 0:
      _params["token_embedder"] = _all_params["token_embedder"]
    if stage_index == num_stages - 1:
      _params["decoder_norm"] = _all_params["decoder_norm"]
      _params["logits_dense"] = _all_params["logits_dense"]
    params_by_stage.append({"params": _params})
  return tuple(params_by_stage)


def split_opt_state_by_stage(num_stages, opt_state):
  # Assumption: Optimizer state consists of mu and nu.
  # https://flax-linen.readthedocs.io/en/latest/guides/model_inspection/model_surgery.html#surgery-with-optimizers
  mu_by_stage = split_params_by_stage(num_stages, opt_state[0].mu)
  nu_by_stage = split_params_by_stage(num_stages, opt_state[0].nu)
  opt_state_by_stage = [
    tuple_update(opt_state, 0, opt_state[0]._replace(mu=mu, nu=nu))
    for mu, nu in zip(mu_by_stage, nu_by_stage)
  ]
  return tuple(opt_state_by_stage)


def split_state_by_stage(num_stages, state):
  params_by_stage = split_params_by_stage(num_stages, state.params)
  opt_state_by_stage = split_opt_state_by_stage(num_stages, state.opt_state)
  return tuple(
    state.replace(step=state.step, params=params, opt_state=opt_state)
    for params, opt_state in zip(params_by_stage, opt_state_by_stage, strict=True)
  )


# update_state is the stage-sharded equivalent of
#   new_state = old_state.apply_gradients(grads=grads)
#
# To work around flax and optax's API and complexity of cross-stage sharding we
# make some heavy-handed assumptions here:
# - params are owned by exactly one stage (i.e. no weight sharing across stages)
# - optimizer state is sharded analogously (no cross-stage dependencies)
def update_state(ctx, old_state_by_stage, grads_by_stage):
  new_state_by_stage = []
  for stage_index, (old_state, grads) in enumerate(
      zip(old_state_by_stage, grads_by_stage, strict=True)):
    params, opt_state = old_state.params, old_state.opt_state
    _update_stage_state = ctx.section((mmpp.SectionKind.Epilogue, stage_index))
    with nvtx.annotate(f'update{stage_index}', color='green'):
      new_params, new_opt_state = _update_stage_state(params, opt_state, grads)
    new_state_by_stage.append(
        old_state.replace(
            step=old_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
    )
  return tuple(new_state_by_stage)


### Transfer state and input data to the corresponding stages' meshes

def transfer(stage_idx, xs):
  ctx = mmpp.get_context()
  if ctx.tracing_for_inference:
    return xs
  stage_mesh = ctx.get_stage_mesh(stage_idx)
  def transfer_one(x):
    sharding = mmpp.sharding_with_mesh(x.sharding, stage_mesh)
    return jax.device_put(x, device=sharding)
  return jax.tree.map(transfer_one, xs)


def split_and_transfer_state(mesh, num_stages, state, in_shard_train, out_shard_train):
  state_by_stage = split_state_by_stage(num_stages, state)
  with mmpp.set_context(mmpp.MmppContext(mesh, tracing_for_inference=False)):
    state_by_stage = tuple(
      transfer(stage_idx, state) for stage_idx, state in enumerate(state_by_stage)
    )

  assert in_shard_train[0] == out_shard_train[0]
  assert isinstance(in_shard_train[0], train_state.TrainState)
  state_shard_by_stage = split_state_by_stage(num_stages, in_shard_train[0])
  in_shard_train = (state_shard_by_stage,) + in_shard_train[1:]
  out_shard_train = (state_shard_by_stage,) + out_shard_train[1:]

  return state_by_stage, in_shard_train, out_shard_train


def transfer_initial_rng(mesh, rng):
  from jax.sharding import NamedSharding, PartitionSpec
  return jax.device_put(rng, device=NamedSharding(mesh, PartitionSpec()))


### Loop over stages and train step

# TODO: When doing the first tracing (to infer shardings) only use num_mubatches==1
# TODO: Make sure we only transfer inputs actually needed by a section
def value_and_grad(ctx, num_stages, num_mubatches, params_by_stage, data, dropout_rng):
  ### Schedule
  tasks = [
    (mubatch_idx, stage_idx, is_fwd)
    for stage_idx in range(num_stages)
    for mubatch_idx in range(num_mubatches)
    for is_fwd in (False, True)
  ]
  # We want to be careful with the order in which we enqueue work, since
  # a single process is managing multiple devices.
  # Assuming a GPipe-like schedule we traverse tasks in the following order:
  #          t=0 t=1 t=2 t=3 t=4 t=5 t=6
  # stage=0    1   2   4   7
  # stage=1        3   5   8  11
  # stage=2            6   9  12  14
  # stage=3               10  13  15  16
  def task_key(task):
    mubatch_idx, stage_idx, is_bwd = task
    if is_bwd:
      stage_idx = -stage_idx
    return (is_bwd, mubatch_idx + stage_idx, stage_idx)
  tasks.sort(key=task_key)

  ### State
  # params_by_stage : stage_idx -> params
  params_by_stage = list(params_by_stage)
  # fwd_input : (mubatch_idx, stage_idx) -> input/activation
  # TODO: Actually slice the input data into separate microbatches
  fwd_input = {
    (mubatch_idx, 0): None
    for mubatch_idx in range(num_mubatches)
  }
  # stashed : (mubatch_idx, stage_idx) -> stashed residuals
  stashed = {}
  # bwd_input : (mubatch_idx, stage_idx) -> activation
  bwd_input = {
    (mubatch_idx, num_stages-1): 1.0
    for mubatch_idx in range(num_mubatches)
  }
  # grads_by_stage : stage_idx -> grads
  grads_by_stage = []
  for stage_idx in range(num_stages):
    _init_stage_grads = ctx.section((mmpp.SectionKind.Prologue, stage_idx))
    with nvtx.annotate(f"init_grads{stage_idx}", color="green"):
      grads = _init_stage_grads()
    grads_by_stage.append(grads)
  # loss : mubatch_idx -> loss
  loss = [None] * num_mubatches
  aux = [None] * num_mubatches

  ### Microbatched forward+backward
  for mubatch_idx, stage_idx, is_bwd in tasks:
    fwd_bwd_str = "B" if is_bwd else "F"
    color = "blue" if is_bwd else "red"
    task_name = f"mub{mubatch_idx}/{fwd_bwd_str}{stage_idx}"
    print(f"TASK {task_name}")
    with nvtx.annotate(task_name, color=color):
      curr_id = (mubatch_idx, stage_idx)
      if not is_bwd:
        ### Forward
        succ_id = (mubatch_idx, stage_idx+1)
        _fwd = ctx.section(
            (mmpp.SectionKind.Forward, stage_idx),
            donate_argnums=(0,1,),
        )
        res = _fwd(
            params_by_stage[stage_idx],
            fwd_input.pop(curr_id),
            transfer(stage_idx, data),  # TODO: only transfer where actually needed
            transfer(stage_idx, dropout_rng),
        )
        params_by_stage[stage_idx], activation, stashed[curr_id] = res[:3]
        if stage_idx == num_stages - 1:
          loss[mubatch_idx] = activation
          aux[mubatch_idx] = res[3]
        else:
          with nvtx.annotate(
              f"Tx mub{mubatch_idx} {stage_idx}->{stage_idx+1}", color="yellow",
          ):
            fwd_input[succ_id] = transfer(stage_idx+1, activation)
        del res
        del activation
      else:
        ### Backward
        succ_id = (mubatch_idx, stage_idx-1)
        _bwd = ctx.section(
            (mmpp.SectionKind.Backward, stage_idx),
            donate_argnums=(0,1,2,3),
        )
        params_by_stage[stage_idx], grads_by_stage[stage_idx], activation_cot = _bwd(
            params_by_stage[stage_idx],
            stashed.pop(curr_id),
            bwd_input.pop(curr_id),
            grads_by_stage[stage_idx],
        )
        if stage_idx-1 >= 0:
          with nvtx.annotate(
              f"Tx mub{mubatch_idx} {stage_idx}->{stage_idx-1}", color="orange",
          ):
            bwd_input[succ_id] = transfer(stage_idx-1, activation_cot)
        del activation_cot

  stack_mean = lambda x: jnp.mean(jnp.stack(x), axis=0)
  loss = stack_mean(loss)
  aux = jax.tree.map(lambda *xs: stack_mean(xs), *aux)
  return params_by_stage, grads_by_stage, (loss, aux)


def profiled_step(fn):
  step, profile_start, profile_end = 0, 4, 6
  @wraps(fn)
  def wrapper(*args, **kwargs):
    nonlocal step
    if step == profile_start:
      libcudart.cudaProfilerStart()
    with nvtx.annotate(f"step{step}", color="white"):
      res = fn(*args, **kwargs)
    if step == profile_end:
      libcudart.cudaProfilerStop()
    step += 1
    return res
  return wrapper


@profiled_step
def train_step(model, config, _state_mesh_shardings, state_by_stage, data, dropout_rng):
  assert config is model.config
  assert not config.gradient_clipping_threshold > 0
  assert not config.optimizer_memory_host_offload
  assert not config.use_dpo
  assert not config.use_multimodal
  assert not config.gradient_accumulation_steps > 1
  assert not config.record_internal_nn_metrics
  assert not config.enable_dropout

  ctx = mmpp.get_context()
  num_stages = model.num_logical_stages
  num_mubatches = config.num_pipeline_microbatches

  # TODO: Reshape data into microbatches, slice out right microbatch
  # TODO: Replicate data and dropout_rng to all stages, process locally
  # TODO: Also donate data slice and rng?
  data = transfer(0, data)

  # Note: value_and_grad donates params; the params_by_stage returned will merely be
  # fresh jax.Arrays containing the same data.
  params_by_stage = tuple(state.params for state in state_by_stage)
  params_by_stage, grads_by_stage, (loss, aux) = value_and_grad(
      ctx, num_stages, num_mubatches, params_by_stage, data, dropout_rng)
  state_by_stage = tuple(
    state.replace(params=params)
    for state, params in zip(state_by_stage, params_by_stage)
  )

  new_state_by_stage = update_state(ctx, state_by_stage, grads_by_stage)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/total_weights": aux["total_weights"],
  }
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }
  return new_state_by_stage, metrics
