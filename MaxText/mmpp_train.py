"""Model definition for mmpp. Derived from train.py and layers/models.py while
dropping support for various configs."""

from functools import lru_cache, partial, wraps
from typing import Callable, Optional, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

import common_types
from layers import embeddings
from layers import linears
from layers import models
from layers import quantizations

import max_utils
import mmpp

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
    assert not cfg.logits_via_embedding                  # no shared embedding
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
    mesh = mmpp.get_context_or_fallback(self.mesh).get_stage_mesh(stage_index)
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


from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info
from jax._src.tree_util import Partial
from jax._src.util import tuple_update

def fwd_and_bwd(
    fun: Callable, argnums: Sequence[int], has_aux: bool = False, jitted: bool = True,
) -> tuple[Callable, Callable]:
  def fwd(*args, **kwargs):
    dbg = debug_info('fwd_and_bwd', fun, args, kwargs)
    f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
    f_partial, dyn_args = argnums_partial(
        f, argnums, args, require_static_args_hashable=False)
    return jax._src.api._vjp(f_partial, *dyn_args, has_aux=has_aux)
  def bwd(f_vjp, outgrad):
    return f_vjp(outgrad)
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
    assert len(args) == 2
    args = tuple_update(args, 0, vjp_pack(args[0]))
    return bwd(*args, **kwargs)
  return wrapper


_fwd_and_bwds_cache = None

def get_fwd_and_bwds(model):
  global _fwd_and_bwds_cache
  if _fwd_and_bwds_cache is None:
    num_stages = model.num_logical_stages
    fwd_and_bwds = []
    for stage_index in range(num_stages):
      fwd, bwd = fwd_and_bwd(
        partial(forward, model, stage_index),
        argnums=(0, 1),  # params and input activations
        has_aux=(stage_index == num_stages - 1),
        jitted=False,
      )
      fwd.__name__ = f"forward{stage_index}"
      bwd.__name__ = f"backward{stage_index}"
      fwd = with_vjp_unpack(fwd)
      bwd = with_vjp_pack(bwd)
      fwd_and_bwds.append((fwd, bwd))
    _fwd_and_bwds_cache = (model, fwd_and_bwds)
  else:
    assert model is _fwd_and_bwds_cache[0]
  return _fwd_and_bwds_cache[1]


### Managing flax state

def split_params_by_stage(num_logical_stages, all_params):
  # Assumption: no params are shared between stages; we specialize to MmppTransformer.
  params_by_stage = []
  _all_params = all_params["params"]
  for stage_index in range(num_logical_stages):
    _params = {}
    layers_name = f"stage{stage_index}_layers"
    _params[layers_name] = _all_params[layers_name]
    if stage_index == 0:
      _params["token_embedder"] = _all_params["token_embedder"]
    if stage_index == num_logical_stages - 1:
      _params["decoder_norm"] = _all_params["decoder_norm"]
      _params["logits_dense"] = _all_params["logits_dense"]
    params_by_stage.append({"params": _params})
  return params_by_stage


def combine_params_by_stage(params_by_stage):
  _params = {}
  all_params = {"params": _params}
  for params in params_by_stage:
    for key, value in params["params"].items():
      assert key not in _params, f"{key=} already present"
      _params[key] = value
  return all_params


def split_opt_state_by_stage(num_stages, opt_state):
  # Assumption: Optimizer state consists of mu and nu.
  # https://flax-linen.readthedocs.io/en/latest/guides/model_inspection/model_surgery.html#surgery-with-optimizers
  mu_by_stage = split_params_by_stage(num_stages, opt_state[0].mu)
  nu_by_stage = split_params_by_stage(num_stages, opt_state[0].nu)
  opt_state_by_stage = [
    tuple_update(opt_state, 0, opt_state[0]._replace(mu=mu, nu=nu))
    for mu, nu in zip(mu_by_stage, nu_by_stage)
  ]
  return opt_state_by_stage


def combine_opt_state_by_stage(opt_state_by_stage):
  _opt_state0 = opt_state_by_stage[0][0]._replace(
    mu=combine_params_by_stage([opt_state[0].mu for opt_state in opt_state_by_stage]),
    nu=combine_params_by_stage([opt_state[0].nu for opt_state in opt_state_by_stage]),
  )
  # Note: Assuming that all components except for opt_state[0] are the same.
  return tuple_update(opt_state_by_stage[0], 0, _opt_state0)


def update_stage_state(tx, params, opt_state, grads):
  # No OWG: https://github.com/google/flax/blob/240a5107c02d60c171098fbc3f2738d8b6f5ba75/flax/training/train_state.py#L108-L110
  assert nn.fp8_ops.OVERWRITE_WITH_GRADIENT not in grads
  updates, new_opt_state = tx.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state


# update_state is the stage-sharded equivalent of
#   new_state = old_state.apply_gradients(grads=grads)
#
# To work around flax and optax's API and complexity of cross-stage sharding we
# make some heavy-handed assumptions here:
# - params are owned by exactly one stage (i.e. no weight sharing across stages)
# - optimizer state is sharded analogously (no cross-stage dependencies)
def update_state(ctx, old_state, grads_by_stage):
  num_stages = len(grads_by_stage)
  params_by_stage = split_params_by_stage(num_stages, old_state.params)
  opt_state_by_stage = split_opt_state_by_stage(num_stages, old_state.opt_state)

  new_params_by_stage = []
  new_opt_state_by_stage = []
  _update_stage_state = partial(update_stage_state, old_state.tx)
  for stage_index, (params, opt_state, grads) in enumerate(zip(params_by_stage, opt_state_by_stage, grads_by_stage)):
    name = (mmpp.SectionKind.Epilogue, stage_index)
    new_params, new_opt_state = ctx.section(name, _update_stage_state)(
        params,
        opt_state,
        grads,
    )
    new_params_by_stage.append(new_params)
    new_opt_state_by_stage.append(new_opt_state)

  return old_state.replace(
      step=old_state.step + 1,
      params=combine_params_by_stage(new_params_by_stage),
      opt_state=combine_opt_state_by_stage(new_opt_state_by_stage),
  )


### Loop over stages and train step

def value_and_grad(ctx, model, params_by_stage, data, dropout_rng):
  num_stages = model.num_logical_stages
  fwd_and_bwds = get_fwd_and_bwds(model)

  act, aux = None, None
  stashed = [None] * num_stages
  for stage_index in range(num_stages):
    fwd, _ = fwd_and_bwds[stage_index]
    name = (mmpp.SectionKind.Forward, stage_index)
    print(f"FWD {fwd.__name__}")
    res = ctx.section(name, fwd)(
        params_by_stage[stage_index],
        act,
        data,
        dropout_rng,
    )
    if stage_index == num_stages - 1:
      act, stashed[stage_index], aux = res
    else:
      act, stashed[stage_index] = res
    # if not mmpp.get_context().use_stage0_mesh_only:
    #   print(f"  -> {act.sharding.spec=} / devices={act.sharding.mesh._flat_devices_tuple}")
  loss = act

  cot_act = jnp.ones(loss.shape)
  grads_by_stage = [None] * num_stages
  for stage_index in reversed(range(num_stages)):
    _, bwd = fwd_and_bwds[stage_index]
    name = (mmpp.SectionKind.Backward, stage_index)
    print(f"BWD {bwd.__name__}")
    grads_by_stage[stage_index], cot_act = ctx.section(name, bwd)(
        stashed[stage_index],
        cot_act,
    )

  return (loss, aux), grads_by_stage


def train_step(model, config, _state_mesh_shardings, state, data, dropout_rng):
  assert config is model.config
  assert not config.gradient_clipping_threshold > 0
  assert not config.optimizer_memory_host_offload
  assert not config.use_dpo
  assert not config.gradient_accumulation_steps > 1
  assert not config.record_internal_nn_metrics
  assert not config.enable_dropout

  ctx = mmpp.get_context()
  num_stages = model.num_logical_stages

  # print("BEFORE SPLIT", jax.tree.map(lambda x: x.shape, state.params))
  params_by_stage = split_params_by_stage(num_stages, state.params)
  # print("AFTER SPLIT", jax.tree.map(lambda x: x.shape, params_by_stage))
  (loss, aux), grads_by_stage = value_and_grad(
      ctx, model, params_by_stage, data, dropout_rng)
  new_state = update_state(ctx, state, grads_by_stage)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/total_weights": aux["total_weights"],
  }
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }
  return new_state, metrics
