"""Model definition for mmpp. Derived from train.py and layers/models.py while
dropping support for various configs."""

from flax import linen as nn
import jax
import jax.numpy as jnp

from MaxText import common_types
from MaxText.layers import embeddings
from MaxText.layers import linears
from MaxText.layers import models
from MaxText.layers import quantizations

from MaxText import max_utils
from MaxText.mmpp import mpmd


Config = common_types.Config
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Quant = quantizations.AqtQuantization

EPS = 1e-8


### The flax model definition

class Transformer(nn.Module):
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
    stage_mesh = mpmd.get_context_or_fallback(self.mesh).get_stage_mesh(stage_index)
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

  # NOTE: This path is only used to initialize the model state.
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
      true_length: int | None = None,
      slot: int | None = None,
  ):
    """
    The transformer implemented in the usual flax way. Note that this is needed for
    model state initialization, but not in mmpp execution to work around flax's implicit
    state management.
    """
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


### The model loss definition outside flax, as used in mmpp

def loss_and_aux_from_logits(config, data, logits):
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent,
      ("activation_embed_and_logits_batch", "activation_length"))
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

# NOTE: `forward` essentially splits the monolithic `model.apply` (enters flax once)
# into one flax entrypoint per stage, `forward(model, stage_index, ...)`, i.e. given
#
#   def apply_model(model, params, data, rng):
#     model.num_logical_stages
#     y = data["inputs"]
#     for stage_index in range(model.num_logical_stages):
#       y = forward(
#           model,
#           stage_index,
#           y,
#           data["inputs_segmentation"],
#           data["inputs_position"],
#       )
#     return y
#
# we have
#
#   model.apply(state, ...data, ...rng) == apply_model(model, state.params, data, rng)
#
# Hence forward breaks the model computation into separate stages that may then be
# executed in an MPMD fashion.
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
