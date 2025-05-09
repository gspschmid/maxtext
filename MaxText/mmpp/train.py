"""Training harness for mmpp.
Primarily concerned with data placement, transfers and pipelining."""

import dataclasses
from functools import partial, wraps
from typing import Any, Callable, Sequence

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from MaxText.mmpp import models
from MaxText.mmpp import mpmd

# Profiling imports
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')
import nvtx

# vjp-related utils imports
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info
from jax._src.tree_util import Partial
from jax._src.util import tuple_update


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
    partial(models.forward, model, stage_index),
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
  stage_mesh = mpmd.get_context().get_stage_mesh(stage_index)
  def zeros_like_param(pi):
    zeros = jnp.zeros(pi.shape, dtype=pi.dtype)
    sharding = mpmd.sharding_with_mesh(pi.sharding, stage_mesh)
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


def get_section_fns(model, state_by_stage) -> dict[mpmd.SectionName, Callable]:
  section_fns = {}
  for stage_index, state in enumerate(state_by_stage):
    fwd, bwd = model_fwd_and_bwd(model, stage_index)
    param_infos = jax.tree.map(lambda x: ParamInfo(x.shape, x.dtype, x.sharding), state.params)
    section_fns[(mpmd.SectionKind.Prologue, stage_index)] = partial(init_stage_grads, param_infos, stage_index)
    section_fns[(mpmd.SectionKind.Forward, stage_index)] = partial(fwd_stage, fwd)
    section_fns[(mpmd.SectionKind.Backward, stage_index)] = partial(bwd_stage, bwd)
    section_fns[(mpmd.SectionKind.Epilogue, stage_index)] = partial(update_stage_state, state.tx)
  return section_fns


### Managing flax and optax state

def split_params_by_stage(num_stages, all_params):
  # Assumption: no params are shared between stages; we specialize to Transformer.
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
    _update_stage_state = ctx.section((mpmd.SectionKind.Epilogue, stage_index))
    with nvtx.annotate(f"update{stage_index}", color="green"):
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
  ctx = mpmd.get_context()
  if ctx.tracing_for_inference:
    return xs
  stage_mesh = ctx.get_stage_mesh(stage_idx)
  def transfer_one(x):
    sharding = mpmd.sharding_with_mesh(x.sharding, stage_mesh)
    return jax.device_put(x, device=sharding)
  return jax.tree.map(transfer_one, xs)


def split_and_transfer_state(mesh, num_stages, state, in_shard_train, out_shard_train):
  state_by_stage = split_state_by_stage(num_stages, state)
  with mpmd.set_context(mpmd.Context(mesh, tracing_for_inference=False)):
    state_by_stage = tuple(
      transfer(stage_idx, state) for stage_idx, state in enumerate(state_by_stage)
    )

  assert in_shard_train[0] == out_shard_train[0]
  assert isinstance(in_shard_train[0], train_state.TrainState)
  state_shard_by_stage = split_state_by_stage(num_stages, in_shard_train[0])
  in_shard_train = (state_shard_by_stage,) + in_shard_train[1:]
  out_shard_train = (state_shard_by_stage,) + out_shard_train[1:]

  return state_by_stage, in_shard_train, out_shard_train


### Loop over stages and train step

def dump_memory_usage_snapshot(state):
  def jax_tree_size_bytes(tree):
    size_bytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
      if isinstance(leaf, (jax.Array, np.ndarray)):
        size_bytes += leaf.size * leaf.dtype.itemsize
    return size_bytes

  def gb(size_bytes):
    return size_bytes / 1024**3

  record = {}

  print("Memory usage:")
  print("  by device:")
  for i, device in enumerate(jax.devices()):
    stats = device.memory_stats()
    used = gb(stats["bytes_in_use"])
    limit = gb(stats["bytes_limit"])
    peak = gb(stats["peak_bytes_in_use"])
    print(
        f"    {device}: {used:7.01f}/{limit:7.01f}GB ({used/limit*100:4.1f}%)"
        f"  |  peak {peak:7.01f}GB ({peak/limit*100:4.1f}%)"
    )
    record[f"device{i}_used_gb"] = used
    record[f"device{i}_limit_gb"] = limit
    record[f"device{i}_peak_gb"] = peak

  print("  by known state:")
  is_leaf = lambda x: not isinstance(x, dict) or "params_by_stage" not in x
  flat_state, _ = jax.tree_util.tree_flatten_with_path(state, is_leaf=is_leaf)
  total_size_bytes = 0
  for path, value in flat_state:
    size_bytes = jax_tree_size_bytes(value)
    total_size_bytes += size_bytes
    print(f"    state{jax.tree_util.keystr(path):24}: {gb(size_bytes):7.01f}GB")
    assert len(path) == 1 and isinstance(path[0], jax.tree_util.DictKey)
    record[f"state_{path[0].key}_gb"] = gb(size_bytes)
  print(f"  => total size                   : {gb(total_size_bytes):7.01f}GB")

  return record


# TODO: Make sure we only transfer inputs actually needed by a section
def value_and_grad(
    ctx, num_stages, num_mubatches, params_by_stage, data_by_stage, dropout_rng,
    print_memory_usage=False,
):
  assert num_stages == len(params_by_stage) == len(data_by_stage)

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
    _init_stage_grads = ctx.section((mpmd.SectionKind.Prologue, stage_idx))
    with nvtx.annotate(f"init_grads{stage_idx}", color="green"):
      grads = _init_stage_grads()
    grads_by_stage.append(grads)
  # loss : mubatch_idx -> loss
  loss = [None] * num_mubatches
  aux = [None] * num_mubatches

  def memory_usage_snapshot(name):
    if not print_memory_usage or ctx.tracing_for_inference:
      return
    state = {
      "params_by_stage": params_by_stage,
      "fwd_input": fwd_input,
      "stashed": stashed,
      "bwd_input": bwd_input,
      "grads_by_stage": grads_by_stage,
      "loss": loss,
      "aux": aux,
    }
    jax.block_until_ready(state)
    record = dump_memory_usage_snapshot(state)
    if name == "start":
      print("MEM,name," + ",".join(k for k in sorted(record.keys())))
    print(f"MEM,{name}," + ",".join(str(record[k]) for k in sorted(record.keys())))

  ### Microbatched forward+backward
  memory_usage_snapshot("start")
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
            (mpmd.SectionKind.Forward, stage_idx),
            donate_argnums=(0,1,2,),
        )
        # TODO: Investigate whether slice and allocating outside the fwd is a bottleneck
        mubatch_data = jax.tree.map(lambda x: x[mubatch_idx], data_by_stage[stage_idx])
        res = _fwd(
            params_by_stage[stage_idx],
            fwd_input.pop(curr_id),
            mubatch_data,
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
            (mpmd.SectionKind.Backward, stage_idx),
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
    memory_usage_snapshot(task_name)

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

  ctx = mpmd.get_context()
  num_stages = model.num_logical_stages
  num_mubatches = 1 if ctx.tracing_for_inference else config.num_pipeline_microbatches

  # TODO: Investigate whether this replication and resharding is a bottleneck
  def reshape_reshard_data(arr):
    arr = arr.reshape((num_mubatches, -1, *arr.shape[1:]))
    return nn.with_logical_constraint(arr, (None, "activation_batch",))
  data = jax.tree.map(reshape_reshard_data, data)
  data_by_stage = tuple(
    transfer(stage_index, data)  # replicate, it's (relatively) cheap and might overlap?
    for stage_index in range(num_stages)
  )
  del data

  # TODO: Also replicate dropout_rng to all stages and donate?

  # Note: value_and_grad donates params; the params_by_stage returned will merely be
  # fresh jax.Arrays containing the same data.
  params_by_stage = tuple(state.params for state in state_by_stage)
  params_by_stage, grads_by_stage, (loss, aux) = value_and_grad(
      ctx, num_stages, num_mubatches, params_by_stage, data_by_stage, dropout_rng)
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


def prepare_state_and_train_step(
    mesh,
    model,
    state,
    init_rng,
    functional_train,
    in_shard_train,
    out_shard_train,
    example_data,
):
  state, in_shard_train, out_shard_train = split_and_transfer_state(
      mesh,
      model.num_logical_stages,
      state,
      in_shard_train,
      out_shard_train,
  )

  # Replicate init_rng
  from jax.sharding import NamedSharding, PartitionSpec
  init_rng = jax.device_put(init_rng, device=NamedSharding(mesh, PartitionSpec()))

  p_train_step = mpmd.transform(
      mesh,
      get_section_fns(model, state),
      functional_train,
      in_shard_train,
      out_shard_train,
      (state, example_data, init_rng),
  )

  return state, init_rng, p_train_step
