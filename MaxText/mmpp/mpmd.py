"""Infrastructure for MPMD execution of a model broken into explicit stages.
Provides APIs to annotate distinct sections of the model where each section will be
executed in an SPMD fashion. Automatically infers shardings at each section boundary
by first compiling the entire model on a single stage's mesh and extracting the
usual SPMD shardings inferred by JAX/XLA."""

import contextlib
import dataclasses
from enum import Enum
from functools import partial, wraps
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.debug import inspect_array_sharding
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def slice_mesh(mesh, axis_name, slice_index):
  axis = mesh.axis_names.index(axis_name)
  devices = mesh.devices.take(indices=slice_index, axis=axis)
  return Mesh(devices, mesh.axis_names[:axis] + mesh.axis_names[axis+1:])


def get_stage_mesh(global_mesh: Mesh, stage_index: int) -> Mesh:
  num_physical_stages = global_mesh.shape["stage"]
  return slice_mesh(global_mesh, "stage", stage_index % num_physical_stages)


def sharding_with_mesh(s, mesh):
  assert isinstance(s, NamedSharding), f"expected NamedSharding, got: {s}"
  return NamedSharding(mesh, s.spec, memory_kind=s.memory_kind)


def adjust_to_stage_mesh(stage_mesh, shardings):
  return jax.tree.map(partial(sharding_with_mesh, mesh=stage_mesh), shardings)


SectionKind = Enum('SectionKind', 'Prologue Forward Backward Epilogue')
StageIndex = int
SectionName = tuple[SectionKind, StageIndex]


# Context provides the state for correctly transforming mmpp stages.
# We effectively execute the MmppTransformer in three variants:
#  1. The usual Flax way, entering via __call__. We only use this for model.init.
#  2. A first pass in mmpp.mpmd.transform to infer shardings and other metadata.
#  3. A second pass in mmpp.mpmd.transform to compile the stages separately.
# As part of these steps, Context modifies which mesh is used and whether
# the model is broken into separate jax.jits. In particular, 1. and 2. use only
# stage 0's mesh, but compile everything in a single jax.jit. Step 3. compiles
# wrap's each stage in separate jax.jit and uses the appropriate meshes.
@dataclasses.dataclass(frozen=True)
class Context:
  mesh: Mesh
  tracing_for_inference: bool
  section_fns: dict[SectionName, Callable] = dataclasses.field(default_factory=dict)
  section_decorator: Optional[Callable[[SectionName, Callable], Callable]] = None
  section_cache: dict[SectionName, Callable] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    assert "stage" in self.mesh.axis_names

  def get_stage_mesh(self, stage_index: int) -> Mesh:
    slice_index = 0 if self.tracing_for_inference else stage_index
    return get_stage_mesh(self.mesh, slice_index)

  def section(self, name: SectionName, **kwargs) -> Callable:
    """Annotates a section and returns the decorated version."""
    assert self.section_decorator
    assert name in self.section_fns, f"unknown section: {name}"
    section_fn = self.section_fns[name]
    if name not in self.section_cache:
      self.section_cache[name] = self.section_decorator(name, section_fn, **kwargs)
    return self.section_cache[name]

  def section_names(self) -> list[SectionName]:
    return list(self.section_fns.keys())


_mmpp_context: Optional[Context] = None

def get_context() -> Context:
  assert _mmpp_context is not None, \
    'Context unavailable. Are you calling from outside mmpp.mpmd.transform?'
  return _mmpp_context

def get_context_or_fallback(mesh) -> Context:
  if _mmpp_context is None:
    return Context(
        mesh=mesh,
        tracing_for_inference=True,
        section_fns={},
        section_decorator=None,
    )
  return _mmpp_context

@contextlib.contextmanager
def set_context(ctx: Context):
  global _mmpp_context
  old_ctx = _mmpp_context
  _mmpp_context = ctx
  try:
    yield
  finally:
    _mmpp_context = old_ctx


def is_jax_partial(x):
  # We carefully separate jax Partials from their data, so that
  # even when the function in the metadata changes due to re-tracing
  # we can specify in and out shardings via flattened pytrees. In this
  # setting the remaining Partials' pytree children are merely dummy
  # values. This allows us to eliminate the Partials from the sharding
  # prefix pytrees.
  return x is None or isinstance(x, jax._src.tree_util.Partial)


def sharding_extractor():
  def should_infer_sharding(x):
    try:
      aval = jax.core.get_aval(x)
      return type(aval) is jax.core.ShapedArray
    except TypeError:
      return False

  def store(shardings, index, sharding):
    shardings[index] = sharding

  def register_store_callbacks(xs):
    xs_flat, xs_tree = jax.tree.flatten(xs, is_leaf=is_jax_partial)
    shardings = [None] * len(xs_flat)
    for index, x in enumerate(xs_flat):
      if should_infer_sharding(x):
        inspect_array_sharding(x, callback=partial(store, shardings, index))
      else:
        # Note that we end up here for jax Partials (because we first treat them
        # as leaves and then decide not to infer a sharding for them). Simply
        # setting the inferred sharding to None means the Partials do not show
        # up in the inferred shardings; jax will treat these as replicated, but
        # all the data packaged in the Partial will have been replaced by dummy
        # values anyways (see vjp_unpack).
        shardings[index] = None
    return shardings, xs_tree

  # TODO: Add support for static args (e.g. via linear_util.WrappedFun)
  def dump_shardings(fun):
    in_shardings, in_tree = None, None
    out_shardings, out_tree = None, None
    @wraps(fun)
    def wrapper(*args):
      nonlocal in_shardings, in_tree
      nonlocal out_shardings, out_tree
      res = fun(*args)
      # NOTE: Shardings are captured only from the first invocation!
      if out_shardings is None:
        in_shardings, in_tree = register_store_callbacks(args)
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


def check_args_mesh(name, stage_mesh, args):
  bad = []
  def collect(path, x):
    if isinstance(x, jax.Array) and not (
      isinstance(x.sharding, NamedSharding) and
      (x_mesh := x.sharding.mesh) == stage_mesh
    ):
      bad.append((path, x.sharding._device_assignment))
  jax.tree.map_with_path(collect, args)
  if bad:
    bad_str = "\n  ".join(f"{path}: {devices}" for path, devices in bad)
    raise ValueError(
        f"expected arguments section {name} to match stage_mesh devices "
        f"({stage_mesh._flat_devices_tuple}) but got:\n  {bad_str}"
    )


def transform(
    mesh, section_fns, step_fn, step_in_shardings, step_out_shardings, example_inputs,
    print_inferred_shardings=False,
):
  # Phase 1: Infer shardings
  print('PHASE1')

  dump_shardings = sharding_extractor()
  in_shardings_thunk = {}
  out_shardings_thunk = {}

  def dump_section_shardings(section_name, section_fn, **kwargs):
    wrapped, in_shardings_thunk[section_name], out_shardings_thunk[section_name] = \
      dump_shardings(section_fn)
    return wrapped

  def make_shape_dtype(arr):
    return jax.ShapeDtypeStruct(
        arr.shape,
        arr.dtype,
        sharding=sharding_with_mesh(arr.sharding, stage0_mesh),
    )

  ctx = Context(
    mesh=mesh,
    tracing_for_inference=True,
    section_fns=section_fns,
    section_decorator=dump_section_shardings,
  )
  stage0_mesh = ctx.get_stage_mesh(0)
  with set_context(ctx):
    jax.jit(
        step_fn,
        in_shardings=adjust_to_stage_mesh(stage0_mesh, step_in_shardings),
        out_shardings=adjust_to_stage_mesh(stage0_mesh, step_out_shardings),
    ).lower(*jax.tree.map(make_shape_dtype, example_inputs)).compile()

  assert all(
    section_name in in_shardings_thunk and section_name in out_shardings_thunk
    for section_name in ctx.section_names()
  )
  section_in_shardings = {
    section_name: thunk()
    for section_name, thunk in in_shardings_thunk.items()
  }
  section_out_shardings = {
    section_name: thunk()
    for section_name, thunk in out_shardings_thunk.items()
  }
  del in_shardings_thunk
  del out_shardings_thunk

  if print_inferred_shardings:
    for section_name in ctx.section_names():
      ins = jax.tree.map(lambda x: x.spec, section_in_shardings[section_name])
      outs = jax.tree.map(lambda x: x.spec, section_out_shardings[section_name])
      print(f'  {section_name}:\n\t{ins=}\n\t{outs=}\n')

  # Phase 2: Produce final jitted sections
  print('PHASE2')

  def jit_with_shardings(
      section_name, section_fn, *, static_argnums=None, donate_argnums=None
  ):
    _, stage_index = section_name
    stage_mesh = get_stage_mesh(mesh, stage_index)
    in_shardings = adjust_to_stage_mesh(stage_mesh, section_in_shardings[section_name])
    out_shardings = adjust_to_stage_mesh(stage_mesh, section_out_shardings[section_name])
    jitted_section_fn = jax.jit(
        section_fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )
    @wraps(jitted_section_fn)
    def apply_stage(*args):
      check_args_mesh(section_name, stage_mesh, args)
      with stage_mesh:
        return jitted_section_fn(*args)
    return apply_stage

  ctx = Context(
    mesh=mesh,
    tracing_for_inference=False,
    section_fns=section_fns,
    section_decorator=jit_with_shardings,
  )
  @wraps(step_fn)
  def wrapper(*args):
    with set_context(ctx):
      return step_fn(*args)

  return wrapper
