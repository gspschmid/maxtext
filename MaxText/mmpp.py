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
      section_fn: Callable,
      **kwargs,
  ) -> Callable:
    """Annotates a section and caches the resulting function."""
    assert self.section_decorator
    if name not in self.section_cache:
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
  def should_infer_sharding(x):
    try:
      aval = jax.core.get_aval(x)
      return type(aval) is jax.core.ShapedArray
    except TypeError:
      return False

  def is_not_jax_partial(x):
    # We carefully separate jax Partials from their data, so that
    # even when the function in the metadata changes due to re-tracing
    # we can specify in and out shardings via flattened pytrees. In this
    # setting the remaining Partials' pytree children are merely dummy
    # values. This allows us to eliminate the Partials from the sharding
    # prefix pytrees.
    return x is None or isinstance(x, jax._src.tree_util.Partial)

  def store(shardings, index, sharding):
    shardings[index] = sharding

  def register_store_callbacks(xs):
    xs_flat, xs_tree = jax.tree.flatten(xs, is_leaf=is_not_jax_partial)
    shardings = [None] * len(xs_flat)
    for index, x in enumerate(xs_flat):
      if should_infer_sharding(x):
        inspect_array_sharding(x, callback=functools.partial(store, shardings, index))
      else:
        shardings[index] = None
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

  # # Imposes shardings on the inputs and output of fun.
  # #
  # # We need this because jax.jit(fun, out_shardings=...) doesn't quite do what
  # # we want: we expect forward funs to return a vjp wrapper that packages the
  # # corresponding backward function. But out_shardings is tree-mapped against
  # # fun's actual output, so the pytrees *must* have the exact same metadata.
  # # This seems infeasible, since we infer out_shardings from a first tracing of
  # # fun and then want to impose it via jit, which will re-trace and thus always
  # # produce distinct metadata. The analogous problem occurs with in_shardings
  # # when jitting the backward function.
  # #
  # # Instead we use the fact that the shardings resulted from tracing the same
  # # functions, so the flattened shardings are correct -- it's just the pytree
  # # metadata we need to discard. Hence our workaround is to rebuild the
  # # shardings pytree using the re-traced functions in_tree and out_tree.
  # #
  # # Note that {in,out}_shardings cannot merely be prefixes of the actual inputs
  # # and outputs as jax.jit usually allows.
  # def with_in_out_shardings(fun, in_shardings, out_shardings):
  #   @functools.wraps(fun)
  #   def wrapper(*args):
  #     in_flat, in_tree = jax.tree.flatten(args)
  #     in_shardings_flat, _ = jax.tree.flatten(in_shardings)
  #     # I'm feeling lucky.
  #     assert len(in_flat) == len(in_shardings_flat)
  #     in_flat = [
  #       jax.lax.with_sharding_constraint(inval, in_sharding)
  #       for inval, in_sharding in zip(in_flat, in_shardings_flat)
  #     ]
  #     args = jax.tree.unflatten(in_tree, in_flat)

  #     out = fun(*args)

  #     out_flat, out_tree = jax.tree.flatten(out)
  #     out_shardings_flat, _ = jax.tree.flatten(out_shardings)
  #     # I'm feeling lucky.
  #     assert len(out_flat) == len(out_shardings_flat)
  #     out_flat = [
  #       jax.lax.with_sharding_constraint(outval, out_sharding)
  #       for outval, out_sharding in zip(out_flat, out_shardings_flat)
  #     ]
  #     out = jax.tree.unflatten(out_tree, out_flat)
  #     return out
  #   return wrapper

  def jit_with_shardings(section_name, section_fn, *, static_argnums=()):
    # # return section_fn
    # # TODO: donate_argnums?
    # # section_fn.__name__ = f"section_{section_name}"
    # section_fn = with_in_out_shardings(
    #     section_fn,
    #     in_shardings[section_name],
    #     out_shardings[section_name],
    # )
    _in_shardings = in_shardings[section_name]
    _out_shardings = out_shardings[section_name]
    # # TODO: Remove this manual plumbing. Replace by utilities in fwd and bwd
    # # that does something like `res[1] = vjp_unpack(res[1])` in the forward
    # # and `args[0] = vjp_pack(args[0])` and replaces the sharding pytree for
    # # the first component by `None``.
    # if section_name.startswith("forward"):
    #   _out_shardings = _out_shardings[:1] + (None,) + _out_shardings[2:]
    # if section_name.startswith("backward"):
    #   _in_shardings = (None,) + _in_shardings[1:]
    return jax.jit(
        section_fn,
        in_shardings=_in_shardings,
        out_shardings=_out_shardings,
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
