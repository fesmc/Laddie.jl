_to_device(backend, a::AbstractArray) =
    (b = KA.allocate(backend, eltype(a), size(a)); copyto!(b, a); b)

# Concrete matrix type of `backend` at precision FT.
_matrix_type(backend, FT) = typeof(KA.allocate(backend, FT, 0, 0))

# The fields of `x`, with every matrix of the source type `A0` moved to `backend`.
# Integer masks, coordinate vectors, ranges, strings and scalars stay where they
# are, since the dispatch is on the source matrix type.
function _moved_fields(x, ::Type{A0}, backend) where {A0}
    mv(a::A0) = _to_device(backend, a)
    mv(a) = a
    return map(fn -> mv(getfield(x, fn)), fieldnames(typeof(x)))
end

# The matrix type on `backend` with the element type of the source matrices `A0`.
_moved_matrix_type(backend, ::Type{A0}) where {A0} = _matrix_type(backend, eltype(A0))

# Grid, Geometry and IOState have every type parameter as a field type, so their
# default constructors re-infer the parameters from the moved fields.
_grid_to_backend(g::Grid{FT,A0}, backend) where {FT,A0} =
    Grid(_moved_fields(g, A0, backend)...)

_geometry_to_backend(g::Geometry{A0}, backend) where {A0} =
    Geometry(_moved_fields(g, A0, backend)...)

_iostate_to_backend(io::IOState{A0}, backend) where {A0} =
    IOState(_moved_fields(io, A0, backend)...)

_var_to_backend(v::Var{LX,LY,A0}, backend) where {LX,LY,A0} =
    Var{LX,LY,_moved_matrix_type(backend, A0)}(_moved_fields(v, A0, backend)...)

_state_to_backend(s::State, backend) =
    State(map(fn -> _var_to_backend(getfield(s, fn), backend), fieldnames(State))...)

# The scheme-dependent slots (GamT, Conv2, PM) are scalars or matrices; the
# matrix ones follow the backend.
function _cache_to_backend(c::Cache{A0,G,C,P}, backend) where {A0,G,C,P}
    A = _moved_matrix_type(backend, A0)
    slot(T) = T <: AbstractArray ? A : T
    return Cache{A,slot(G),slot(C),slot(P)}(_moved_fields(c, A0, backend)...)
end

# Reconstruct a forcing struct with all float arrays moved to backend, so the
# kernels can read the ambient profiles and T_ice_base on the device.  The
# unparameterized constructor (typename wrapper) re-infers the type parameters
# from the moved arrays.
_forcing_to_backend(f::CavityForcing, backend) = CavityForcing(
    _forcing_to_backend(f.ocean, backend),
    _forcing_to_backend(f.ice, backend),
)

function _forcing_to_backend(
    f::F,
    backend,
) where {F<:Union{AbstractOceanForcing,AbstractIceForcing}}
    fields = map(fieldnames(F)) do fn
        v = getfield(f, fn)
        v isa AbstractArray && eltype(v) <: AbstractFloat ? _to_device(backend, v) : v
    end
    Base.typename(F).wrapper(fields...)
end

"""
$(TYPEDSIGNATURES)

Return a new model with all floating-point arrays transferred to `backend`.
The original model is not modified; always assign the result:

```julia
using CUDA
model = Model(Grid(mask, z_draft, dx, dy); forcing)
model = to_backend(model, CUDABackend())
sim = Simulation(model)
```

Prefer passing `backend` directly to `Grid` or `build_isomip` where possible —
it avoids the redundant CPU allocation:
```julia
sim = build_isomip(CUDABackend(); FT = Float32)
```
"""
function to_backend(m::Model, backend)
    new_grid = _grid_to_backend(getfield(m, :grid), backend)
    new_geometry = _geometry_to_backend(getfield(m, :geometry), backend)
    new_state = _state_to_backend(getfield(m, :state), backend)
    new_cache = _cache_to_backend(getfield(m, :cache), backend)
    new_forcing = _forcing_to_backend(getfield(m, :forcing), backend)
    Model(
        new_grid,
        new_geometry,
        new_state,
        new_cache,
        getfield(m, :params),
        getfield(m, :boundary),
        new_forcing,
    )
end

"""
$(TYPEDEF)

Run a simulation through [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl):
the time step is traced, compiled with XLA and executed in batches of steps, on the
GPU or the CPU that Reactant targets.  Requires `using Reactant, CUDA` (CUDA.jl is
needed even for the CPU target, since Reactant compiles the kernels through it).

```julia
using Laddie, Reactant, CUDA
sim = build_isomip(CPU(); FT = Float32)          # build and bootstrap on the CPU
rsim = to_backend(sim, ReactantBackend())        # move to Reactant
run!(rsim; days = 30)                            # compiled on the first call
```

Results agree with the KernelAbstractions backends to round-off, not bit for bit:
XLA reorders floating-point operations.

# Fields
$(TYPEDFIELDS)
"""
struct ReactantBackend{F}
    "how XLA may fuse the kernels of a step (see the Reactant docs page); `:auto` picks the measured best"
    fusion::F
end
ReactantBackend(; fusion = :auto) = ReactantBackend(fusion)

# Implemented by the Reactant extension: the KernelAbstractions backend that
# Reactant arrays report, and the batched execution.
function _reactant_ka_backend end
function _reactant_execution end
_reactant_ka_backend(::Any) = error(
    "ReactantBackend requires the Reactant extension: run `using Reactant, CUDA` first",
)
_reactant_execution(b) = _reactant_ka_backend(b)

to_backend(m::Model, b::ReactantBackend) = to_backend(m, _reactant_ka_backend(b))
_iostate_to_backend(io::IOState, b::ReactantBackend) =
    _iostate_to_backend(io, _reactant_ka_backend(b))
_execution(b::ReactantBackend) = _reactant_execution(b)
