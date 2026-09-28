_to_device(backend, a::AbstractArray) =
    (b = KA.allocate(backend, eltype(a), size(a)); copyto!(b, a); b)

# `x` with every field of its float matrix type `A` moved to `backend`.  Integer
# masks, coordinate vectors, ranges, strings and scalars stay where they are, and so
# do the scalar slots of the cache (`GamT`, `Conv2`, `PM` under some schemes).
_matrices_to_backend(x, ::Type{A}, backend) where {A} =
    _mapfields(y -> y isa A ? _to_device(backend, y) : y, x)

_grid_to_backend(g::Grid{FT,A}, backend) where {FT,A} = _matrices_to_backend(g, A, backend)
_geometry_to_backend(g::Geometry{A}, backend) where {A} = _matrices_to_backend(g, A, backend)
_cache_to_backend(c::Cache{A}, backend) where {A} = _matrices_to_backend(c, A, backend)

_state_to_backend(s::State, backend) =
    _mapfields(v -> _mapfields(a -> _to_device(backend, a), v), s)

# The accumulators of the time-averaged output, a NamedTuple of fields.
_iostate_to_backend(io::IOState, backend) =
    _mapfields(x -> x isa NamedTuple ? map(a -> _to_device(backend, a), x) : x, io)

# Every float array of the forcing, so the kernels can read the ambient profiles and
# T_ice_base on the device.
_forcing_to_backend(f::CavityForcing, backend) =
    _mapfields(x -> _forcing_to_backend(x, backend), f)
_forcing_to_backend(f::Union{AbstractOceanForcing,AbstractIceForcing}, backend) =
    _mapfields(f) do x
        x isa AbstractArray && eltype(x) <: AbstractFloat ? _to_device(backend, x) : x
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

With a `mesh` (a `Reactant.Sharding.Mesh`), the grid is split over several devices:
every field is sharded along `partition`, one mesh axis name (or `nothing`) per grid
axis, and XLA exchanges the halos between the devices.  The default `partition`
splits the second grid axis (contiguous slabs) over a one-axis mesh, and both grid
axes over a two-axis mesh.  Each split grid axis must be divisible by its number of
devices: crop with `MinRectangleDomainCropping(; multiple)` to round the grid up.
The native kernels of the GPU default cannot be split, so on a mesh `fusion = :auto`
raises them: `:kernel` over one mesh axis, `:xla` over two.

```julia
mesh = Reactant.Sharding.Mesh(collect(0:3), (:y,))          # 4 devices
crop = MinRectangleDomainCropping(; multiple = (1, 4))
grid = Grid(mask, z_draft, dx, dy; domain_cropping = crop)
rsim = to_backend(Simulation(Model(grid; forcing)), ReactantBackend(; mesh))
```

Results agree with the KernelAbstractions backends to round-off, not bit for bit:
XLA reorders floating-point operations.

# Fields
$(TYPEDFIELDS)
"""
struct ReactantBackend{F,M,P}
    "how XLA may fuse the kernels of a step (see the Reactant docs page); `:auto` picks the measured best"
    fusion::F
    "device mesh to shard the grid over, a `Reactant.Sharding.Mesh`; `nothing` for one device"
    mesh::M
    "mesh axis name (or `nothing`) each grid axis is split along; `nothing` for the default"
    partition::P
end
ReactantBackend(; fusion = :auto, mesh = nothing, partition = nothing) =
    ReactantBackend(fusion, mesh, partition)

# Implemented by the Reactant extension: what `to_backend` moves the arrays with (the
# KernelAbstractions backend that Reactant arrays report, or a sharded placement over
# the mesh), and the batched execution.
function _reactant_device end
function _reactant_execution end
_reactant_device(::Any) = error(
    "ReactantBackend requires the Reactant extension: run `using Reactant, CUDA` first",
)
_reactant_execution(b) = _reactant_device(b)

to_backend(m::Model, b::ReactantBackend) = to_backend(m, _reactant_device(b))
_iostate_to_backend(io::IOState, b::ReactantBackend) =
    _iostate_to_backend(io, _reactant_device(b))
_execution(b::ReactantBackend) = _reactant_execution(b)
