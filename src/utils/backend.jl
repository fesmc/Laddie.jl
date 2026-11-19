# ============================================================================
# Kernel launches
# ============================================================================

# Launch `kernel!(out, args...)` over the full extent of `out`, on its backend.  Every
# kernel writes its first argument, and every array has the grid's size.
# CPU: long blocks along the first (column-major inner) index keep the inner loops
# vectorisable, and a grid smaller than one block runs on a single task instead of
# paying the spawn overhead — measured 1.1–1.8× faster than (8, 8) from 80×20 to
# 1000×1000 cells on 1–16 threads.
_workgroup(::CPU) = (128, 32)
_workgroup(::Any) = (32, 8)   # GPU: 256 threads, warp-aligned x-dimension

launch!(kernel!, out, args...) = launch_range!(kernel!, size(out), out, args...)

# Launch a stencil kernel over the interior cells only, the full array minus its
# one-cell border ring.  The kernel offsets its index by one past the ring, so
# `i ± 1`, `j ± 1` always stay inside the array and need no periodic wrap: the
# border ring is never active, and a stencil output there was only ever multiplied
# by a zero mask.  Plain `i ± 1` also keeps the indices affine, which lets Reactant
# raise the neighbour reads to slices; for the wrapped reads that remain, see the
# forms Reactant handles at `_xp1`.
launch_interior!(kernel!, out, args...) = launch_range!(kernel!, size(out) .- 2, out, args...)

# Launch over `ndrange`, a block of `out` that the kernel places itself.  `launch!`
# and `launch_interior!` go through here, and so does every backend method (the
# Reactant extension's).
function launch_range!(kernel!, ndrange, out, args...)
    backend = _launch_backend(KA.get_backend(out))
    kernel!(backend, _workgroup(backend))(out, args...; ndrange)
    return nothing
end

# CPU: static thread assignment (`@threads :static` instead of one `@spawn` per
# chunk), so the same chunk of the grid lands on the same thread in every kernel
# and its arrays stay in that core's cache.  With dynamic assignment the chunks
# migrate between cores from kernel to kernel; on the 241×466 ASE grid that made
# 2 threads no faster than 1, and 8 threads only 1.7× (3.3× static, with the
# threads pinned).  `:static` cannot run inside another threaded region, so a
# launch from within `@threads` falls back to dynamic assignment.
_launch_backend(backend) = backend
_launch_backend(backend::CPU) =
    ccall(:jl_in_threaded_region, Cint, ()) == 0 ? CPU(; static = true) : backend

# ============================================================================
# Moving models between backends
# ============================================================================

_to_device(backend, a::AbstractArray) =
    (b = KA.allocate(backend, eltype(a), size(a)); copyto!(b, _flush_subnormals(a)); b)
# To the host, the array is copied first and flushed there: the flush's `map` would read a
# device array element by element, which Reactant's arrays refuse (the CPU copy that
# `meltstats`, the log line, the output and the restarts of a Reactant run read).
_to_device(::CPU, a::AbstractArray) = _flush_subnormals(Array(a))

# Subnormal floats set to zero.  A velocity at rest decays geometrically under drag
# and, in Float32 on CUDA (which keeps subnormals), can end up at ~1e-42 rather than
# 0 (pan-Antarctic restarts at front corners).  XLA flushes subnormals to zero on the
# GPU while `_safe_div` and `_safe_sqrt` test for an exact zero, so such a value makes
# the reverse-mode gradient NaN.  Physically they are zero.
_flush_subnormals(a::AbstractArray{T}) where {T<:AbstractFloat} =
    map(x -> ifelse(abs(x) < floatmin(T), zero(T), x), a)
_flush_subnormals(a) = a

# `x` with every field of its float matrix type `A` moved to `backend`.  Integer
# masks, coordinate vectors, ranges, strings and scalars stay where they are, and so
# do the scalar slots of the cache (`GamT`, `Conv2`, `PM` under some schemes).
_matrices_to_backend(x, ::Type{A}, backend) where {A} =
    _mapfields(y -> y isa A ? _to_device(backend, y) : y, x)

_grid_to_backend(g::Grid{FT,A}, backend) where {FT,A} = _matrices_to_backend(g, A, backend)
_geometry_to_backend(g::Geometry{A}, backend) where {A} = _matrices_to_backend(g, A, backend)
_cache_to_backend(c::Cache{A}, backend) where {A} = _matrices_to_backend(c, A, backend)

# Moves that are the identity on the CPU, where grids and models are first built.
_grid_on(grid, ::CPU) = grid
_grid_on(grid, backend) = _grid_to_backend(grid, backend)
_cpu_grid(grid, ::CPU) = grid
_cpu_grid(grid, backend) = _grid_to_backend(grid, CPU())
_model_on(m, ::CPU) = m
_model_on(m, backend) = to_backend(m, backend)

_state_to_backend(s::State, backend) =
    _mapfields(v -> _mapfields(a -> _to_device(backend, a), v), s)

# The accumulators of the time-averaged output, a NamedTuple of fields.
_iostate_to_backend(io::IOState, backend) =
    _mapfields(x -> x isa NamedTuple ? map(a -> _to_device(backend, a), x) : x, io)

# Every non-integer real array of the forcing (floats, and ForwardDiff `Dual`s, which
# are `Real` but not `AbstractFloat`), so the kernels can read the ambient profiles
# and T_ice_farfield on the device.
_forcing_to_backend(f::CavityForcing, backend) =
    _mapfields(x -> _forcing_to_backend(x, backend), f)
_forcing_to_backend(f::Union{AbstractOceanForcing,AbstractIceForcing}, backend) =
    _mapfields(f) do x
        x isa AbstractArray && eltype(x) <: Real && !(eltype(x) <: Integer) ?
        _to_device(backend, x) : x
    end
# The basin map is an integer array the ambient kernel reads, so it moves too.
_forcing_to_backend(f::PerBasinOceanForcing1D, backend) =
    _mapfields(x -> x isa AbstractArray ? _to_device(backend, x) : x, f)
# Both ends of a ramp move; the moved model gets its own weight.
_forcing_to_backend(f::RampedOceanForcing, backend) = RampedOceanForcing(
    _forcing_to_backend(f.from, backend), _forcing_to_backend(f.to, backend), f.start, f.duration, Ref(f.weight[]))

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
    return Model(
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

With a `mesh` (a `Reactant.Sharding.Mesh`) the grid is split over several devices along
`partition`; each split axis must be divisible by its number of devices. Results agree with
the KernelAbstractions backends to round-off. See [Reactant backend](@ref).

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

# The time series of shelf means are not compiled into the Reactant step loop.
_check_series_backend(output, backend) = nothing
_check_series_backend(output, ::ReactantBackend) =
    output.tsday === nothing || throw(
        ArgumentError(
            "time series (OutputConfig tsday = $(output.tsday)) are not available under " *
            "ReactantBackend; leave tsday at its default `nothing`",
        ),
    )
