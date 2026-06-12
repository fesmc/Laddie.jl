
# ============================================================================
# Backend transfer
# ============================================================================

_to_device(backend, a::AbstractArray) =
    (b = KA.allocate(backend, eltype(a), size(a)); copyto!(b, a); b)

# Reconstruct an immutable Grid{FT, A} with all float arrays moved to backend.
# The integer mask field is left on CPU (used for host-side branching only).
# Dispatches mv on the source array type A0 so scalars and Matrix{Int} pass through.
function _grid_to_backend(g::Grid{FT, A0}, backend) where {FT, A0}
    A = typeof(_to_device(backend, g.tmask))
    mv(a::A0) = _to_device(backend, a)
    mv(a) = a
    Grid{FT, A}(map(fn -> mv(getfield(g, fn)), fieldnames(typeof(g)))...)
end

# Reconstruct a forcing struct with all float vectors moved to backend.
# Required so update_ambient_fields! can index Tz/Sz with GPU index arrays.
function _forcing_to_backend(f::F, backend) where {F<:AbstractForcing}
    fields = map(fieldnames(F)) do fn
        v = getfield(f, fn)
        v isa AbstractVector && eltype(v) <: AbstractFloat ? _to_device(backend, v) : v
    end
    F(fields...)
end

function _var_to_backend(v::Var{LX, LY, FT, A0}, backend) where {LX, LY, FT, A0}
    A = typeof(_to_device(backend, v.past))
    Var{LX, LY, FT, A}(
        _to_device(backend, v.past),
        _to_device(backend, v.present),
        _to_device(backend, v.future),
    )
end

function _state_to_backend(s::State{FT, A0}, backend) where {FT, A0}
    D = _var_to_backend(s.D, backend)
    State{FT, typeof(D.past)}(
        D,
        _var_to_backend(s.U, backend),
        _var_to_backend(s.V, backend),
        _var_to_backend(s.T, backend),
        _var_to_backend(s.S, backend),
    )
end

# Reconstruct IOState with accumulator matrices moved to backend; counters,
# strings, and the CPU-resident x/y coordinate vectors pass through unchanged.
function _iostate_to_backend(io::IOState{FT, A0}, backend) where {FT, A0}
    A = typeof(KA.allocate(backend, FT, 0, 0))
    mv(a::A0) = _to_device(backend, a)
    mv(a) = a
    IOState{FT, A}(map(fn -> mv(getfield(io, fn)), fieldnames(typeof(io)))...)
end

# Dispatches mv on the source matrix type A0; scalars (FT or Int) fall through.
function _cache_to_backend(c::Cache{FT, A0, GamT0, Conv2_0}, backend) where {FT, A0, GamT0, Conv2_0}
    A     = typeof(_to_device(backend, c.melt))
    GamT  = GamT0  <: AbstractArray ? A : GamT0
    Conv2 = Conv2_0 <: AbstractArray ? A : Conv2_0
    mv(a::A0) = _to_device(backend, a)
    mv(a) = a
    Cache{FT, A, GamT, Conv2}(map(fn -> mv(getfield(c, fn)), fieldnames(typeof(c)))...)
end

"""
$(TYPEDSIGNATURES)

Return a new model with all floating-point arrays transferred to `backend`.
The original model is not modified; always assign the result:

```julia
using CUDA
m = build_isomip()
m = to_backend(m, CUDABackend())
run!(m; days = 5.0)
```

Prefer passing `backend` directly to `build_isomip` where possible — it
avoids the redundant CPU allocation:
```julia
m = build_isomip(CUDABackend(); FT = Float32)
```
"""
function to_backend(m::Model, backend)
    new_io      = _iostate_to_backend(getfield(m, :io),   backend)
    new_grid    = _grid_to_backend(getfield(m, :grid),    backend)
    new_state   = _state_to_backend(getfield(m, :state),  backend)
    new_cache   = _cache_to_backend(getfield(m, :cache),  backend)
    new_forcing = _forcing_to_backend(getfield(m, :forcing), backend)
    Model(new_io, getfield(m, :rc), new_grid, new_state, new_cache,
          getfield(m, :params), new_forcing)
end

# Backward-compatible alias — returns a NEW Model; assign the result:
#   m = to_backend!(m, backend)
to_backend!(m::Model, backend) = to_backend(m, backend)
