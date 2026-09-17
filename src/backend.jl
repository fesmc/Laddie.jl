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

_grid_to_backend(g::Grid{FT,A0}, backend) where {FT,A0} =
    Grid{FT,_matrix_type(backend, FT)}(_moved_fields(g, A0, backend)...)

_geometry_to_backend(g::Geometry{FT,A0}, backend) where {FT,A0} =
    Geometry{FT,_matrix_type(backend, FT)}(_moved_fields(g, A0, backend)...)

_iostate_to_backend(io::IOState{FT,A0}, backend) where {FT,A0} =
    IOState{FT,_matrix_type(backend, FT)}(_moved_fields(io, A0, backend)...)

_var_to_backend(v::Var{LX,LY,FT,A0}, backend) where {LX,LY,FT,A0} =
    Var{LX,LY,FT,_matrix_type(backend, FT)}(_moved_fields(v, A0, backend)...)

_state_to_backend(s::State{FT}, backend) where {FT} = State{FT,_matrix_type(backend, FT)}(
    map(fn -> _var_to_backend(getfield(s, fn), backend), fieldnames(State))...,
)

# The scheme-dependent slots (GamT, Conv2, PM) are scalars or matrices; the
# matrix ones follow the backend.
function _cache_to_backend(c::Cache{FT,A0,G,C,P}, backend) where {FT,A0,G,C,P}
    A = _matrix_type(backend, FT)
    slot(T) = T <: AbstractArray ? A : T
    return Cache{FT,A,slot(G),slot(C),slot(P)}(_moved_fields(c, A0, backend)...)
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
