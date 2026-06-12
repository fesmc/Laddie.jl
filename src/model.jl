# Properties resolved before the field-forwarding chain in getproperty.
const _RESERVED_PROPS = (:io, :rc, :grid, :state, :cache, :params, :forcing, :FT, :ny, :nx)

# The flat forwarding layer resolves `m.field` by searching the sub-structs in
# a fixed order, so a field name appearing in two of them would be silently
# shadowed by whichever comes first.  Reject such configurations outright —
# this matters mostly for user-defined AbstractForcing types.  Called from the
# inner constructor so no construction path can bypass it.
function _check_property_collisions(io, rc, grid, state, cache, params, forcing)
    seen = Dict{Symbol, String}()
    for (label, x) in (
        ("Grid", grid), ("State", state), ("Cache", cache), ("Params", params),
        ("RunConfig", rc), (string(nameof(typeof(forcing))), forcing), ("IOState", io),
    )
        for fn in fieldnames(typeof(x))
            fn in _RESERVED_PROPS && error(
                "field `$fn` of $label collides with the reserved Model property `$fn`")
            haskey(seen, fn) && error(
                "Model property forwarding is ambiguous: field `$fn` exists in both $(seen[fn]) and $label")
            seen[fn] = label
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

The top-level model container.  Constructed by `build_isomip`; advanced by `run!`.

`A` is the concrete matrix type (`Matrix{FT}` on CPU, `CuArray{FT,2}` on GPU).
Use `to_backend(m, backend)` to obtain a model on a different backend —
it returns a new `Model` with the appropriate `A`.

Fields are accessed directly on `m` through a flat forwarding layer:

| Access pattern | Source struct | Examples |
|----------------|--------------|---------|
| `m.D`, `m.U`, `m.V`, `m.T`, `m.S` | `State` | `m.D.present`, `m.U.past` |
| `m.melt`, `m.entr`, `m.drho`, `m.Ta`, `m.Sa`, … | `Cache` | `m.melt .* spy` |
| `m.tmask`, `m.zb`, `m.dx`, `m.dy`, … | `Grid` | `m.tmask .> 0` |
| `m.dt`, `m.f`, `m.Cd`, `m.Ah`, `m.minD`, … | `Params` | `m.dt` |
| `m.name`, `m.saveday`, `m.save_D`, … | `RunConfig` | `m.rc.saveday` |
| `m.Tz`, `m.Sz`, `m.z` | Forcing | ambient profile arrays |
| `m.t`, `m.count`, `m.rundir`, `m.x`, `m.Dav`, … | `IOState` | runtime I/O state |
`m.FT` returns the floating-point type (`Float64` or `Float32`).

`Grid` and `Params` are immutable after construction.  `Cache`, `State`, and
`IOState` fields are mutable and updated in place each time step.
"""
mutable struct Model{
    FT,
    A<:AbstractMatrix{FT},
    F<:AbstractForcing,
    EP<:AbstractEntrainmentParam,
    MP<:AbstractMeltParam,
    CS<:AbstractConvectionScheme,
    OB<:AbstractOpenBoundary,
    GL<:AbstractGroundingLineBC,
    C<:Cache,
}
    io     :: IOState{FT, A}
    rc     :: RunConfig
    grid   :: Grid{FT, A}
    state  :: State{FT, A}
    cache  :: C
    params :: Params{FT, EP, MP, CS, OB, GL}
    forcing:: F

    function Model{FT, A, F, EP, MP, CS, OB, GL, C}(
        io, rc, grid, state, cache, params, forcing,
    ) where {
        FT, A<:AbstractMatrix{FT}, F<:AbstractForcing,
        EP<:AbstractEntrainmentParam, MP<:AbstractMeltParam,
        CS<:AbstractConvectionScheme, OB<:AbstractOpenBoundary,
        GL<:AbstractGroundingLineBC, C<:Cache,
    }
        _check_property_collisions(io, rc, grid, state, cache, params, forcing)
        new{FT, A, F, EP, MP, CS, OB, GL, C}(io, rc, grid, state, cache, params, forcing)
    end
end

function Model(
    io     ::IOState{FT, A},
    rc     ::RunConfig,
    grid   ::Grid{FT, A},
    state  ::State{FT, A},
    cache  ::C,
    params ::Params{FT, EP, MP, CS, OB, GL},
    forcing::F,
) where {FT, A, C<:Cache, F<:AbstractForcing, EP, MP, CS, OB, GL}
    Model{FT, A, F, EP, MP, CS, OB, GL, C}(io, rc, grid, state, cache, params, forcing)
end

function Base.getproperty(m::Model{FT}, k::Symbol) where {FT}
    # Direct struct fields — fast path
    k === :io      && return getfield(m, :io)
    k === :rc      && return getfield(m, :rc)
    k === :grid    && return getfield(m, :grid)
    k === :state   && return getfield(m, :state)
    k === :cache   && return getfield(m, :cache)
    k === :params  && return getfield(m, :params)
    k === :forcing && return getfield(m, :forcing)
    k === :FT      && return FT
    # Interior dimensions derived from grid (total minus 2 border cells)
    k === :ny && return getfield(m, :grid).Ny - 2
    k === :nx && return getfield(m, :grid).Nx - 2
    # Grid: geometry, masks, stagger denominators
    g = getfield(m, :grid)
    hasfield(typeof(g), k) && return getfield(g, k)
    # State: prognostic Var objects
    s = getfield(m, :state)
    hasfield(typeof(s), k) && return getfield(s, k)
    # Cache: mutable scratch / diagnostic arrays
    c = getfield(m, :cache)
    hasfield(typeof(c), k) && return getfield(c, k)
    # Params: physical constants + parameterization objects
    p = getfield(m, :params)
    hasfield(typeof(p), k) && return getfield(p, k)
    # RunConfig: static run + I/O configuration
    r = getfield(m, :rc)
    hasfield(typeof(r), k) && return getfield(r, k)
    # Forcing: ambient T/S profiles on the uniform z-grid
    f = getfield(m, :forcing)
    hasfield(typeof(f), k) && return getfield(f, k)
    # IOState: runtime I/O state (counters, accumulators, coordinates)
    io = getfield(m, :io)
    hasfield(typeof(io), k) && return getfield(io, k)
    error("Model has no property `$k`")
end

function Base.setproperty!(m::Model, k::Symbol, v)
    # Cache is mutable — all physics scratch arrays live here
    c = getfield(m, :cache)
    if hasfield(typeof(c), k)
        setfield!(c, k, v)
        return
    end
    # State is mutable — D/U/V/T/S Var objects can be replaced
    s = getfield(m, :state)
    if hasfield(typeof(s), k)
        setfield!(s, k, v)
        return
    end
    # IOState: runtime I/O counters, paths, and accumulators
    io = getfield(m, :io)
    if hasfield(typeof(io), k)
        setfield!(io, k, convert(fieldtype(typeof(io), k), v))
        return
    end
    error("Model has no settable property `$k`")
end