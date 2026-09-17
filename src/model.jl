# Properties resolved before the field-forwarding chain in getproperty.
const _RESERVED_PROPS =
    (:grid, :geometry, :state, :cache, :params, :boundary, :forcing, :FT, :ny, :nx)

# The flat forwarding layer resolves `m.field` by searching the sub-structs in
# a fixed order, so a field name appearing in two of them would be silently
# shadowed by whichever comes first.  Reject such configurations outright —
# this matters mostly for user-defined forcing types.  The `CavityForcing` wrapper
# itself is transparent: forwarding descends into its `ocean` and `ice` members, so
# those are what have to be collision-free, not the wrapper.  Called from the inner
# constructor so no construction path can bypass it.
function _check_property_collisions(grid, geometry, state, cache, params, forcing)
    seen = Dict{Symbol,String}()
    for (label, x) in (
        ("Grid", grid),
        ("Geometry", geometry),
        ("State", state),
        ("Cache", cache),
        ("Params", params),
        (string(nameof(typeof(forcing.ocean))), forcing.ocean),
        (string(nameof(typeof(forcing.ice))), forcing.ice),
    )
        for fn in fieldnames(typeof(x))
            fn in _RESERVED_PROPS && error(
                "field `$fn` of $label collides with the reserved Model property `$fn`",
            )
            haskey(seen, fn) && error(
                "Model property forwarding is ambiguous: field `$fn` exists in both $(seen[fn]) and $label",
            )
            seen[fn] = label
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

The model: geometry, physics and prognostic state.  It knows nothing about time
integration — wrap it in a [`Simulation`](@ref) to advance it with `run!`.

`A` is the concrete matrix type (`Matrix{FT}` on CPU, `CuArray{FT,2}` on GPU).
Use `to_backend(m, backend)` to obtain a model on a different backend —
it returns a new `Model` with the appropriate `A`.

Fields are accessed directly on `m` through a flat forwarding layer:

| Access pattern | Source struct | Examples |
|----------------|--------------|---------|
| `m.D`, `m.U`, `m.V`, `m.T`, `m.S` | `State` | `m.D.present`, `m.U.past` |
| `m.melt`, `m.entr`, `m.drho`, `m.Ta`, `m.Sa`, … | `Cache` | `m.melt .* m.seconds_per_year` |
| `m.mask`, `m.z_draft`, `m.dx`, `m.x`, … | `Grid` | raw mask, draft, spacing, coordinates |
| `m.tmask`, `m.umask`, `m.resolved_mask`, `m.dzdx`, … | `Geometry` | `m.tmask .> 0` |
| `m.C_d`, `m.A_h`, `m.D_min`, … | `Params` | `m.C_d` |
| `m.f`, `m.fu`, `m.fv` | `Geometry` | Coriolis at T-, u- and v-points |
| `m.boundary` | `BoundaryConditions` | `m.boundary.land` (not flattened) |
| `m.Tz`, `m.Sz`, `m.z` | Forcing (ocean) | ambient profile arrays |
| `m.T_ice_base` | Forcing (ice) | basal ice temperature field |
`m.FT` returns the floating-point type (`Float64` or `Float32`).

`Grid`, `Geometry` and `Params` are immutable after construction.  `Cache` and
`State` fields are mutable and updated in place each time step.
"""
mutable struct Model{
    FT,
    A<:AbstractMatrix{FT},
    F<:CavityForcing,
    P<:Params{FT},
    B<:BoundaryConditions,
    C<:Cache,
}
    grid::Grid{FT,A}
    geometry::Geometry{FT,A}
    state::State{FT,A}
    cache::C
    params::P
    boundary::B
    forcing::F

    function Model{FT,A,F,P,B,C}(
        grid,
        geometry,
        state,
        cache,
        params,
        boundary,
        forcing,
    ) where {
        FT,
        A<:AbstractMatrix{FT},
        F<:CavityForcing,
        P<:Params{FT},
        B<:BoundaryConditions,
        C<:Cache,
    }
        _check_property_collisions(grid, geometry, state, cache, params, forcing)
        new{FT,A,F,P,B,C}(grid, geometry, state, cache, params, boundary, forcing)
    end
end

function Model(
    grid::Grid{FT,A},
    geometry::Geometry{FT,A},
    state::State{FT,A},
    cache::C,
    params::P,
    boundary::B,
    forcing::F,
) where {FT,A,C<:Cache,F<:CavityForcing,P<:Params{FT},B<:BoundaryConditions}
    Model{FT,A,F,P,B,C}(grid, geometry, state, cache, params, boundary, forcing)
end

function Base.getproperty(m::Model{FT}, k::Symbol) where {FT}
    # Direct struct fields — fast path
    k === :grid && return getfield(m, :grid)
    k === :geometry && return getfield(m, :geometry)
    k === :state && return getfield(m, :state)
    k === :cache && return getfield(m, :cache)
    k === :params && return getfield(m, :params)
    k === :boundary && return getfield(m, :boundary)
    k === :forcing && return getfield(m, :forcing)
    k === :FT && return FT
    # Interior dimensions derived from grid (total minus 2 border cells)
    k === :ny && return getfield(m, :grid).Ny - 2
    k === :nx && return getfield(m, :grid).Nx - 2
    # Grid: cell layout, raw mask, draft, bed, coordinates
    g = getfield(m, :grid)
    hasfield(typeof(g), k) && return getfield(g, k)
    # Geometry: resolved mask, derived masks, stagger denominators, slope, Coriolis
    gm = getfield(m, :geometry)
    hasfield(typeof(gm), k) && return getfield(gm, k)
    # State: prognostic Var objects
    s = getfield(m, :state)
    hasfield(typeof(s), k) && return getfield(s, k)
    # Cache: mutable scratch / diagnostic arrays
    c = getfield(m, :cache)
    hasfield(typeof(c), k) && return getfield(c, k)
    # Params: physical constants + parameterization objects
    p = getfield(m, :params)
    hasfield(typeof(p), k) && return getfield(p, k)
    # Forcing: ambient T/S profiles on the uniform z-grid, then the ice state.
    # `CavityForcing` is a container, not a namespace — `m.ocean` / `m.ice` are not
    # forwarded, only the members' own fields.
    f = getfield(m, :forcing)
    fo = getfield(f, :ocean)
    hasfield(typeof(fo), k) && return getfield(fo, k)
    fi = getfield(f, :ice)
    hasfield(typeof(fi), k) && return getfield(fi, k)
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
    error("Model has no settable property `$k`")
end
