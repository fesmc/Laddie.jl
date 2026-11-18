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

# One type parameter per field, with no element type shared between them: each
# component can then change its array type on its own (a Reactant trace turns the
# matrices into traced arrays but leaves the scalar `Params` as they are).
"""
$(TYPEDEF)

The model: geometry, physics and prognostic state.  It knows nothing about time
integration — wrap it in a [`Simulation`](@ref) to advance it with `run!`.

The fields are `Matrix{FT}` on CPU and `CuArray{FT,2}` on GPU.
Use `to_backend(m, backend)` to obtain a model on a different backend —
it returns a new `Model` with the matching array types.

Fields are accessed directly on `m` through a flat forwarding layer:

| Access pattern | Source struct | Examples |
|----------------|--------------|---------|
| `m.H`, `m.U`, `m.V`, `m.T`, `m.S` | `State` | `m.H.present`, `m.U.past` |
| `m.melt`, `m.entr`, `m.drho`, `m.Ta`, `m.Sa`, … | `Cache` | `m.melt .* m.seconds_per_year` |
| `m.mask`, `m.z_draft`, `m.dx`, `m.x`, … | `Grid` | input mask, draft, spacing, coordinates |
| `m.tmask`, `m.umask`, `m.resolved_mask`, `m.dzdx`, … | `Geometry` | `m.tmask .> 0` |
| `m.C_d`, `m.A_h`, `m.H_min`, … | `Params` | `m.C_d` |
| `m.f`, `m.fu`, `m.fv` | `Geometry` | Coriolis at T-, u- and v-points |
| `m.boundary` | `BoundaryConditions` | `m.boundary.land` (not flattened) |
| `m.Tz`, `m.Sz`, `m.z` | Forcing (ocean) | ambient profile arrays |
| `m.T_ice_base` | Forcing (ice) | ice temperature above the basal boundary layer (see `PrescribedIceForcing`) |

`m.FT` returns the floating-point type (`Float64` or `Float32`) of the scalar
parameters, which `Model` checks against the grid and forcing, and `m.nx`, `m.ny`
the interior cell counts.

`Grid`, `Geometry` and `Params` are immutable after construction.  `Cache` and
`State` fields are mutable and updated in place each time step.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Model{
    G<:Grid,
    GE<:Geometry,
    S<:State,
    C<:Cache,
    P<:Params,
    B<:BoundaryConditions,
    F<:CavityForcing,
}
    "the [`Grid`](@ref): cell layout, mask, ice draft, bed and coordinates"
    grid::G
    "masks, wall indicators, ice-base slope and Coriolis field derived from the grid"
    geometry::GE
    "the prognostic fields `H`, `U`, `V`, `T`, `S`"
    state::S
    "diagnostic and scratch fields, updated in place every step"
    cache::C
    "physical constants and parameterisation choices ([`Params`](@ref))"
    params::P
    "boundary conditions ([`BoundaryConditions`](@ref))"
    boundary::B
    "ocean and ice forcing ([`CavityForcing`](@ref))"
    forcing::F

    function Model{G,GE,S,C,P,B,F}(
        grid,
        geometry,
        state,
        cache,
        params,
        boundary,
        forcing,
    ) where {
        G<:Grid,
        GE<:Geometry,
        S<:State,
        C<:Cache,
        P<:Params,
        B<:BoundaryConditions,
        F<:CavityForcing,
    }
        _check_property_collisions(grid, geometry, state, cache, params, forcing)
        return new{G,GE,S,C,P,B,F}(grid, geometry, state, cache, params, boundary, forcing)
    end
end

Model(
    grid::G,
    geometry::GE,
    state::S,
    cache::C,
    params::P,
    boundary::B,
    forcing::F,
) where {G,GE,S,C,P,B,F} =
    Model{G,GE,S,C,P,B,F}(grid, geometry, state, cache, params, boundary, forcing)

function Base.getproperty(m::Model, k::Symbol)
    # Direct struct fields — fast path
    k === :grid && return getfield(m, :grid)
    k === :geometry && return getfield(m, :geometry)
    k === :state && return getfield(m, :state)
    k === :cache && return getfield(m, :cache)
    k === :params && return getfield(m, :params)
    k === :boundary && return getfield(m, :boundary)
    k === :forcing && return getfield(m, :forcing)
    k === :FT && return _float_type(getfield(m, :params))
    # Interior dimensions derived from grid (total minus 2 border cells)
    k === :ny && return getfield(m, :grid).Ny - 2
    k === :nx && return getfield(m, :grid).Nx - 2
    # Grid: cell layout, raw mask, draft, bed, coordinates
    g = getfield(m, :grid)
    hasfield(typeof(g), k) && return getfield(g, k)
    # Geometry: resolved mask, derived masks, wall indicators, slope, Coriolis
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

# Everything `getproperty` answers to, in its lookup order, so that REPL completion
# shows the forwarded names too.
function Base.propertynames(m::Model, ::Bool = false)
    names(x) = fieldnames(typeof(x))
    f = getfield(m, :forcing)
    return (
        fieldnames(Model)...,
        :FT,
        :nx,
        :ny,
        names(getfield(m, :grid))...,
        names(getfield(m, :geometry))...,
        names(getfield(m, :state))...,
        names(getfield(m, :cache))...,
        names(getfield(m, :params))...,
        names(getfield(f, :ocean))...,
        names(getfield(f, :ice))...,
    )
end

function Base.setproperty!(m::Model, k::Symbol, v)
    # Cache is mutable — all physics scratch arrays live here
    c = getfield(m, :cache)
    if hasfield(typeof(c), k)
        setfield!(c, k, v)
        return
    end
    # State is mutable — H/U/V/T/S Var objects can be replaced
    s = getfield(m, :state)
    if hasfield(typeof(s), k)
        setfield!(s, k, v)
        return
    end
    error("Model has no settable property `$k`")
end

# ============================================================================
# Model constructor: validation, cropping and assembly
# ============================================================================

_float_type(::Params{FT}) where {FT} = _value_type(FT)
# The number type a scalar parameter holds: the type itself, except for a scalar
# traced by Reactant (see `trace_parameters`), whose extension returns the float
# type it wraps.
_value_type(::Type{T}) where {T} = T
# The type the parameters are stored as (the Reactant number type once traced).
_scalar_type(::Params{FT}) where {FT} = FT
# Whether `trace_parameters` has turned the scalar parameters into traced numbers.
_is_traced(p::Params) = _float_type(p) !== _scalar_type(p)
_float_type(f::Union{OceanForcing1D,OceanForcing2D,PerBasinOceanForcing1D}) = eltype(f.Tz)
_float_type(f::RampedOceanForcing) = _float_type(f.from)

# Per-cell ambient profiles cover the grid's input arrays and are cropped with them;
# a single profile is the same everywhere.
_crop_ocean_forcing(o::AbstractOceanForcing, sz, r, c) = o
function _crop_ocean_forcing(o::OceanForcing2D, sz, r, c)
    size(o.Tz)[1:2] == sz || throw(
        ArgumentError(
            "the OceanForcing2D profiles are $(size(o.Tz)[1:2]) cells but the mask is $sz; " *
            "they must cover the full domain including the border ring",
        ),
    )
    return OceanForcing2D(o.Tz[r, c, :], o.Sz[r, c, :], o.z, o.dz, o.z0)
end
function _crop_ocean_forcing(o::PerBasinOceanForcing1D, sz, r, c)
    size(o.basin) == sz || throw(
        ArgumentError(
            "the PerBasinOceanForcing1D basin map is $(size(o.basin)) cells but the mask is $sz; " *
            "it must cover the full domain including the border ring",
        ),
    )
    return PerBasinOceanForcing1D(o.Tz, o.Sz, o.basin[r, c], o.z, o.dz, o.z0)
end
# Both ends of a ramp are cropped alike; the weight is kept, not copied.
_crop_ocean_forcing(o::RampedOceanForcing, sz, r, c) = RampedOceanForcing(
    _crop_ocean_forcing(o.from, sz, r, c), _crop_ocean_forcing(o.to, sz, r, c), o.start, o.duration, o.weight)
_float_type(f::CavityForcing) = _float_type(f.ocean)

# A user input given as a scalar or a field: a matrix must cover the grid's input
# arrays (`sz`, the full domain including the border ring), before cropping, and
# anything else must be a real scalar.  `name` and `desc` word the errors.
function _check_scalar_or_field(x, sz, name, desc)
    if x isa AbstractMatrix
        size(x) == sz || throw(
            ArgumentError(
                "$name is $(size(x)) but the mask is $sz; a 2D $desc must cover the " *
                "full domain including the border ring",
            ),
        )
    elseif !(x isa Real)
        throw(ArgumentError("$name must be a real scalar or a matrix, got $(typeof(x))"))
    end
    return x
end

# Bring a user-supplied ice temperature `T_ice_base` onto the full domain: a scalar is
# broadcast, a matrix is checked against the grid's input size and converted to FT,
# so `_crop_ice_forcing` can then slice it with the grid's crop ranges.  Materialising once at build keeps
# the melt kernel to a single indexed path instead of a scalar and a field variant.
function _expand_ice_forcing(ice::PrescribedIceForcing, sz, FT)
    T = _check_scalar_or_field(ice.T_ice_base, sz, "T_ice_base", "ice temperature")
    Tb = T isa AbstractMatrix ? FT.(T) : fill(FT(T), sz)
    any(isnan, Tb) && throw(ArgumentError("T_ice_base contains NaN"))
    all(<=(0), Tb) || throw(
        ArgumentError(
            "T_ice_base must be at or below 0 °C (it is ice); found a maximum of " *
            "$(maximum(Tb)) °C",
        ),
    )
    return PrescribedIceForcing(Tb)
end

_crop_ice_forcing(ice::PrescribedIceForcing, r, c) =
    PrescribedIceForcing(ice.T_ice_base[r, c])

# Unknown ice forcings pass through untouched; they are responsible for supplying
# their own grid-shaped `T_ice_base`.
_expand_ice_forcing(ice::AbstractIceForcing, sz, FT) = ice
_crop_ice_forcing(ice::AbstractIceForcing, r, c) = ice

# FixedGamTMelting: the cache reports the constant exchange velocities, which the
# kernels read from the parameter (`_exchange_velocities`).
_init_exchange_velocities!(cache, ::AbstractMelting) = nothing
function _init_exchange_velocities!(cache, mp::FixedGamTMelting)
    cache.gamT, cache.gamS = _fixed_exchange_velocities(mp, typeof(cache.gamT))
    return
end

# Fill the Cache's prescribed melt field (m s⁻¹, grid-shaped) from a
# `PrescribedMelting` given in m yr⁻¹ as a scalar or a full-domain matrix.  Other
# melt schemes carry no such field.
_init_prescribed_melt!(cache, ::AbstractMelting, grid, params) = nothing
function _init_prescribed_melt!(cache, mp::PrescribedMelting, grid, params)
    M = _check_scalar_or_field(mp.melt, grid.input_size, "PrescribedMelting melt", "melt rate")
    all(isfinite, M) || throw(ArgumentError("PrescribedMelting melt must be finite"))
    r, c = grid.crop
    rate = M isa AbstractMatrix ? M[r, c] : M
    cache.melt_prescribed .= rate ./ params.seconds_per_year
    return
end

# Shape checks.  These run *before* preprocessing and cropping, because the
# cropping slices every input with index ranges derived from the mask: a size
# mismatch there would surface as an opaque BoundsError instead of an ArgumentError.
function _validate_input_shapes(mask, z_draft, z_bed)
    (size(mask, 1) >= 3 && size(mask, 2) >= 3) || throw(
        ArgumentError(
            "mask must be at least 3×3 — interior cells plus the one-cell border " *
            "ring — got $(size(mask))",
        ),
    )
    size(z_draft) == size(mask) || throw(
        ArgumentError(
            "z_draft and mask must have the same size, got $(size(z_draft)) vs $(size(mask))",
        ),
    )
    z_bed === nothing ||
        size(z_bed) == size(mask) ||
        throw(
            ArgumentError(
                "z_bed and mask must have the same size, got $(size(z_bed)) vs $(size(mask))",
            ),
        )
    return
end

# Mask semantics.  These run on the grid's cropped mask, before any modelling
# choice: a shelf cell on the border ring is invalid whatever the gaps treatment.
function _validate_grid_mask(mask)
    bad = setdiff(unique(mask), 0:4)
    isempty(bad) || throw(
        ArgumentError(
            "mask may only contain 0 (ocean), 1 (land), 2 (grounded), 3 (shelf), 4 (gap); found $(sort(bad))",
        ),
    )
    any(==(3), mask) || throw(
        ArgumentError(
            "mask contains no floating-shelf cells (value 3) — nothing to simulate",
        ),
    )
    _validate_border(mask, ==(3))
    return
end

# Active cells (shelf, and gaps the boundary condition keeps) on the border ring.
function _validate_border(mask, active)
    on_border =
        any(active, @view mask[1, :]) ||
        any(active, @view mask[end, :]) ||
        any(active, @view mask[:, 1]) ||
        any(active, @view mask[:, end])
    on_border && throw(
        ArgumentError(
            "active cells (3 = shelf, 4 = gap) on the domain border: the stencils skip " *
            "the outermost ring, so it must be ocean/land/grounded (0–2)",
        ),
    )
    return
end

# The stencils are launched over the interior only (`launch_interior!`), so nothing
# is computed on the border ring.  That is exact when every ring cell next to an
# active cell is a wall (land or grounded ice), but not at an ice front: the
# velocity faces between an active cell and an ocean ring cell would never be
# updated.  Diagonal neighbours count, since the momentum stencils read them.
function _validate_front_margin(mask, active)
    nx, ny = size(mask)
    on_ring(i, j) = i == 1 || i == nx || j == 1 || j == ny
    for j = 2:(ny-1), i = 2:(nx-1)
        (i in (2, nx - 1) || j in (2, ny - 1)) && active(mask[i, j]) || continue
        for dj = -1:1, di = -1:1
            ii, jj = i + di, j + dj
            if on_ring(ii, jj) && mask[ii, jj] == 0
                throw(
                    ArgumentError(
                        "active cell ($i, $j) borders the ocean cell ($ii, $jj) of the " *
                        "domain border: an ice front must keep at least one ocean cell " *
                        "between it and the outermost ring. Pad the mask with ocean, or " *
                        "crop with MinRectangleDomainCropping(margin ≥ 2)",
                    ),
                )
            end
        end
    end
    return
end

# Checks that need the model's choices: the gap-resolved mask (a gap on the border
# is legal under SinkGapsBC, which demotes it to ocean) and precision agreement.
function _validate_model_inputs(resolved_mask, forcing, params, FT)
    _validate_border(resolved_mask, v -> v == 3 || v == 4)
    _validate_front_margin(resolved_mask, v -> v == 3 || v == 4)
    _float_type(params) === FT || throw(
        ArgumentError(
            "params is Params{$(_float_type(params))} but the grid is Grid{$FT}; " *
            "construct the parameters with Params(; FT = $FT, ...) or build the grid with the matching FT",
        ),
    )
    forcing.ocean isa Union{OceanForcing1D,OceanForcing2D,PerBasinOceanForcing1D,RampedOceanForcing} || throw(
        ArgumentError(
            "Model needs an OceanForcing1D (one ambient profile for the whole " *
            "domain), a PerBasinOceanForcing1D (one per basin), an OceanForcing2D " *
            "(one per cell) or a RampedOceanForcing between two of them; got $(typeof(forcing.ocean)).",
        ),
    )
    _float_type(forcing) === FT || throw(
        ArgumentError(
            "forcing holds $(_float_type(forcing)) profiles but the grid is Grid{$FT}; " *
            "construct the forcing with FT = $FT or build the grid with the matching FT",
        ),
    )
    return
end

"""
$(TYPEDSIGNATURES)

Assemble a `Model` on `grid` from a forcing, a parameter set and boundary
conditions, and set its initial prognostic fields.  Wrap the result in a
[`Simulation`](@ref) to run it.

The model derives everything that depends on its choices from the grid and keeps
it in its `Geometry`: the gap-resolved mask (`boundary.gaps` decides whether gap
cells `4` are active or demoted to open ocean, so the same grid can drive both
treatments), the active-cell and velocity masks, the wall indicators, the
ice-base slope (`params.ice_slope`) and the Coriolis field (`params.coriolis`).

Land (`1`) and grounded ice (`2`) are both walls to the plume and are unioned into
`grd`, but they stay distinct throughout: only `2` is a grounding line (`gl`, and
the `boundary.grounding_line` slip condition), while `1` is rock (`lnd`, and
`boundary.land`).  Both default to LADDIE v2's wall (slip factor 1) and can be set
independently.

# Keywords
- `forcing` (required): a `CavityForcing`, or an ocean forcing alone (e.g.
  `ISOMIPForcing`, `OceanForcing1D`), which is paired with a uniform
  `PrescribedIceForcing($(DEFAULT_T_ICE_BASE))`.  A 2D ice temperature `T_ice_base` must
  cover the grid's input arrays (`grid.input_size`); it is cropped with the grid.
- `params`: a `Params` object with all physical constants and parameterizations
  (default `Params(; FT)` at the grid's precision).  A 2D latitude in
  `params.coriolis` is cropped with the grid, like the ice temperature.
- `boundary`: a [`BoundaryConditions`](@ref) (default `BoundaryConditions()`: LADDIE
  v2's walls, otherwise the conditions both LADDIE versions share).

The model lives on the grid's backend and precision; `params` and `forcing` must
match that precision (an `ArgumentError` is thrown otherwise).  It holds its
initial fields only; secondary fields (melt, entrainment, …) are computed when a
`Simulation` is constructed from it, because they depend on the time step.

The outermost ring of the grid is never computed by the stencils.  It must hold no
active cell (3 or 4), and no active cell may border one of its ocean cells: an ice
front needs at least one ocean cell between it and the ring.  Walls (land or
grounded ice) may sit on the ring right next to the shelf.  An `ArgumentError` is
thrown otherwise; [`MinRectangleDomainCropping`](@ref) with its minimum margin of 2
always satisfies this.

# Example
```julia
mask    = build_laddie_mask(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
z_draft = ice_base_depth(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
grid    = Grid(mask, z_draft, 2000.0, 2000.0)
model   = Model(grid; forcing = ISOMIPForcing(:warm),
                boundary = BoundaryConditions(; land = FreeSlipLand()))
sim     = Simulation(model; dt = 210.0)
run!(sim; days = 30)
```
"""
function Model(
    grid::Grid{FT};
    forcing,
    params::Params = Params(; FT),
    boundary::BoundaryConditions = BoundaryConditions(),
) where {FT}
    forcing = _as_cavity_forcing(forcing)
    boundary = _mapfields(bc -> _promote_param(bc, FT), boundary)
    backend = KA.get_backend(grid.z_draft)
    # The model is assembled on the CPU and moved in one go, so its derived fields
    # are computed exactly as on a CPU run whatever the grid's backend.
    grid = _cpu_grid(grid, backend)
    r, c = grid.crop
    # Full-domain fields are validated against the grid's input size, then cropped
    # with the grid's own ranges.  Materialising the ice temperature once keeps the
    # melt kernel to a single indexed path instead of a scalar and a field variant.
    ice = _crop_ice_forcing(_expand_ice_forcing(forcing.ice, grid.input_size, FT), r, c)
    f_t = _coriolis_field(params.coriolis, grid.input_size, FT)[r, c]
    mask = _apply_gaps_bc(grid.mask, boundary.gaps)
    _validate_model_inputs(mask, forcing, params, FT)
    # The ice forcing is grid-shaped from here on, so `m.T_ice_base` lines up with
    # `m.z_draft` and the melt kernel can index it directly; so are per-cell ambient
    # profiles.
    forcing = CavityForcing(_crop_ocean_forcing(forcing.ocean, grid.input_size, r, c), ice)
    geometry = Geometry(mask, grid.z_draft, f_t, grid.dx, grid.dy; FT, ice_slope = params.ice_slope)
    nx_total, ny_total = size(mask)
    state = State(FT, nx_total, ny_total)
    cache = Cache(
        FT,
        typeof(params.melting),
        typeof(params.convection_scheme),
        typeof(params.frazil_ice),
        nx_total,
        ny_total,
        typeof(boundary.gaps),
    )
    _init_gaps!(cache, boundary.gaps, mask)
    _init_prescribed_melt!(cache, params.melting, grid, params)
    _init_exchange_velocities!(cache, params.melting)
    m = Model(grid, geometry, state, cache, params, boundary, forcing)
    _initialize_prognostics!(m)
    m = _model_on(m, backend)
    return m
end
