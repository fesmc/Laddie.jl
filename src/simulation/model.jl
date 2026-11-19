# The parts of a Model whose fields `m.<name>` reaches, in lookup order.  `CavityForcing`
# is a container, not a namespace: its `ocean` and `ice` members are searched, not the
# wrapper (`m.ocean` is not a property).
_parts(grid, geometry, state, cache, params, forcing) =
    (grid, geometry, state, cache, params, getfield(forcing, :ocean), getfield(forcing, :ice))
_parts(m) = _parts(getfield(m, :grid), getfield(m, :geometry), getfield(m, :state), getfield(m, :cache),
                   getfield(m, :params), getfield(m, :forcing))

# Properties answered before the forwarding: the fields of `Model` and these.
const _DERIVED_PROPS = (:FT, :nx, :ny)

# The forwarding takes the first part that holds a name, so a name in two parts would be
# shadowed silently.  Such configurations, mostly from user forcing types, are rejected
# here, in the inner constructor, so that no construction path bypasses the check.
function _check_property_collisions(parts...)
    seen = Dict{Symbol,String}()
    for x in _parts(parts...)
        label = string(nameof(typeof(x)))
        for fn in fieldnames(typeof(x))
            (fn in fieldnames(Model) || fn in _DERIVED_PROPS) &&
                error("field `$fn` of $label collides with the reserved Model property `$fn`")
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

The model: geometry, physics and prognostic state, on the grid's backend and precision. It
knows nothing about time stepping; wrap it in a [`Simulation`](@ref) to run it, and move it
with [`to_backend`](@ref). Fields of its parts are read directly on `m`:

| Access pattern | Source struct | Examples |
|----------------|--------------|---------|
| `m.H`, `m.U`, `m.V`, `m.T`, `m.S` | `State` | `m.H.present`, `m.U.past` |
| `m.melt`, `m.entr`, `m.drho`, `m.Ta`, `m.Sa`, … | `Cache` | `m.melt .* m.seconds_per_year` |
| `m.mask`, `m.z_draft`, `m.dx`, `m.x`, … | `Grid` | input mask, draft, spacing, coordinates |
| `m.tmask`, `m.umask`, `m.resolved_mask`, `m.dzdx`, … | `Geometry` | `m.tmask .> 0` |
| `m.C_d`, `m.A_h`, `m.H_min`, … | `Params` | `m.C_d` |
| `m.f`, `m.fu`, `m.fv` | `Geometry` | Coriolis at T-, u- and v-points |
| `m.boundary` | `BoundaryConditions` | `m.boundary.land_slip` (not flattened) |
| `m.Tz`, `m.Sz`, `m.z` | Forcing (ocean) | ambient profile arrays |
| `m.T_ice_farfield` | Forcing (ice) | ice temperature above the basal boundary layer (see `PrescribedIceForcing`) |

`m.FT` is the float type of the scalar parameters; `m.nx`, `m.ny` the interior cell counts.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Model{G<:Grid,GE<:Geometry,S<:State,C<:Cache,P<:Params,B<:BoundaryConditions,F<:CavityForcing}
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

    function Model{G,GE,S,C,P,B,F}(grid, geometry, state, cache, params, boundary, forcing) where {G,GE,S,C,P,B,F}
        _check_property_collisions(grid, geometry, state, cache, params, forcing)
        return new{G,GE,S,C,P,B,F}(grid, geometry, state, cache, params, boundary, forcing)
    end
end

Model(grid::G, geometry::GE, state::S, cache::C, params::P, boundary::B, forcing::F) where {G,GE,S,C,P,B,F} =
    Model{G,GE,S,C,P,B,F}(grid, geometry, state, cache, params, boundary, forcing)

# `m.k` from the first part that holds `k`.  Recursive over the parts, so that with a
# constant `k` the lookup folds to one field access.
@inline _forward(k, x, rest...) = hasfield(typeof(x), k) ? getfield(x, k) : _forward(k, rest...)
@inline _forward(k) = error("Model has no property `$k`")

function Base.getproperty(m::Model, k::Symbol)
    hasfield(typeof(m), k) && return getfield(m, k)
    k === :FT && return _float_type(getfield(m, :params))
    # Interior dimensions: the grid's minus the border ring.
    k === :nx && return getfield(m, :grid).Nx - 2
    k === :ny && return getfield(m, :grid).Ny - 2
    return _forward(k, _parts(m)...)
end

# Everything `getproperty` answers to, in its lookup order, so that REPL completion
# shows the forwarded names too.
Base.propertynames(m::Model, ::Bool = false) =
    (fieldnames(Model)..., _DERIVED_PROPS..., (fn for x in _parts(m) for fn in fieldnames(typeof(x)))...)

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
# A user-defined ocean forcing: its precision is not known, so it is not checked.
_float_type(::AbstractOceanForcing) = nothing

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

# Bring a user-supplied ice temperature `T_ice_farfield` onto the full domain: a scalar is
# broadcast, a matrix is checked against the grid's input size and converted to FT,
# so `_crop_ice_forcing` can then slice it with the grid's crop ranges.  Materialising once at build keeps
# the melt kernel to a single indexed path instead of a scalar and a field variant.
function _expand_ice_forcing(ice::PrescribedIceForcing, sz, FT)
    T = _check_scalar_or_field(ice.T_ice_farfield, sz, "T_ice_farfield", "ice temperature")
    Tb = T isa AbstractMatrix ? FT.(T) : fill(FT(T), sz)
    any(isnan, Tb) && throw(ArgumentError("T_ice_farfield contains NaN"))
    all(<=(0), Tb) || throw(
        ArgumentError(
            "T_ice_farfield must be at or below 0 °C (it is ice); found a maximum of " *
            "$(maximum(Tb)) °C",
        ),
    )
    return PrescribedIceForcing(Tb)
end

_crop_ice_forcing(ice::PrescribedIceForcing, r, c) =
    PrescribedIceForcing(ice.T_ice_farfield[r, c])

# Unknown ice forcings pass through untouched; they are responsible for supplying
# their own grid-shaped `T_ice_farfield`.
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
    forcing.ocean isa AbstractOceanForcing || throw(
        ArgumentError(
            "Model needs an AbstractOceanForcing, such as an OceanForcing1D (one ambient profile " *
            "for the whole domain), a PerBasinOceanForcing1D (one per basin), an OceanForcing2D " *
            "(one per cell) or a RampedOceanForcing between two of them; got $(typeof(forcing.ocean)).",
        ),
    )
    forcing_type = _float_type(forcing)
    (forcing_type === nothing || forcing_type === FT) || throw(
        ArgumentError(
            "forcing holds $forcing_type profiles but the grid is Grid{$FT}; " *
            "construct the forcing with FT = $FT or build the grid with the matching FT",
        ),
    )
    return
end

"""
$(TYPEDSIGNATURES)

Assemble a `Model` on `grid` from a forcing, a parameter set and boundary conditions, and set
its initial prognostic fields. The model derives its `Geometry` (gap-resolved mask,
masks, wall indicators, ice-base slope, Coriolis field) from the grid and these choices.

# Keywords
- `forcing` (required): a [`CavityForcing`](@ref), or an ocean forcing alone, paired with
  `PrescribedIceForcing($(DEFAULT_T_ICE_FARFIELD))`.
- `params`: a [`Params`](@ref) at the grid's precision (default `Params(; FT)`).
- `boundary`: a [`BoundaryConditions`](@ref) (default `BoundaryConditions()`).

Full-domain 2D inputs (ice temperature, latitude, ocean forcing) are cropped with the grid.
The border ring must hold no active cell, and an ice front needs an ocean cell between it and
the ring ([`MinRectangleDomainCropping`](@ref) ensures both); an `ArgumentError` is thrown
otherwise, or when the precisions differ.

# Example
```julia
mask    = build_laddie_mask(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
z_draft = ice_base_depth(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
grid    = Grid(mask, z_draft, 2000.0, 2000.0)
model   = Model(grid; forcing = ISOMIPForcing(:warm),
                boundary = BoundaryConditions(; land_slip = FreeSlipAtLand()))
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
    f_t = coriolis_field(params.coriolis, grid.input_size, FT)[r, c]
    mask = gaps_mask(grid.mask, boundary.gaps)
    _validate_model_inputs(mask, forcing, params, FT)
    # The ice forcing is grid-shaped from here on, so `m.T_ice_farfield` lines up with
    # `m.z_draft` and the melt kernel can index it directly; so are per-cell ambient
    # profiles.
    forcing = CavityForcing(_crop_ocean_forcing(forcing.ocean, grid.input_size, r, c), ice)
    geometry = Geometry(mask, grid.z_draft, f_t, grid.dx, grid.dy; FT, ice_slope = params.ice_slope)
    nx_total, ny_total = size(mask)
    state = State(FT, nx_total, ny_total)
    cache = Cache(
        FT,
        typeof(params.melting),
        typeof(params.convection),
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
