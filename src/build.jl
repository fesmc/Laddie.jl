# ============================================================================
# General model builder
# ============================================================================

_float_type(::Params{FT}) where {FT} = FT
_float_type(f::OceanForcing1D) = eltype(f.Tz)
_float_type(f::CavityForcing) = _float_type(f.ocean)

# Bring a user-supplied basal ice temperature onto the full domain: a scalar is
# broadcast, a matrix is checked against the mask and converted to FT.  Runs
# before cropping, on the same footing as `z_draft_raw`, so `_crop_ice_forcing`
# can then slice it with the mask's own ranges.  Materialising once at build keeps
# the melt kernel to a single indexed path instead of a scalar and a field variant.
function _expand_ice_forcing(ice::PrescribedIceForcing, sz, FT)
    T = ice.T_ice_base
    if T isa AbstractMatrix
        size(T) == sz || throw(
            ArgumentError(
                "T_ice_base is $(size(T)) but the mask is $sz; a 2D basal ice " *
                "temperature must cover the full domain including the border ring",
            ),
        )
    elseif !(T isa Real)
        throw(
            ArgumentError("T_ice_base must be a real scalar or a matrix, got $(typeof(T))"),
        )
    end
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

# Shape and spacing checks.  These run *before* preprocessing and cropping, because
# `_crop_domain` slices z_draft_raw/z_bed_raw with index ranges derived from the mask:
# a size mismatch there surfaces as an opaque BoundsError from an internal helper
# instead of the ArgumentError the caller should see.
function _validate_input_shapes(mask, z_draft_raw, z_bed_raw, dx, dy)
    (size(mask, 1) >= 3 && size(mask, 2) >= 3) || throw(
        ArgumentError(
            "mask must be at least 3×3 — interior cells plus the one-cell border ring — got $(size(mask))",
        ),
    )
    size(z_draft_raw) == size(mask) || throw(
        ArgumentError(
            "z_draft_raw and mask must have the same size, got $(size(z_draft_raw)) vs $(size(mask))",
        ),
    )
    z_bed_raw === nothing ||
        size(z_bed_raw) == size(mask) ||
        throw(
            ArgumentError(
                "z_bed_raw and mask must have the same size, got $(size(z_bed_raw)) vs $(size(mask))",
            ),
        )
    (dx > 0 && dy > 0) ||
        throw(ArgumentError("dx and dy must be positive, got dx = $dx, dy = $dy"))
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
            "active cells (3 = shelf, 4 = gap) on the domain border: the stencils wrap " *
            "periodically, so the outermost ring must be ocean/land/grounded (0–2)",
        ),
    )
    return
end

# Checks that need the model's choices: the gap-resolved mask (a gap on the border
# is legal under SinkGapsBC, which demotes it to ocean) and precision agreement.
function _validate_model_inputs(resolved_mask, forcing, params, FT)
    _validate_border(resolved_mask, v -> v == 3 || v == 4)
    _float_type(params) === FT || throw(
        ArgumentError(
            "params is Params{$(_float_type(params))} but the grid is Grid{$FT}; " *
            "construct the parameters with Params(; FT = $FT, ...) or build the grid with the matching FT",
        ),
    )
    forcing.ocean isa OceanForcing1D || throw(
        ArgumentError(
            "Model needs an OceanForcing1D (one ambient profile for the whole " *
            "domain); got $(typeof(forcing.ocean)). A laterally varying ambient field " *
            "is not implemented.",
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
ice-base slope (`gradient`) and the Coriolis field (`params.coriolis`).

Land (`1`) and grounded ice (`2`) are both walls to the plume and are unioned into
`grd`, but they stay distinct throughout: only `2` is a grounding line (`gl`, and
the `boundary.grounding_line` slip condition), while `1` is rock (`lnd`, and
`boundary.land`).  Both slip conditions default to the global `Params.slip` (free
slip) and can be set independently.

# Keywords
- `forcing` (required): a `CavityForcing`, or an ocean forcing alone (e.g.
  `ISOMIPForcing`, `OceanForcing1D`), which is paired with a uniform
  `PrescribedIceForcing($(DEFAULT_T_ICE_BASE))`.  A 2D basal ice temperature must
  cover the grid's input arrays (`grid.input_size`); it is cropped with the grid.
- `params`: a `Params` object with all physical constants and parameterizations
  (default `Params(; FT)` at the grid's precision).  A 2D latitude in
  `params.coriolis` is cropped with the grid, like the ice temperature.
- `boundary`: a [`BoundaryConditions`](@ref) (default: all LADDIE v1.x conditions).
- `gradient`: ice-base slope stencil, [`JlGradient`](@ref) (default) or
  [`PyGradient`](@ref).

The model lives on the grid's backend and precision; `params` and `forcing` must
match that precision (an `ArgumentError` is thrown otherwise).  It holds its
initial fields only; secondary fields (melt, entrainment, …) are computed when a
`Simulation` is constructed from it, because they depend on the time step.

# Example
```julia
mask    = build_laddie_mask(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
z_draft = ice_base_depth(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
grid    = Grid(mask, z_draft, 2000.0, 2000.0)
model   = Model(grid; forcing = ISOMIPForcing(Float64, :warm),
                boundary = BoundaryConditions(; grounding_line = NoSlipGL()))
sim     = Simulation(model; dt = 210.0)
run!(sim; days = 30)
```
"""
function Model(
    grid::Grid{FT};
    forcing,
    params::Params = Params(; FT),
    boundary::BoundaryConditions = BoundaryConditions(),
    gradient = JlGradient(),
) where {FT}
    forcing = _as_cavity_forcing(forcing)
    boundary = BoundaryConditions(map(bc -> _promote_param(bc, FT), _bc_tuple(boundary))...)
    backend = KA.get_backend(grid.z_draft)
    # The model is assembled on the CPU and moved in one go, so its derived fields
    # are computed exactly as on a CPU run whatever the grid's backend.
    grid = backend === CPU() ? grid : _grid_to_backend(grid, CPU())
    r, c = grid.crop
    # Full-domain fields are validated against the grid's input size, then cropped
    # with the grid's own ranges.  Materialising the ice temperature once keeps the
    # melt kernel to a single indexed path instead of a scalar and a field variant.
    ice = _crop_ice_forcing(_expand_ice_forcing(forcing.ice, grid.input_size, FT), r, c)
    f_t = _coriolis_field(params.coriolis, grid.input_size, FT)[r, c]
    mask = _apply_gaps_bc(grid.mask, boundary.gaps)
    _validate_model_inputs(mask, forcing, params, FT)
    # The ice forcing is grid-shaped from here on, so `m.T_ice_base` lines up with
    # `m.z_draft` and the melt kernel can index it directly.
    forcing = CavityForcing(forcing.ocean, ice)
    geometry = Geometry(mask, grid.z_draft, f_t, grid.dx, grid.dy; FT, gradient)
    nx_total, ny_total = size(mask)
    state = State(FT, nx_total, ny_total)
    cache = Cache(
        FT,
        typeof(params.melting),
        typeof(params.convection_scheme),
        nx_total,
        ny_total,
    )
    m = Model(grid, geometry, state, cache, params, boundary, forcing)
    _initialize_prognostics!(m)
    backend === CPU() || (m = to_backend(m, backend))
    return m
end

# ============================================================================
# ISOMIP+ geometry builder
# ============================================================================

"""
$(TYPEDSIGNATURES)

Convenience constructor for the idealised ISOMIP+ channel geometry
(Asay-Davis et al. 2016).  Builds the mask and ice draft analytically, then the
`Grid`, `Model` and a ready-to-run [`Simulation`](@ref) of it.

# Arguments
- `backend`: KernelAbstractions backend.  Default `CPU()`; use `CUDABackend()`
  for NVIDIA GPU, `ROCBackend()` for AMD, `MetalBackend()` for Apple Silicon.
- `nx`, `ny`: interior cell counts in x and y (default 240 × 40).
- `dx`, `dy`: cell size in metres (default 2 km).
- `xgl`: grounding-line x-position in metres (default 20 km).
- `xfront`: ice-front x-position in metres (default 460 km).
- `z_draft_gl`, `z_draft_front`: ice-draft depth at grounding line and ice front in
  metres (default −720 m and −200 m).
- `isomipcond`: `:warm` (1 °C at depth) or `:cold` (nearly freezing).
- `ice_forcing`: an `AbstractIceForcing` supplying the basal ice temperature
  (default: uniform `PrescribedIceForcing($(DEFAULT_T_ICE_BASE))`).
- `FT`: floating-point precision type (default `Float64`; use `Float32` for GPU).
- `params`: `Params` object (default: ISOMIP+-canonical values).
- `domain_cropping`, `preprocess`: forwarded to `Grid`.
- `boundary`, `gradient`: forwarded to `Model`.
- any other keyword (`dt`, `tstep`, `cfl`, `nu`, `stop`, `output`, `restart`,
  `debug`) is forwarded to [`Simulation`](@ref).

```julia
sim = build_isomip(; isomipcond = :cold, tstep = AdaptiveDt())
run!(sim; days = 30)
sim.model.melt
```
"""
function build_isomip(
    backend = CPU();
    FT = Float64,
    nx = 240,
    ny = 40,
    dx = 2000.0,
    dy = 2000.0,
    xgl = 20_000.0,
    xfront = 460_000.0,
    z_draft_gl = -720.0,
    z_draft_front = -200.0,
    isomipcond = :warm,
    ice_forcing = PrescribedIceForcing(),
    params = nothing,
    boundary = BoundaryConditions(),
    gradient = JlGradient(),
    domain_cropping = NoDomainCropping(),
    preprocess = AbstractPreprocess[],
    simulation_kwargs...,
)
    nx_total, ny_total = nx + 2, ny + 2
    mask = zeros(Int, nx_total, ny_total)
    z_draft_raw = zeros(FT, nx_total, ny_total)
    xgl_ft = FT(xgl);
    xfront_ft = FT(xfront)
    zgl_ft = FT(z_draft_gl);
    zfr_ft = FT(z_draft_front)
    for j = 1:ny, i = 1:nx
        x = FT((i - 1) * dx)
        ip, jp = i + 1, j + 1
        if x < xgl_ft
            mask[ip, jp] = 2
        elseif x <= xfront_ft
            mask[ip, jp] = 3
            z_draft_raw[ip, jp] =
                zgl_ft + (zfr_ft - zgl_ft) * (x - xgl_ft) / (xfront_ft - xgl_ft)
        end
    end
    mask[1, :] .= 1;
    mask[end, :] .= 1
    mask[:, 1] .= 1;
    mask[:, end] .= 1

    forcing = CavityForcing(ISOMIPForcing(FT, isomipcond), ice_forcing)
    _params =
        isnothing(params) ?
        Params(;
            FT,
            entrainment = LambertEntrainment(FT(2.5)),
            melting = FixedGamTMelting(FT(0.00018)),
            convection_scheme = ResetToAmbient(FT(0.005)),
            max_layer_thickness = NoMaxLayerThickness(),
        ) : params

    grid = Grid(mask, z_draft_raw, dx, dy; preprocess, domain_cropping, backend, FT)
    model = Model(grid; forcing, params = _params, boundary, gradient)
    return Simulation(model; simulation_kwargs...)
end
