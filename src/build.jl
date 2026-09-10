# ============================================================================
# General model builder
# ============================================================================

_float_type(::Params{FT}) where {FT} = FT
_float_type(f::AbstractForcing) = eltype(f.Tz)

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
    z_bed_raw === nothing || size(z_bed_raw) == size(mask) || throw(
        ArgumentError(
            "z_bed_raw and mask must have the same size, got $(size(z_bed_raw)) vs $(size(mask))",
        ),
    )
    (dx > 0 && dy > 0) ||
        throw(ArgumentError("dx and dy must be positive, got dx = $dx, dy = $dy"))
    return
end

# Mask semantics and precision checks.  These run after gap resolution and cropping,
# on the geometry the model will actually be built from.
function _validate_build_inputs(mask, z_draft_raw, dx, dy, forcing, params, FT)
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
    _is_active(v) = v == 3 || v == 4
    border_shelf =
        any(_is_active, @view mask[1, :]) ||
        any(_is_active, @view mask[end, :]) ||
        any(_is_active, @view mask[:, 1]) ||
        any(_is_active, @view mask[:, end])
    border_shelf && throw(
        ArgumentError(
            "active cells (3 = shelf, 4 = gap) on the domain border: the stencils wrap " *
            "periodically, so the outermost ring must be ocean/land/grounded (0–2)",
        ),
    )
    _float_type(params) === FT || throw(
        ArgumentError(
            "params is Params{$(_float_type(params))} but Model was called with FT = $FT; " *
            "construct the parameters with Params(; FT = $FT, ...) or pass the matching FT",
        ),
    )
    _float_type(forcing) === FT || throw(
        ArgumentError(
            "forcing holds $(_float_type(forcing)) profiles but Model was called with FT = $FT; " *
            "construct the forcing with FT = $FT or pass the matching FT",
        ),
    )
    return
end

"""
$(TYPEDSIGNATURES)

General model constructor.  Assembles and initialises a `Model` from an
arbitrary domain mask and ice-draft, a forcing profile, and a parameter set.

# Mask convention
| Value | Meaning |
|-------|---------|
| `0`   | open ocean (outside domain, passive) |
| `1`   | land — exposed bedrock, and the one-cell border ring |
| `2`   | grounded ice (sets inflow boundary for the plume) |
| `3`   | floating ice shelf (active plume cells) |
| `4`   | ice-shelf gap (ice-free but active; see [`AbstractGapsBC`](@ref)) |

Land (`1`) and grounded ice (`2`) are both walls to the plume and are unioned into
`Grid.grd`, but they stay distinct throughout: only `2` is a grounding line
(`Grid.gl`, and the `Params.grline_bc` slip condition), while `1` is rock
(`Grid.lnd`, and `Params.land_bc`).  Both slip conditions default to the global
`Params.slip` (free slip) and can be set independently; keeping the two masks
apart also matters because an ice-free island misclassified as ocean turns into
an open-boundary sink inside the cavity.

Value `4` marks an ice-free cell *inside* the ice-shelf domain — a gap opened by
melt-through.  Whether it is simulated as an active cell or demoted to open ocean
is decided by `Params.gaps_bc`, so the same mask can drive both treatments.

The `mask` and `z_draft_raw` arrays must include the full domain with the one-cell
border ring, i.e. size `(ny+2, nx+2)` where `ny × nx` are the interior cells.

`z_draft_raw` gives the ice-base depth in metres (negative downward) at each cell;
values at non-shelf cells (mask ≠ 3) are ignored and zeroed internally.

# Arguments
- `mask`:    integer mask matrix, size `(ny+2, nx+2)`.
- `z_draft_raw`:  raw ice-draft matrix (same size); need not be pre-processed.
- `dx`, `dy`: cell spacing in metres.
- `forcing`: an `AbstractForcing` (e.g. `ISOMIPForcing`, `ProfileForcing`).
- `params`:  a `Params` object with all physical constants and parameterizations.
- `backend`: KernelAbstractions backend (default `CPU()`).
- `FT`:      floating-point precision type (default `Float64`); must match the
  precision of `params` and `forcing` (an `ArgumentError` is thrown otherwise).
- `config`:      `RunConfig` (default: `RunConfig()`, I/O disabled).

The returned model is fully initialised and ready for `run!`.

# Example
```julia
mask   = build_laddie_mask(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
z_draft_raw = ice_base_depth(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
forcing = ISOMIPForcing(Float64, :warm)
params  = Params()
m = Model(mask, z_draft_raw, 2000.0, 2000.0, forcing, params)
run!(m; days=30)
```
"""
function Model(
    mask::AbstractMatrix{Int},
    z_draft_raw::AbstractMatrix,
    dx::Real,
    dy::Real,
    forcing::AbstractForcing,
    params::Params;
    backend = CPU(),
    FT = Float64,
    config = RunConfig(),
    gradient = JlGradient(),
    z_bed_raw = nothing,
    domain_cropping = MinRectangleDomainCropping(),
    preprocess = AbstractPreprocess[],
)
    _validate_input_shapes(mask, z_draft_raw, z_bed_raw, dx, dy)
    for p in preprocess
        preprocess!(mask, p)
    end
    # Gap resolution runs on the cleaned geometry but before cropping: `preprocess!` is
    # size-preserving, so a ConnectedGapsBC reference footprint lines up with the mask
    # here, whereas after cropping it would not.
    mask = _apply_gaps_bc(mask, params.gaps_bc)
    mask, z_draft_raw, z_bed_raw = _crop_domain(mask, z_draft_raw, z_bed_raw, domain_cropping)
    _validate_build_inputs(mask, z_draft_raw, dx, dy, forcing, params, FT)
    ny_total, nx_total = size(mask)
    nx, ny = nx_total - 2, ny_total - 2

    z_draft = _adjust_z_draft(mask, z_draft_raw, FT)
    z_bed = if z_bed_raw === nothing
        fill(FT(-Inf), ny_total, nx_total)  # no upper cap on D
    else
        FT.(z_bed_raw)
    end
    grid = Grid(mask, z_draft, z_bed, FT(dx), FT(dy); FT, gradient)
    state = State(FT, ny_total, nx_total)
    cache =
        Cache(FT, typeof(params.melting), typeof(params.convection_scheme), ny_total, nx_total)

    io = IOState(FT, collect(FT(dx) .* (1:nx)), collect(FT(dy) .* (1:ny)))
    m = Model(io, config, grid, state, cache, params, forcing)
    m.dt = params.dt0   # runtime dt starts at the configured initial step

    if config.saveday > 0
        create_rundir!(m)
    end
    if config.fromrestart
        m.drho = zero(grid.tmask)
        m.Tf = zero(grid.tmask)
        m.melt = zero(grid.tmask)
        m.Tb = zero(grid.tmask)
        init_from_restart!(m)
    else
        _initialize_prognostics!(m)
    end
    backend === CPU() || (m = to_backend(m, backend))
    if config.saveday > 0
        prepare_output!(m)
    end
    return m
end

# ============================================================================
# ISOMIP+ geometry builder
# ============================================================================

"""
$(TYPEDSIGNATURES)

Convenience constructor for the idealised ISOMIP+ channel geometry
(Asay-Davis et al. 2016).  Builds the mask and ice draft analytically, then
delegates to `Model`.

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
- `FT`: floating-point precision type (default `Float64`; use `Float32` for GPU).
- `params`: `Params` object (default: ISOMIP+-canonical values).
- `config`: `RunConfig` object (default: `RunConfig()`, I/O disabled).
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
    params = nothing,
    config = RunConfig(),
    gradient = JlGradient(),
    domain_cropping = NoDomainCropping(),
    preprocess = AbstractPreprocess[],
)
    ny_total, nx_total = ny + 2, nx + 2
    mask = zeros(Int, ny_total, nx_total)
    z_draft_raw = zeros(FT, ny_total, nx_total)
    xgl_ft = FT(xgl);
    xfront_ft = FT(xfront)
    zgl_ft = FT(z_draft_gl);
    zfr_ft = FT(z_draft_front)
    for j = 1:ny, i = 1:nx
        x = FT((i - 1) * dx)
        jp, ip = j + 1, i + 1
        if x < xgl_ft
            mask[jp, ip] = 2
        elseif x <= xfront_ft
            mask[jp, ip] = 3
            z_draft_raw[jp, ip] =
                zgl_ft + (zfr_ft - zgl_ft) * (x - xgl_ft) / (xfront_ft - xgl_ft)
        end
    end
    mask[1, :] .= 1;
    mask[end, :] .= 1
    mask[:, 1] .= 1;
    mask[:, end] .= 1

    forcing = ISOMIPForcing(FT, isomipcond)
    _params =
        isnothing(params) ?
        Params(;
            FT,
            entrainment = LambertEntrainment(FT(2.5)),
            melting = FixedGamTMelting(FT(0.00018)),
            convection_scheme = ResetToAmbient(FT(0.005)),
            open_bc = ZeroGradientInflow(),
            max_layer_thickness = TopographicMaxLayerThickness(),
        ) : params

    return Model(mask, z_draft_raw, dx, dy, forcing, _params; backend, FT, config, gradient, domain_cropping, preprocess)
end
