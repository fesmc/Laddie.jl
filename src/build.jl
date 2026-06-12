
# ============================================================================
# General model builder
# ============================================================================

"""
$(TYPEDSIGNATURES)

General model constructor.  Assembles and initialises a `Model` from an
arbitrary domain mask and ice-draft, a forcing profile, and a parameter set.

# Mask convention
| Value | Meaning |
|-------|---------|
| `0`   | open ocean (outside domain, passive) |
| `1`   | land / boundary (one-cell border ring) |
| `2`   | grounded ice (sets inflow boundary for the plume) |
| `3`   | floating ice shelf (active plume cells) |

The `mask` and `zb_raw` arrays must include the full domain with the one-cell
border ring, i.e. size `(ny+2, nx+2)` where `ny × nx` are the interior cells.

`zb_raw` gives the ice-base depth in metres (negative downward) at each cell;
values at non-shelf cells (mask ≠ 3) are ignored and zeroed internally.

# Arguments
- `mask`:    integer mask matrix, size `(ny+2, nx+2)`.
- `zb_raw`:  raw ice-draft matrix (same size); need not be pre-processed.
- `dx`, `dy`: cell spacing in metres.
- `forcing`: an `AbstractForcing` (e.g. `ISOMIPForcing`, `LinearForcing`, `FileForcing`).
- `params`:  a `Params` object with all physical constants and parameterizations.
- `backend`: KernelAbstractions backend (default `CPU()`).
- `FT`:      floating-point precision type (default `Float64`).
- `rc`:      `RunConfig` (default: `RunConfig()`, I/O disabled).

The returned model is fully initialised and ready for `run!`.

# Example
```julia
mask   = build_laddie_mask(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
zb_raw = ice_base_depth(bed, thickness; rho_ice=917.0, rho_sw=1028.0)
forcing = ISOMIPForcing(Float64, :warm)
params  = Params()
m = build_model(mask, zb_raw, 2000.0, 2000.0, forcing, params)
run!(m; days=30)
```
"""
function build_model(
    mask    :: AbstractMatrix{Int},
    zb_raw  :: AbstractMatrix,
    dx      :: Real,
    dy      :: Real,
    forcing :: AbstractForcing,
    params  :: Params;
    backend = CPU(),
    FT      = Float64,
    rc      = RunConfig(),
)
    ny_total, nx_total = size(mask)
    nx, ny = nx_total - 2, ny_total - 2

    zb    = _adjust_zb(mask, zb_raw, FT)
    grid  = Grid(mask, zb, FT(dx), FT(dy); FT)
    state = State(FT, ny_total, nx_total)
    cache = Cache(FT, typeof(params.meltpar), typeof(params.convpar), ny_total, nx_total)

    io = IOState(FT, collect(FT(dx) .* (1:nx)), collect(FT(dy) .* (1:ny)))
    m  = Model(io, rc, grid, state, cache, params, forcing)

    if rc.saveday > 0
        create_rundir!(m)
    end
    if rc.fromrestart
        m.drho = zero(grid.tmask)
        m.Tf   = zero(grid.tmask)
        m.melt = zero(grid.tmask)
        m.Tb   = zero(grid.tmask)
        init_from_restart!(m)
    else
        _initialize_prognostics!(m)
    end
    backend === CPU() || (m = to_backend(m, backend))
    if rc.saveday > 0
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
delegates to `build_model`.

# Arguments
- `backend`: KernelAbstractions backend.  Default `CPU()`; use `CUDABackend()`
  for NVIDIA GPU, `ROCBackend()` for AMD, `MetalBackend()` for Apple Silicon.
- `nx`, `ny`: interior cell counts in x and y (default 240 × 40).
- `dx`, `dy`: cell size in metres (default 2 km).
- `xgl`: grounding-line x-position in metres (default 20 km).
- `xfront`: ice-front x-position in metres (default 460 km).
- `zb_gl`, `zb_front`: ice-draft depth at grounding line and ice front in
  metres (default −720 m and −200 m).
- `isomipcond`: `:warm` (1 °C at depth) or `:cold` (nearly freezing).
- `FT`: floating-point precision type (default `Float64`; use `Float32` for GPU).
- `params`: `Params` object (default: ISOMIP+-canonical values).
- `rc`: `RunConfig` object (default: `RunConfig()`, I/O disabled).
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
    zb_gl = -720.0,
    zb_front = -200.0,
    isomipcond = :warm,
    params = nothing,
    rc = RunConfig(),
)
    ny_total, nx_total = ny + 2, nx + 2
    mask   = zeros(Int, ny_total, nx_total)
    zb_raw = zeros(FT, ny_total, nx_total)
    xgl_ft = FT(xgl);   xfront_ft = FT(xfront)
    zgl_ft = FT(zb_gl); zfr_ft    = FT(zb_front)
    for j = 1:ny, i = 1:nx
        x = FT((i - 1) * dx)
        jp, ip = j + 1, i + 1
        if x < xgl_ft
            mask[jp, ip] = 2
        elseif x <= xfront_ft
            mask[jp, ip] = 3
            zb_raw[jp, ip] =
                zgl_ft + (zfr_ft - zgl_ft) * (x - xgl_ft) / (xfront_ft - xgl_ft)
        end
    end
    mask[1, :] .= 1;   mask[end, :] .= 1
    mask[:, 1] .= 1;   mask[:, end] .= 1

    forcing = ISOMIPForcing(FT, isomipcond)
    _params = isnothing(params) ? Params(;
        FT,
        entpar  = GasparEntrainment(FT(2.5)),
        meltpar = FixedGamT(FT(0.00018)),
        convpar = ResetToAmbient(FT(0.005)),
        openbc  = ZeroGradientInflow(),
    ) : params

    return build_model(mask, zb_raw, dx, dy, forcing, _params; backend, FT, rc)
end