# ============================================================================
# Reproductions of published LADDIE v1 and v2 setups.
#
# Two levels per experiment:
#   - `Params<Setup>v1|v2(; FT, kwargs...)` returns the `Params` of the run; any
#     keyword overrides a field.
#   - `Simulation<Setup>v1|v2(backend; datadir, ...)` builds the whole run (grid,
#     forcing, boundary conditions, time step, duration) from the geometry the run
#     left behind under `datadir` (default `papers/` in the package root).
#
# Where the paper and the code that produced its results disagree, the values
# follow the code (the bundled run configs) and the docstring warns.  Digests of
# the papers, with the config files these values come from, are in
# papers/lambert-2023, papers/lambert-2026 and papers/jesse-2026.
#
# LADDIE v2 runs on an unstructured mesh with FB-RK3 time stepping; Laddie.jl
# reproduces its parameters, not its numerics.
# ============================================================================

const DEFAULT_PAPERS_DIR = joinpath(dirname(@__DIR__), "papers")

# The ice temperature of the v1 runs (`Ti` in every config) and of the standalone
# v2 runs (UPSY's default uniform ice temperature, 270 K).
const T_ICE_BASE_V1 = -25.0
const T_ICE_BASE_V2 = 270.0 - 273.15

# Merge the setup's values with the caller's overrides into a `Params`.
_reproduction_params(setup::NamedTuple, FT, overrides) =
    Params(; FT, merge(setup, values(overrides))...)

# ----------------------------------------------------------------------------
# LADDIE v1: values shared by every v1 config (`code/config*.toml` and the v1 run
# of Lambert 2026, Sect. 3.1).
# ----------------------------------------------------------------------------

_v1_params(; A_h, K_h, H_min, u_tide) = (
    g = 9.81,
    C_d = 2.5e-3,
    C_d_top = 1.1e-3,
    A_h,
    K_h,
    max_detrainment = 0.5,
    H_min,
    v_cut = 1.414,
    u_tide,
    rho_freshwater = 1000.0,
    rho0_seawater = 1028.0,
    rho_ice = 910.0,
    L = 3.34e5,
    c_p = 3974.0,
    c_i = 2009.0,
    alpha = 3.733e-5,
    beta = 7.843e-4,
    l1 = -5.73e-2,
    l2 = 8.32e-2,
    l3 = 7.61e-4,
    H_init = 10.0,
    dT_init = 0.0,
    dS_init = -0.1,
    seconds_per_year = 31_536_000.0,
    entrainment = GasparEntrainment(2.5),
    melting = TurbulentGamTMelting(),
    convection_scheme = ResetToAmbient(0.005),
    ice_conduction = HollandJenkinsConduction(),
    equation_of_state = LinearEOS(),
    lateral_viscosity = PrescribedLateralViscosity(),
    momentum_advection = CentredMomentumAdvection(),
    tracer_form = AdvectiveTracerForm(),
    coriolis = CoriolisParameter0D(-1.37e-4),
)

# One partial-slip factor of 1 on every wall (`slip = 1`), zero-gradient inflow at
# the ice front (`boundop = 1`).
_v1_boundary() = BoundaryConditions(;
    open_ocean = ZeroGradientInflow(),
    grounding_line = PartialSlipGL(1.0),
    land = PartialSlipLand(1.0),
)

# v1 `forcing.py`, option 'tanh': a smooth thermocline between the surface freezing
# point (at S0) and T1, salinity compensating a density profile ∝ √|z|.
function _v1_tanh_forcing(; T1, z0, z1, S0 = 34.0, drho0 = 0.01, FT)
    alpha, beta, rho0, l1, l2 = 3.733e-5, 7.843e-4, 1028.0, -5.73e-2, 8.32e-2
    z = collect(-5000.0:1.0:-1.0)
    T0 = l1 * S0 + l2
    Tz = @. T1 + (T0 - T1) * (1 + tanh((z - z0) / z1)) / 2
    Sz = @. S0 + alpha * (Tz - T0) / beta + drho0 * sqrt(abs(z)) / (beta * rho0)
    return OceanForcing1D(Tz, Sz, z; FT)
end

# v1 `forcing.py`, option 'linear': linear in T and S from the surface (T at the
# freezing point of S0) to (T1, S1) at depth z0, continued linearly below.
function _v1_linear_forcing(; T1, S0, S1, z0, FT)
    l1, l2 = -5.73e-2, 8.32e-2
    z = collect(-5000.0:1.0:-1.0)
    T0 = l1 * S0 + l2
    Tz = @. T0 + z * (T1 - T0) / z0
    Sz = @. S0 + z * (S1 - S0) / z0
    return OceanForcing1D(Tz, Sz, z; FT)
end

# A v1 output file holds the computational domain the run used: its mask and ice
# draft (`zvar`), already cut to the shelf.  v1 then adds a one-cell ring of closed
# border (`add_border`, mask 1), which is Laddie.jl's border ring.
function _read_v1_geometry(path; zvar)
    isfile(path) || throw(ArgumentError("geometry file not found: $path"))
    mask, z = NCDatasets.NCDataset(path) do ds
        Int.(coalesce.(ds["mask"][:, :], 0)), Float64.(coalesce.(ds[zvar][:, :], 0.0))
    end
    return _pad_ring(mask, 1), _pad_ring(z, 0.0)
end

function _pad_ring(a::AbstractMatrix, v)
    out = fill(convert(eltype(a), v), size(a) .+ 2)
    out[2:end-1, 2:end-1] .= a
    return out
end

function _v1_simulation(
    backend,
    mask,
    z_draft,
    dx;
    FT,
    params,
    ocean,
    dt,
    days,
    boundary = _v1_boundary(),
    simulation_kwargs,
)
    grid = Grid(mask, z_draft, dx, dx; backend, FT, domain_cropping = NoDomainCropping())
    forcing = CavityForcing(ocean, PrescribedIceForcing(T_ICE_BASE_V1))
    model = Model(grid; forcing, params, boundary, gradient = PyGradient())
    return Simulation(
        model;
        merge((; dt, nu = 0.8, stop = FixedSimulationEnd(days)), values(simulation_kwargs))...,
    )
end

const _V1_WARNING = """
!!! warning "Paper differs"
    The entrainment parameter is μ = 2.5 here, as in every bundled v1 config;
    [Lambert et al. (2023, Table 1)](@cite lambert_modelling_2023) state μ = 0.5.  The layer starts at the ambient
    temperature (`dT_init = 0`, as in the configs); the paper starts it 0.1 °C
    below.  `max_detrainment = 0.5` and `v_cut = 1.414` are code values the paper
    does not mention.
"""

"""
    ParamsCrossonDotsonv1(; FT = Float64, kwargs...)

`Params` of the Crosson–Dotson run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023), Table 2:
Δx = 500 m), with any field overridden by `kwargs`.  `A_h = K_h = 25`, `H_min = 2.8`,
`u_tide = 0.01`, `C_d_top = 1.1e-3`, [Jenkins (1991)](@cite jenkins_onedimensional_1991) transfer coefficients.

$_V1_WARNING
See also [`SimulationCrossonDotsonv1`](@ref).
"""
ParamsCrossonDotsonv1(; FT = Float64, kwargs...) =
    _reproduction_params(_v1_params(; A_h = 25.0, K_h = 25.0, H_min = 2.8, u_tide = 0.01), FT, kwargs)

"""
    ParamsFRISv1(; FT = Float64, kwargs...)

`Params` of the Filchner–Ronne run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023), Table 2:
Δx = 1 km), with any field overridden by `kwargs`.  `A_h = K_h = 50`, `H_min = 4.4`,
`u_tide = 0.1`, otherwise as [`ParamsCrossonDotsonv1`](@ref).

$_V1_WARNING
See also [`SimulationFRISv1`](@ref).
"""
ParamsFRISv1(; FT = Float64, kwargs...) =
    _reproduction_params(_v1_params(; A_h = 50.0, K_h = 50.0, H_min = 4.4, u_tide = 0.1), FT, kwargs)

"""
    ParamsPIGv1(; FT = Float64, kwargs...)

`Params` of the Pine Island run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023), App. C: "the
same model tuning as Crosson–Dotson", Δx = 500 m), i.e. those of
[`ParamsCrossonDotsonv1`](@ref).

$_V1_WARNING
See also [`SimulationPIGv1`](@ref).
"""
ParamsPIGv1(; FT = Float64, kwargs...) = ParamsCrossonDotsonv1(; FT, kwargs...)

"""
    ParamsASEv1(; FT = Float64, kwargs...)

`Params` of the LADDIE v1 Amundsen Sea run that [Lambert et al. (2026, Sect. 3.1)](@cite lambert_one-layer_2026)
compare v2 against, from its config (`results/laddie1/ASE/config_ASE.toml`):
`A_h = 20`, `K_h = 4`, `H_min = 1`, `u_tide = 0.01`, `C_d_top = 1.1e-3`.

!!! warning "Paper differs"
    The config sets f = −1.37 × 10⁻⁴ s⁻¹; [Lambert et al. (2026)](@cite lambert_one-layer_2026) state
    −1.34 × 10⁻⁴ s⁻¹ for both model versions.  The paper gives no viscosity or
    diffusivity for this run.

See also [`SimulationASEv1`](@ref) and [`SimulationASEv1Theoretical`](@ref).
"""
ParamsASEv1(; FT = Float64, kwargs...) =
    _reproduction_params(_v1_params(; A_h = 20.0, K_h = 4.0, H_min = 1.0, u_tide = 0.01), FT, kwargs)

"""
    SimulationCrossonDotsonv1(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The Crosson–Dotson run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023)), on the 500 m domain
stored in `datadir/lambert-2023/data/CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc`.

- forcing: v1 'tanh' profile, T from the surface freezing point (S0 = 34) to 0.4 °C,
  thermocline at −500 m with scale 250 m, `drho0 = 0.01` (fitted to the file's
  ambient fields to 1e−4 °C); `T_ice_base = -25 °C`
- `dt = 120` s, 50 days, Robert–Asselin `nu = 0.8`
- v1 walls (partial slip 1), the v1 ice-slope stencil [`PyGradient`](@ref)

`params` defaults to [`ParamsCrossonDotsonv1`](@ref); other keywords go to
[`Simulation`](@ref) and override `dt`, `nu` and `stop`.

!!! warning "Paper differs"
    The run behind the stored output lasted 50 days (averaged over days 45–50);
    the paper reports a 30-day equilibration.  See also [`ParamsCrossonDotsonv1`](@ref).
"""
function SimulationCrossonDotsonv1(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsCrossonDotsonv1(; FT),
    simulation_kwargs...,
)
    path = joinpath(datadir, "lambert-2023", "data", "CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc")
    mask, z_draft = _read_v1_geometry(path; zvar = "draft")
    ocean = _v1_tanh_forcing(; T1 = 0.4, z0 = -500.0, z1 = 250.0, FT)
    return _v1_simulation(backend, mask, z_draft, 500.0; FT, params, ocean, dt = 120.0, days = 50.0,
                          simulation_kwargs)
end

"""
    SimulationFRISv1(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The Filchner–Ronne run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023)), on the 1 km domain stored
in `datadir/lambert-2023/data/FRIS_1.0_linear_S134.8_T1-2.3_720.nc`.

- forcing: v1 'linear' profile, T from the surface freezing point to −2.3 °C and S from
  34.5 to 34.8 psu at −2000 m (matches the file's ambient fields to round-off);
  `T_ice_base = -25 °C`
- `dt = 240` s, 720 days, Robert–Asselin `nu = 0.8`
- v1 walls (partial slip 1), [`PyGradient`](@ref)

`params` defaults to [`ParamsFRISv1`](@ref); other keywords go to [`Simulation`](@ref).

!!! note
    The paper builds the 1 km grid by averaging 2 × 2 BedMachine cells.  If that was
    done with the v1 code's `coarsen`, the run used Δx = 500 m in its derivatives
    (see [`SimulationASEv1`](@ref)); the stored file cannot tell, so Δx = 1 km is
    used here.
"""
function SimulationFRISv1(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsFRISv1(; FT),
    simulation_kwargs...,
)
    path = joinpath(datadir, "lambert-2023", "data", "FRIS_1.0_linear_S134.8_T1-2.3_720.nc")
    mask, z_draft = _read_v1_geometry(path; zvar = "draft")
    ocean = _v1_linear_forcing(; T1 = -2.3, S0 = 34.5, S1 = 34.8, z0 = -2000.0, FT)
    return _v1_simulation(backend, mask, z_draft, 1000.0; FT, params, ocean, dt = 240.0, days = 720.0,
                          simulation_kwargs)
end

"""
    SimulationPIGv1(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The Pine Island run of LADDIE v1.0 ([Lambert et al. 2023](@cite lambert_modelling_2023), App. C), on the 500 m domain
stored in `datadir/lambert-2023/data/PIG.nc`.

- geometry: the file holds no mask, so it is rebuilt from the stored ice thickness
  and bed by flotation with BedMachine's densities (917 and 1027 kg m⁻³)
- forcing: v1 'tanh' as Crosson–Dotson, but 1.0 °C at depth and the thermocline at
  −450 m; `T_ice_base = -25 °C`
- `dt = 120` s, 30 days, Robert–Asselin `nu = 0.8`
- v1 walls (partial slip 1), [`PyGradient`](@ref)

`params` defaults to [`ParamsPIGv1`](@ref); other keywords go to [`Simulation`](@ref).

!!! note
    The paper gives only the deep temperature and thermocline depth; the thermocline
    scale (250 m), `S0` and `drho0` are taken from the Crosson–Dotson profile it says
    this one adapts.
"""
function SimulationPIGv1(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsPIGv1(; FT),
    simulation_kwargs...,
)
    path = joinpath(datadir, "lambert-2023", "data", "PIG.nc")
    isfile(path) || throw(ArgumentError("geometry file not found: $path"))
    bed, thickness, zb = NCDatasets.NCDataset(path) do ds
        f(v) = Float64.(coalesce.(ds[v][:, :], 0.0))
        f("B"), f("H"), f("zb")
    end
    mask = build_laddie_mask(bed, thickness; rho_ice = 917.0, rho_sw = 1027.0)
    z_draft = _pad_ring(zb, 0.0)
    ocean = _v1_tanh_forcing(; T1 = 1.0, z0 = -450.0, z1 = 250.0, FT)
    return _v1_simulation(backend, mask, z_draft, 500.0; FT, params, ocean, dt = 120.0, days = 30.0,
                          simulation_kwargs)
end

# The v1 run of Lambert et al. (2026, Sect. 3.1) reads a 500 m geometry with
# `coarsen = 2`.  The v1 code takes `dx` from the input grid *before* coarsening and
# never updates it (`geometry.py`, `read_geom`), so the run was on the 1 km grid
# stored in its output but with every derivative divided by 500 m.
function _ase_v1_simulation(backend, dx; FT, datadir, params, simulation_kwargs)
    path = joinpath(datadir, "lambert-2026", "results", "laddie1", "ASE", "output_000020.nc")
    mask, z_draft = _read_v1_geometry(path; zvar = "zb")
    ocean = _v1_tanh_forcing(; T1 = 1.0, z0 = -450.0, z1 = 100.0, FT)
    return _v1_simulation(backend, mask, z_draft, dx; FT, params, ocean, dt = 30.0, days = 20.0,
                          simulation_kwargs)
end

"""
    SimulationASEv1(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The LADDIE v1 Amundsen Sea run of [Lambert et al. (2026, Sect. 3.1)](@cite lambert_one-layer_2026) **as it was run**,
on the 1 km domain stored in `datadir/lambert-2026/results/laddie1/ASE/output_000020.nc`.

The config reads a 500 m geometry with `coarsen = 2`, but the v1 code sets the grid
spacing before coarsening and never updates it.  The run was therefore on a 1 km grid
with Δx = 500 m in every derivative, and so is this simulation: `Grid(…, 500, 500)` on
the 1 km cells.  [`SimulationASEv1Theoretical`](@ref) is the consistent 1 km run.

- forcing: v1 'tanh', surface freezing point (S0 = 34) to 1.0 °C, thermocline at
  −450 m with scale 100 m, `drho0 = 0.01`; `T_ice_base = -25 °C`
- `dt = 30` s, 20 days, Robert–Asselin `nu = 0.8`
- v1 walls (partial slip 1), [`PyGradient`](@ref)

`params` defaults to [`ParamsASEv1`](@ref); other keywords go to [`Simulation`](@ref).
"""
SimulationASEv1(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsASEv1(; FT),
    simulation_kwargs...,
) = _ase_v1_simulation(backend, 500.0; FT, datadir, params, simulation_kwargs)

"""
    SimulationASEv1Theoretical(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The LADDIE v1 Amundsen Sea run of [Lambert et al. (2026, Sect. 3.1)](@cite lambert_one-layer_2026) **as it was meant to
run**: as [`SimulationASEv1`](@ref), but with the grid spacing consistent with the 1 km
cells.
"""
SimulationASEv1Theoretical(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsASEv1(; FT),
    simulation_kwargs...,
) = _ase_v1_simulation(backend, 1000.0; FT, datadir, params, simulation_kwargs)

# ----------------------------------------------------------------------------
# LADDIE v2: values shared by every v2 run config (`results/<run>/*.cfg` of
# Lambert 2026) and UPSY's hard-wired constants (`UPSY/basic/parameters.f90`).
#
# v2's `laddie_viscosity` is the coefficient of its shear-scaled viscosity, and the
# linear wall friction uses the same number, hence `A_h = C_visc`.  v2 floors the
# density contrast of its entrainment at the same buoyancy minimum as its convection,
# and caps detrainment at 1e-3 m s⁻¹ (`laddie_physics.f90`, `maxdetr`).
# ----------------------------------------------------------------------------

_v2_params(; C_d_top, melting, C_visc, K_h, H_min, H_init, v_cut, H_max, buoyancy_min) = (
    g = 9.81,
    C_d = 2.5e-3,
    C_d_top,
    A_h = C_visc,
    K_h,
    max_detrainment = 1e-3,
    drho_floor = buoyancy_min / 1027.0,
    H_min,
    v_cut,
    u_tide = 0.01,
    rho_freshwater = 1000.0,
    rho0_seawater = 1027.0,
    rho_ice = 917.0,
    L = 3.335e5,
    c_p = 3974.0,
    c_i = 2009.0,
    alpha = 3.733e-5,
    beta = 7.843e-4,
    l1 = -5.73e-2,
    l2 = 8.32e-2,
    l3 = 7.61e-4,
    H_init,
    dT_init = 0.0,
    dS_init = -0.1,
    seconds_per_year = 31_536_000.0,
    entrainment = GasparEntrainment(2.5),
    melting,
    convection_scheme = ClampDensity(buoyancy_min),
    ice_conduction = HollandJenkinsConduction(),
    equation_of_state = LinearEOS(),
    max_layer_thickness = AbsoluteMaxLayerThickness(H_max),
    lateral_viscosity = NonlinearLateralViscosity(C_visc),
    front_pressure = TruncatedDepthGradient(),
    momentum_advection = UpstreamMomentumAdvection(),
    tracer_form = FluxTracerForm(),
    coriolis = CoriolisParameter0D(-1.37e-4),
)

# No slip on every wall, no momentum across walls.
_v2_boundary(; gaps = SinkGapsBC()) =
    BoundaryConditions(; gaps, wall_advection = NoWallAdvection())

# v2 samples its idealised profiles on the ocean vertical grid (0:dz:depth_max, m
# below sea level) and interpolates linearly between levels, flat beyond them.
function _v2_profile_forcing(T, S; dz = 100.0, depth_max = 1500.0, FT)
    d = collect(0.0:dz:depth_max)
    return OceanForcing1D(T.(d), S.(d), -d; FT)
end

# UFEMISM 'TANH' (`ocean_idealised.f90`): surface freezing point at S0 = 34 to
# `T_deep`, thermocline at depth `z_tc` with scale `z_scale`, salinity compensating
# a density profile ∝ √depth (ρ_sw = 1027).
function _v2_tanh_forcing(; T_deep, z_tc = 450.0, z_scale = 100.0, FT, kwargs...)
    alpha, beta, rho_sw, S0, drho0 = 3.733e-5, 7.843e-4, 1027.0, 34.0, 0.01
    T_surf = -5.73e-2 * S0 + 8.32e-2
    T(d) = T_surf + (T_deep - T_surf) * (1 + tanh((d - z_tc) / z_scale)) / 2
    S(d) = S0 + alpha * (T(d) - T_surf) / beta + drho0 * sqrt(d) / (beta * rho_sw)
    return _v2_profile_forcing(T, S; FT, kwargs...)
end

# UFEMISM 'ISOMIP' warm: linear from (−1.9 °C, 33.8 psu) at the surface to
# (1.0 °C, 34.7 psu) at 720 m.
_v2_isomip_forcing(; FT) =
    _v2_profile_forcing(d -> -1.9 + 2.9 * d / 720, d -> 33.8 + 0.9 * d / 720; FT)

# A v2 gridded output (`laddie_output_grid.nc`) holds the run's ice thickness `Hi`,
# draft `Hib` and thickness above flotation `TAF`, remapped from the mesh to a regular
# grid.  Floating ice is shelf.  The remapping smears the mask along the grounding line
# and the domain edge (ISOMIP+: 512 floating cells where v2 had no layer, 104 with a
# layer that do not float); taking the shelf from v2's layer instead turns those cells
# into walls and fits v2 worse (ISOMIP+ melt pattern correlation 0.71 against 0.93).
# Ice-free cells are ocean (the file stores no bed), and a border ring of land closes
# the domain as the mesh border does.
function _read_v2_grid_geometry(path)
    isfile(path) || throw(ArgumentError("geometry file not found: $path"))
    Hi, Hib, TAF, dx = NCDatasets.NCDataset(path) do ds
        f(v) = Float64.(coalesce.(ds[v][:, :], 0.0))
        f("Hi"), f("Hib"), f("TAF"), Float64(ds["x"][2] - ds["x"][1])
    end
    mask = map((h, taf) -> h > 0 ? (taf > 0 ? 2 : 3) : 0, Hi, TAF)
    return _pad_ring(mask, 1), _pad_ring(Hib, 0.0), dx
end

function _v2_simulation(
    backend,
    mask,
    z_draft,
    dx;
    FT,
    params,
    ocean,
    dt,
    days,
    boundary = _v2_boundary(),
    simulation_kwargs,
)
    grid = Grid(mask, z_draft, dx, dx; backend, FT)
    forcing = CavityForcing(ocean, PrescribedIceForcing(T_ICE_BASE_V2))
    model = Model(grid; forcing, params, boundary)
    return Simulation(
        model;
        merge((; dt, stop = FixedSimulationEnd(days)), values(simulation_kwargs))...,
    )
end

const _V2_WARNING = """
!!! warning "Paper differs"
    UPSY hard-wires ρ₀ = 1027 kg m⁻³ and L = 3.335 × 10⁵ J kg⁻¹, used here; Lambert
    et al. (2026, Table 1) state 1028 and 3.34 × 10⁵.  The ice temperature of the
    standalone runs is UPSY's default 270 K (−3.15 °C), not v1's −25 °C.
"""

"""
    ParamsISOMIPv2(; FT = Float64, kwargs...)

`Params` of the ISOMIP+ Ocean1 run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026), Sect. 3.2.1),
from its config (`results/isomip_ocean1/isomip_ocean1.cfg`): uniform transfer
coefficient Γ_T = 2.78 × 10⁻² ([`UStarGamTMelting`](@ref)), `C_d_top = 2.5e-3`,
viscosity coefficient 0.1, `K_h = 1`, `H_min = H_init = 1`, `v_cut = 1`, `H ≤ 1500` m.

$_V2_WARNING
!!! warning "Paper differs"
    The config sets the viscosity coefficient to 0.1 and `K_h = 1`; Table 1 of the
    paper lists 1 and 4 as the defaults used.

See also [`SimulationISOMIPv2`](@ref).
"""
ParamsISOMIPv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 2.5e-3,
        melting = UStarGamTMelting(2.78e-2),
        C_visc = 0.1,
        K_h = 1.0,
        H_min = 1.0,
        H_init = 1.0,
        v_cut = 1.0,
        H_max = 1500.0,
        buoyancy_min = 5e-3,
    ),
    FT,
    kwargs,
)

"""
    ParamsAISv2(; FT = Float64, kwargs...)

`Params` of the pan-Antarctic run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026), Sect. 3.2.2),
from `laddie2-description-main/Antarctica.cfg` (the run left no config of its own;
this is the one whose 3 km triangles match the paper): [Jenkins (1991)](@cite jenkins_onedimensional_1991) transfer
coefficients, `C_d_top = 1.1e-3`, viscosity coefficient 0.1, `K_h = 1`, `H_min = 1`,
`H_init = 4`, `v_cut = 1`, `H ≤ 10000` m, buoyancy floor 0.02 kg m⁻³.

$_V2_WARNING
!!! warning "Paper differs"
    The config sets the viscosity coefficient to 0.1 and `K_h = 1`; Table 1 lists 1
    and 4.  The buoyancy floor of 0.02 kg m⁻³ is four times that of the other runs.

No `SimulationAISv2`: the run's BedMachine v3 geometry and ISMIP6 forcing are not
bundled.
"""
ParamsAISv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 1.1e-3,
        melting = TurbulentGamTMelting(),
        C_visc = 0.1,
        K_h = 1.0,
        H_min = 1.0,
        H_init = 4.0,
        v_cut = 1.0,
        H_max = 10000.0,
        buoyancy_min = 2e-2,
    ),
    FT,
    kwargs,
)

"""
    ParamsASEv2(; FT = Float64, kwargs...)

`Params` of the Amundsen Sea run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026), Sect. 3.1), from
its config (`results/ASE/ASE.cfg`): [Jenkins (1991)](@cite jenkins_onedimensional_1991) transfer coefficients,
`C_d_top = 1.1e-3`, viscosity coefficient 1, `K_h = 4`, `H_min = H_init = 1`,
`v_cut = 1.4`, `H ≤ 1500` m.

$_V2_WARNING
!!! warning "Paper differs"
    The config sets f = −1.37 × 10⁻⁴ s⁻¹; the paper states −1.34 × 10⁻⁴ s⁻¹.

See also [`SimulationASEv2`](@ref) and its v1 counterpart [`ParamsASEv1`](@ref).
"""
ParamsASEv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 1.1e-3,
        melting = TurbulentGamTMelting(),
        C_visc = 1.0,
        K_h = 4.0,
        H_min = 1.0,
        H_init = 1.0,
        v_cut = 1.4,
        H_max = 1500.0,
        buoyancy_min = 5e-3,
    ),
    FT,
    kwargs,
)

"""
    ParamsPIGv2(; FT = Float64, kwargs...)

`Params` of the 120 m Pine Island run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026),
Sect. 3.3.2), from its config (`results/PIG_rema3/PIG.cfg`): [Jenkins (1991)](@cite jenkins_onedimensional_1991) transfer
coefficients, `C_d_top = 1.1e-3`, viscosity coefficient 0.1, `K_h = 0.1`,
`H_min = H_init = 1`, `v_cut = 1.4`, `H ≤ 1500` m.

$_V2_WARNING
!!! warning "Paper differs"
    The paper says the configuration is that of its Table 1 (viscosity 1, `K_h = 4`);
    the config uses 0.1 and 0.1.

No `SimulationPIGv2`: the run's REMA/Bedmap v3 geometry is not bundled.
"""
ParamsPIGv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 1.1e-3,
        melting = TurbulentGamTMelting(),
        C_visc = 0.1,
        K_h = 0.1,
        H_min = 1.0,
        H_init = 1.0,
        v_cut = 1.4,
        H_max = 1500.0,
        buoyancy_min = 5e-3,
    ),
    FT,
    kwargs,
)

"""
    ParamsMISOMIPv2(; FT = Float64, kwargs...)

`Params` of the coupled UFEMISM–LADDIE MISOMIP+ run of [Lambert et al. (2026, Sect. 4)](@cite lambert_one-layer_2026),
from its config (`results/misomip_laddie/misomip_laddie.cfg`): Γ_T = 2.03 × 10⁻²
([`UStarGamTMelting`](@ref)), `C_d_top = 2.5e-3`, viscosity coefficient 0.1,
`K_h = 1`, `H_min = H_init = 1`, `v_cut = 1`, `H ≤ 1500` m.  Laddie.jl runs it
standalone, as a sanity check.

$_V2_WARNING
See also [`SimulationMISOMIPv2`](@ref).
"""
ParamsMISOMIPv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 2.5e-3,
        melting = UStarGamTMelting(2.03e-2),
        C_visc = 0.1,
        K_h = 1.0,
        H_min = 1.0,
        H_init = 1.0,
        v_cut = 1.0,
        H_max = 1500.0,
        buoyancy_min = 5e-3,
    ),
    FT,
    kwargs,
)

"""
    ParamsMISOMIPGapsv2(; FT = Float64, kwargs...)

`Params` of the coupled UFEMISM–LADDIE 2.0 runs of [Jesse et al. (2026)](@cite jesse_ice-sheet_2026), from their
Table A1 (no run config is bundled): Γ_T = 3 × 10⁻² ([`UStarGamTMelting`](@ref)),
`C_d_top = 2.5e-3`, viscosity coefficient and `K_h` 10, `H_min = 1`.  The values the
table leaves out (`H_init = 1`, `v_cut = 1`, `H ≤ 1500` m, buoyancy floor
0.005 kg m⁻³) are those of the MISOMIP+ run of [Lambert et al. (2026)](@cite lambert_one-layer_2026), see
[`ParamsMISOMIPv2`](@ref).  Laddie.jl runs it standalone, as a sanity check.

$_V2_WARNING
See also [`SimulationMISOMIPGapsv2`](@ref).
"""
ParamsMISOMIPGapsv2(; FT = Float64, kwargs...) = _reproduction_params(
    _v2_params(;
        C_d_top = 2.5e-3,
        melting = UStarGamTMelting(3.0e-2),
        C_visc = 10.0,
        K_h = 10.0,
        H_min = 1.0,
        H_init = 1.0,
        v_cut = 1.0,
        H_max = 1500.0,
        buoyancy_min = 5e-3,
    ),
    FT,
    kwargs,
)

"""
    SimulationISOMIPv2(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The ISOMIP+ Ocean1 run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026), Sect. 3.2.1), on the 2 km
Ocean1 geometry remapped to a regular grid in
`datadir/lambert-2026/results/isomip_ocean1/laddie_output_grid.nc`; the shelf is the
ice that floats there.

- forcing: ISOMIP+ warm, sampled every 100 m as v2 does; `T_ice_base = -3.15 °C`
- `dt = 90` s, 20 days
- v2 walls (no slip, no momentum across walls)

`params` defaults to [`ParamsISOMIPv2`](@ref); other keywords go to [`Simulation`](@ref).
Laddie.jl integrates with leapfrog on a C-grid, not v2's FB-RK3 on a mesh.
"""
function SimulationISOMIPv2(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsISOMIPv2(; FT),
    simulation_kwargs...,
)
    path = joinpath(datadir, "lambert-2026", "results", "isomip_ocean1", "laddie_output_grid.nc")
    mask, z_draft, dx = _read_v2_grid_geometry(path)
    return _v2_simulation(backend, mask, z_draft, dx; FT, params, ocean = _v2_isomip_forcing(; FT),
                          dt = 90.0, days = 20.0, simulation_kwargs)
end

"""
    SimulationASEv2(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The Amundsen Sea run of LADDIE v2.0 ([Lambert et al. 2026](@cite lambert_one-layer_2026), Sect. 3.1), on its geometry
remapped to the regular 1 km grid in `datadir/lambert-2026/results/ASE/laddie_output_grid.nc`
(the run used 1.2 km triangles); the shelf is the ice that floats there.

- forcing: UFEMISM 'TANH', surface freezing point to 1.0 °C, thermocline at 450 m with
  scale 100 m, sampled every 100 m; `T_ice_base = -3.15 °C`
- `dt = 30` s, 20 days
- v2 walls (no slip, no momentum across walls)

`params` defaults to [`ParamsASEv2`](@ref); other keywords go to [`Simulation`](@ref).
Its v1 counterpart is [`SimulationASEv1`](@ref).
"""
function SimulationASEv2(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsASEv2(; FT),
    simulation_kwargs...,
)
    path = joinpath(datadir, "lambert-2026", "results", "ASE", "laddie_output_grid.nc")
    mask, z_draft, dx = _read_v2_grid_geometry(path)
    ocean = _v2_tanh_forcing(; T_deep = 1.0, FT)
    return _v2_simulation(backend, mask, z_draft, dx; FT, params, ocean, dt = 30.0, days = 20.0,
                          simulation_kwargs)
end

# The MISOMIP+ ice geometry after the UFEMISM spin-up of Lambert et al. (2026), last
# time slice of its gridded output (2 km).
function _read_misomip_spinup(datadir)
    path = joinpath(datadir, "lambert-2026", "results", "misomip_spinup_1", "main_output_ANT_grid.nc")
    isfile(path) || throw(ArgumentError("geometry file not found: $path"))
    bed, thickness, dx = NCDatasets.NCDataset(path) do ds
        f(v) = Float64.(coalesce.(ds[v][:, :, end], 0.0))
        f("Hb"), f("Hi"), Float64(ds["x"][2] - ds["x"][1])
    end
    mask = build_laddie_mask(bed, thickness; rho_ice = 917.0, rho_sw = 1027.0)
    z_draft = ice_base_depth(bed, thickness; rho_ice = 917.0, rho_sw = 1027.0)
    return mask, z_draft, dx
end

"""
    SimulationMISOMIPv2(backend = CPU(); FT = Float64, datadir, params, simulation_kwargs...)

The initial LADDIE spin-up of the coupled MISOMIP+ run of [Lambert et al. (2026,
Sect. 4)](@cite lambert_one-layer_2026), standalone on the spun-up ice geometry (last slice of
`datadir/lambert-2026/results/misomip_spinup_1/main_output_ANT_grid.nc`, 2 km; the
coupled run used 1 km triangles on the shelf).  A sanity check, not a coupled run.

- forcing: UFEMISM 'TANH', surface freezing point to 1.5 °C, thermocline at 450 m with
  scale 100 m, sampled every 100 m; `T_ice_base = -3.15 °C`
- `dt = 60` s, 10 days (the config's initial LADDIE run)
- v2 walls (no slip, no momentum across walls)

`params` defaults to [`ParamsMISOMIPv2`](@ref); other keywords go to [`Simulation`](@ref).

!!! warning "Paper differs"
    The config sets 1.5 °C at depth; the paper describes the forcing of Jesse et al.
    (2025), with 1.2 °C.
"""
function SimulationMISOMIPv2(
    backend = CPU();
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsMISOMIPv2(; FT),
    simulation_kwargs...,
)
    mask, z_draft, dx = _read_misomip_spinup(datadir)
    ocean = _v2_tanh_forcing(; T_deep = 1.5, FT)
    return _v2_simulation(backend, mask, z_draft, dx; FT, params, ocean, dt = 60.0, days = 10.0,
                          simulation_kwargs)
end

const _JESSE_DEEP_T = (ocean1 = -1.0, ocean2 = 0.0, ocean3 = 1.0)

"""
    SimulationMISOMIPGapsv2(backend = CPU(); ocean = :ocean3, gaps = SinkGapsBC(), FT = Float64,
                            datadir, params, simulation_kwargs...)

The initial LADDIE spin-up of the coupled runs of [Jesse et al. (2026)](@cite jesse_ice-sheet_2026), standalone, as a
sanity check.  [Jesse et al.](@cite jesse_ice-sheet_2026)'s spun-up geometry is not bundled (their output is on the
mesh only); this uses the MISOMIP+ spin-up of [Lambert et al. (2026)](@cite lambert_one-layer_2026) instead, as in
[`SimulationMISOMIPv2`](@ref), whose grounding line also sits near x = 450 km.

- `ocean`: `:ocean1`, `:ocean2` or `:ocean3`, deep temperature −1, 0 or 1 °C under a
  thermocline at 450 m (UFEMISM 'TANH', which keeps the density profile identical
  across the three, as the paper requires); `T_ice_base = -3.15 °C`
- `gaps`: [`SinkGapsBC`](@ref) (the paper's "sink", LADDIE's default) or
  [`ConnectedGapsBC`](@ref) ("connected"); without gaps in the initial geometry both
  give the same run
- `dt = 100` s, 20 days

`params` defaults to [`ParamsMISOMIPGapsv2`](@ref); other keywords go to [`Simulation`](@ref).

!!! warning "Paper differs"
    The paper fixes the surface at −1.8 °C; UFEMISM 'TANH' puts it at the freezing
    point of 34 psu, −1.865 °C.
"""
function SimulationMISOMIPGapsv2(
    backend = CPU();
    ocean = :ocean3,
    gaps = SinkGapsBC(),
    FT = Float64,
    datadir = DEFAULT_PAPERS_DIR,
    params = ParamsMISOMIPGapsv2(; FT),
    simulation_kwargs...,
)
    haskey(_JESSE_DEEP_T, ocean) ||
        throw(ArgumentError("ocean must be :ocean1, :ocean2 or :ocean3, got :$ocean"))
    mask, z_draft, dx = _read_misomip_spinup(datadir)
    forcing = _v2_tanh_forcing(; T_deep = _JESSE_DEEP_T[ocean], FT)
    return _v2_simulation(backend, mask, z_draft, dx; FT, params, ocean = forcing, dt = 100.0,
                          days = 20.0, boundary = _v2_boundary(; gaps), simulation_kwargs)
end
