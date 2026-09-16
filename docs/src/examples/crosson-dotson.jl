#=
# Crosson–Dotson: reproducing LADDIE v1 (Lambert et al., 2023)

This script reproduces the published Crosson–Dotson simulation of
[Lambert et al. (2023)](https://doi.org/10.5194/tc-17-3203-2023) — the run behind Figs. 3–4
of that paper — and compares the result to the reference output when it is available.

The code on this page is **not executed** when the documentation is built: it needs
BedMachine v2 on disk and the reference file, and the 50-day run at 500 m takes about
10 minutes on 16 CPU threads. The figure and numbers below come from running this exact
script locally. It is the configuration to copy for a realistic cavity, and the settings
below are the reference's, not Laddie.jl's defaults.

Reference diagnostics (5-day average over days 45–50, over the shelf mask):
mean melt 9.82 m yr⁻¹, max melt 114.3 m yr⁻¹, `D` between 2.7 and 510 m, max speed 0.68 m s⁻¹.

Points worth carrying to other setups:

  * `nu = 0.8`. The Robert–Asselin coefficient is a stability parameter, not a detail:
    `nu = 0.1` at `dt = 120 s` is unstable on this geometry and inflates the mean melt
    rate roughly sevenfold. The paper makes the same remark (Sect. 2.1.3).
  * No upper bound on the layer thickness (the default, [`NoMaxLayerThickness`](@ref)).
    Capping `D` at the water column collapses melt by ~90 % here, because the clamp
    removes volume without removing momentum or heat; see the warning on
    [`AbstractMaxLayerThickness`](@ref).
  * `ClampDensity` convection — the buoyancy floor — is what the reference does. The
    default `ResetToAmbient` is v1.1's `convop = 1`, and it is worth −0.07 % on this run.
  * `PyGradient` matches the reference's `np.gradient` ice-base slope. The default
    `JlGradient` is mask-aware and arguably better, and costs −0.8 % here.
=#

using Laddie
using NCDatasets
using Printf
using Statistics
using CairoMakie

FT = Float64

# Optional GPU. `CUDA` is not a dependency of Laddie, so this stays opt-in.
const USE_GPU = false
if USE_GPU
    using CUDA
    backend = CUDABackend()
else
    using KernelAbstractions
    backend = CPU()
end

# Paths to the BedMachine geometry and (optionally) the reference output.
const BEDMACHINE =
    "/home/jan/Documents/projects/esm-datasets/data/topography/src/BedMachineAntarctica-v2.nc"
const REFERENCE = joinpath(
    pkgdir(Laddie), "papers", "lambert-2023", "data",
    "CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc",
)

# ## Geometry
#
# The reference cuts Crosson–Dotson out of BedMachine v2 with
# `isel(x=3445:3705, y=7730:8065)` (0-based, end-exclusive) at the native 500 m spacing,
# with no coarsening and no smoothing of the ice draft.
#
# Mask conventions differ by one value: BedMachine marks ice-free land `1`, which the
# reference relabels as grounded (`2`); Laddie.jl keeps `1` as land, a wall that can take
# its own slip condition ([`NoSlipLand`](@ref)). This domain contains no ice-free land, so
# the two agree here.

xs, ys = 3446:3705, 7731:8065          # 1-based equivalents of the reference's slices

ds = Dataset(BEDMACHINE)
## NCDatasets hands back [x, y], which is Laddie.jl's own order; BedMachine stores y
## descending, so only the y axis needs flipping.
xy(v) = Float64.(coalesce.(Array(ds[v][xs, ys]), NaN))[:, end:-1:1]
bed       = xy("bed")
surface   = xy("surface")
thickness = xy("thickness")
mask      = Int.(xy("mask"))
close(ds)

dx = dy = 500.0

## Ice-base draft, and the reference's shallow-draft clamp of −10 m (Laddie.jl's own
## `_adjust_z_draft` would clamp at −1 m, worth +0.0 % here).
z_draft = surface .- thickness
z_draft[(mask .== 3) .& (z_draft .> -10.0)] .= -10.0

## No border ring is added: the domain already ends in ocean and grounded ice on all four
## sides, which is what the periodic stencils need. Hence `NoDomainCropping`.
grid = Grid(mask, z_draft, dx, dy; domain_cropping = NoDomainCropping(), backend, FT)

# ## Forcing: the analytic two-layer profile of the paper
#
# `tanh(ztcl = 500, Tdeep = 0.4, z1 = 250)`: a thermocline centred at 500 m depth with a
# 250 m scale, from the surface freezing point to +0.4 °C at depth. Salinity compensates a
# prescribed quadratic density profile, so the stratification is stable by construction.

l1, l2 = -5.73e-2, 8.32e-2              # liquidus coefficients, as in Params()
alpha, beta, rho0 = 3.733e-5, 7.843e-4, 1028.0
S0, T_deep, z_tcl, z_scale, drho0 = 34.0, 0.4, -500.0, 250.0, 0.01

z = collect(-5000.0:1.0:-1.0)
T0 = l1 * S0 + l2                       # surface freezing temperature
Tz = @. T_deep + (T0 - T_deep) * (1 + tanh((z - z_tcl) / z_scale)) / 2
Sz = @. S0 + alpha * (Tz - T0) / beta + drho0 * sqrt(abs(z)) / (beta * rho0)

forcing = CavityForcing(OceanForcing1D(Tz, Sz, z; FT), PrescribedIceForcing(-25.0))

# ## Parameters and run
#
# `Ah = Kh = 25` m² s⁻¹ and `D_min = 2.8` m are the paper's resolution-dependent and tuned
# values for 500 m (Table 2); `C_d_top = 1.1e-3` is the other tuning parameter.

params = Params(; FT,
    A_h = 25.0, K_h = 25.0,
    C_d = 2.5e-3, C_d_top = 1.1e-3,
    D_min = 2.8, u_tide = 0.01, slip = 1.0,
    max_detrainment = 0.5, v_cut = 1.414,
    D_init = 10.0, dT_init = -0.1, dS_init = -0.1,
    coriolis = CoriolisParameter0D(-1.37e-4),
    entrainment = LambertEntrainment(2.5),          # Gaspar/Gladish mu = 2.5
    melting = TurbulentGamTMelting(13.8, 2432.0, 1.95e-6),
    convection_scheme = ClampDensity(0.005),        # the reference's buoyancy floor
    max_layer_thickness = NoMaxLayerThickness(),    # the default; do not cap D here
)

model = Model(grid; forcing, params, gradient = PyGradient())
sim = Simulation(model; dt = 120.0, nu = 0.8)

## The reference spins up for 50 days and reports the average of the last 5.
run!(sim; days = 45.0)

m = sim.model
acc = Dict(k => zero(m.tmask) for k in ("melt", "D", "T", "S"))
nsteps = round(Int, 5 * 86400 / sim.clock.dt)
for _ in 1:nsteps
    time_step!(sim)
    acc["melt"] .+= m.melt
    acc["D"] .+= m.D.present
    acc["T"] .+= m.T.present
    acc["S"] .+= m.S.present
end
avg = Dict(k => Array(v) ./ nsteps for (k, v) in acc)
avg["melt"] .*= 86400 * 365.25          # m s⁻¹ → m yr⁻¹

shelf = Array(m.imask) .> 0
@printf "mean melt %.3f m/yr, max melt %.1f m/yr, D in [%.1f, %.0f] m\n" mean(avg["melt"][shelf]) maximum(avg["melt"][shelf]) minimum(avg["D"][shelf]) maximum(avg["D"][shelf])

# ## Comparison with the reference output
#
# The reference file stores the same 5-day average, on the same grid, so the comparison is
# cell by cell. Expect agreement to well under 1 % in the mean — the residual is dominated
# by the reference's own code-level quirks, not by the physics.

if isfile(REFERENCE)
    dsr = Dataset(REFERENCE)
    ref = Dict(v => coalesce.(Array(dsr[v]), NaN) for v in ("melt", "D", "T", "S"))
    ref_shelf = Array(dsr["tmask"]) .== 1
    close(dsr)

    @printf "reference: mean melt %.3f m/yr, max %.1f m/yr\n" mean(ref["melt"][ref_shelf]) maximum(ref["melt"][ref_shelf])
    for v in ("melt", "D", "T", "S")
        d = abs.(avg[v][ref_shelf] .- ref[v][ref_shelf])
        @printf "  %-4s mean |Δ| %8.4f   max |Δ| %8.3f\n" v mean(d) maximum(d)
    end
else
    @info "Reference output not found at $REFERENCE — skipping the comparison."
end

# ## Figure
#
# Melt rate and layer thickness, Laddie.jl against the reference, averaged over days 45–50.
# The figure is written to `docs/src/assets/` and committed, since the docs build does
# not run this script.

if isfile(REFERENCE)
    x_km = (0:size(mask, 1)-1) .* dx ./ 1e3
    y_km = (0:size(mask, 2)-1) .* dy ./ 1e3
    on_shelf(A) = map((s, a) -> s ? a : NaN, ref_shelf, A)

    fig = Figure(size = (1100, 900))
    rows = (("melt", "melt (m/yr)", (0.0, 60.0), 5.0, :inferno),
            ("D", "D (m)", (0.0, 200.0), 5.0, :viridis))
    for (i, (v, label, crange, dmax, cmap)) in enumerate(rows)
        panels = ((avg[v], "Laddie.jl", crange, cmap),
                  (ref[v], "reference", crange, cmap),
                  (avg[v] .- ref[v], "Laddie.jl − reference", (-dmax, dmax), :RdBu))
        for (j, (A, title, cr, cm)) in enumerate(panels)
            ax = Axis(fig[i, 2j-1]; title = "$label: $title", aspect = DataAspect(),
                      xlabel = i == 2 ? "x (km)" : "", ylabel = j == 1 ? "y (km)" : "")
            hm = heatmap!(ax, x_km, y_km, on_shelf(A); colormap = cm, colorrange = cr)
            Colorbar(fig[i, 2j], hm)
        end
    end
    save(joinpath(pkgdir(Laddie), "docs", "src", "assets", "crosson-dotson.png"), fig)
end

# ## Results
#
# Output of the script above (CPU, 16 threads, Float64; 10 min wall time), compared with
# the reference file over its shelf mask:
#
# |                      | Laddie.jl | reference |
# |:---------------------|----------:|----------:|
# | mean melt (m yr⁻¹)   | 9.839     | 9.818     |
# | max melt (m yr⁻¹)    | 114.2     | 114.3     |
# | `D` range (m)        | 2.8–509   | 2.7–510   |
#
# The mean melt rate agrees to +0.21 %. Cell-by-cell mean absolute differences are
# 0.13 m yr⁻¹ in melt, 0.62 m in `D`, 0.006 °C in `T` and 0.0015 in `S`.
#
# ![Crosson–Dotson: Laddie.jl vs reference](../assets/crosson-dotson.png)
