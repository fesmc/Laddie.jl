using Laddie
using NCDatasets
using CairoMakie
using DelimitedFiles
using KernelAbstractions
using CUDA

# =============================================================================
# Ocean forcing profile
# =============================================================================
# CSV layout: column 1 = T or S value, column 2 = depth in km (negative down).
# The T and S profiles were digitised separately, so they sample different
# depths — interpolate S onto the temperature depths before building the
# forcing.

fn_T = "assets/crosson-dotson-T.csv"
fn_S = "assets/crosson-dotson-S.csv"

T_data = readdlm(fn_T, ',', skipstart = 1)
S_data = readdlm(fn_S, ',', skipstart = 1)

z_forc = T_data[:, 2] .* 1e3    # km → m
T_forc = T_data[:, 1]
S_forc = Laddie._interp1d(reverse(S_data[:, 2] .* 1e3), reverse(S_data[:, 1]), z_forc)

# ProfileForcing sorts by depth and resamples to the 1-m grid the model needs,
# with flat extrapolation beyond the data range.
forcing = ProfileForcing(T_forc, S_forc, z_forc)

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "Temperature (°C)", ylabel = "Depth (m)")
ax2 = Axis(fig[1, 2], xlabel = "Salinity (PSU)",   ylabel = "Depth (m)")
lines!(ax1, forcing.Tz, forcing.z)
lines!(ax2, forcing.Sz, forcing.z)
fig

# =============================================================================
# BedMachine geometry
# =============================================================================
i1, i2 = 3445, 3720
j1, j2 = 7732, 8070
fn_topo = "/home/jan/Documents/projects/esm-datasets/data/topography/src/BedMachineAntarctica-v4.nc"
ds    = Dataset(fn_topo)
z_bed = Float64.(Array(ds["bed"][i1:i2, j2:-1:j1]))
h_ice = Float64.(Array(ds["thickness"][i1:i2, j2:-1:j1]))
close(ds)

dx = 500.0   # BedMachine v3 resolution: 500 m
dy = 500.0




# Derive the 4-class LADDIE mask and ice-base draft from BedMachine arrays.
# build_laddie_mask classifies each interior cell as:
#   0 = open ocean, 1 = border, 2 = grounded ice, 3 = floating shelf
# ice_base_depth returns the ice-draft elevation in metres (negative below sea level).
mask = build_laddie_mask(z_bed, h_ice)
fig_map = Figure()
ax1 = Axis(fig_map[1, 1], aspect = DataAspect())
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
Colorbar(fig_map[2, 1], hm, vertical = false, flipaxis = false, label = "LADDIE mask class",
    ticks = ([0, 1, 2, 3], ["ocean", "border", "grounded", "shelf"]))
fig_map

n = fill_ocean_holes!(mask)
n > 0 && @info "fill_ocean_holes!: reclassified $n isolated ocean cells as grounded"
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
fig_map

n = fill_shelf_holes!(mask)
n > 0 && @info "fill_shelf_holes!: reclassified $n isolated shelf cells as grounded"
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
fig_map

n = fill_small_shelf_patches!(mask, 10)
n > 0 && @info "fill_small_shelf_patches!: removed $n cells in undersized shelf patches"
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
fig_map

n = fill_small_grounded_patches!(mask, 3)
n > 0 && @info "fill_small_grounded_patches!: reclassified $n cells in undersized isolated grounded patches"
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
fig_map


zb       = ice_base_depth(z_bed, h_ice)
z_bed_m  = bed_elevation(z_bed)

z_bed_ocean = copy(z_bed)
z_bed_ocean[h_ice .> 0] .= NaN
cmap_ocean = cgrad([:midnightblue, :cornflowerblue])
z_bed_grounded = copy(z_bed)
z_bed_grounded[h_ice .<= 0] .= NaN
cmap_grounded = cgrad([:gray20, :gray80])
zb_plot = copy(zb)
zb_plot[mask .< 3] .= NaN
cmap_shelfbase = cgrad(:tempo, rev = true)

ax2 = Axis(fig_map[1, 2], aspect = DataAspect())
heatmap!(ax2, z_bed_ocean ./ 1f3, colormap = cmap_ocean, colorrange = (-1, 1))
heatmap!(ax2, z_bed_grounded ./ 1f3, colormap = cmap_grounded, colorrange = (-1, 1))
heatmap!(ax2, zb_plot ./ 1f3, colormap = cmap_shelfbase, colorrange = (-2, 0))
fig_map

# =============================================================================
# Build and run model — snapshot all fields + masks every 5th step
# =============================================================================
# mp = PrescribedMelt{Float64}()
mp = TurbulentGamT()
# mp = FixedGamT(0.00018)
params = Params(; dt = 120, A_h = 25, K_h = 25, D_min = 2.8, nu = 0.1, D_init = 2.8,
    tstep = AdaptiveDt(), melting = mp, grline_bc = NoSlipGL(),
    max_layer_thickness = AbsoluteMaxLayerThickness(30))

m = build_model(mask, zb, dx, dy, forcing, params;
    z_bed_raw = z_bed_m,
    config = RunConfig(; saveday = 0.1, dbg = DebugConfig(check_nans = true)),
    backend = CUDABackend(),
)
run!(m; days = 3, verbose = true)


# Colors sampled directly from the source figure (pale cyan -> blue -> dark navy
# -> magenta -> orange -> pale yellow), 40 stops, light->dark->light diverging map
colors = [
    colorant"#ebfcfc", colorant"#d8f2f3", colorant"#bce4e4", colorant"#9dd4d7",
    colorant"#83c5d3", colorant"#6fb5ce", colorant"#5fa4c5", colorant"#5194c1",
    colorant"#4783bb", colorant"#4071b3", colorant"#3e61ab", colorant"#3e509a",
    colorant"#3c3f84", colorant"#323267", colorant"#2a2850", colorant"#1b1a37",
    colorant"#210b4b", colorant"#350960", colorant"#450a68", colorant"#560f6d",
    colorant"#64156e", colorant"#741b6d", colorant"#842069", colorant"#932669",
    colorant"#a32d61", colorant"#b33259", colorant"#c23a50", colorant"#d04447",
    colorant"#db503b", colorant"#e55c30", colorant"#ee6923", colorant"#f47d15",
    colorant"#f98c09", colorant"#fc9f06", colorant"#feb117", colorant"#fbc42b",
    colorant"#f3d848", colorant"#f4ea6e", colorant"#f4f992", colorant"#fbffa2",
]
cmap = cgrad(colors)

fig_melt = Figure()
ax = Axis(fig_melt[1, 1], aspect = DataAspect())
hidedecorations!(ax)
hm = heatmap!(
    ax,
    Array(m.cache.melt) .* (3600 * 24 * 365.25),
    colormap = cmap,
    colorrange = (-10, 100),
    colorscale = Makie.Symlog10(0.3),
    lowclip    = colors[1],
    highclip   = colors[end],
)
Colorbar(fig_melt[2, 1], hm;
    vertical   = false,
    flipaxis   = false,
    ticks      = [-10, -3, -1, -0.3, 0, 0.3, 1, 3, 10, 30, 100],
    label      = L"\mathrm{Freezing\ /\ Melt\ rate\ } \dot{n}\ [\mathrm{m\ yr^{-1}}]",
    width = Relative(0.5),
    height = 20,
)
fig_melt

mx, mn, sp = meltstats(m)
println("max melt = $(round(mx, digits=2)) m/yr,  mean = $(round(mn, digits=2)) m/yr")