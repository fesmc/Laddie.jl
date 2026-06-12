using Laddie
using NCDatasets
using CairoMakie
using DelimitedFiles

# =============================================================================
# Ocean forcing profile
# =============================================================================
# CSV layout: column 1 = T or S value, column 2 = depth in km (negative down).
# The T and S profiles were digitised separately, so they sample different
# depths — interpolate S onto the temperature depths before building the
# forcing.

fn_T = "docs/assets/crosson-dotson-T.csv"
fn_S = "docs/assets/crosson-dotson-S.csv"

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
display(fig)

# =============================================================================
# BedMachine geometry
# =============================================================================
i1, i2 = 3400, 3700
j1, j2 = 7800, 8200
fn_topo = "/home/jan/pCloudSync/PhD/Projects/Isostasy/GRDMIP/GRDMIP-Paleo/preprocessing/topography/data/BedMachine/BedMachineAntarctica-v3.nc"
ds    = Dataset(fn_topo)
z_bed = Float64.(Array(ds["bed"][i1:i2, j1:j2]))
h_ice = Float64.(Array(ds["thickness"][i1:i2, j1:j2]))
close(ds)

dx = 500.0   # BedMachine v3 resolution: 500 m
dy = 500.0

# Derive the 4-class LADDIE mask and ice-base draft from BedMachine arrays.
# build_laddie_mask classifies each interior cell as:
#   0 = open ocean, 1 = border, 2 = grounded ice, 3 = floating shelf
# ice_base_depth returns the ice-draft elevation in metres (negative below sea level).
mask = build_laddie_mask(z_bed, h_ice)
zb   = ice_base_depth(z_bed, h_ice)

# =============================================================================
# Build and run model
# =============================================================================
params = Params(; FT = Float64)

m = build_model(mask, zb, dx, dy, forcing, params)
run!(m; days = 5.0)

mx, mn, sp = meltstats(m)
println("max melt = $(round(mx, digits=2)) m/yr,  mean = $(round(mn, digits=2)) m/yr")
