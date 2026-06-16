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
j1, j2 = 7710, 8070
fn_topo = "/home/jan/pCloudSync/PhD/Projects/Isostasy/GRDMIP/GRDMIP-Paleo/preprocessing/topography/data/BedMachine/BedMachineAntarctica-v3.nc"
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
zb   = ice_base_depth(z_bed, h_ice)

z_bed_ocean = copy(z_bed)
z_bed_ocean[h_ice .> 0] .= NaN
cmap_ocean = cgrad([:midnightblue, :cornflowerblue])
z_bed_grounded = copy(z_bed)
z_bed_grounded[h_ice .<= 0] .= NaN
cmap_grounded = cgrad([:gray20, :gray80])
zb[mask .< 3] .= NaN
cmap_shelfbase = cgrad(:tempo, rev = true)

fig_map = Figure()
ax1 = Axis(fig_map[1, 1], aspect = DataAspect())
hm = heatmap!(
    ax1,
    mask,
    colormap = cgrad(:viridis, range(0, stop = 1, length = 5), categorical = true),
    colorrange = (0, 3))
Colorbar(fig_map[2, 1], hm, vertical = false, flipaxis = false, label = "LADDIE mask class",
    ticks = ([0, 1, 2, 3], ["ocean", "border", "grounded", "shelf"]))

ax2 = Axis(fig_map[1, 2], aspect = DataAspect())
heatmap!(ax2, z_bed_ocean ./ 1f3, colormap = cmap_ocean, colorrange = (-1, 1))
heatmap!(ax2, z_bed_grounded ./ 1f3, colormap = cmap_grounded, colorrange = (-1, 1))
heatmap!(ax2, zb ./ 1f3, colormap = cmap_shelfbase, colorrange = (-2, 0))
fig_map


# h_af = h_above_flotation.(z_bed, h_ice)
# z_base = ice_base.(z_bed, h_ice, h_af)
# shelf_mask = h_ice .> 0 .&& h_af .< 0
# fig_map = Figure()


# =============================================================================
# Build and run model — snapshot all fields + masks every 5th step
# =============================================================================
params = Params(; tstep = FixedDt())

m = build_model(mask, zb, dx, dy, forcing, params)
run!(m; days = 5)

# The built-in periodic output (RunConfig `saveday`) writes day-AVERAGED fields
# and names files by whole day, so a sub-daily cadence would overwrite a single
# file.  For instantaneous per-step snapshots we step the model in 5-step chunks
# and write our own NetCDF — all prognostics, the melt/entrainment fields, and
# the masks — with a sequential filename.  Tune `nstep` / `total_days` below.
outdir = joinpath("output", "crosson-dotson")
mkpath(outdir)

xc = collect((0:m.nx-1) .* dx)
yc = collect((0:m.ny-1) .* dy)
strip_halo(a) = Array(a)[2:end-1, 2:end-1]

function save_snapshot(m, path, t_days)
    NCDataset(path, "c") do ds
        defDim(ds, "x", m.nx)
        defDim(ds, "y", m.ny)
        defVar(ds, "x", Float64, ("x",))[:] = xc
        defVar(ds, "y", Float64, ("y",))[:] = yc
        ds.attrib["time_days"] = t_days
        fields = (
            ("D",    m.D.present,          "m"),
            ("U",    m.U.present,          "m s-1"),
            ("V",    m.V.present,          "m s-1"),
            ("T",    m.T.present,          "degC"),
            ("S",    m.S.present,          "psu"),
            ("melt", m.melt .* Laddie.spy, "m yr-1"),
            ("entr", m.entr .* Laddie.spy, "m yr-1"),
            ("ent2", m.ent2 .* Laddie.spy, "m yr-1"),
            ("detr", m.detr .* Laddie.spy, "m yr-1"),
            ("Tb",   m.Tb,                 "degC"),
            ("Ta",   m.Ta,                 "degC"),
            ("zb",   m.zb,                 "m"),
        )
        for (name, field, units) in fields
            defVar(ds, name, Float64, ("y", "x"); attrib = ["units" => units])[:, :] =
                strip_halo(field)
        end
        defVar(ds, "tmask", Int32, ("y", "x"))[:, :] = Int32.(strip_halo(m.tmask))
        defVar(ds, "mask", Int32, ("y", "x"))[:, :] = Int32.(strip_halo(m.mask))
    end
end

nstep      = 5                              # snapshot cadence (time steps)
total_days = 5.0                            # total run length (days)
chunk      = nstep * params.dt0 / 86400.0   # 5 steps expressed in days
ndumps     = round(Int, total_days / chunk)

save_snapshot(m, joinpath(outdir, "snap_00000.nc"), 0.0)   # initial state
for k = 1:ndumps
    @show k
    run!(m; days = chunk, verbose = false)
    save_snapshot(m, joinpath(outdir, "snap_$(lpad(k, 5, '0')).nc"), k * chunk)
end

mx, mn, sp = meltstats(m)
println("max melt = $(round(mx, digits=2)) m/yr,  mean = $(round(mn, digits=2)) m/yr")
