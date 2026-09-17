using JLD2
using NCDatasets
using Printf
using TOML


# ============================================================================
# DebugConfig — optional debug options (all fields default to off).
# ============================================================================

"""
$(TYPEDEF)

Optional debug configuration passed via `Simulation(model; debug = DebugConfig(...))`.

Set `check_nans = true` to check every prognostic variable for NaNs over the
shelf mask after each sub-step of `leapfrog_step!`.  When a NaN is found the
run errors immediately, naming the first variable to blow up and the time.
"""
Base.@kwdef struct DebugConfig
    check_nans::Bool = false
end

# ============================================================================
# OutputConfig — static output configuration (all fields have defaults).
# ============================================================================

"""
$(TYPEDEF)

Output configuration of a [`Simulation`](@ref): run directory, output cadence,
restart cadence, and which fields are written.  All fields have sensible
defaults; a plain `OutputConfig()` disables file I/O (`saveday = 0`).

Set `saveday > 0` to enable file I/O: time-averaged fields in `output.nc` every
`saveday` days, a line of diagnostics in the log every `diagday` days, and a JLD2
restart every `restday` days (plus one at the end of every `run!`).  Event times
are absolute on the simulation clock, so a simulation advanced by several `run!`
calls writes on the same cadence as one advanced by a single call.

```julia
OutputConfig(; name = "warm", saveday = 1.0, save_entr = true)
```

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct OutputConfig
    "run name; files go to `joinpath(resultdir, name)`"
    name::String = "run"
    "averaging and output interval in days; `0` disables all file I/O"
    saveday::Float64 = 0.0
    "interval of the log-file diagnostics in days (must be positive)"
    diagday::Float64 = 1.0
    "interval of the restart files in days (must be positive)"
    restday::Float64 = 30.0
    "parent directory of the run directory"
    resultdir::String = "./output/"
    "log file name inside the run directory"
    logfilename::String = "log.txt"
    "reserved: an existing run directory is currently always reused"
    forcenewdir::Bool = true
    "write `Ut`, the x-velocity averaged onto T-points (m s⁻¹)"
    save_Ut::Bool = true
    "write `Uu`, the x-velocity on its own u-points (m s⁻¹)"
    save_Uu::Bool = false
    "write `Vt`, the y-velocity averaged onto T-points (m s⁻¹)"
    save_Vt::Bool = true
    "write `Vv`, the y-velocity on its own v-points (m s⁻¹)"
    save_Vv::Bool = false
    "write `D`, the layer thickness (m)"
    save_D::Bool = true
    "write `T`, the layer temperature (°C)"
    save_T::Bool = true
    "write `S`, the layer salinity (psu)"
    save_S::Bool = true
    "write `melt`, the basal melt rate (m yr⁻¹)"
    save_melt::Bool = true
    "write `entr`, the entrainment rate (m yr⁻¹)"
    save_entr::Bool = false
    "write `ent2`, the extra entrainment that keeps `D ≥ D_min` (m yr⁻¹)"
    save_ent2::Bool = false
    "write `detr`, the detrainment rate (m yr⁻¹)"
    save_detr::Bool = false
    "write `Tbase`, the temperature at the ice–ocean interface (°C)"
    save_Tbase::Bool = false
    "write `Tamb`, the ambient temperature at the layer base (°C)"
    save_Tamb::Bool = false
    "write `gammaT`, the turbulent heat-exchange velocity (m s⁻¹)"
    save_gammaT::Bool = false
    "write `ustar`, the friction velocity at the ice base (m s⁻¹)"
    save_ustar::Bool = false
    "write `drho`, the reduced density contrast with the ambient water (–)"
    save_drho::Bool = false
    "write `convection`, the fraction of time a cell was convectively unstable"
    save_convection::Bool = false
    "write the static masks: `mask` and the margin flags `at_isf`, `at_grl`, `at_lnd`, `at_gap`"
    save_mask::Bool = true
    "write the static ice draft `z_draft` (m)"
    save_zb::Bool = true
end

# ============================================================================
# IOState{FT, A} — mutable runtime I/O state: counters, run directory, log, and
# time-average accumulators.  Owned by the Simulation;
# the simulated time itself lives in its `Clock`.
# A is the concrete matrix type (matches Grid/State/Cache).  Accumulators are
# allocated 0×0 at construction; `prepare_output!` replaces the enabled ones
# with full-size device arrays (the `save_*` flags in OutputConfig guard access).
# The cell-centre coordinates written to NetCDF live on the Grid.
# ============================================================================

mutable struct IOState{FT,A<:AbstractMatrix{FT}}
    # Time-average accumulation window
    count::Int        # steps accumulated since the last output write
    t_accum::Float64    # simulated time accumulated since the last write (s)
    time_index::Int     # number of time slices written to output.nc so far
    # Next-event times for periodic I/O (s on the simulation clock)
    nextsave::Float64
    nextdiag::Float64
    nextrest::Float64
    # Run directory and log
    rundir::String
    logfile::String
    walltime_start::Float64
    # Restart file this simulation was started from ("" for a fresh start)
    restartfile::String
    # Time-average accumulators
    Utav::A
    Uuav::A
    Vtav::A
    Vvav::A
    Dav::A
    Tav::A
    Sav::A
    meltav::A
    entrav::A
    ent2av::A
    detrav::A
    Tbav::A
    Taav::A
    gamTav::A
    ustarav::A
    drhoav::A
    convav::A
end

# The time-averaged output fields.  `flag` is the `OutputConfig` switch and `acc`
# the `IOState` accumulator; `src` reads the field from the model, or is `Val(:t_u)`
# / `Val(:t_v)` for a velocity averaged onto T-points; `per_year` marks rates
# (m s⁻¹) written in m yr⁻¹.
_outfield(flag, acc, name, units, long, src; per_year = false) =
    (; flag, acc, name, units, long, src, per_year)
const _OUTPUT_FIELDS = (
    _outfield(:save_Ut, :Utav, "Ut", "m s-1", "x-velocity on t-grid", Val(:t_u)),
    _outfield(:save_Uu, :Uuav, "Uu", "m s-1", "x-velocity on u-grid", m -> m.U.present),
    _outfield(:save_Vt, :Vtav, "Vt", "m s-1", "y-velocity on t-grid", Val(:t_v)),
    _outfield(:save_Vv, :Vvav, "Vv", "m s-1", "y-velocity on v-grid", m -> m.V.present),
    _outfield(:save_D, :Dav, "D", "m", "mixed-layer thickness", m -> m.D.present),
    _outfield(:save_T, :Tav, "T", "degC", "layer-averaged temperature", m -> m.T.present),
    _outfield(:save_S, :Sav, "S", "psu", "layer-averaged salinity", m -> m.S.present),
    _outfield(:save_melt, :meltav, "melt", "m yr-1", "basal melt rate", m -> m.melt;
              per_year = true),
    _outfield(:save_entr, :entrav, "entr", "m yr-1", "entrainment rate", m -> m.entr;
              per_year = true),
    _outfield(:save_ent2, :ent2av, "ent2", "m yr-1", "additional entrainment", m -> m.ent2;
              per_year = true),
    _outfield(:save_detr, :detrav, "detr", "m yr-1", "detrainment rate", m -> m.detr;
              per_year = true),
    _outfield(:save_Tbase, :Tbav, "Tbase", "degC", "temperature at ice base", m -> m.Tb),
    _outfield(:save_Tamb, :Taav, "Tamb", "degC", "ambient temperature at layer base",
              m -> m.Ta),
    _outfield(:save_gammaT, :gamTav, "gammaT", "m s-1", "turbulent heat exchange velocity",
              m -> m.gamT),
    _outfield(:save_ustar, :ustarav, "ustar", "m s-1", "friction velocity at the ice base",
              m -> m.ustar),
    _outfield(:save_drho, :drhoav, "drho", "1",
              "reduced density contrast with ambient, (rho_a - rho)/rho_0", m -> m.drho),
    _outfield(:save_convection, :convav, "convection", "1",
              "fraction of time convectively unstable", m -> m.convection),
)

# ============================================================================
# Run directory + log
# ============================================================================

function _print2log(sim, text)
    isempty(sim.io.logfile) && return
    elapsed = time() - sim.io.walltime_start
    h = floor(Int, elapsed / 3600)
    rem = elapsed - 3600h
    mn = floor(Int, rem / 60)
    sc = rem - 60mn
    open(sim.io.logfile, "a") do f
        write(f, @sprintf("[%02d:%02d:%04.1f] %s\n", h, mn, sc, text))
    end
end

# One-line record of an adaptive-dt change (no-op when I/O is disabled).  Kept
# here because Printf is imported in this file; called by the dt controller.
_log_dt_change!(sim, dt_old, dt_new, cfl) = _print2log(
    sim,
    @sprintf(
        "%.3f days: dt %.1f → %.1f s (CFL %.2f)",
        _t_days(sim),
        Float64(dt_old),
        Float64(dt_new),
        cfl
    )
)

"""
$(TYPEDSIGNATURES)

Create the output directory at `joinpath(output.resultdir, output.name)` (if it
does not exist yet) and open the log file, which is appended to.  An existing
directory is reused whatever `output.forcenewdir` says, and its `output.nc` is
replaced by the new run's; the run metadata never overwrites an earlier file.
"""
function create_rundir!(sim)
    rundir = joinpath(sim.output.resultdir, sim.output.name)
    if sim.output.forcenewdir || !isdir(rundir)
        mkpath(rundir)
    end
    sim.io.rundir = rundir
    sim.io.logfile = joinpath(rundir, sim.output.logfilename)
    sim.io.walltime_start = time()
    _print2log(sim, "Run directory: $(rundir)")
    return sim
end

# ============================================================================
# Run provenance metadata
# ============================================================================

_toml_value(v::Bool) = v
_toml_value(v::Integer) = Int(v)
_toml_value(v::AbstractFloat) = Float64(v)
_toml_value(v::String) = v
_toml_value(v::Symbol) = String(v)
_toml_value(::Any) = nothing   # arrays etc. are skipped

# Type name + all TOML-representable fields of a struct.
function _scalar_fields(x)
    d = Dict{String,Any}("type" => string(nameof(typeof(x))))
    for fn in fieldnames(typeof(x))
        v = _toml_value(getfield(x, fn))
        v === nothing || (d[string(fn)] = v)
    end
    return d
end

# Forcing metadata.  The profiles themselves are arrays, which `_scalar_fields`
# skips, so record their ranges explicitly — otherwise the entry would say only
# which type was used and a warm run would be indistinguishable from a cold one.
function _forcing_metadata(f::CavityForcing)
    o, i = f.ocean, f.ice
    ocean = _scalar_fields(o)
    if hasproperty(o, :Tz)
        ocean["T_range"] = [Float64(x) for x in extrema(o.Tz)]
        ocean["S_range"] = [Float64(x) for x in extrema(o.Sz)]
        ocean["z_range"] = [Float64(x) for x in extrema(o.z)]
        ocean["nz"] = length(o.z)
    end
    ice = _scalar_fields(i)
    if hasproperty(i, :T_ice_base)
        ice["T_ice_base_range"] = [Float64(x) for x in extrema(i.T_ice_base)]
    end
    return Dict{String,Any}("ocean" => ocean, "ice" => ice)
end

# Write the effective configuration of this run — parameters, forcing, grid,
# time integration, output, precision, backend, package/Julia versions — so any
# output directory can be traced back to what produced it.  Never overwrites: a
# continuation run into the same directory gets run_metadata_1.toml, _2.toml, ...
function _write_run_metadata(sim)
    m = sim.model
    p = getfield(m, :params)
    params_d = _scalar_fields(p)
    params_d["entrainment"] = _scalar_fields(p.entrainment)
    params_d["melt"] = _scalar_fields(p.melting)
    if p.melting isa PrescribedMelting && p.melting.melt isa AbstractArray
        params_d["melt"]["melt_range"] = [Float64(x) for x in extrema(p.melting.melt)]
    end
    params_d["convection"] = _scalar_fields(p.convection_scheme)
    # Recorded because the choice can change the melt field by an order of magnitude
    # on a real cavity (see AbstractMaxLayerThickness).
    params_d["max_layer_thickness"] = _scalar_fields(p.max_layer_thickness)
    b = getfield(m, :boundary)
    boundary_d = Dict{String,Any}(
        "open_ocean" => _scalar_fields(b.open_ocean),
        "grounding_line" => _scalar_fields(b.grounding_line),
        "land" => _scalar_fields(b.land),
        "gaps" => _scalar_fields(b.gaps),
    )
    params_d["lateral_viscosity"] = _scalar_fields(p.lateral_viscosity)
    params_d["laplacian_weights"] = _scalar_fields(p.laplacian_weights)
    params_d["front_pressure"] = _scalar_fields(p.front_pressure)
    # A 2D latitude is an array, which `_scalar_fields` skips; record its range so
    # the entry says more than just which option was chosen.
    params_d["coriolis"] = _scalar_fields(p.coriolis)
    if p.coriolis isa CoriolisParameter2D && p.coriolis.lat isa AbstractArray
        params_d["coriolis"]["lat_range"] = [Float64(x) for x in extrema(p.coriolis.lat)]
    end
    sim_d = Dict{String,Any}(
        "dt0" => Float64(sim.clock.dt),
        "nu" => Float64(sim.nu),
        "time_stepper" => _scalar_fields(sim.tstep),
        "cfl" => _scalar_fields(sim.cfl),
        "stop" => _scalar_fields(sim.stop),
        "restart" => sim.io.restartfile,
        "debug" => _scalar_fields(sim.debug),
    )
    meta = Dict{String,Any}(
        "run" => Dict{String,Any}(
            "created" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()),
            "julia_version" => string(VERSION),
            "laddie_version" => string(pkgversion(@__MODULE__)),
            "backend" => string(nameof(typeof(KA.get_backend(m.tmask)))),
            "float_type" => string(m.FT),
            "t_start_days" => _t_days(sim),
        ),
        "grid" => Dict{String,Any}(
            "nx" => m.nx,
            "ny" => m.ny,
            "dx" => Float64(m.dx),
            "dy" => Float64(m.dy),
        ),
        "forcing" => _forcing_metadata(getfield(m, :forcing)),
        "params" => params_d,
        "boundary" => boundary_d,
        "simulation" => sim_d,
        "output" => _scalar_fields(sim.output),
    )
    path = joinpath(sim.io.rundir, "run_metadata.toml")
    k = 1
    while isfile(path)
        path = joinpath(sim.io.rundir, "run_metadata_$(k).toml")
        k += 1
    end
    open(io -> TOML.print(io, meta), path, "w")
    _print2log(sim, "Wrote run metadata → $(basename(path))")
    return
end

# ============================================================================
# Output preparation
# ============================================================================

"""
$(TYPEDSIGNATURES)

Initialise time-average accumulators and the next output/diagnostic/restart
event times.  Must be called after the leapfrog is bootstrapped and after
`create_rundir!`.
"""
function prepare_output!(sim)
    m = sim.model
    out, spd = sim.output, Float64(m.seconds_per_day)
    sim.io.nextsave = sim.clock.time + out.saveday * spd
    sim.io.nextdiag = sim.clock.time + out.diagday * spd
    sim.io.nextrest = sim.clock.time + out.restday * spd
    sim.io.count = 0
    sim.io.t_accum = 0.0
    sim.io.time_index = 0

    # Allocate on same device and with same FT as the model arrays.
    # Full grid size (including halos) so _accum! can do bare .+= without
    # border-stripping; halos are masked out when writing to NetCDF.
    z = zero(m.tmask)
    for f in _OUTPUT_FIELDS
        getfield(out, f.flag) && setfield!(sim.io, f.acc, copy(z))
    end
    _write_run_metadata(sim)
    _create_output_file!(sim)
    # Write the initial state as the first time slice before any stepping.
    _accum!(sim)
    _write_output!(sim, _t_days(sim))
    _reset_accum!(sim)
    return sim
end

# ============================================================================
# Per-step accumulation helpers
# ============================================================================

_int(a) = Array(a)[2:(end-1), 2:(end-1)]

# t-grid velocity accumulation fused with the staggered average — avoids the
# two circshift allocations per step that im_half()/jm_half() would cost.  Accumulation
# is dt-weighted (× dt) so the time average is correct when dt varies.
@kernel function _accum_ut_kernel!(av, @Const(U), Nx, dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(av)
        half = FT(0.5)
        im1 = _xm1(i, Nx)
        av[i, j] += (U[i, j] + U[im1, j]) * half * dt
    end
end

@kernel function _accum_vt_kernel!(av, @Const(V), Ny, dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(av)
        half = FT(0.5)
        jm1 = _ym1(j, Ny)
        av[i, j] += (V[i, j] + V[i, jm1]) * half * dt
    end
end

function _accum!(sim)
    m, io = sim.model, sim.io
    dt = sim.clock.dt
    io.count += 1
    io.t_accum += dt
    for f in _OUTPUT_FIELDS
        getfield(sim.output, f.flag) && _accum_field!(getfield(io, f.acc), f.src, m, dt)
    end
end

_accum_field!(av, src, m, dt) = (av .+= src(m) .* dt)
_accum_field!(av, ::Val{:t_u}, m, dt) =
    launch!(_accum_ut_kernel!, av, av, m.U.present, size(av, 1), dt)
_accum_field!(av, ::Val{:t_v}, m, dt) =
    launch!(_accum_vt_kernel!, av, av, m.V.present, size(av, 2), dt)

function _reset_accum!(sim)
    io = sim.io
    io.count = 0
    io.t_accum = 0.0
    for f in _OUTPUT_FIELDS
        fill!(getfield(io, f.acc), 0)   # disabled accumulators are 0×0 — no-op
    end
end

# ============================================================================
# Time-average output
# ============================================================================

# Create output.nc once at the start of a run: defines all dimensions,
# coordinate variables, and time-varying/static field variables.  Time-varying
# fields are 3D (x, y, time) in Julia order with an unlimited time dimension — which
# NCDatasets writes as (time, y, x) on disk, the CF layout ncview expects; static fields
# (mask, z_draft) are 2D and written here.
function _create_output_file!(sim)
    m = sim.model
    path = joinpath(sim.io.rundir, "output.nc")
    NCDataset(path, "c") do ds
        defDim(ds, "x", m.nx)
        defDim(ds, "y", m.ny)
        defDim(ds, "time", Inf)   # unlimited

        defVar(ds, "x", Float64, ("x",); attrib = ["units" => "m"])[:] = m.x
        defVar(ds, "y", Float64, ("y",); attrib = ["units" => "m"])[:] = m.y
        defVar(
            ds,
            "time",
            Float64,
            ("time",);
            attrib = [
                "units" => "days",
                "long_name" => "simulation time (end of averaging window)",
            ],
        )
        defVar(
            ds,
            "walltime",
            Float64,
            ("time",);
            attrib = [
                "units" => "s",
                "long_name" => "elapsed wall-clock time since run start",
            ],
        )

        ds.attrib["model"] = "Laddie.jl"
        ds.attrib["saveday"] = sim.output.saveday

        # Time-varying fields — data appended each save interval
        function dv(name, units, longname)
            defVar(
                ds,
                name,
                Float64,
                ("x", "y", "time");
                fillvalue = NaN,
                attrib = ["units" => units, "long_name" => longname],
            )
        end
        for f in _OUTPUT_FIELDS
            getfield(sim.output, f.flag) && dv(f.name, f.units, f.long)
        end

        # Static fields — written once
        if sim.output.save_mask
            defVar(ds, "mask", Int32, ("x", "y"))[:, :] = Int32.(_int(m.resolved_mask))
            flag!(name, long, cells) =
                defVar(ds, name, Int8, ("x", "y"); attrib = ["long_name" => long])[:, :] =
                    Int8.(_int(cells))
            # Under ConnectedGapsBC a gap (mask 4) is active but not ocean, so it does
            # not mark an ice front: `at_isf` then traces only the outer edge of the
            # connected region, which is what the calving front actually is.
            flag!(
                "at_isf",
                "active cell at ice-shelf front (ocean neighbour)",
                (m.tmask .> 0) .& (m.ocnxm1 .+ m.ocnxp1 .+ m.ocnym1 .+ m.ocnyp1 .> 0),
            )
            # Wall diagnostics are split by wall type so a margin can be told apart
            # at a glance: `at_grl` is the grounding line (grounded ice, mask 2) and
            # `at_lnd` is rock (land, mask 1) — an island shore or an ice-free coast.
            # A cell may carry more than one of at_isf/at_grl/at_lnd; that is genuine
            # where a shelf cell has several different neighbours.
            mask_c = m.resolved_mask
            shelf_next_to(v) =
                (mask_c .== 3) .& (
                    (xm1(mask_c) .== v) .| (xp1(mask_c) .== v) .| (ym1(mask_c) .== v) .|
                    (yp1(mask_c) .== v)
                )
            flag!(
                "at_grl",
                "shelf cell at grounding line (grounded-ice neighbour)",
                shelf_next_to(2),
            )
            flag!("at_lnd", "shelf cell at a land margin (bedrock neighbour)", shelf_next_to(1))
            # The internal margin opened by melt-through.  All zeros under
            # SinkGapsBC, which demotes gaps to ocean when the model is built — so
            # this field also records which gap treatment ran.
            flag!("at_gap", "shelf cell at a melt-through gap (gap neighbour)", shelf_next_to(4))
        end
        if sim.output.save_zb
            defVar(ds, "z_draft", Float64, ("x", "y"); attrib = ["units" => "m"])[:, :] =
                _int(m.z_draft)
        end
    end
    _print2log(sim, "Created output file → output.nc")
end

# Append one time-average slice to output.nc.  The dt-weighted accumulators are
# divided by the accumulated window length; the time coordinate stores the end
# of the averaging window in days.
function _write_output!(sim, t_days)
    m = sim.model
    n = sim.io.t_accum
    tmask_int = _int(m.tmask)
    sim.io.time_index += 1
    k = sim.io.time_index
    path = joinpath(sim.io.rundir, "output.nc")

    NCDataset(path, "a") do ds
        ds["time"][k] = t_days
        ds["walltime"][k] = time() - sim.io.walltime_start

        function wv(name, av, scale)
            av_int = _int(av)
            FT0 = zero(eltype(av_int))
            ds[name][:, :, k] = ifelse.(tmask_int .> 0, av_int ./ n .* scale, FT0)
        end

        for f in _OUTPUT_FIELDS
            getfield(sim.output, f.flag) || continue
            wv(f.name, getfield(sim.io, f.acc), f.per_year ? m.seconds_per_year : 1.0)
        end
    end
    _print2log(sim, @sprintf("%.3f days: appended output → output.nc (step %d)", t_days, k))
end

# A periodic event is due once the clock reaches the next event time.  The
# half-step tolerance mirrors the run! stopping rule (round-half-up), so a fixed
# dt fires at the same steps an integer `t % interval == 0` test would.
_event_due(sim, next) = sim.clock.time + sim.clock.dt / 2 >= next

"""
$(TYPEDSIGNATURES)

Accumulate model fields into time averages and write a NetCDF output file
at every `output.saveday`-day interval.  Called once per time step inside `run!`;
the final partial window is flushed by `run!` after the loop.
"""
function savefields!(sim)
    _accum!(sim)
    if _event_due(sim, sim.io.nextsave)
        _write_output!(sim, _t_days(sim))
        _reset_accum!(sim)
        sim.io.nextsave += sim.output.saveday * sim.model.seconds_per_day
    end
end

# Flush any unwritten accumulation as a final output file (end of run).
function flush_output!(sim)
    sim.io.count > 0 || return
    _write_output!(sim, _t_days(sim))
    _reset_accum!(sim)
end

# ============================================================================
# Restart I/O
# ============================================================================

function _write_restart!(sim, t_days)
    m = sim.model
    filename = joinpath(sim.io.rundir, @sprintf("restart_%06.0f.jld2", t_days))

    _v(var) =
        (past = Array(var.past), present = Array(var.present), future = Array(var.future))
    # Save the current dt so an adaptive run resumes at the step it left off
    # (the saved leapfrog levels are separated by this dt); FixedDt saves dt0.
    jldsave(
        filename;
        t_days,
        dt = Float64(sim.clock.dt),
        D = _v(m.D),
        U = _v(m.U),
        V = _v(m.V),
        T = _v(m.T),
        S = _v(m.S),
    )

    cp(filename, joinpath(sim.io.rundir, "restart_latest.jld2"); force = true)
    _print2log(sim, @sprintf("%.3f days: saved restart → %s", t_days, basename(filename)))
end

"""
$(TYPEDSIGNATURES)

Write a JLD2 restart file containing all three leapfrog levels of D, U, V,
T, S at every `output.restday`-day interval.  Called once per time step inside
`run!`; the final state is written by `run!` after the loop.  Also writes
`restart_latest.jld2`.  Arrays are moved to CPU before saving so the file is
backend-agnostic; the native `FT` precision is preserved.
"""
function saverestart!(sim)
    _event_due(sim, sim.io.nextrest) || return
    _write_restart!(sim, _t_days(sim))
    sim.io.nextrest += sim.output.restday * sim.model.seconds_per_day
end

"""
$(TYPEDSIGNATURES)

Load D, U, V, T, S from the JLD2 restart file at `path` into all three leapfrog
levels of the model's `Var` structs, set the clock to the restart time, then call
`update_secondary_fields!` and one bootstrap integration step.

Geometry (masks, z_draft) must already be initialised before calling this.
"""
function init_from_restart!(sim, path::AbstractString)
    m = sim.model
    sim.io.restartfile = path
    jldopen(path, "r") do f
        sim.clock.time = f["t_days"] * Float64(m.seconds_per_day)
        # Resume at the saved dt when present (adaptive runs); older files
        # without it keep the dt the simulation was constructed with.
        haskey(f, "dt") && (sim.clock.dt = m.FT(f["dt"]))
        for (name, var) in (("D", m.D), ("U", m.U), ("V", m.V), ("T", m.T), ("S", m.S))
            data = f[name]
            var.past .= data.past
            var.present .= data.present
            var.future .= data.future
        end
    end
    update_secondary_fields!(m, sim.clock.dt)
    leapfrog_step!(sim, 1)
    _print2log(sim, "Restarted from $(path) at $(_t_days(sim)) days")
    return sim
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
$(TYPEDSIGNATURES)

Write a one-line diagnostic to the log file at every `output.diagday`-day interval.
"""
function printdiags(sim)
    _event_due(sim, sim.io.nextdiag) || return
    m = sim.model
    sim.io.nextdiag += sim.output.diagday * m.seconds_per_day
    t_days = _t_days(sim)

    tmask = Array(m.tmask)
    D = Array(m.D.present)
    melt = Array(m.melt)
    entr = Array(m.entr)
    ent2 = Array(m.ent2)
    detr = Array(m.detr)
    convD = Array(m.convD)
    drho = Array(m.drho)
    conv = Array(m.convection)
    U = Array(m.U.present)
    V = Array(m.V.present)

    dxdy = m.dx * m.dy
    area = sum(tmask) * dxdy

    d_Dav = sum(D .* tmask) * dxdy / area
    icecells = findall(tmask .> 0)
    d_Dmin = minimum(D[icecells])
    d_Dmax = maximum(D[icecells])

    # Melt statistics over ice-covered cells, as reported by `meltstats`.
    stats = meltstats(m)
    d_Mmax = stats.max_meltrate
    d_Mav = stats.mean_meltrate
    d_Mtot = stats.total_melt

    total = sum((melt .+ entr .+ ent2 .- detr) .* tmask) * dxdy
    d_MWF = total > 0 ? 100.0 * sum(melt .* tmask) * dxdy / total : 0.0

    d_Etot = 1e-6 * sum(entr .* tmask) * dxdy
    d_E2tot = 1e-6 * sum(ent2 .* tmask) * dxdy
    d_DEtot = 1e-6 * sum(detr .* tmask) * dxdy
    d_PSI = -1e-6 * sum(convD .* tmask) * dxdy

    # max t-grid speed without the im/jm circshift allocations
    nx, ny = size(U)
    d_Vmax = 0.0
    for j = 1:ny, i = 1:nx
        tmask[i, j] > 0 || continue
        im1 = i == 1 ? nx : i - 1
        jm1 = j == 1 ? ny : j - 1
        u_t = (U[i, j] + U[im1, j]) / 2
        v_t = (V[i, j] + V[i, jm1]) / 2
        spd = sqrt(u_t^2 + v_t^2)
        spd > d_Vmax && (d_Vmax = spd)
    end

    d_drho = 1000.0 * minimum(ifelse.(tmask .> 0, drho, 100.0))
    d_conv = sum(conv .* (tmask .> 0))

    line = @sprintf(
        "%8.3f days || %5.1f [%4.2f %4.0f] m || %5.2f | %3.0f m/yr | %7.2f Gt/yr || %5.2f %% || %5.3f + %5.3f - %5.3f | %5.3f Sv || %3.2f m/s || %5.5f %3.0f []",
        t_days,
        d_Dav,
        d_Dmin,
        d_Dmax,
        d_Mav,
        d_Mmax,
        d_Mtot,
        d_MWF,
        d_Etot,
        d_E2tot,
        d_DEtot,
        d_PSI,
        d_Vmax,
        d_drho,
        d_conv
    )
    _print2log(sim, line)
end
