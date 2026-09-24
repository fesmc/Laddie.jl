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

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct DebugConfig{B}
    "check the prognostic fields for NaNs after each leapfrog sub-step (default `false`)"
    check_nans::B = false
    DebugConfig{B}(check_nans) where {B} = new{B}(check_nans)
end

DebugConfig(check_nans) = DebugConfig{Bool}(check_nans)

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

`fields` names the variables of `output.nc`.  Time averages:

| name | field | units |
|------|-------|-------|
| `:Ut`, `:Vt` | x- and y-velocity averaged onto T-points | m s⁻¹ |
| `:Uu`, `:Vv` | x- and y-velocity on their own u- and v-points | m s⁻¹ |
| `:D`, `:T`, `:S` | layer thickness, temperature, salinity | m, °C, psu |
| `:melt` | basal melt rate | m yr⁻¹ |
| `:entr`, `:detr` | entrainment and detrainment rates | m yr⁻¹ |
| `:ent2` | the extra entrainment that keeps `D ≥ D_min` | m yr⁻¹ |
| `:Tbase`, `:Tamb` | temperature at the ice–ocean interface, ambient temperature at the layer base | °C |
| `:gammaT` | turbulent heat-exchange velocity | m s⁻¹ |
| `:ustar` | friction velocity at the ice base | m s⁻¹ |
| `:drho` | reduced density contrast with the ambient water | – |
| `:convection` | fraction of time a cell was convectively unstable | – |

Static fields, written once: `:mask` (the mask and the margin flags `at_isf`,
`at_grl`, `at_lnd`, `at_gap`) and `:z_draft` (the ice draft, m).

```julia
OutputConfig(; name = "warm", saveday = 1.0, fields = (:D, :melt, :entr))
```

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct OutputConfig{T,F}
    "run name; files go to `joinpath(resultdir, name)`"
    name::String = "run"
    "averaging and output interval in days; `0` disables all file I/O"
    saveday::T = 0.0
    "interval of the log-file diagnostics in days (must be positive)"
    diagday::T = 1.0
    "interval of the restart files in days (must be positive)"
    restday::T = 30.0
    "parent directory of the run directory"
    resultdir::String = "./output/"
    "log file name inside the run directory"
    logfilename::String = "log.txt"
    "variables written to `output.nc` (see the table above)"
    fields::F = (:Ut, :Vt, :D, :T, :S, :melt, :mask, :z_draft)
    OutputConfig{T,F}(args...) where {T,F} = new{T,F}(args...)
end

const _STATIC_FIELDS = (:mask, :z_draft)

# Positional (and keyword) construction converts the intervals to one float type and
# the field names to a tuple of symbols, and rejects unknown names.
function OutputConfig(name, saveday, diagday, restday, resultdir, logfilename, fields)
    days = promote(float(saveday), float(diagday), float(restday))
    fields = Tuple(Symbol.(fields))
    known = (keys(_OUTPUT_FIELDS)..., _STATIC_FIELDS...)
    for f in fields
        f in known || throw(ArgumentError("unknown output field :$f; use one of $known"))
    end
    return OutputConfig{eltype(days),typeof(fields)}(
        name,
        days...,
        resultdir,
        logfilename,
        fields,
    )
end

# ============================================================================
# IOState{I, T, C} — mutable runtime I/O state.  Owned by the Simulation.
# ============================================================================

"""
$(TYPEDEF)

Runtime I/O state of a [`Simulation`](@ref): counters, next-event times, run
directory, log, and the time-average accumulators.  The simulated time itself lives
in the [`Clock`](@ref), and the cell-centre coordinates written to NetCDF on the
[`Grid`](@ref).

# Fields
$(TYPEDFIELDS)
"""
mutable struct IOState{I,T,C}
    "steps accumulated since the last output write"
    count::I
    "simulated time accumulated since the last output write (s)"
    t_accum::T
    "number of time slices written to `output.nc` so far"
    time_index::I
    "clock time of the next output write (s)"
    nextsave::T
    "clock time of the next log diagnostics (s)"
    nextdiag::T
    "clock time of the next restart write (s)"
    nextrest::T
    "run directory (empty while I/O is disabled)"
    rundir::String
    "path of the log file"
    logfile::String
    "wall-clock time at which the run started (s since the epoch)"
    walltime_start::T
    "restart file this simulation was started from (empty for a fresh start)"
    restartfile::String
    "the dt-weighted sums over the averaging window of the time-averaged output fields, by name (full-size model arrays)"
    acc::C
end

# The time-averaged output fields, by output name.  `src` reads the field from the
# model, or is `Val(:t_u)` / `Val(:t_v)` for a velocity averaged onto T-points;
# `per_year` marks rates (m s⁻¹) written in m yr⁻¹.
_outfield(units, long, src; per_year = false) = (; units, long, src, per_year)
const _OUTPUT_FIELDS = (
    Ut = _outfield("m s-1", "x-velocity on t-grid", Val(:t_u)),
    Uu = _outfield("m s-1", "x-velocity on u-grid", m -> m.U.present),
    Vt = _outfield("m s-1", "y-velocity on t-grid", Val(:t_v)),
    Vv = _outfield("m s-1", "y-velocity on v-grid", m -> m.V.present),
    D = _outfield("m", "mixed-layer thickness", m -> m.D.present),
    T = _outfield("degC", "layer-averaged temperature", m -> m.T.present),
    S = _outfield("psu", "layer-averaged salinity", m -> m.S.present),
    melt = _outfield("m yr-1", "basal melt rate", m -> m.melt; per_year = true),
    entr = _outfield("m yr-1", "entrainment rate", m -> m.entr; per_year = true),
    ent2 = _outfield("m yr-1", "additional entrainment", m -> m.ent2; per_year = true),
    detr = _outfield("m yr-1", "detrainment rate", m -> m.detr; per_year = true),
    Tbase = _outfield("degC", "temperature at ice base", m -> m.Tb),
    Tamb = _outfield("degC", "ambient temperature at layer base", m -> m.Ta),
    gammaT = _outfield("m s-1", "turbulent heat exchange velocity", m -> m.gamT),
    ustar = _outfield("m s-1", "friction velocity at the ice base", m -> m.ustar),
    drho = _outfield(
        "1",
        "reduced density contrast with ambient, (rho_a - rho)/rho_0",
        m -> m.drho,
    ),
    convection = _outfield("1", "fraction of time convectively unstable", m -> m.convection),
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
        _float64(dt_old),
        _float64(dt_new),
        cfl
    )
)

"""
$(TYPEDSIGNATURES)

Create the output directory at `joinpath(output.resultdir, output.name)` (if it
does not exist yet) and open the log file, which is appended to.  An existing
directory is reused, and its `output.nc` is
replaced by the new run's; the run metadata never overwrites an earlier file.
"""
function create_rundir!(sim)
    rundir = joinpath(sim.output.resultdir, sim.output.name)
    mkpath(rundir)
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
_toml_value(v::Real) = _float64(v)   # floats, and ForwardDiff duals
_toml_value(v::String) = v
_toml_value(v::Symbol) = String(v)
_toml_value(v::Tuple) = [_toml_value(x) for x in v]
_toml_value(::Any) = nothing   # functions etc. are skipped

# A struct as a TOML table: its type name, its numbers, strings and symbols, nested
# structs (the parameterisation objects) as tables of their own, and the value range
# of a numeric array as `<field>_range` — a warm and a cold forcing profile then
# differ in the metadata, not just in the type they share.
function _scalar_fields(x)
    d = Dict{String,Any}("type" => string(nameof(typeof(x))))
    for fn in fieldnames(typeof(x))
        v = getfield(x, fn)
        if v isa AbstractArray{<:Real}
            isempty(v) || (d["$(fn)_range"] = [_float64(y) for y in extrema(v)])
        elseif _is_table(v)
            d[string(fn)] = _scalar_fields(v)
        else
            t = _toml_value(v)
            t === nothing || (d[string(fn)] = t)
        end
    end
    return d
end
_is_table(v) =
    isstructtype(typeof(v)) &&
    !(v isa Union{Number,AbstractString,Symbol,Tuple,AbstractArray,Function})

# Write the effective configuration of this run — parameters, forcing, grid,
# time integration, output, precision, backend, package/Julia versions — so any
# output directory can be traced back to what produced it.  Never overwrites: a
# continuation run into the same directory gets run_metadata_1.toml, _2.toml, ...
function _write_run_metadata(sim)
    m = sim.model
    sim_d = Dict{String,Any}(
        "dt0" => _float64(sim.clock.dt),
        "nu" => _float64(sim.nu),
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
            "dx" => _float64(m.dx),
            "dy" => _float64(m.dy),
        ),
        "forcing" => _scalar_fields(getfield(m, :forcing)),
        "params" => _scalar_fields(getfield(m, :params)),
        "boundary" => _scalar_fields(getfield(m, :boundary)),
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
    out, spd = sim.output, _float64(m.seconds_per_day)
    sim.io.nextsave = sim.clock.time + out.saveday * spd
    sim.io.nextdiag = sim.clock.time + out.diagday * spd
    sim.io.nextrest = sim.clock.time + out.restday * spd
    sim.io.count = 0
    sim.io.t_accum = 0.0
    sim.io.time_index = 0
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

_int(a) = _primal.(view(Array(a), 2:(size(a, 1)-1), 2:(size(a, 2)-1)))

# t-grid velocity accumulation fused with the staggered average — avoids the
# two circshift allocations per step that im_half()/jm_half() would cost.  Accumulation
# is dt-weighted (× dt) so the time average is correct when dt varies.
@kernel function _accum_ut_kernel!(av, @Const(U), Nx, dt)
    dt = _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(av)
        half = FT(0.5)
        im1 = _xm1(i, Nx)
        av[i, j] += (U[i, j] + U[im1, j]) * half * dt
    end
end

@kernel function _accum_vt_kernel!(av, @Const(V), Ny, dt)
    dt = _val(dt)
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
    io.t_accum += _primal(dt)
    for (name, acc) in pairs(io.acc)
        _accum_field!(acc, _OUTPUT_FIELDS[name].src, m, dt)
    end
end

_accum_field!(av, src, m, dt) = (av .+= src(m) .* dt)
_accum_field!(av, ::Val{:t_u}, m, dt) =
    launch!(_accum_ut_kernel!, av, m.U.present, size(av, 1), dt)
_accum_field!(av, ::Val{:t_v}, m, dt) =
    launch!(_accum_vt_kernel!, av, m.V.present, size(av, 2), dt)

function _reset_accum!(sim)
    io = sim.io
    io.count = 0
    io.t_accum = 0.0
    foreach(acc -> fill!(acc, 0), io.acc)
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

        defVar(ds, "x", Float64, ("x",); attrib = ["units" => "m"])[:] = _primal.(m.x)
        defVar(ds, "y", Float64, ("y",); attrib = ["units" => "m"])[:] = _primal.(m.y)
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
        for name in keys(sim.io.acc)
            f = _OUTPUT_FIELDS[name]
            dv(string(name), f.units, f.long)
        end

        # Static fields — written once
        if :mask in sim.output.fields
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
                (m.tmask .> 0) .& next_to_ocean(m.ocn),
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
            flag!(
                "at_lnd",
                "shelf cell at a land margin (bedrock neighbour)",
                shelf_next_to(1),
            )
            # The internal margin opened by melt-through.  All zeros under
            # SinkGapsBC, which demotes gaps to ocean when the model is built — so
            # this field also records which gap treatment ran.
            flag!(
                "at_gap",
                "shelf cell at a melt-through gap (gap neighbour)",
                shelf_next_to(4),
            )
        end
        if :z_draft in sim.output.fields
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

        for (name, acc) in pairs(sim.io.acc)
            scale = _OUTPUT_FIELDS[name].per_year ? _primal(m.seconds_per_year) : 1.0
            wv(string(name), acc, scale)
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

Write the time averages accumulated since the last write to `output.nc`, once every
`output.saveday` days.  Called by `run!` after each batch of steps, which also
accumulates the averages; the final partial window is flushed after the loop.
"""
function savefields!(sim)
    _event_due(sim, sim.io.nextsave) || return
    _write_output!(sim, _t_days(sim))
    _reset_accum!(sim)
    sim.io.nextsave += sim.output.saveday * _primal(sim.model.seconds_per_day)
    return
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

    _v(var) = (
        past = _primal.(Array(var.past)),
        present = _primal.(Array(var.present)),
        future = _primal.(Array(var.future)),
    )
    # Save the current dt so an adaptive run resumes at the step it left off
    # (the saved leapfrog levels are separated by this dt); FixedDt saves dt0.
    jldsave(
        filename;
        t_days,
        dt = _float64(sim.clock.dt),
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
    sim.io.nextrest += sim.output.restday * _primal(sim.model.seconds_per_day)
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
        sim.clock.time = f["t_days"] * _float64(m.seconds_per_day)
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
    _bootstrap_leapfrog!(sim)
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
    m = _diag_model(sim.exec, sim)
    sim.io.nextdiag += sim.output.diagday * _primal(m.seconds_per_day)
    t_days = _t_days(sim)

    # Device reductions over the scratch field `diag`, so no field is copied to the host.
    d = m.diag
    active_sum(x) = (@. d = x * m.tmask; sum(d))
    dxdy = m.dx * m.dy
    area = sum(m.tmask) * dxdy

    d_Dav = active_sum(m.D.present) * dxdy / area
    @. d = ifelse(m.tmask > 0, m.D.present, Inf)
    d_Dmin = minimum(d)
    @. d = ifelse(m.tmask > 0, m.D.present, -Inf)
    d_Dmax = maximum(d)

    # Melt statistics over ice-covered cells, and the largest T-point speed, as
    # reported by `meltstats`.
    stats = meltstats(m)
    d_Mmax = stats.max_meltrate
    d_Mav = stats.mean_meltrate
    d_Mtot = stats.total_melt
    d_Vmax = stats.max_speed

    @. d = (m.melt + m.entr + m.ent2 - m.detr) * m.tmask
    total = sum(d) * dxdy
    d_MWF = total > 0 ? 100.0 * active_sum(m.melt) * dxdy / total : 0.0

    d_Etot = 1e-6 * active_sum(m.entr) * dxdy
    d_E2tot = 1e-6 * active_sum(m.ent2) * dxdy
    d_DEtot = 1e-6 * active_sum(m.detr) * dxdy
    d_PSI = -1e-6 * active_sum(m.convD) * dxdy

    @. d = ifelse(m.tmask > 0, m.drho, 100.0)
    d_drho = 1000.0 * minimum(d)
    @. d = m.convection * (m.tmask > 0)
    d_conv = sum(d)

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
