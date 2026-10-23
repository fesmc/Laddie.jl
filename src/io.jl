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
| `:H`, `:T`, `:S` | layer thickness, temperature, salinity | m, °C, psu |
| `:melt` | basal melt rate | m yr⁻¹ |
| `:entr`, `:detr` | entrainment and detrainment rates | m yr⁻¹ |
| `:ent2` | the extra entrainment that keeps `H ≥ H_min` | m yr⁻¹ |
| `:detr_cap` | the detrainment that keeps `H` under `max_layer_thickness` | m yr⁻¹ |
| `:frazil` | frazil-ice formation, freshwater equivalent (zero unless `RelaxedFrazilIce`) | m yr⁻¹ |
| `:Tbase`, `:Tamb` | temperature at the ice–ocean interface, ambient temperature at the layer base | °C |
| `:gammaT` | turbulent heat-exchange velocity | m s⁻¹ |
| `:ustar` | friction velocity at the ice base | m s⁻¹ |
| `:drho` | reduced density contrast with the ambient water | – |
| `:convection` | fraction of time a cell was convectively unstable | – |
| `:supercooling` | ``T - T_f(S, z_b)``, the layer temperature above its freezing point at the ice base (negative where supercooled) | °C |

Static fields, written once: `:mask` (the mask and the margin flags `at_isf`,
`at_grl`, `at_lnd`, `at_gap`) and `:z_draft` (the ice draft, m).

```julia
OutputConfig(; name = "warm", saveday = 1.0, fields = (:H, :melt, :entr))
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
    fields::F = (:Ut, :Vt, :H, :T, :S, :melt, :mask, :z_draft)
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
    H = _outfield("m", "mixed-layer thickness", m -> m.H.present),
    T = _outfield("degC", "layer-averaged temperature", m -> m.T.present),
    S = _outfield("psu", "layer-averaged salinity", m -> m.S.present),
    melt = _outfield("m yr-1", "basal melt rate", m -> m.melt; per_year = true),
    entr = _outfield("m yr-1", "entrainment rate", m -> m.entr; per_year = true),
    ent2 = _outfield("m yr-1", "additional entrainment", m -> m.ent2; per_year = true),
    detr = _outfield("m yr-1", "detrainment rate", m -> m.detr; per_year = true),
    detr_cap = _outfield(
        "m yr-1",
        "detrainment keeping H under the maximum layer thickness",
        m -> m.detr_cap;
        per_year = true,
    ),
    frazil = _outfield(
        "m yr-1",
        "frazil-ice formation rate, freshwater equivalent",
        m -> m.frazil;
        per_year = true,
    ),
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
    supercooling = _outfield(
        "degC",
        "layer temperature above the freezing point at the ice base, T - Tf(S, z_draft)",
        Val(:supercooling),
    ),
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

# Add the current fields to the averages: on the device, the dt-weighted fields
# (`_accum_fields!`, which the Reactant extension traces into its step loop); on the
# host, the counters of the averaging window (`_count_accum!`).
function _accum!(sim)
    _accum_fields!(sim.io.acc, sim.model, sim.clock.dt)
    _count_accum!(sim.io, sim.clock.dt)
    return
end

function _accum_fields!(accs, m, dt)
    for (name, acc) in pairs(accs)
        _accum_field!(acc, _OUTPUT_FIELDS[name].src, m, dt)
    end
    return
end

function _count_accum!(io, dt)
    io.count += 1
    io.t_accum += _primal(dt)
    return
end

_accum_field!(av, src, m, dt) = (av .+= src(m) .* dt)
_accum_field!(av, ::Val{:t_u}, m, dt) =
    launch!(_accum_ut_kernel!, av, m.U.present, size(av, 1), dt)
_accum_field!(av, ::Val{:t_v}, m, dt) =
    launch!(_accum_vt_kernel!, av, m.V.present, size(av, 2), dt)
# T − Tf(S, z_b), with the liquidus of the melt kernel; negative where supercooled.
_accum_field!(av, ::Val{:supercooling}, m, dt) =
    (av .+= (m.T.present .- (m.l1 .* m.S.present .+ m.l2 .+ m.l3 .* m.z_draft)) .* dt)

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
_event_due(sim, next) = _event_due(sim.clock.time, _primal(sim.clock.dt) / 2, next)
_event_due(t, half, next) = t + half >= next

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
    # The grid's active cells, crop and full-domain size let a finer run start
    # from this one (`init_from_restart!(...; refine)`).
    g = m.grid
    jldsave(
        filename;
        t_days,
        dt = _float64(sim.clock.dt),
        dx = _float64(g.dx),
        tmask = Array(m.tmask) .> 0,
        crop = (extrema(g.crop[1]), extrema(g.crop[2])),
        input_size = Tuple(g.input_size),
        map(_v, _prognostics(m))...,
    )

    cp(filename, joinpath(sim.io.rundir, "restart_latest.jld2"); force = true)
    _print2log(sim, @sprintf("%.3f days: saved restart → %s", t_days, basename(filename)))
end

"""
$(TYPEDSIGNATURES)

Write a JLD2 restart file containing all three leapfrog levels of H, U, V,
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
    init_from_restart!(sim, path; refine = 1, coarse_grid = nothing)

Load H, U, V, T, S from the JLD2 restart file at `path` into all three leapfrog
levels of the model's `Var` structs, set the clock to the restart time, then call
`update_secondary_fields!` and one bootstrap integration step.  The saved CPU
arrays are copied to the model's backend (CPU or GPU) and converted to its
element type, so a restart written at one precision or on one backend loads on
another.  [`Simulation`](@ref) calls it for its `restart` keyword.

With `refine = f > 1`, `path` is the restart of a run whose cells are `f` times
coarser, on the same full domain: each coarse cell holds `f × f` fine ones, and
the fine domain, border ring included, is `f .* (coarse size .- 2) .+ 2` cells.
Each fine cell takes H, T and S of the coarse cell it lies in where that cell was
active, and each fine velocity point the cell-centred velocity of that coarse cell
(the mean of its two faces), on the fine velocity masks.  Fine cells whose coarse
cell was not active (a finer shelf reaches further) keep the state `sim` was built
with, so call this on a simulation started from rest.  dt is the restart's divided
by `f`.  The two runs may crop their domains differently.  The coarse grid is read
from the restart file; files written before it was stored there need the coarse
run's [`Grid`](@ref) as `coarse_grid`.

Geometry (masks, z_draft) must already be initialised before calling this.
"""
function init_from_restart!(sim, path::AbstractString; refine::Integer = 1, coarse_grid = nothing)
    refine ≥ 1 || throw(ArgumentError("refine must be a positive integer, got $refine"))
    m = sim.model
    sim.io.restartfile = path
    jldopen(path, "r") do f
        sim.clock.time = f["t_days"] * _float64(m.seconds_per_day)
        # Resume at the saved dt when present (adaptive runs); older files
        # without it keep the dt the simulation was constructed with.
        haskey(f, "dt") && (sim.clock.dt = m.FT(f["dt"] / refine))
        fine = refine == 1 ? (name, a, dst) -> a : _refiner(m, f, refine, coarse_grid)
        for (name, var) in pairs(_prognostics(m))
            # Restarts from before the layer thickness was renamed D → H store it as "D".
            data = f[name == :H && !haskey(f, "H") ? "D" : String(name)]
            # The file holds plain CPU arrays: convert each level to the model's
            # element type (FT, or a `Dual` with zero derivatives), then copy it to
            # the model's device.  A broadcast from a CPU array into a device array
            # fails to compile on the GPU.
            for level in (:past, :present, :future)
                dst = getfield(var, level)
                src = fine(name, getfield(data, level), dst)
                size(src) == size(dst) || throw(
                    ArgumentError(
                        "the restart's $name is $(size(src)) cells but the model's is $(size(dst)); " *
                        "to start from a coarser run, pass `refine`",
                    ),
                )
                copyto!(dst, convert(Array{eltype(dst)}, src))
            end
        end
    end
    _bootstrap_leapfrog!(m, sim.clock.dt; _nan_check(sim)...)
    refined = refine == 1 ? "" : " (refined $(refine)×)"
    _print2log(sim, "Restarted from $(path)$(refined) at $(_t_days(sim)) days")
    return sim
end

# The coarse run's grid as the restart file stores it, or as `coarse_grid` gives it
# for files written before it was stored: its active cells (cropped), crop ranges,
# full-domain size and spacing (`nothing` when unknown).
function _coarse_layout(f, coarse_grid)
    if haskey(f, "tmask")
        (r, c) = f["crop"]
        return (; tmask = f["tmask"], crop = (r[1]:r[2], c[1]:c[2]), input_size = f["input_size"], dx = f["dx"])
    end
    coarse_grid === nothing && throw(
        ArgumentError(
            "this restart file does not store its grid (written before refinement was " *
            "supported); pass the coarse run's Grid as `coarse_grid`",
        ),
    )
    g = coarse_grid
    tmask = (g.mask .== 3) .| (g.mask .== 4)   # active cells: shelf and gaps
    return (; tmask, crop = g.crop, input_size = Tuple(g.input_size), dx = _float64(g.dx))
end

# A function `(name, coarse level, model level) -> fine level` that refines a
# coarse restart onto the model's grid, `f` times finer (`init_from_restart!`).
function _refiner(m, file, f, coarse_grid)
    co = _coarse_layout(file, coarse_grid)
    g = m.grid
    fine_size = f .* (co.input_size .- 2) .+ 2
    Tuple(g.input_size) == fine_size || throw(
        ArgumentError(
            "the model's full domain is $(Tuple(g.input_size)) cells, but $(f)× the coarse " *
            "$(co.input_size) (border ring kept) is $fine_size",
        ),
    )
    co.dx === nothing || isapprox(co.dx, f * _float64(g.dx); rtol = 1e-6) || throw(
        ArgumentError("the restart's dx is $(co.dx) m, not $(f)× the model's $(_float64(g.dx)) m"),
    )
    # Coarse cropped index of the coarse cell holding each fine cell of the model's
    # crop, the border rings mapping onto each other; 0 outside the coarse crop
    # (inactive there: cropping keeps every active cell).
    function parents(fine_range, n_fine, n_coarse, coarse_range)
        map(fine_range) do i
            p = i == 1 ? 1 : i == n_fine ? n_coarse : (i - 2) ÷ f + 2
            p in coarse_range ? p - first(coarse_range) + 1 : 0
        end
    end
    I = parents(g.crop[1], g.input_size[1], co.input_size[1], co.crop[1])
    J = parents(g.crop[2], g.input_size[2], co.input_size[2], co.crop[2])
    pick(a, i, j, default) = (I[i] > 0 && J[j] > 0) ? a[I[i], J[j]] : default
    active_c = co.tmask
    wet = [pick(active_c, i, j, false) for i in eachindex(I), j in eachindex(J)]
    tmask = Array(m.tmask) .> 0
    umask, vmask = Array(m.umask), Array(m.vmask)
    return function (name, a, dst)
        if name in (:U, :V)
            # Faces → cell centres: the mean of a cell's two faces, the west
            # (south) one being the previous index.
            c = name == :U ? (a .+ circshift(a, (1, 0))) ./ 2 : (a .+ circshift(a, (0, 1))) ./ 2
            c = ifelse.(active_c, c, zero(eltype(c)))
            vmask_ = name == :U ? umask : vmask
            return [pick(c, i, j, zero(eltype(c))) * vmask_[i, j] for i in eachindex(I), j in eachindex(J)]
        else
            keep = Array(dst)
            return [wet[i, j] & tmask[i, j] ? pick(a, i, j, keep[i, j]) : keep[i, j]
                    for i in eachindex(I), j in eachindex(J)]
        end
    end
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

    d_Hav = active_sum(m.H.present) * dxdy / area
    @. d = ifelse(m.tmask > 0, m.H.present, Inf)
    d_Hmin = minimum(d)
    @. d = ifelse(m.tmask > 0, m.H.present, -Inf)
    d_Hmax = maximum(d)

    # Melt statistics over ice-covered cells, and the largest T-point speed, as
    # reported by `meltstats`.
    stats = meltstats(m)
    d_Mmax = stats.max_meltrate
    d_Mav = stats.mean_meltrate
    d_Mtot = stats.total_melt
    d_Vmax = stats.max_speed

    @. d = (m.melt + m.entr + m.ent2 - m.detr - m.detr_cap - m.frazil) * m.tmask
    total = sum(d) * dxdy
    d_MWF = total > 0 ? 100.0 * active_sum(m.melt) * dxdy / total : 0.0

    d_Etot = 1e-6 * active_sum(m.entr) * dxdy
    d_E2tot = 1e-6 * active_sum(m.ent2) * dxdy
    d_DEtot = 1e-6 * active_sum(m.detr) * dxdy
    d_PSI = -1e-6 * active_sum(m.convH) * dxdy

    @. d = ifelse(m.tmask > 0, m.drho, 100.0)
    d_drho = 1000.0 * minimum(d)
    @. d = m.convection * (m.tmask > 0)
    d_conv = sum(d)

    line = @sprintf(
        "%8.3f days || %5.1f [%4.2f %4.0f] m || %5.2f | %3.0f m/yr | %7.2f Gt/yr || %5.2f %% || %5.3f + %5.3f - %5.3f | %5.3f Sv || %3.2f m/s || %5.5f %3.0f []",
        t_days,
        d_Hav,
        d_Hmin,
        d_Hmax,
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
