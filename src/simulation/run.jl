# ============================================================================
# Time loop
# ============================================================================

# Abort with a clear message as soon as the integration produces non-finite
# values, instead of silently stepping NaNs for the rest of the run.  `d` holds the
# sync-point diagnostics; `t`/`nt` count the steps of the current `run!` call.
function _check_blowup(sim, d, t, nt)
    m = sim.model
    d.finite && return
    error(
        "Simulation blew up: non-finite values in H/U/V/T/S at step $t/$nt " *
        "(≈ day $(round(_t_days(sim), digits = 2))). Common causes: time step too " *
        "large for this grid (dt = $(_primal(sim.clock.dt)) s, dx = $(m.dx) m) or unstable " *
        "forcing. Reduce dt in Simulation, or check the inputs.",
    )
end

"""
$(TYPEDSIGNATURES)

Advance `sim` by one leapfrog time step: `advance_leapfrog!` → `leapfrog_step!`
(2×dt, which ends each momentum step with `clamp_velocities!`) →
`apply_robert_asselin_filter!`, then move the clock forward by `dt`.
No I/O, no CFL control, no blow-up check — those belong to [`run!`](@ref).

Returns `sim`.
"""
function time_step!(sim::Simulation)
    _set_forcing_time!(sim)
    _step_model!(sim.model, sim.clock.dt, sim.nu; _nan_check(sim)...)
    _tick!(sim.clock)
    return sim
end

# Bring a time-dependent forcing (a `RampedOceanForcing`) to the clock's time, from the
# host; a no-op for every other forcing.
_set_forcing_time!(sim) = _set_forcing_time!(
    getfield(sim.model, :forcing).ocean, sim.clock.time / _float64(_primal(sim.model.seconds_per_day)))

# Move `clock` forward by one step of its dt: the host side of `time_step!`.
function _tick!(clock)
    clock.time += _primal(clock.dt)
    clock.iteration += 1
    return clock
end

# ============================================================================
# Execution hooks of `run!`.  The native (KernelAbstractions) versions step one
# time step at a time and compute the diagnostics directly on the model; the
# Reactant extension replaces them with compiled programs.
# ============================================================================

# Steps to take before `run!` must act on the host again.
_batch_length(::NativeExecution, sim, r) = 1

# Advance `n` steps, accumulating the output averages after each one.
function _advance_batch!(::NativeExecution, sim, n, io_on)
    for _ = 1:n
        time_step!(sim)
        io_on && _accum!(sim)
    end
    return
end

# Everything `run!` reads from the device at a sync point, in one call: whether the
# prognostics are finite, the largest δρ·H (for the worst-case CFL), the melt
# statistics, the largest active H, and the reductions of the CFL number.  The
# Reactant extension compiles it into one program.
_diagnostics(m, cfl) = (
    finite = _prognostics_finite(m),
    Hdrho = _max_Hdrho(m),
    stats = meltstats(m),
    Hmax = _max_active_H(m),
    cfl = cfl_reductions(m, cfl),
)

# The sync-point diagnostics of `sim` as host numbers, with the CFL reductions
# combined into the CFL number of the current dt.
_sync_diagnostics(::NativeExecution, sim) = _on_host(sim, _diagnostics(sim.model, sim.cfl))
_on_host(sim, d) =
    (; finite = Bool(d.finite), d.Hdrho, d.stats, d.Hmax, cfl = _cfl_number(sim, d.cfl))

# `&`, not `&&`: also evaluated on traced values by the Reactant extension.
_prognostics_finite(m) =
    all(isfinite, m.H.present) & all(isfinite, m.U.present) & all(isfinite, m.V.present) &
    all(isfinite, m.T.present) & all(isfinite, m.S.present)
function _max_active_H(m)
    @. m.diag = ifelse(m.tmask > 0, m.H.present, m.FT(-Inf))
    return maximum(m.diag)
end
# The model the log diagnostics (`printdiags`) are computed on.
_diag_model(::NativeExecution, sim) = sim.model

# ============================================================================
# Host events of `run!`
# ============================================================================

"""
$(TYPEDEF)

The progress of one [`run!`](@ref) call.  Steps and simulated seconds count from
where the call started, while the clock and the I/O event times are absolute.

# Fields
$(TYPEDFIELDS)
"""
mutable struct RunProgress{I,T,B}
    "steps taken by this call"
    step::I
    "simulated seconds of this call"
    elapsed::T
    "hard time cap of this call (s)"
    total::T
    "steps between sync points (blow-up check, CFL, dt control, progress diagnostics)"
    checkint::I
    "`elapsed` of the next steady-state sample; `Inf` when the criterion needs none"
    next_steady::T
    "mean melt rate of the previous steady-state sample (m yr⁻¹), `NaN` before the first"
    prev_mean::T
    "whether file I/O is on"
    io_on::B
end

# The host events due once `step` steps and `elapsed` seconds of the call have been
# taken, with the clock at `t`; `half` is half the time step, the tolerance of every
# event time (round-half-up).  `run!` tests them after each batch, and
# `_steps_to_next_event` replays them step by step, so that a batched run meets every
# event at the same step as a one-step-at-a-time run.
_sync_due(r, step, elapsed, half) = step % r.checkint == 0 || elapsed + half >= r.total
_sample_due(r, elapsed, half) = elapsed + half >= r.next_steady
_output_due(io, t, half) =
    _event_due(t, half, io.nextsave) || _event_due(t, half, io.nextdiag) ||
    _event_due(t, half, io.nextrest) || _event_due(t, half, io.nextseries)

# The number of steps until the next host event.
function _steps_to_next_event(sim, r)
    dt = _primal(sim.clock.dt)
    half = dt / 2
    t, elapsed = sim.clock.time, r.elapsed
    k = 0
    while true
        k += 1
        t += dt
        elapsed += dt
        (_sync_due(r, r.step + k, elapsed, half) || _sample_due(r, elapsed, half) ||
         r.io_on && _output_due(sim.io, t, half)) && return k
    end
end

# Steady-state early stop: compare the mean melt rate with the previous daily sample
# and schedule the next one.  Returns whether the run should stop.
function _steady_sample!(sim, r, until, mean_melt)
    if steady_reached(until, mean_melt, r.prev_mean)
        _print2log(
            sim,
            "$(round(_t_days(sim), digits = 3)) days: steady state reached " *
            "(relative Δ mean melt < $(until.tol))",
        )
        return true
    end
    r.prev_mean = mean_melt
    r.next_steady += _float64(sim.model.seconds_per_day)
    return false
end

# The diagnostics attached to the progress bar.
function _progress_values(sim, d)
    mx, mn, sp, gt = d.stats
    return Tuple{String,Any}[
        ("simulated days", _r(_t_days(sim), 2)),
        ("melt mean/max [m/yr]", string(_r(mn, 2), " / ", _r(mx, 2))),
        ("Hmax [m]", _r(d.Hmax, 1)),
        ("total melt [Gt/yr]", _r(gt, 2)),
        ("|u|max [m/s]", _r(sp, 3)),
        ("dt [s] / CFL", string(_r(sim.clock.dt, 1), " / ", _r(d.cfl, 3))),
    ]
end

"""
$(TYPEDSIGNATURES)

Advance simulation `sim` until `until` is met, an [`AbstractSimulationEnd`](@ref) measured from
the current clock time (default `sim.stop`; `days = 30` is short for
`until = FixedSimulationEnd(t_end = 30)`). The clock is never reset, so successive calls
continue the run. Each step is one [`time_step!`](@ref); output, diagnostics and restarts are
written when due (see [Output and restarts](@ref)), and the call ends with a restart.
`verbose` shows a progress bar. The run warns when the worst-case CFL number exceeds 1 and
aborts on non-finite fields.

Returns `sim` for chaining.
"""
function run!(sim::Simulation; days = nothing, until = nothing, verbose = true)
    m = sim.model
    clock = sim.clock
    if days !== nothing && until !== nothing
        throw(ArgumentError("pass either `days` or `until`, not both"))
    end
    _is_traced(m.params) && throw(
        ArgumentError(
            "run! needs plain scalar parameters; a model from `trace_parameters` is " *
            "for programs compiled with `reactant_compile`",
        ),
    )
    until =
        until !== nothing ? until :
        days !== nothing ? FixedSimulationEnd(t_end = _float64(days)) : sim.stop
    total = end_seconds(until, _primal(m.seconds_per_day))  # hard time cap (s) for this call
    io_on = sim.output.saveday > 0
    # Predictive adaptive dt: rescue a too-large dt before the first step
    # (no-op for FixedTimeStepper). nt/checkint below then reflect the adjusted dt.
    init_adaptive_dt!(sim, sim.time_stepper)
    nt = round(Int, total / clock.dt)
    cfl = _cfl_worstcase(sim)
    cfl > 1.0 && @warn "Worst-case CFL (advection at v_cut + gravity wave) is " *
          "$(round(cfl, digits = 2)) > 1 (dt = $(_primal(clock.dt)) s, dx = $(m.dx) m, " *
          "dy = $(m.dy) m); the run is likely unstable — reduce dt or coarsen the grid."
    backend = nameof(typeof(KA.get_backend(m.tmask)))
    # Progress is tracked in simulated seconds (nt is only an estimate under
    # adaptive dt); update! sets the absolute position from `elapsed` each step.
    prog = Progress(
        round(Int, total);
        desc = "[$backend] $(m.nx)×$(m.ny) interior, ~$nt steps: ",
        enabled = verbose,
        showspeed = true,
    )
    showvals = Tuple{String,Any}[]
    # Steady-state sampling compares the mean melt rate on a fixed daily cadence
    # (independent of run length, so `tol` means the same thing for any cap), and is
    # disabled (next_steady = Inf) unless the criterion needs it.
    next_steady = needs_melt_sample(until) ? _float64(m.seconds_per_day) : Inf
    steps_per_day = max(1, round(Int, _primal(m.seconds_per_day) / clock.dt))
    checkint = _check_interval(sim.time_stepper, nt, steps_per_day)
    r = RunProgress(0, 0.0, _float64(total), checkint, next_steady, NaN, io_on)
    # Time cap (round-half-up rule: round(total/dt) steps for fixed dt); a
    # SteadyStateSimulationEnd may break out earlier once the mean melt rate is steady.
    while r.elapsed + _primal(clock.dt) / 2 < r.total
        # Steps up to the next host event (1 on the KernelAbstractions backends;
        # a compiled batch under Reactant), with the output accumulation.
        n = _batch_length(sim.exec, sim, r)
        _advance_batch!(sim.exec, sim, n, io_on)
        r.step += n
        for _ = 1:n
            r.elapsed += _primal(clock.dt)
        end
        if io_on
            saveseries!(sim)
            savefields!(sim)
            printdiags(sim)
            saverestart!(sim)
        end
        half = _primal(clock.dt) / 2
        sample = _sample_due(r, r.elapsed, half)
        sync = _sync_due(r, r.step, r.elapsed, half)
        if sample || sync
            # Device reductions force a GPU sync, so they run only here: once per
            # simulated day for the steady-state sample, and at the check cadence
            # (~5 %, or every `ncheck` steps under AdaptiveTimeStepper, and at least daily) for the blow-up check,
            # the CFL monitor, the controller and the progress diagnostics.  All of
            # them see the clean stepped state, before any dt re-bootstrap.
            d = _sync_diagnostics(sim.exec, sim)
            sample && _steady_sample!(sim, r, until, d.stats.mean_melt) && break
            if sync
                _check_blowup(sim, d, r.step, nt)
                # Adjust dt for the upcoming steps (no-op under FixedTimeStepper).
                adapt_dt!(sim, sim.time_stepper, d.cfl)
                verbose && (showvals = _progress_values(sim, d))
            end
        end
        update!(prog, round(Int, r.elapsed); showvalues = showvals)
    end
    # Flush the final partial average and write the end-of-run restart.
    if io_on
        flush_output!(sim)
        _write_restart!(sim, _t_days(sim))
    end
    finish!(prog)
    return sim
end
