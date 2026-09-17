using ProgressMeter

# =================================================================
# Driver
# =================================================================
"""
$(TYPEDSIGNATURES)

Return melt-rate and velocity statistics for the current state of a model or
simulation.
All reductions execute on the device (GPU-safe); results are returned as
CPU scalars.

- `max_meltrate`  — maximum basal melt rate, m yr⁻¹
- `mean_meltrate` — area-mean basal melt rate over floating ice, m yr⁻¹
- `max_speed`     — maximum depth-averaged current speed |u|, m s⁻¹

The melt statistics average over `imask` (ice-covered cells), not `tmask`.  Under
[`ConnectedGapsBC`](@ref) the two differ: gap cells are dynamically active but ice-free
and melt exactly zero, so averaging over `tmask` would dilute the mean by the gap area.
This diverges from LADDIE v2, whose `domain_a` integration includes gaps; with no gaps
in the mask the two masks coincide and the statistics are unchanged.  `max_speed` stays
on `tmask` — the layer really does flow through gaps, and that flow is the point.
"""
meltstats(sim::Simulation) = meltstats(sim.model)

function meltstats(m::Model)
    imask = m.imask
    n = sum(imask)
    meltyr = m.melt .* m.seconds_per_year
    max_meltrate = maximum(meltyr .* imask)
    mean_meltrate = sum(meltyr .* imask) / n
    max_speed =
        maximum(sqrt.(im_half(m.U.present) .^ 2 .+ jm_half(m.V.present) .^ 2) .* m.tmask)
    return max_meltrate, mean_meltrate, max_speed
end

# Dispatch on Simulation.cfl — selects ConservativeCFL or ExactCFL.
_cfl_number(sim) = _cfl_number(sim, sim.cfl)

# Conservative: global max of |U|, |V|, and c taken independently, then combined.
# Overestimates the true CFL but cheap (three scalar reductions).
function _cfl_number(sim, ::ConservativeCFL)
    m = sim.model
    FT = m.FT
    umax = maximum(abs.(m.U.present) .* m.umask)
    vmax = maximum(abs.(m.V.present) .* m.vmask)
    gDdrho = maximum(m.drho .* m.D.present .* m.tmask)
    c = sqrt(m.g * max(zero(FT), gDdrho))
    return Float64(sim.clock.dt) * (
        (Float64(umax) + Float64(c)) / Float64(m.dx) +
        (Float64(vmax) + Float64(c)) / Float64(m.dy)
    )
end

# Exact: per-cell CFL using T-point-interpolated velocities; maximum over active cells.
# Tighter than ConservativeCFL but allocates temporaries proportional to grid size.
function _cfl_number(sim, ::ExactCFL)
    m = sim.model
    FT = m.FT
    U_T = im_half(m.U.present)
    V_T = jm_half(m.V.present)
    c = sqrt.(m.g .* max.(zero(FT), m.drho .* m.D.present))
    cfl_cell =
        Float64(sim.clock.dt) .* (
            abs.(U_T) ./ Float64(m.dx) .+ abs.(V_T) ./ Float64(m.dy) .+ c ./ Float64(m.dx) .+
            c ./ Float64(m.dy)
        ) .* m.tmask
    return Float64(maximum(cfl_cell))
end

# Worst-case CFL: the advective term uses the velocity cap `v_cut` instead of the
# actual speed.  At startup the flow is ~stationary, so the actual CFL is tiny
# and useless for sizing dt0; this bounds the advective CFL the developing flow
# can ever reach (it cannot exceed v_cut), while keeping the real gravity-wave
# term.  Used by the preemptive startup rescue, not the in-loop controller.
function _cfl_worstcase(sim)
    m = sim.model
    FT = m.FT
    v_cut = Float64(m.v_cut)
    gDdrho = maximum(m.drho .* m.D.present .* m.tmask)
    c = Float64(sqrt(m.g * max(zero(FT), gDdrho)))
    return Float64(sim.clock.dt) *
           ((v_cut + c) / Float64(m.dx) + (v_cut + c) / Float64(m.dy))
end

# Abort with a clear message as soon as the integration produces non-finite
# values, instead of silently stepping NaNs for the rest of the run.  `t`/`nt`
# count the steps of the current `run!` call.
function _check_blowup(sim, t, nt)
    m = sim.model
    (
        all(isfinite, m.D.present) &&
        all(isfinite, m.U.present) &&
        all(isfinite, m.V.present)
    ) && return
    error(
        string(
            "Simulation blew up: non-finite values in D/U/V at step ",
            t,
            "/",
            nt,
            " (≈ day ",
            round(_t_days(sim), digits = 2),
            "). Common causes: time step too",
            " large for this grid (dt = ",
            sim.clock.dt,
            " s, dx = ",
            m.dx,
            " m) or unstable",
            " forcing. Reduce dt in Simulation, or check the inputs.",
        ),
    )
end


"""
$(TYPEDSIGNATURES)

Advance `sim` by one leapfrog time step: `advance_leapfrog!` → `leapfrog_step!`
(2×dt) → `clamp_velocities!` → `apply_robert_asselin_filter!`, then move the
clock forward by `dt`.  No I/O, no CFL control, no blow-up check — those belong
to [`run!`](@ref).

Returns `sim`.
"""
function time_step!(sim::Simulation)
    advance_leapfrog!(sim)
    leapfrog_step!(sim, 2)
    clamp_velocities!(sim.model)
    apply_robert_asselin_filter!(sim)
    sim.clock.time += sim.clock.dt
    sim.clock.iteration += 1
    return sim
end

"""
$(TYPEDSIGNATURES)

Advance simulation `sim` until the criterion `until` is met.

`until` is an [`AbstractSimulationEnd`](@ref) measured from the clock time at
which `run!` is called: [`FixedSimulationEnd`](@ref) runs for a fixed duration,
[`SteadyStateEnd`](@ref) stops early once the mean melt rate is quasi-steady.  It
defaults to `sim.stop`; as a shorthand, `run!(sim; days = 30.0)` is equivalent to
`run!(sim; until = FixedSimulationEnd(t_end = 30.0))`.

The clock is never reset, so successive calls continue the same simulation:
`run!(sim; days = 10); run!(sim; days = 10)` ends at day 20, and output and
restart files are stamped accordingly.  Each call rounds its own duration to a
whole number of steps, so a sequence of calls matches one long call exactly only
when the durations are multiples of `dt`.

Each step is one [`time_step!`](@ref).  When `sim.output.saveday > 0`,
`savefields!`, `printdiags`, and `saverestart!` are also called, and the call
ends by flushing the partial averaging window and writing a restart.  When
`verbose = true`, a progress bar with throughput and ETA is displayed;
melt/thickness/speed diagnostics attached to the bar refresh every ~5 % of
steps (they are device reductions, so they are deliberately not per-step).

Before stepping, a warning is emitted if the advective CFL number at the
velocity cap `v_cut` exceeds 1.  Every ~5 % of steps the prognostic fields are
checked for non-finite values; on blow-up the run aborts with an error
instead of integrating NaNs.

Returns `sim` for chaining.
"""
function run!(sim::Simulation; days = nothing, until = nothing, verbose = true)
    m = sim.model
    clock = sim.clock
    FT = m.FT
    if days !== nothing && until !== nothing
        throw(ArgumentError("pass either `days` or `until`, not both"))
    end
    until =
        until !== nothing ? until :
        days !== nothing ? FixedSimulationEnd(t_end = Float64(days)) : sim.stop
    total = _end_seconds(until, m.seconds_per_day)  # hard time cap (s) for this call
    io_on = sim.output.saveday > 0
    # Predictive adaptive dt: rescue a too-large dt before the first step
    # (no-op for FixedDt). nt/checkint below then reflect the adjusted dt.
    _init_adaptive_dt!(sim, sim.tstep)
    nt = round(Int, total / clock.dt)
    checkint = _check_interval(sim.tstep, nt)
    cfl = _cfl_worstcase(sim)
    cfl > 1.0 && @warn string(
        "Worst-case CFL (advection at v_cut + gravity wave) is ",
        round(cfl, digits = 2),
        " > 1 (dt = ",
        clock.dt,
        " s, dx = ",
        m.dx,
        " m, dy = ",
        m.dy,
        " m); the run is likely unstable — reduce dt or coarsen the grid.",
    )
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
    # Steady-state sampling: compare the mean melt rate on a fixed daily cadence
    # (independent of run length, so `tol` means the same thing for any cap).
    # Disabled (next_steady = Inf) unless the criterion needs it.
    prev_mean = NaN
    next_steady = _needs_melt_sample(until) ? Float64(m.seconds_per_day) : Inf
    # Steps and simulated seconds of *this call*: the stopping rule, the check
    # cadence and the progress bar are all relative to where the call started,
    # while the clock and the I/O event times are absolute.
    step = 0
    elapsed = 0.0
    # Time cap (round-half-up rule: round(total/dt) steps for fixed dt); a
    # SteadyStateEnd may break out earlier once the mean melt rate is steady.
    while elapsed + clock.dt / 2 < total
        step += 1
        time_step!(sim)
        elapsed += clock.dt
        if io_on
            savefields!(sim)
            printdiags(sim)
            saverestart!(sim)
        end
        # Steady-state early stop: sample the mean melt rate once per simulated
        # day (before any dt re-bootstrap, so it sees the clean stepped state)
        # and stop when its relative change falls below the tolerance.  No-op
        # for FixedSimulationEnd (next_steady = Inf).
        if elapsed + clock.dt / 2 >= next_steady
            _, mean_melt, _ = meltstats(m)
            if _steady_reached(until, mean_melt, prev_mean)
                _print2log(
                    sim,
                    string(
                        round(_t_days(sim), digits = 3),
                        " days: steady state reached (relative Δ mean melt < ",
                        until.tol,
                        ")",
                    ),
                )
                break
            end
            prev_mean = mean_melt
            next_steady += Float64(m.seconds_per_day)
        end
        # Device-reduction diagnostics force a GPU sync, so they run only at
        # this cadence (~5 %, or every `ncheck` steps under AdaptiveDt — the
        # blow-up check, the CFL monitor, the controller, and the progress
        # diagnostics all share this one sync point).
        if step % checkint == 0 || elapsed + clock.dt / 2 >= total
            _check_blowup(sim, step, nt)
            # Adjust dt for the upcoming steps (no-op under FixedDt); after I/O
            # and the blow-up check, so both see the clean stepped state.
            cfl = (verbose || _adapts(sim.tstep)) ? _cfl_number(sim) : 0.0
            _maybe_adapt_dt!(sim, sim.tstep, cfl)
            if verbose
                mx, mn, sp = meltstats(m)
                Dmax = maximum(ifelse.(m.tmask .> 0, m.D.present, FT(-Inf)))
                showvals = [
                    ("simulated days", round(_t_days(sim), digits = 2)),
                    (
                        "melt mean/max [m/yr]",
                        string(round(mn, digits = 2), " / ", round(mx, digits = 2)),
                    ),
                    ("Dmax [m]", round(Dmax, digits = 1)),
                    ("|u|max [m/s]", round(sp, digits = 3)),
                    (
                        "dt [s] / CFL",
                        string(round(clock.dt, digits = 1), " / ", round(cfl, digits = 3)),
                    ),
                ]
            end
        end
        update!(prog, round(Int, elapsed); showvalues = showvals)
    end
    # Flush the final partial average and write the end-of-run restart.
    if io_on
        flush_output!(sim)
        _write_restart!(sim, _t_days(sim))
    end
    finish!(prog)
    return sim
end
