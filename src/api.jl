using ProgressMeter

# =================================================================
# Driver
# =================================================================
"""
$(TYPEDSIGNATURES)

Return melt-rate and velocity statistics for the current model state.
All reductions execute on the device (GPU-safe); results are returned as
CPU scalars.

- `max_meltrate`  — maximum basal melt rate, m yr⁻¹
- `mean_meltrate` — area-mean basal melt rate over floating ice, m yr⁻¹
- `max_speed`     — maximum depth-averaged current speed |u|, m s⁻¹
"""
function meltstats(m)
    mask = m.tmask
    n = sum(mask)
    meltyr = m.melt .* spy
    max_meltrate = maximum(meltyr .* mask)
    mean_meltrate = sum(meltyr .* mask) / n
    max_speed = maximum(sqrt.(im_half(m.U.present) .^ 2 .+ jm_half(m.V.present) .^ 2) .* mask)
    return max_meltrate, mean_meltrate, max_speed
end

# Dispatch on RunConfig.cfl — selects ConservativeCFL or ExactCFL.
_cfl_number(m) = _cfl_number(m, m.cfl)

# Conservative: global max of |U|, |V|, and c taken independently, then combined.
# Overestimates the true CFL but cheap (three scalar reductions).
function _cfl_number(m, ::ConservativeCFL)
    FT = m.FT
    umax = maximum(ifelse.(m.umask .> 0, abs.(m.U.present), zero(FT)))
    vmax = maximum(ifelse.(m.vmask .> 0, abs.(m.V.present), zero(FT)))
    gDdrho = maximum(ifelse.(m.tmask .> 0, m.drho .* m.D.present, zero(FT)))
    c = sqrt(m.g * max(zero(FT), gDdrho))
    return Float64(m.dt) * (
        (Float64(umax) + Float64(c)) / Float64(m.dx) +
        (Float64(vmax) + Float64(c)) / Float64(m.dy)
    )
end

# Exact: per-cell CFL using T-point-interpolated velocities; maximum over active cells.
# Tighter than ConservativeCFL but allocates temporaries proportional to grid size.
function _cfl_number(m, ::ExactCFL)
    FT = m.FT
    U_T = im_half(m.U.present)
    V_T = jm_half(m.V.present)
    c = sqrt.(m.g .* max.(zero(FT), m.drho .* m.D.present))
    cfl_cell = Float64(m.dt) .* (
        abs.(U_T) ./ Float64(m.dx) .+
        abs.(V_T) ./ Float64(m.dy) .+
        c ./ Float64(m.dx) .+
        c ./ Float64(m.dy)
    ) .* m.tmask
    return Float64(maximum(cfl_cell))
end

# Worst-case CFL: the advective term uses the velocity cap `v_cut` instead of the
# actual speed.  At startup the flow is ~stationary, so the actual CFL is tiny
# and useless for sizing dt0; this bounds the advective CFL the developing flow
# can ever reach (it cannot exceed v_cut), while keeping the real gravity-wave
# term.  Used by the preemptive startup rescue, not the in-loop controller.
function _cfl_worstcase(m)
    FT = m.FT
    v_cut = Float64(m.v_cut)
    gDdrho = maximum(ifelse.(m.tmask .> 0, m.drho .* m.D.present, zero(FT)))
    c = Float64(sqrt(m.g * max(zero(FT), gDdrho)))
    return Float64(m.dt) * ((v_cut + c) / Float64(m.dx) + (v_cut + c) / Float64(m.dy))
end

# Abort with a clear message as soon as the integration produces non-finite
# values, instead of silently stepping NaNs for the rest of the run.
function _check_blowup(m, t, nt)
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
            round(_t_days(m), digits = 2),
            "). Common causes: time step too",
            " large for this grid (dt = ",
            m.dt,
            " s, dx = ",
            m.dx,
            " m) or unstable",
            " forcing. Reduce dt in Params, or check the inputs.",
        ),
    )
end

"""
$(TYPEDSIGNATURES)

Advance model `m` forward in time until the criterion `until` is met.

`until` is an [`AbstractSimulationEnd`](@ref): [`FixedSimulationEnd`](@ref) runs
for a fixed duration, [`SteadyStateEnd`](@ref) stops early once the mean melt
rate is quasi-steady.  As a shorthand, `run!(m; days = 30.0)` is equivalent to
`run!(m; until = FixedSimulationEnd(t_end = 30.0))`; with neither given the run
lasts `m.config.days`.

Each step applies `advance_leapfrog!` → `leapfrog_step!` (2×dt) →
`clamp_velocities!` → `apply_robert_asselin_filter!`.  When `m.config.saveday > 0`,
`savefields!`, `printdiags`, and `saverestart!` are also called.  When
`verbose = true`, a progress bar with throughput and ETA is displayed;
melt/thickness/speed diagnostics attached to the bar refresh every ~5 % of
steps (they are device reductions, so they are deliberately not per-step).

Before stepping, a warning is emitted if the advective CFL number at the
velocity cap `v_cut` exceeds 1.  Every ~5 % of steps the prognostic fields are
checked for non-finite values; on blow-up the run aborts with an error
instead of integrating NaNs.

Returns `m` for chaining.
"""
function run!(m; days = nothing, until = nothing, verbose = true)
    if days !== nothing && until !== nothing
        throw(ArgumentError("pass either `days` or `until`, not both"))
    end
    until =
        until !== nothing ? until :
        FixedSimulationEnd(t_end = days !== nothing ? Float64(days) : m.config.days)
    total = _end_seconds(until)                    # hard time cap (s) for this run
    io_on = m.config.saveday > 0
    # Fresh time accounting for this run; next-event times are relative to it.
    m.t = 0
    m.t_sim = 0.0
    if io_on
        m.nextsave = m.saveday * 86400.0
        m.nextdiag = m.diagday * 86400.0
        m.nextrest = m.restday * 86400.0
    end
    # Predictive adaptive dt: rescue a too-large dt0 before the first step
    # (no-op for FixedDt). nt/checkint below then reflect the adjusted dt.
    _init_adaptive_dt!(m, m.tstep)
    nt = round(Int, total / m.dt)
    checkint = _check_interval(m.tstep, nt)
    cfl = _cfl_worstcase(m)
    cfl > 1.0 && @warn string(
        "Worst-case CFL (advection at v_cut + gravity wave) is ",
        round(cfl, digits = 2),
        " > 1 (dt = ",
        m.dt,
        " s, dx = ",
        m.dx,
        " m, dy = ",
        m.dy,
        " m); the run is likely unstable — reduce dt or coarsen the grid.",
    )
    backend = nameof(typeof(KA.get_backend(m.tmask)))
    # Progress is tracked in simulated seconds (nt is only an estimate under
    # adaptive dt); update! sets the absolute position from t_sim each step.
    prog = Progress(
        round(Int, total);
        desc = "[$backend] $(m.ny)×$(m.nx) interior, ~$nt steps: ",
        enabled = verbose,
        showspeed = true,
    )
    showvals = Tuple{String,Any}[]
    # Steady-state sampling: compare the mean melt rate on a fixed daily cadence
    # (independent of run length, so `tol` means the same thing for any cap).
    # Disabled (next_steady = Inf) unless the criterion needs it.
    prev_mean = NaN
    next_steady = _needs_melt_sample(until) ? 86400.0 : Inf
    # Time cap (round-half-up rule: round(total/dt) steps for fixed dt); a
    # SteadyStateEnd may break out earlier once the mean melt rate is steady.
    while m.t_sim + m.dt / 2 < total
        m.t += 1
        advance_leapfrog!(m)
        leapfrog_step!(m, 2)
        clamp_velocities!(m)
        apply_robert_asselin_filter!(m)
        m.t_sim += m.dt
        if io_on
            savefields!(m)
            printdiags(m)
            saverestart!(m)
        end
        # Steady-state early stop: sample the mean melt rate once per simulated
        # day (before any dt re-bootstrap, so it sees the clean stepped state)
        # and stop when its relative change falls below the tolerance.  No-op
        # for FixedSimulationEnd (next_steady = Inf).
        if m.t_sim + m.dt / 2 >= next_steady
            _, mean_melt, _ = meltstats(m)
            if _steady_reached(until, mean_melt, prev_mean)
                _print2log(
                    m,
                    string(
                        round(_t_days(m), digits = 3),
                        " days: steady state reached (relative Δ mean melt < ",
                        until.tol,
                        ")",
                    ),
                )
                break
            end
            prev_mean = mean_melt
            next_steady += 86400.0
        end
        # Device-reduction diagnostics force a GPU sync, so they run only at
        # this cadence (~5 %, or every `ncheck` steps under AdaptiveDt — the
        # blow-up check, the CFL monitor, the controller, and the progress
        # diagnostics all share this one sync point).
        if m.t % checkint == 0 || m.t_sim + m.dt / 2 >= total
            _check_blowup(m, m.t, nt)
            # Adjust dt for the upcoming steps (no-op under FixedDt); after I/O
            # and the blow-up check, so both see the clean stepped state.
            cfl = (verbose || _adapts(m.tstep)) ? _cfl_number(m) : 0.0
            _maybe_adapt_dt!(m, m.tstep, cfl)
            if verbose
                mx, mn, sp = meltstats(m)
                Dmax = maximum(ifelse.(m.tmask .> 0, m.D.present, -Inf))
                showvals = [
                    ("simulated days", round(_t_days(m), digits = 2)),
                    (
                        "melt mean/max [m/yr]",
                        string(round(mn, digits = 2), " / ", round(mx, digits = 2)),
                    ),
                    ("Dmax [m]", round(Dmax, digits = 1)),
                    ("|u|max [m/s]", round(sp, digits = 3)),
                    (
                        "dt [s] / CFL",
                        string(round(m.dt, digits = 1), " / ", round(cfl, digits = 3)),
                    ),
                ]
            end
        end
        update!(prog, round(Int, m.t_sim); showvalues = showvals)
    end
    # Flush the final partial average and write the end-of-run restart.
    if io_on
        flush_output!(m)
        _write_restart!(m, _t_days(m))
    end
    finish!(prog)
    return m
end