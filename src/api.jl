using ProgressMeter

# ============================================================================
# Driver
# ============================================================================

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
    return maximum(meltyr .* mask),
    sum(meltyr .* mask) / n,
    maximum(sqrt.(im(m.U.present) .^ 2 .+ jm(m.V.present) .^ 2) .* mask)
end

# Abort with a clear message as soon as the integration produces non-finite
# values, instead of silently stepping NaNs for the rest of the run.
function _check_blowup(m, t, nt)
    (all(isfinite, m.D.present) && all(isfinite, m.U.present) &&
     all(isfinite, m.V.present)) && return
    t_days = m.t_start + t * m.dt / 86400
    error(string(
        "Simulation blew up: non-finite values in D/U/V at step ", t, "/", nt,
        " (≈ day ", round(t_days, digits = 2), "). Common causes: time step too",
        " large for this grid (dt = ", m.dt, " s, dx = ", m.dx, " m) or unstable",
        " forcing. Reduce dt in Params, or check the inputs."))
end

"""
$(TYPEDSIGNATURES)

Advance model `m` forward in time for `days` days (default: `m.rc.days`).

Each step applies `advance_leapfrog!` → `leapfrog_step!` (2×dt) →
`clamp_velocities!` → `apply_robert_asselin_filter!`.  When `m.rc.saveday > 0`,
`savefields!`, `printdiags`, and `saverestart!` are also called.  When
`verbose = true`, a progress bar with throughput and ETA is displayed;
melt/thickness/speed diagnostics attached to the bar refresh every ~5 % of
steps (they are device reductions, so they are deliberately not per-step).

Before stepping, a warning is emitted if the advective CFL number at the
velocity cap `vcut` exceeds 1.  Every ~5 % of steps the prognostic fields are
checked for non-finite values; on blow-up the run aborts with an error
instead of integrating NaNs.

Returns `m` for chaining.
"""
function run!(m; days = nothing, verbose = true)
    n_days = days !== nothing ? Float64(days) : m.rc.days
    nt = round(Int, n_days * 86400 / m.dt)
    diagint = max(1, nt ÷ 20)
    io_on = m.rc.saveday > 0
    io_on && (m.nt = nt)
    cfl = Float64(m.dt) * Float64(m.vcut) * (1.0 / Float64(m.dx) + 1.0 / Float64(m.dy))
    cfl > 1.0 && @warn string(
        "Advective CFL number at the velocity cap is ", round(cfl, digits = 2),
        " > 1 (dt = ", m.dt, " s, vcut = ", m.vcut, " m/s, dx = ", m.dx,
        " m, dy = ", m.dy, " m); the run is likely unstable — reduce dt or coarsen the grid.")
    backend = nameof(typeof(KA.get_backend(m.tmask)))
    prog = Progress(nt;
        desc = "[$backend] $(m.ny)×$(m.nx) interior, $nt steps: ",
        enabled = verbose, showspeed = true)
    showvals = Tuple{String, Any}[]
    for t = 1:nt
        io_on && (m.t = t)
        advance_leapfrog!(m)
        leapfrog_step!(m, 2)
        clamp_velocities!(m)
        apply_robert_asselin_filter!(m)
        if io_on
            savefields!(m)
            printdiags(m)
            saverestart!(m)
        end
        if t % diagint == 0 || t == nt
            _check_blowup(m, t, nt)
            # These diagnostics are device reductions (each forces a GPU
            # sync), so they refresh only at this ~5 % cadence; the bar and
            # ETA update every step via next! at negligible cost.
            if verbose
                mx, mn, sp = meltstats(m)
                Dmax = maximum(ifelse.(m.tmask .> 0, m.D.present, -Inf))
                showvals = [
                    ("simulated days", round(m.t_start + t * m.dt / 86400, digits = 2)),
                    ("melt mean/max [m/yr]", string(round(mn, digits = 2), " / ", round(mx, digits = 2))),
                    ("Dmax [m]", round(Dmax, digits = 1)),
                    ("|u|max [m/s]", round(sp, digits = 3)),
                ]
            end
        end
        next!(prog; showvalues = showvals)
    end
    finish!(prog)
    return m
end