
# =================================================================
# Driver
# =================================================================
"""
$(TYPEDSIGNATURES)

Return melt-rate and velocity statistics for the current state of a model or
simulation, as a named tuple.  All reductions execute on the device (GPU-safe);
results are returned as CPU scalars.

- `max_meltrate`  — maximum basal melt rate, m yr⁻¹
- `mean_meltrate` — area-mean basal melt rate over ice-covered cells, m yr⁻¹
- `max_speed`     — maximum depth-averaged current speed |u|, m s⁻¹
- `total_melt`    — area-integrated basal melt (mass flux), Gt yr⁻¹

Melt rates are freshwater equivalent, the unit the model integrates; multiply by
`rho_freshwater / rho_ice` for ice equivalent.  `total_melt` is the corresponding
mass flux, `Σ melt · dx · dy · rho_freshwater`, and is the same in either unit.

The fields keep their order, so `max_melt, mean_melt, max_speed = meltstats(sim)`
still destructures the first three.

The melt statistics average over `imask` (ice-covered cells), not `tmask`.  Under
[`ConnectedGapsBC`](@ref) the two differ: gap cells are dynamically active but ice-free
and melt exactly zero, so averaging over `tmask` would dilute the mean by the gap area.
This diverges from LADDIE v2, whose `domain_a` integration includes gaps; with no gaps
in the mask the two masks coincide and the statistics are unchanged.  `max_speed` stays
on `tmask` — the layer really does flow through gaps, and that flow is the point.
"""
meltstats(sim::Simulation) = meltstats(sim.model)

const _KG_PER_GT = 1e12     # kilogrammms per gigatonne

function meltstats(m::Model)
    imask = m.imask
    d = m.diag
    n = sum(imask)
    @. d = m.melt * m.seconds_per_year * imask
    max_meltrate = maximum(d)
    melt_sum = sum(d)                             # m yr⁻¹ summed over ice cells
    mean_meltrate = melt_sum / n
    total_melt = melt_sum * m.dx * m.dy * m.rho_freshwater / _KG_PER_GT
    _launch_tpoint_diag!(_speed_norm_kernel!, m)
    max_speed = maximum(d)
    return (; max_meltrate, mean_meltrate, max_speed, total_melt)
end

# Per-cell diagnostics on T-points, written into `m.diag`. Velocities are
# averaged onto the T-point as `im_half(U)`, `jm_half(V)`. Both kernels share
# one argument list (see `_launch_tpoint_diag!`), so the speed ignores some of it.
@kernel function _speed_norm_kernel!(
    out,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(D),
    @Const(tmask),
    g,
    dx,
    dy,
    Nx,
    Ny,
)
    g, dx, dy = _val(g), _val(dx), _val(dy)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u = (U[i, j] + U[_xm1(i, Nx), j]) / 2
        v = (V[i, j] + V[i, _ym1(j, Ny)]) / 2
        out[i, j] = _safe_sqrt(u^2 + v^2) * tmask[i, j]
    end
end

# The CFL number of a cell per unit time step: advection plus the internal
# gravity-wave speed c = √(g·max(0, δρ·D)) in both directions.
@kernel function _cfl_rate_kernel!(
    out,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(D),
    @Const(tmask),
    g,
    dx,
    dy,
    Nx,
    Ny,
)
    g, dx, dy = _val(g), _val(dx), _val(dy)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(g)
        u = (U[i, j] + U[_xm1(i, Nx), j]) / 2
        v = (V[i, j] + V[i, _ym1(j, Ny)]) / 2
        c = _safe_sqrt(g * max(zero(FT), drho[i, j] * D[i, j]))
        out[i, j] = (abs(u) / dx + abs(v) / dy + c / dx + c / dy) * tmask[i, j]
    end
end

function _launch_tpoint_diag!(kernel!, m)
    nx, ny = size(m.diag)
    launch!(
        kernel!,
        m.diag,
        m.U.present,
        m.V.present,
        m.drho,
        m.D.present,
        m.tmask,
        m.g,
        m.dx,
        m.dy,
        nx,
        ny,
    )
end

# The CFL number of the current state, for the dt controller.  `Simulation.cfl`
# selects ConservativeCFL or ExactCFL.  The device reductions and the host formula
# that combines them are separate functions, so that the Reactant extension can
# compile the reductions and still combine them the same way.
_cfl_number(sim) =
    _float64(sim.clock.dt) *
    _cfl_rate(sim.model, sim.cfl, _cfl_reductions(sim.model, sim.cfl))

# Largest δρ·D over the active domain, for the gravity-wave speed.
function _max_Ddrho(m)
    @. m.diag = m.drho * m.D.present * m.tmask
    return maximum(m.diag)
end

# Conservative: global max of |U|, |V|, and c taken independently, then combined.
# Overestimates the true CFL but cheap (three scalar reductions).
function _cfl_reductions(m, ::ConservativeCFL)
    d = m.diag
    @. d = abs(m.U.present) * m.umask
    umax = maximum(d)
    @. d = abs(m.V.present) * m.vmask
    vmax = maximum(d)
    c = sqrt(m.g * max(zero(m.FT), _max_Ddrho(m)))
    return (umax, vmax, c)
end
_cfl_rate(m, ::ConservativeCFL, (umax, vmax, c)) =
    (_float64(umax) + _float64(c)) / _float64(m.dx) +
    (_float64(vmax) + _float64(c)) / _float64(m.dy)

# Exact: per-cell CFL using T-point-interpolated velocities; maximum over active
# cells.  Tighter than ConservativeCFL at the cost of one kernel pass.
function _cfl_reductions(m, ::ExactCFL)
    _launch_tpoint_diag!(_cfl_rate_kernel!, m)
    return (maximum(m.diag),)
end
_cfl_rate(m, ::ExactCFL, (rate,)) = _float64(rate)

# Worst-case CFL: the advective term uses the velocity cap `v_cut` instead of the
# actual speed.  At startup the flow is ~stationary, so the actual CFL is tiny
# and useless for sizing dt0; this bounds the advective CFL the developing flow
# can ever reach (it cannot exceed v_cut), while keeping the real gravity-wave
# term.  Used by the preemptive startup rescue, not the in-loop controller.
function _cfl_worstcase(sim)
    m = sim.model
    FT = m.FT
    v_cut = _float64(m.v_cut)
    c = _float64(sqrt(m.g * max(zero(FT), _max_Ddrho(sim.exec, sim))))
    return _float64(sim.clock.dt) *
           ((v_cut + c) / _float64(m.dx) + (v_cut + c) / _float64(m.dy))
end

# Abort with a clear message as soon as the integration produces non-finite
# values, instead of silently stepping NaNs for the rest of the run.  `t`/`nt`
# count the steps of the current `run!` call.
function _check_blowup(sim, t, nt)
    m = sim.model
    _prognostics_finite(sim.exec, sim) && return
    error(
        "Simulation blew up: non-finite values in D/U/V at step $t/$nt " *
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
    _step_model!(sim)
    sim.clock.time += _primal(sim.clock.dt)
    sim.clock.iteration += 1
    return sim
end

"""
$(TYPEDSIGNATURES)

Advance `model` by `n` leapfrog steps of `dt` seconds (Robert–Asselin coefficient
`nu`), without a [`Simulation`](@ref): no clock, output, dt control or blow-up check.
The model must already be bootstrapped, e.g. taken from a simulation after
construction (`sim.model`).  Returns `model`.

This is the model as a plain function of its inputs, for automatic differentiation:
with ForwardDiff through a model built at `FT = Dual`, and with Enzyme through a
program compiled by [`reactant_compile`](@ref), where `n` is a traced number and the
steps run as a traced loop.

```julia
loss(model, dt, n) = (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))
```

Under Reactant, the traced method also takes `checkpoints`, the memory budget of
reverse-mode differentiation; and `integrate!(model, sched)` replays an adaptive dt
schedule (see [`adaptive_schedule`](@ref)).
"""
function integrate!(model, dt, n; nu = 0.8)
    s = _stepping_view(model, dt, nu)
    for _ = 1:n
        _step_model!(s)
    end
    return model
end

# What the step functions read from a Simulation (they take `sim` untyped).
_stepping_view(model, dt, nu) =
    (; model, clock = (; dt), nu = model.FT(nu), debug = (; check_nans = false))

# One leapfrog step of the model: `time_step!` without the clock.
function _step_model!(sim)
    advance_leapfrog!(sim)
    leapfrog_step!(sim, 2)
    apply_robert_asselin_filter!(sim)
    return
end

"""
$(TYPEDSIGNATURES)

Compile `f(args...)` with Reactant for the arrays of a model moved to a
[`ReactantBackend`](@ref), with the kernel settings Laddie needs (requires `using
Reactant, CUDA`).  Returns the compiled function; call it with arguments of the same
types and sizes.

The kernels are raised to XLA operations before any differentiation
(`raise_first = true`), so `f` may call `Enzyme.autodiff`. `fusion` is one of the
strategies of `ReactantBackend`; only `:xla` can be differentiated.  Other keywords
go to `Reactant.compile`.

The scalar parameters of a model are compiled in as constants: the program ignores
the parameters of the model it is called with.  Pass a model from
[`trace_parameters`](@ref) to make them inputs of the program, which can then be
called with other parameter values and differentiated with respect to them.

```julia
using Laddie, Reactant, CUDA
using Reactant: Enzyme
sim = to_backend(build_isomip(CPU()), ReactantBackend())
loss(model, dt, n) = (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))
dmodel = Enzyme.make_zero(sim.model)
dmodel.forcing.ocean.Tz .= 1                       # direction: uniform warming
fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                     Enzyme.Const(dt), Enzyme.Const(n))
dt, n = ConcreteRNumber(sim.clock.dt), ConcreteRNumber(100)
dloss = only(reactant_compile(fwd, sim.model, dmodel, dt, n)(sim.model, dmodel, dt, n))
```
"""
function reactant_compile end

"""
$(TYPEDEF)

The dt schedule of an adaptive run, as segments of constant dt, made by
[`adaptive_schedule`](@ref) and replayed by `integrate!(model, sched)`.  The arrays have
a fixed capacity; entries past `nseg` are unused.

# Fields
$(TYPEDFIELDS)
"""
struct DtSchedule{V,W,N}
    "dt of each segment (s)"
    dt::V
    "number of steps in each segment"
    steps::W
    "number of segments in use"
    nseg::N
end

"""
    adaptive_schedule(model, dt; days, stepper = AdaptiveDt(), cfl = ExactCFL(), nu = 0.8,
                      maxsegments = 10_000) -> DtSchedule

Run `model` (on a [`ReactantBackend`](@ref); requires `using Reactant, CUDA`) for `days`
from the step `dt` (s), adapting dt with `stepper` as `run!` does, and return the dt
schedule it took.  Advances `model`.

Differentiating through the dt controller is neither possible (Enzyme cannot reverse
a loop whose length depends on the data) nor wanted (the controller is not smooth), so
an adaptive run is differentiated in two passes, like the ForwardDiff path does it
implicitly: this function fixes the schedule, and `integrate!(model, sched)` inside
[`reactant_compile`](@ref) replays it with dt as data.  The gradient is that of the run
with this dt sequence.

```julia
fresh() = trace_parameters(to_backend(build_isomip(CPU()), ReactantBackend()).model)
sched = adaptive_schedule(fresh(), 120.0; days = 15)                 # pass 1
function loss(model, sched)                                          # pass 2
    melt = integrate!(model, sched; means = (:melt,)).melt
    return sum(melt .* model.imask) / sum(model.imask)
end
prog = reactant_compile(loss, fresh(), sched)       # differentiable, as with integrate!
```

`integrate!(model, sched; means = (), nu = 0.8, checkpoints = 20)` returns the
time-weighted means over the run of the fields named in `means` (e.g. `:melt`, `:D`),
as a `NamedTuple`, or `model` when `means` is empty.  The controller checks the CFL
every `stepper.ncheck` steps counted from the start of the call, and re-bootstraps the
leapfrog when dt changes, as `run!`; the last step may end past `days`.
"""
function adaptive_schedule end

integrate!(model, sched::DtSchedule; kwargs...) = throw(ArgumentError(
    "integrate!(model, ::DtSchedule) replays a schedule inside a program compiled by " *
    "`reactant_compile` (requires `using Reactant, CUDA`)"))

"""
    trace_parameters(model; overrides...)

A model whose scalar parameters ([`Params`](@ref), including the floats of its
parameterisation objects) are Reactant numbers, so that a program compiled by
[`reactant_compile`](@ref) takes them as inputs rather than constants (requires
`using Reactant, CUDA`).  The model shares its arrays with `model`.  Keywords replace
fields of `Params`: numbers are converted to the model precision, parameterisation
objects are converted as by `Params`.

The compiled program then runs for any parameter values without recompiling, and
Enzyme differentiates it with respect to a parameter through a tangent model seeded
with `trace_parameters` too.  Parameters that only act when the model is built
(`coriolis`, `D_init`, `dT_init`, `dS_init`) have no effect on the program.

```julia
using Laddie, Reactant, CUDA
using Reactant: Enzyme
sim = to_backend(build_isomip(CPU(); FT = Float64), ReactantBackend())
model = trace_parameters(sim.model)
loss(model, dt, n) = (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))
dmodel = trace_parameters(Enzyme.make_zero(model); C_d = 1)      # direction: C_d
fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                     Enzyme.Const(dt), Enzyme.Const(n))
dt, n = ConcreteRNumber(sim.clock.dt), ConcreteRNumber(100)
dloss = only(reactant_compile(fwd, model, dmodel, dt, n)(model, dmodel, dt, n))
```

Do not `run!` a simulation on such a model: `run!` compiles its own programs, and its
host-side dt control and diagnostics expect plain numbers.
"""
function trace_parameters end

# ============================================================================
# Execution hooks of `run!`.  The native (KernelAbstractions) versions step one
# time step at a time and compute the diagnostics directly on the model; the
# Reactant extension replaces them with compiled programs.
# ============================================================================

# Steps to take before `run!` must act on the host again.
_batch_length(::NativeExecution, sim, step, checkint, elapsed, total, next_steady, io_on) = 1

# The number of steps until the next host event: the check cadence, the end of the
# run, a steady-state sample, or (with output) an output, diagnostics or restart
# time.  Replays the host arithmetic of `run!` step by step, so a batched run meets
# every event at the same step as a one-step-at-a-time run.
function _steps_to_next_event(sim, step, checkint, elapsed, total, next_steady, io_on)
    dt = _primal(sim.clock.dt)
    t = sim.clock.time
    io = sim.io
    k = 0
    while true
        k += 1
        t += dt
        elapsed += dt
        half = dt / 2
        (step + k) % checkint == 0 && return k
        (elapsed + half >= total || elapsed + half >= next_steady) && return k
        io_on && (t + half >= io.nextsave || t + half >= io.nextdiag ||
                  t + half >= io.nextrest) && return k
    end
end

# Advance `n` steps, accumulating the output averages after each one.
function _advance_batch!(::NativeExecution, sim, n, io_on)
    for _ = 1:n
        time_step!(sim)
        io_on && _accum!(sim)
    end
    return
end

_prognostics_finite(::NativeExecution, sim) = _prognostics_finite(sim.model)
_sync_cfl_number(::NativeExecution, sim) = _cfl_number(sim)
_meltstats(::NativeExecution, sim) = meltstats(sim.model)
_max_Ddrho(::NativeExecution, sim) = _max_Ddrho(sim.model)
_max_active_D(::NativeExecution, sim) = _max_active_D(sim.model)

# `&`, not `&&`: also evaluated on traced values by the Reactant extension.
_prognostics_finite(m) =
    all(isfinite, m.D.present) & all(isfinite, m.U.present) & all(isfinite, m.V.present)
function _max_active_D(m)
    @. m.diag = ifelse(m.tmask > 0, m.D.present, m.FT(-Inf))
    return maximum(m.diag)
end
# The model the log diagnostics (`printdiags`) are computed on.
_diag_model(::NativeExecution, sim) = sim.model

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

Each step is one [`time_step!`](@ref).  When `sim.output.saveday > 0`, the
output fields are accumulated after every step, `savefields!`, `printdiags` and
`saverestart!` write whenever their interval is due, and the call ends by flushing
the partial averaging window and writing a restart.  When
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
    _float_type(m.params) === _scalar_type(m.params) || throw(
        ArgumentError(
            "run! needs plain scalar parameters; a model from `trace_parameters` is " *
            "for programs compiled with `reactant_compile`",
        ),
    )
    until =
        until !== nothing ? until :
        days !== nothing ? FixedSimulationEnd(t_end = _float64(days)) : sim.stop
    total = _end_seconds(until, _primal(m.seconds_per_day))  # hard time cap (s) for this call
    io_on = sim.output.saveday > 0
    # Predictive adaptive dt: rescue a too-large dt before the first step
    # (no-op for FixedDt). nt/checkint below then reflect the adjusted dt.
    _init_adaptive_dt!(sim, sim.tstep)
    nt = round(Int, total / clock.dt)
    checkint = _check_interval(sim.tstep, nt)
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
    # Steady-state sampling: compare the mean melt rate on a fixed daily cadence
    # (independent of run length, so `tol` means the same thing for any cap).
    # Disabled (next_steady = Inf) unless the criterion needs it.
    prev_mean = NaN
    next_steady = _needs_melt_sample(until) ? _float64(m.seconds_per_day) : Inf
    # Steps and simulated seconds of *this call*: the stopping rule, the check
    # cadence and the progress bar are all relative to where the call started,
    # while the clock and the I/O event times are absolute.
    step = 0
    elapsed = 0.0
    # Time cap (round-half-up rule: round(total/dt) steps for fixed dt); a
    # SteadyStateEnd may break out earlier once the mean melt rate is steady.
    while elapsed + clock.dt / 2 < total
        # Steps up to the next host event (1 on the KernelAbstractions backends;
        # a compiled batch under Reactant), with the output accumulation.
        n = _batch_length(sim.exec, sim, step, checkint, elapsed, total, next_steady, io_on)
        _advance_batch!(sim.exec, sim, n, io_on)
        step += n
        for _ = 1:n
            elapsed += _primal(clock.dt)
        end
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
            _, mean_melt, _ = _meltstats(sim.exec, sim)
            if _steady_reached(until, mean_melt, prev_mean)
                _print2log(
                    sim,
                    "$(round(_t_days(sim), digits = 3)) days: steady state reached " *
                    "(relative Δ mean melt < $(until.tol))",
                )
                break
            end
            prev_mean = mean_melt
            next_steady += _float64(m.seconds_per_day)
        end
        # Device-reduction diagnostics force a GPU sync, so they run only at
        # this cadence (~5 %, or every `ncheck` steps under AdaptiveDt — the
        # blow-up check, the CFL monitor, the controller, and the progress
        # diagnostics all share this one sync point).
        if step % checkint == 0 || elapsed + clock.dt / 2 >= total
            _check_blowup(sim, step, nt)
            # Adjust dt for the upcoming steps (no-op under FixedDt); after I/O
            # and the blow-up check, so both see the clean stepped state.
            cfl = (verbose || _adapts(sim.tstep)) ? _sync_cfl_number(sim.exec, sim) : 0.0
            _maybe_adapt_dt!(sim, sim.tstep, cfl)
            if verbose
                mx, mn, sp, gt = _meltstats(sim.exec, sim)
                Dmax = _max_active_D(sim.exec, sim)
                showvals = [
                    ("simulated days", _r(_t_days(sim), 2)),
                    ("melt mean/max [m/yr]", string(_r(mn, 2), " / ", _r(mx, 2))),
                    ("Dmax [m]", _r(Dmax, 1)),
                    ("total melt [Gt/yr]", _r(gt, 2)),
                    ("|u|max [m/s]", _r(sp, 3)),
                    ("dt [s] / CFL", string(_r(clock.dt, 1), " / ", _r(cfl, 3))),
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
