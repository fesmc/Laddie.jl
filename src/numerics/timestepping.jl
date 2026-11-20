
# ============================================================================
# Time steppers — control how `dt` evolves during a run.  Held by the
# `Simulation` (`Simulation(model; time_stepper = ...)`); the runtime step lives in its
# `Clock` as `sim.clock.dt`.
# ============================================================================
"""
Abstract supertype for the time-step control of a [`Simulation`](@ref).  Pass a
concrete instance as `Simulation(model; time_stepper = ...)`: [`FixedTimeStepper`](@ref) (the
default) or [`AdaptiveTimeStepper`](@ref).
"""
abstract type AbstractTimeStepper end

"""
$(TYPEDEF)

Constant time step (the default): `dt` stays at the value the `Simulation` was built with, as
in LADDIE v1.

Select via `Simulation(model; time_stepper = FixedTimeStepper())` (the default).
"""
struct FixedTimeStepper <: AbstractTimeStepper end

"""
$(TYPEDEF)

CFL-limited adaptive time stepping. Every `ncheck` steps the advective–gravity-wave CFL number
is measured and `dt` scaled by `factor = (cfl_target / cfl)^q`: at once when above the target,
by at most `max_growth` when below `grow_hyst * cfl_target`, and not at all in between. Each
change re-bootstraps the leapfrog, so sample fields in one long `run!` call, not many short
ones; see [Time stepping](@ref).

Select via `Simulation(model; time_stepper = AdaptiveTimeStepper(; cfl_target = 0.4))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct AdaptiveTimeStepper{FT,I} <: AbstractTimeStepper
    "CFL setpoint the controller aims for (default `0.3`)"
    cfl_target::FT = 0.3
    "response exponent: `1` is CFL-exact, `< 1` damps, `> 1` overshoots (default `1.0`)"
    q::FT = 1.0
    "largest dt increase per adjustment, e.g. `1.1` = +10 % (default `1.1`)"
    max_growth::FT = 1.1
    "grow only when `cfl < grow_hyst * cfl_target` (default `0.8`)"
    grow_hyst::FT = 0.8
    "steps between CFL checks (default `20`)"
    ncheck::I = 20
    "lower clamp on dt (s, default `1.0`)"
    dtmin::FT = 1.0
    "upper clamp on dt (s, default `1000.0`)"
    dtmax::FT = 1000.0
    AdaptiveTimeStepper{FT,I}(args...) where {FT,I} = new{FT,I}(args...)
end

# The float fields are promoted to one type, so `AdaptiveTimeStepper(; dtmax = 1000)` works.
function AdaptiveTimeStepper(cfl_target, q, max_growth, grow_hyst, ncheck, dtmin, dtmax)
    f = promote(map(float, (cfl_target, q, max_growth, grow_hyst, dtmin, dtmax))...)
    return AdaptiveTimeStepper{eltype(f),typeof(ncheck)}(f[1:4]..., ncheck, f[5:6]...)
end

# (Float-type promotion of the stepper to the model's FT is handled generically by
# `_promote_param` in params.jl, called from the `Simulation` constructor.)

# Proposed dt from the predictive CFL controller, clamped to [dtmin, dtmax]: shrunk
# above the target, grown well below it, and the input dt unchanged inside the
# hysteresis band or when there is no usable CFL signal.  `allow_grow = false`
# (startup rescue) only ever shrinks, so a small initial CFL (e.g. zero velocity at
# t = 0) can never inflate dt0.  Branch-free and computed in the type of `dt`, so
# that the Reactant extension runs the same rule on traced numbers; `run!` calls it
# in Float64.
function _controller_dt(ts::AdaptiveTimeStepper, dt, cfl; allow_grow::Bool)
    FT = typeof(dt)
    c(x) = FT(_primal(x))
    target = c(ts.cfl_target)
    usable = (cfl > zero(cfl)) & isfinite(cfl)
    factor = (target / ifelse(usable, cfl, one(cfl)))^c(ts.q)
    grow = allow_grow ? cfl < c(ts.grow_hyst) * target : false
    dtn = ifelse(cfl > target, dt * factor,                            # above target → shrink now
                 ifelse(grow, dt * min(factor, c(ts.max_growth)), dt)) # well below → grow slowly
    return clamp(ifelse(usable, dtn, dt), c(ts.dtmin), c(ts.dtmax))
end

# Apply a controller decision: when dt actually changes, set it, re-bootstrap
# the leapfrog at the new dt, and log the change.  Returns whether dt changed.
function _apply_dt!(sim, ts::AdaptiveTimeStepper, cfl; allow_grow::Bool)
    dt_old = sim.clock.dt
    dt_new = sim.model.FT(_controller_dt(ts, _float64(dt_old), cfl; allow_grow))
    dt_new == dt_old && return false
    sim.clock.dt = dt_new
    _rebootstrap!(sim)
    _log_dt_change!(sim, dt_old, dt_new, cfl)
    return true
end

# Cadence of the shared blow-up/CFL/controller sync point.  FixedTimeStepper checks every
# ~5 % of the run.  AdaptiveTimeStepper checks at least every
# `ncheck` steps, and more often on short runs so a too-large dt0 is caught
# before it can blow up.  Both check at least once per simulated day, so that the
# blow-up check does not thin out on long runs.
_check_interval(::AbstractTimeStepper, nt, steps_per_day) = min(max(1, nt ÷ 20), steps_per_day)
_check_interval(ts::AdaptiveTimeStepper, nt, steps_per_day) =
    min(clamp(ts.ncheck, 1, max(1, nt ÷ 20)), steps_per_day)

"""
    adapt_dt!(sim, stepper, cfl)

Adjust `sim.clock.dt` from the measured CFL number `cfl`; `run!` calls it every check
interval. A no-op under [`FixedTimeStepper`](@ref); [`AdaptiveTimeStepper`](@ref) scales `dt`
toward its target and never rejects a step.
"""
function adapt_dt! end

adapt_dt!(sim, ::FixedTimeStepper, cfl) = nothing
adapt_dt!(sim, ts::AdaptiveTimeStepper, cfl) =
    (_apply_dt!(sim, ts, cfl; allow_grow = true); nothing)

"""
    init_adaptive_dt!(sim, stepper)

Before the first step of `run!`, shrink a too-large `dt` from the worst-case CFL number
(advection at `v_cut`), since a layer at rest underestimates the flow to come. A no-op under
[`FixedTimeStepper`](@ref).
"""
function init_adaptive_dt! end

init_adaptive_dt!(sim, ::FixedTimeStepper) = nothing
function init_adaptive_dt!(sim, ts::AdaptiveTimeStepper)
    _apply_dt!(sim, ts, _cfl_worstcase(sim); allow_grow = false)
    return
end

# ============================================================================
# CFL diagnostics — select how the in-loop CFL number is computed.
# Held by the Simulation as its `cfl` field; dispatches `_cfl_number`.
# ============================================================================
"""
Abstract supertype for how the in-loop CFL number is computed (for the progress
display and for [`AdaptiveTimeStepper`](@ref)).  Pass a concrete instance as
`Simulation(model; cfl = ...)`: [`ExactCFL`](@ref) (the default) or
[`ConservativeCFL`](@ref).
"""
abstract type AbstractCFL end

"""
$(TYPEDEF)

Conservative CFL estimate: takes `max|U|`, `max|V|`, and `max c` independently
over the active domain, then adds them.  This overestimates the true CFL (the
worst velocity and worst wave speed rarely occur at the same cell) but avoids a
per-cell kernel pass.
"""
struct ConservativeCFL <: AbstractCFL end

"""
$(TYPEDEF)

Per-cell CFL: interpolates U and V to T-points, computes the CFL at every
active cell, and returns the maximum (the default).  Tighter than
[`ConservativeCFL`](@ref) at the cost of one kernel pass.
"""
struct ExactCFL <: AbstractCFL end

# ============================================================================
# CFL number
# ============================================================================

# The CFL number of a cell per unit time step: advection plus the internal
# gravity-wave speed c = √(g·max(0, δρ·H)) in both directions.  Launched by
# `_launch_tpoint_diag!`, with the argument list of `_speed_norm_kernel!`.
@kernel function _cfl_rate_kernel!(
    out,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(H),
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
        c = _safe_sqrt(g * max(zero(FT), drho[i, j] * H[i, j]))
        out[i, j] = (abs(u) / dx + abs(v) / dy + c / dx + c / dy) * tmask[i, j]
    end
end

# The CFL number of the current state, for the dt controller.  `Simulation.cfl`
# selects ConservativeCFL or ExactCFL.  The device reductions and the host formula
# that combines them are separate functions, so that the Reactant extension can
# compile the reductions and still combine them the same way.
_cfl_number(sim, reductions = cfl_reductions(sim.model, sim.cfl)) =
    _float64(sim.clock.dt) * cfl_rate(sim.model, sim.cfl, reductions)

# Largest δρ·H over the active domain, for the gravity-wave speed.
function _max_Hdrho(m)
    @. m.diag = m.drho * m.H.present * m.tmask
    return maximum(m.diag)
end

"""
    cfl_reductions(m, cfl) -> Tuple

The device reductions (a tuple of scalars) that [`cfl_rate`](@ref) needs under the
[`AbstractCFL`](@ref) `cfl`, taken at a sync point of `run!`.
"""
function cfl_reductions end

# Conservative: global max of |U|, |V|, and c taken independently, then combined.
# Overestimates the true CFL but cheap (three scalar reductions).
function cfl_reductions(m, ::ConservativeCFL)
    d = m.diag
    @. d = abs(m.U.present) * m.umask
    umax = maximum(d)
    @. d = abs(m.V.present) * m.vmask
    vmax = maximum(d)
    c = sqrt(m.g * max(zero(m.FT), _max_Hdrho(m)))
    return (umax, vmax, c)
end

"""
    cfl_rate(m, cfl, reductions)

The CFL rate (s⁻¹; the CFL number is `rate · dt`) under the [`AbstractCFL`](@ref) `cfl`,
from the `reductions` of [`cfl_reductions`](@ref).
"""
function cfl_rate end

cfl_rate(m, ::ConservativeCFL, (umax, vmax, c)) =
    (_float64(umax) + _float64(c)) / _float64(m.dx) +
    (_float64(vmax) + _float64(c)) / _float64(m.dy)

# Exact: per-cell CFL using T-point-interpolated velocities; maximum over active
# cells.  Tighter than ConservativeCFL at the cost of one kernel pass.
function cfl_reductions(m, ::ExactCFL)
    _launch_tpoint_diag!(_cfl_rate_kernel!, m)
    return (maximum(m.diag),)
end
cfl_rate(m, ::ExactCFL, (rate,)) = _float64(rate)

# Worst-case CFL: the advective term uses the velocity cap `v_cut` instead of the
# actual speed.  At startup the flow is ~stationary, so the actual CFL is tiny
# and useless for sizing dt0; this bounds the advective CFL the developing flow
# can ever reach (it cannot exceed v_cut), while keeping the real gravity-wave
# term.  Used by the preemptive startup rescue, not the in-loop controller.
function _cfl_worstcase(sim)
    m = sim.model
    FT = m.FT
    v_cut = _float64(m.v_cut)
    c = _float64(sqrt(m.g * max(zero(FT), _sync_diagnostics(sim.exec, sim).Hdrho)))
    return _float64(sim.clock.dt) *
           ((v_cut + c) / _float64(m.dx) + (v_cut + c) / _float64(m.dy))
end
