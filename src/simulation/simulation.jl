# ============================================================================
# Simulation — the model plus everything that advances it in time: the clock,
# the time stepper, the CFL diagnostic, the Robert–Asselin filter, the stopping
# criterion, output, and restart.  `Model` holds geometry, physics and state
# only; nothing in it knows about `dt`.
# ============================================================================

# ============================================================================
# Simulation-end criteria — decide when `run!` stops.  Set as the `stop` keyword
# of `Simulation`, or per call as the `until` keyword of `run!`, and dispatched in
# the time loop.  The `days` keyword is the shorthand for
# `until = FixedSimulationEnd(t_end = days)`.
# ============================================================================

"""
Abstract supertype for the stopping criterion of `run!`.  Pass a concrete
instance as `Simulation(model; stop = ...)` or `run!(sim; until = ...)`:
[`FixedSimulationEnd`](@ref) (run a set
duration) or [`SteadyStateSimulationEnd`](@ref) (stop once the solution settles).
"""
abstract type AbstractSimulationEnd end

"""
$(TYPEDEF)

Stop after a fixed simulated duration `t_end` (days, the duration of this `run!`
call).  This is the default and exactly reproduces `run!(sim; days = t_end)`.

Select via `Simulation(model; stop = FixedSimulationEnd(t_end = 30.0))` or
`run!(sim; until = FixedSimulationEnd(t_end = 30.0))`.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct FixedSimulationEnd{T} <: AbstractSimulationEnd
    "duration of the `run!` call (days, default `30`)"
    t_end::T = 30.0
    FixedSimulationEnd{T}(t_end) where {T} = new{T}(t_end)
end

# Positional (and keyword) construction converts to floating point, as the
# former `Float64` field did: `FixedSimulationEnd(t_end = 30)` holds 30.0.
FixedSimulationEnd(t_end) = (t = float(t_end); FixedSimulationEnd{typeof(t)}(t))

"""
$(TYPEDEF)

Stop early once the cavity reaches a quasi-steady state, or after `t_end` days,
whichever comes first.  Steadiness is detected when the **relative** change in
the domain-mean basal melt rate (over floating ice) between successive
diagnostic samples falls below `tol`:

```
|m̄_(k+1) − m̄_k| / |m̄_k| < tol
```

`tol` is dimensionless, so it means the same thing in a warm cavity (melt of
order 100 m yr⁻¹) and a cold one (order 1 m yr⁻¹).  Samples are taken once per
simulated day (not every step), which is cheap, robust to the leapfrog
computational mode, and independent of the run length.  `t_end` is a safety cap on
the duration of the `run!` call.

Select via `run!(sim; until = SteadyStateSimulationEnd(tol = 1e-3, t_end = 365.0))`.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct SteadyStateSimulationEnd{T} <: AbstractSimulationEnd
    "relative change in the mean melt rate between daily samples below which the run stops (default `1e-3`)"
    tol::T = 1e-3
    "cap on the duration of the `run!` call (days, default `365`)"
    t_end::T = 365.0
    SteadyStateSimulationEnd{T}(tol, t_end) where {T} = new{T}(tol, t_end)
end

SteadyStateSimulationEnd(tol, t_end) =
    (p = promote(float(tol), float(t_end)); SteadyStateSimulationEnd{eltype(p)}(p...))

# Hard time cap in seconds (both criteria carry one).
end_seconds(e::AbstractSimulationEnd, spd) = e.t_end * spd

# Whether `run!` must sample the mean melt rate at each diagnostic check.
needs_melt_sample(::AbstractSimulationEnd) = false
needs_melt_sample(::SteadyStateSimulationEnd) = true

# Early-stop test from successive domain-mean melt samples (m yr⁻¹).
# FixedSimulationEnd never stops early; SteadyStateSimulationEnd stops once the relative
# change drops below tol (the first sample, with no predecessor, never stops).
steady_reached(::AbstractSimulationEnd, mean_new, mean_prev) = false
function steady_reached(e::SteadyStateSimulationEnd, mean_new, mean_prev)
    (isfinite(mean_prev) && mean_prev != 0) || return false
    return abs(mean_new - mean_prev) / abs(mean_prev) < e.tol
end

"""
$(TYPEDEF)

Simulated time.  Owned by a [`Simulation`](@ref) and never reset by `run!`, so
successive `run!` calls continue from where the previous one stopped.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Clock{FT,T,I}
    "simulated time since the origin (s); a restarted simulation starts at the restart time"
    time::T
    "time steps taken by this simulation (not restored from a restart)"
    iteration::I
    "current time step (s); varies under [`AdaptiveTimeStepper`](@ref)"
    dt::FT
end

# I/O state on the model's backend, with a zeroed accumulator for each time-averaged
# output field when file I/O is on, and for each field behind the time series when
# these are on too.  Full grid size (including the border ring), so `_accum!` can do
# bare `.+=`; the ring is masked out when writing to NetCDF.
function IOState(m::Model, output::OutputConfig)
    names = output.saveday > 0 ? filter(in(keys(_OUTPUT_FIELDS)), output.fields) : ()
    series = output.saveday > 0 && output.tsday !== nothing ? _SERIES_FIELDS : ()
    return IOState(
        0,       # count
        0.0,     # t_accum
        0,       # time_index
        0.0,     # nextsave
        0.0,     # nextdiag
        0.0,     # nextrest
        "",      # rundir
        "",      # logfile
        0.0,     # walltime_start
        "",      # restartfile
        NamedTuple{names}(map(_ -> zero(m.tmask), names)),
        Inf,     # nextseries
        0.0,     # ts_t_accum
        NamedTuple{series}(map(_ -> zero(m.tmask), series)),
        _SeriesRow[],   # ts_rows
        0,       # ts_index
    )
end

# How `run!` advances a simulation: step by step with the KernelAbstractions
# kernels (`NativeExecution`), or in compiled batches of steps (the Reactant
# extension's execution, selected by `to_backend(sim, ReactantBackend())`).
abstract type AbstractExecution end
struct NativeExecution <: AbstractExecution end

# The execution a simulation moved to `backend` runs with.
_execution(backend) = NativeExecution()

"""
$(TYPEDEF)

A [`Model`](@ref) together with its time integration and I/O.  Construct with
[`Simulation(model; ...)`](@ref Simulation(::Model)) and advance with [`run!`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct Simulation{
    M<:Model,
    CL<:Clock,
    TS<:AbstractTimeStepper,
    C<:AbstractCFL,
    N,
    E<:AbstractSimulationEnd,
    O<:OutputConfig,
    IOS<:IOState,
    D<:DebugConfig,
    X<:AbstractExecution,
}
    "the model being integrated"
    model::M
    "simulated time, iteration count and current `dt`"
    clock::CL
    "how `dt` evolves: [`FixedTimeStepper`](@ref) or [`AdaptiveTimeStepper`](@ref)"
    time_stepper::TS
    "how the in-loop CFL number is computed: [`ExactCFL`](@ref) or [`ConservativeCFL`](@ref)"
    cfl::C
    "Robert–Asselin filter coefficient"
    nu::N
    "default stopping criterion of `run!`"
    stop::E
    "output cadence, field selection and run directory"
    output::O
    "runtime I/O state: accumulators, next-event times, run directory, log"
    io::IOS
    "debug options"
    debug::D
    "how `run!` advances the model: step by step (KernelAbstractions) or in compiled batches (Reactant)"
    exec::X
end

"""
$(TYPEDSIGNATURES)

Wrap `model` in a simulation, initialise its leapfrog time levels, and (when
`output.saveday > 0`) create the run directory and write the initial state.

# Keywords
- `dt`: initial time step in seconds (default 210).  Under [`FixedTimeStepper`](@ref) it
  stays constant; under [`AdaptiveTimeStepper`](@ref) it is the starting point.
- `time_stepper`: [`FixedTimeStepper`](@ref) (default) or [`AdaptiveTimeStepper`](@ref).
- `cfl`: [`ExactCFL`](@ref) (default) or [`ConservativeCFL`](@ref).
- `nu`: Robert–Asselin filter coefficient (default 0.8).
- `stop`: default stopping criterion of [`run!`](@ref), a
  [`FixedSimulationEnd`](@ref) (default, 30 days) or [`SteadyStateSimulationEnd`](@ref).
- `output`: an [`OutputConfig`](@ref); the default disables file I/O.
- `restart`: path to a JLD2 restart file, or `nothing` (default) to start from
  the model's initial state.  Restores the prognostic fields, the clock time
  and the time step.
- `debug`: a [`DebugConfig`](@ref).

Constructing a simulation takes the first (bootstrap) step of the leapfrog
scheme on `model`, so build one simulation per model.

```julia
grid = Grid(mask, z_draft, dx, dy)
model = Model(grid; forcing, params = Params())
sim = Simulation(model; dt = 120.0, time_stepper = AdaptiveTimeStepper(), stop = FixedSimulationEnd(t_end = 90.0))
run!(sim)
```
"""
function Simulation(
    model::Model;
    dt = 210.0,
    time_stepper = FixedTimeStepper(),
    cfl = ExactCFL(),
    nu = 0.8,
    stop = FixedSimulationEnd(),
    output = OutputConfig(),
    restart = nothing,
    debug = DebugConfig(),
)
    FT = model.FT
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    io = IOState(model, output)
    sim = Simulation(
        model,
        Clock(0.0, 0, FT(dt)),
        _promote_param(time_stepper, FT),
        cfl,
        FT(nu),
        stop,
        output,
        io,
        debug,
        NativeExecution(),
    )
    output.saveday > 0 && create_rundir!(sim)
    if restart === nothing
        _bootstrap_leapfrog!(model, sim.clock.dt; _nan_check(sim)...)
    else
        init_from_restart!(sim, restart)
    end
    output.saveday > 0 && prepare_output!(sim)
    return sim
end


# Absolute simulation time in days, including the restart offset, so output and
# restart files of a continuation run never collide with those it restarted from.
_t_days(sim::Simulation) = _t_days(sim.clock, sim.model)
_t_days(clock::Clock, m) = clock.time / _primal(m.seconds_per_day)

# The NaN-check keywords of the step functions (`leapfrog_step!`) for `sim`.
_nan_check(sim::Simulation) = (; check_nans = sim.debug.check_nans, sim.clock)

"""
$(TYPEDSIGNATURES)

Return a new simulation whose model and I/O accumulators live on `backend`.
The clock, time stepper and output configuration are carried over; the original
simulation is not modified.

With a [`ReactantBackend`](@ref), `run!` then advances the simulation in compiled
batches of steps (requires `using Reactant, CUDA`); the time series of
[`OutputConfig`](@ref) (`tsday`) are not available there.
"""
function to_backend(sim::Simulation, backend)
    _check_series_backend(sim.output, backend)
    c = sim.clock
    return Simulation(
        to_backend(sim.model, backend),
        Clock(c.time, c.iteration, c.dt),
        sim.time_stepper,
        sim.cfl,
        sim.nu,
        sim.stop,
        sim.output,
        _iostate_to_backend(sim.io, backend),
        sim.debug,
        _execution(backend),
    )
end
