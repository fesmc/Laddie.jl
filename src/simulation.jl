# ============================================================================
# Simulation — the model plus everything that advances it in time: the clock,
# the time stepper, the CFL diagnostic, the Robert–Asselin filter, the stopping
# criterion, output, and restart.  `Model` holds geometry, physics and state
# only; nothing in it knows about `dt`.
# ============================================================================

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
    "current time step (s); varies under [`AdaptiveDt`](@ref)"
    dt::FT
end

# I/O state on the model's backend, with a zeroed accumulator for each time-averaged
# output field when file I/O is on.  Full grid size (including the border ring), so
# `_accum!` can do bare `.+=`; the ring is masked out when writing to NetCDF.
function IOState(m::Model, output::OutputConfig)
    names = output.saveday > 0 ? filter(in(keys(_OUTPUT_FIELDS)), output.fields) : ()
    IOState(
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
    "how `dt` evolves: [`FixedDt`](@ref) or [`AdaptiveDt`](@ref)"
    tstep::TS
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
- `dt`: initial time step in seconds (default 210).  Under [`FixedDt`](@ref) it
  stays constant; under [`AdaptiveDt`](@ref) it is the starting point.
- `tstep`: [`FixedDt`](@ref) (default) or [`AdaptiveDt`](@ref).
- `cfl`: [`ExactCFL`](@ref) (default) or [`ConservativeCFL`](@ref).
- `nu`: Robert–Asselin filter coefficient (default 0.8).
- `stop`: default stopping criterion of [`run!`](@ref), a
  [`FixedSimulationEnd`](@ref) (default, 30 days) or [`SteadyStateEnd`](@ref).
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
sim = Simulation(model; dt = 120.0, tstep = AdaptiveDt(), stop = FixedSimulationEnd(t_end = 90.0))
run!(sim)
```
"""
function Simulation(
    model::Model;
    dt = 210.0,
    tstep = FixedDt(),
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
        _promote_param(tstep, FT),
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
        _bootstrap_leapfrog!(sim)
    else
        init_from_restart!(sim, restart)
    end
    output.saveday > 0 && prepare_output!(sim)
    return sim
end


# Absolute simulation time in days, including the restart offset, so output and
# restart files of a continuation run never collide with those it restarted from.
_t_days(sim::Simulation) = sim.clock.time / _primal(sim.model.seconds_per_day)

"""
$(TYPEDSIGNATURES)

Return a new simulation whose model and I/O accumulators live on `backend`.
The clock, time stepper and output configuration are carried over; the original
simulation is not modified.

With a [`ReactantBackend`](@ref), `run!` then advances the simulation in compiled
batches of steps (requires `using Reactant, CUDA`).
"""
function to_backend(sim::Simulation, backend)
    c = sim.clock
    Simulation(
        to_backend(sim.model, backend),
        Clock(c.time, c.iteration, c.dt),
        sim.tstep,
        sim.cfl,
        sim.nu,
        sim.stop,
        sim.output,
        _iostate_to_backend(sim.io, backend),
        sim.debug,
        _execution(backend),
    )
end
