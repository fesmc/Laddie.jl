# ============================================================================
# Differentiation: the model as a function of its inputs
# ============================================================================

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
    nu = model.FT(nu)
    for _ = 1:n
        _step_model!(model, dt, nu)
    end
    return model
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
time-weighted means over the run of the fields named in `means` (e.g. `:melt`, `:H`),
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
(`coriolis`, `H_init`, `dT_init`, `dS_init`) have no effect on the program.

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
