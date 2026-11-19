# ============================================================================
# Differentiation: the model as a function of its inputs
# ============================================================================

"""
$(TYPEDSIGNATURES)

Advance `model` by `n` leapfrog steps of `dt` seconds (Robert–Asselin coefficient
`nu`), without a [`Simulation`](@ref): no clock, output, dt control or blow-up check.
The model must already be bootstrapped, e.g. taken from a simulation after
construction (`sim.model`).  Returns `model`.

The model as a plain function of its inputs, for automatic differentiation: with ForwardDiff
through a model built at `FT = Dual`, and with Enzyme inside [`reactant_compile`](@ref), where
`n` may be traced. Under Reactant it also takes `checkpoints` (the reverse-mode memory budget),
and `integrate!(model, sched)` replays an [`adaptive_schedule`](@ref).
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

Kernels are raised to XLA before differentiation, so `f` may call `Enzyme.autodiff`; `fusion` is
a strategy of `ReactantBackend` (only `:xla` differentiates), and other keywords go to
`Reactant.compile`. A model's scalar parameters are compiled in as constants unless the model
comes from [`trace_parameters`](@ref). Examples on the [Reactant backend](@ref) page.
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
    adaptive_schedule(model, dt; days, stepper = AdaptiveTimeStepper(), cfl = ExactCFL(), nu = 0.8,
                      maxsegments = 10_000) -> DtSchedule

Run `model` (on a [`ReactantBackend`](@ref); requires `using Reactant, CUDA`) for `days` from
the step `dt` (s), adapting dt with `stepper` as `run!` does, and return the dt schedule it
took. Advances `model`. The first of two passes that differentiate an adaptive run: the dt
controller cannot be differentiated, so `integrate!(model, sched; means = (), nu = 0.8,
checkpoints = 20)` inside [`reactant_compile`](@ref) replays the schedule with dt as data,
returning the time-weighted means of the fields in `means`. See [Reactant backend](@ref).
"""
function adaptive_schedule end

integrate!(model, sched::DtSchedule; kwargs...) = throw(ArgumentError(
    "integrate!(model, ::DtSchedule) replays a schedule inside a program compiled by " *
    "`reactant_compile` (requires `using Reactant, CUDA`)"))

"""
    trace_parameters(model; overrides...)

A model whose scalar parameters ([`Params`](@ref), including the floats of its schemes) are
Reactant numbers, so a program compiled by [`reactant_compile`](@ref) takes them as inputs and
can be differentiated with respect to them (requires `using Reactant, CUDA`). Shares its
arrays with `model`; keywords replace fields of `Params`. Parameters that act only when the
model is built (`coriolis`, `H_init`, `dT_init`, `dS_init`) have no effect. Do not `run!` it.
"""
function trace_parameters end
