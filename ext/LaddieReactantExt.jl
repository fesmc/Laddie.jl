module LaddieReactantExt

# Runs a Laddie simulation through Reactant.jl: `to_backend(sim, ReactantBackend())`
# moves the arrays to Reactant, and `run!` then advances the simulation in compiled
# batches of steps (a traced `@trace for` loop), with the dt control, blow-up check
# and diagnostics compiled as small programs of their own.  See
# `laddie-roadmap/reactant.md` for the background and the measurements.

using Laddie
using Laddie: ReactantBackend, Model, Simulation
using Reactant
using Reactant: @compile, @trace, ConcreteRNumber
using CUDA
using KernelAbstractions
const KA = KernelAbstractions
const Adapt = KA.Adapt

# ============================================================================
# Kernel launches while tracing
# ============================================================================

# How the kernels of a step become GPU code (measurements: docs/src/reactant.md).
#   :native   the kernels stay CUDA code, compiled by CUDA.jl and called from the XLA
#             program (not raised).  As fast as the KernelAbstractions kernels, with
#             the step loop inside one compiled program.  GPU only; Enzyme cannot
#             differentiate it.
# The other strategies raise the kernels to XLA operations, which XLA then fuses.
# Without guidance it fuses a whole step into a few giant fusions that recompute
# intermediate fields per output cell; an `optimization_barrier` on a kernel's arrays
# forces them to memory and splits the fusions there.
#   :kernel   a barrier on every kernel's arrays after it runs (≈ one fusion per kernel)
#   :xla      no barriers: XLA's heuristics decide.  The only strategy Enzyme can
#             differentiate (barriers have no derivative rule).
# A barrier before each stencil kernel instead (`:stencil`, removed 2026-09-24) was
# slower than `:kernel` on every grid: 2.2× at 2000×2000.
const FUSION_STRATEGIES = (:native, :kernel, :xla)
# The strategy of the program being traced; set by `_compile` around each trace.
const TRACING_FUSION = Ref(:kernel)

# `:auto`: native kernels on the GPU; on the CPU, where XLA must run raised code,
# a barrier per kernel.
function _fusion(f::Symbol)
    f === :auto || f in FUSION_STRATEGIES || throw(ArgumentError(
        "unknown fusion strategy :$f; use :auto or one of $FUSION_STRATEGIES"))
    f === :auto || return f
    platform = Reactant.XLA.platform_name(Reactant.XLA.default_backend())
    return platform == "cuda" ? :native : :kernel
end

function _barrier!(args...)
    arrs = [a for a in args if a isa Reactant.TracedRArray]
    isempty(arrs) && return nothing
    for (a, b) in zip(arrs, Reactant.Ops.optimization_barrier(arrs...))
        a.mlir_data = b.mlir_data
    end
    return nothing
end

# Raised kernels are launched with no workgroup, so Reactant takes the whole ndrange
# as one block: with Laddie's (32, 8) GPU workgroup the tiling leaks into the raised
# program as gathers and transposes, and the step runs about 2× slower.  Native
# kernels keep that workgroup, as on the CUDA backend.
function _launch_traced!(kernel!, ndrange, args)
    f = TRACING_FUSION[]
    backend = KA.get_backend(first(args))
    k = f === :native ? kernel!(backend, Laddie._workgroup(backend)) : kernel!(backend)
    k(args...; ndrange)
    f === :kernel && _barrier!(args...)
    return nothing
end
Laddie.launch!(kernel!, out::Reactant.AnyTracedRArray, args...) =
    _launch_traced!(kernel!, size(out), (out, args...))
Laddie.launch_interior!(kernel!, out::Reactant.AnyTracedRArray, args...) =
    _launch_traced!(kernel!, size(out) .- 2, (out, args...))

# Reactant 0.2.286: `ReactantCUDAExt.Const{T,N,AS}` stores a `CuTracedArray`
# without its `Size` parameter, so the field is abstract and every `@Const` read
# becomes a dynamic call that fails to compile.  `@Const` is only a read-only hint:
# pass the array through unwrapped.  Defined before the first compile rather than in
# `__init__`, since Reactant's CUDA extension may load after this one.
const CONST_WORKAROUND = Ref(false)
function _const_workaround!()
    CONST_WORKAROUND[] && return
    cuext = Base.get_extension(Reactant, :ReactantCUDAExt)
    cuext === nothing && error("Reactant's CUDA extension is not loaded; run `using CUDA`")
    @eval Adapt.adapt_storage(::KA.ConstAdaptor, a::$(cuext.CuTracedArray)) = a
    CONST_WORKAROUND[] = true
    return
end

# ============================================================================
# Backend and execution
# ============================================================================

Laddie._reactant_ka_backend(::ReactantBackend) =
    Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend()

"""
Batched execution through Reactant: the compiled programs of one simulation, and a
CPU mirror of its model for the log diagnostics.
"""
mutable struct ReactantExecution <: Laddie.AbstractExecution
    "fusion strategy of the compiled programs (one of `FUSION_STRATEGIES`)"
    fusion::Symbol
    "compiled programs, by name"
    programs::Dict{Symbol,Any}
    "sync-point diagnostics of the current state, or `nothing` when stale"
    diag::Any
    "CPU copy of the model for `printdiags`, created on first use"
    mirror::Any
end

Laddie._reactant_execution(b::ReactantBackend) =
    ReactantExecution(_fusion(b.fusion), Dict{Symbol,Any}(), nothing, nothing)

# Extra keyword arguments for `Reactant.compile` (e.g. `xla_debug_options`), for
# experiments with XLA's code generation; see `benchmark/reactant/fusion.jl`.
const EXTRA_COMPILE_OPTIONS = Ref{Any}((;))

_compile(f, exec, args...) = _compile_with(f, exec.fusion, args, EXTRA_COMPILE_OPTIONS[])

function _compile_with(f, fusion, args, kwargs)
    _const_workaround!()
    TRACING_FUSION[] = fusion
    try
        # `invokelatest`: the `@Const` workaround may have just been `@eval`ed.
        kw = merge((; raise = fusion !== :native), kwargs)
        return Base.invokelatest(Reactant.compile, f, args; kw...)
    finally
        TRACING_FUSION[] = :kernel
    end
end

# Raised before differentiation (`raise_first`), so Enzyme sees XLA operations, not
# opaque kernel calls.
function Laddie.reactant_compile(f, args...; fusion = :xla, kwargs...)
    fusion = _fusion(fusion)
    fusion === :native || (kwargs = merge((; raise_first = true), kwargs))
    return _compile_with(f, fusion, args, kwargs)
end

# ============================================================================
# Traced parameters
# ============================================================================

# The float type a traced scalar wraps: `m.FT` of a model from `trace_parameters`.
Laddie._value_type(::Type{T}) where {T<:Reactant.RNumber} = Reactant.unwrapped_eltype(T)

# Every float in `x` (a parameter or parameterisation object) as a Reactant number
# of precision FT.  Objects are rebuilt with their float type parameters traced;
# arrays, integers and field-less singletons stay.
_traced(x::Reactant.RNumber, FT) = x
_traced(x::AbstractFloat, FT) = ConcreteRNumber(FT(x))
_traced(x::Number, FT) = x
function _traced(x, FT)
    (x isa AbstractArray || fieldcount(typeof(x)) == 0) && return x
    fields = map(fn -> _traced(getfield(x, fn), FT), fieldnames(typeof(x)))
    T = _traced_type(typeof(x), FT)
    # Built without the constructor, as `Enzyme.make_zero` builds a tangent: a
    # validating constructor (`TurbulentGamTMelting`) rejects the zero tangent.
    fieldtypes(T) == map(typeof, fields) ||
        return Base.typename(typeof(x)).wrapper(fields...)
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, Any[fields...], length(fields))::T
end
# Every field type is a type parameter, so tracing the floats means tracing the
# float parameters.
_traced_type(T::DataType, FT) =
    Base.typename(T).wrapper{map(p -> p isa Type && p <: AbstractFloat ?
                                      typeof(ConcreteRNumber(FT(0))) : p, T.parameters)...}

_override(v::Number, FT) = FT(v)
_override(v, FT) = Laddie._promote_param(v, FT)

function Laddie.trace_parameters(model::Model; overrides...)
    p = model.params
    FT = model.FT
    names = fieldnames(typeof(p))
    for k in keys(overrides)
        k in names || throw(ArgumentError("Params has no field `$k`"))
    end
    fields = map(names) do fn
        v = haskey(overrides, fn) ? _override(overrides[fn], FT) : getfield(p, fn)
        return _traced(v, FT)
    end
    return Model(model.grid, model.geometry, model.state, model.cache,
                 Laddie.Params(fields...), model.boundary, model.forcing)
end

# `integrate!` inside a trace: the steps become a traced loop.  Reverse mode through a
# loop whose trip count is only known at run time needs checkpointing (revolve):
# without it Enzyme stores every step in a buffer of dynamic size, which XLA cannot
# compile.  The checkpoints cost nothing in the primal and in forward mode.
const DEFAULT_CHECKPOINTS = 20
Laddie.integrate!(model, dt, n::Reactant.TracedRNumber; nu = 0.8,
                  checkpoints = DEFAULT_CHECKPOINTS) =
    (_steps!(model, (), dt, n, model.FT(nu), ();
             checkpointing = Reactant.Binomial(checkpoints)); model)

# ============================================================================
# Adaptive dt, differentiable: record the schedule, then replay it
# ============================================================================

# AdaptiveDt's rule (`Laddie._controller_dt` with `allow_grow = true`), branch-free on
# traced numbers.
function _controller_dt(ts::Laddie.AdaptiveDt, dt, cfl)
    FT = typeof(dt)
    target, q = FT(ts.cfl_target), FT(ts.q)
    ok = (cfl > zero(cfl)) & isfinite(cfl)
    r = (target / ifelse(ok, cfl, one(cfl)))^q
    dtn = ifelse(cfl > target, dt * r,
                 ifelse(cfl < FT(ts.grow_hyst) * target, dt * min(r, FT(ts.max_growth)), dt))
    return clamp(ifelse(ok, dtn, dt), FT(ts.dtmin), FT(ts.dtmax))
end

# CFL rate (s⁻¹) on traced fields: `Laddie._cfl_rate` without the host conversions.
function _cfl_rate(m, ::Laddie.ExactCFL)
    Laddie._launch_tpoint_diag!(Laddie._cfl_rate_kernel!, m)
    return maximum(m.diag)
end
function _cfl_rate(m, cfl::Laddie.ConservativeCFL)
    umax, vmax, c = Laddie._cfl_reductions(m, cfl)
    return (umax + c) / m.dx + (vmax + c) / m.dy
end

# Re-bootstrap where dt changed.  A traced `if` is fine here (no differentiation).
function _rebootstrap_if!(model, changed, dt, nu)
    @trace track_numbers = false if changed
        Laddie._collapse_and_bootstrap!(Laddie._stepping_view(model, dt, nu))
    end
    return nothing
end

# Pass 1: the adaptive run.  Loop state: time, dt, step count and segment index as
# floats (exact for integers), and the segments as fixed-capacity arrays.
function _schedule!(model, dt0, tend, ts, cfl, nu, cap)
    FT = model.FT
    idx = FT.(Reactant.Ops.iota(Int, [cap]; iota_dimension = 1) .+ 1)
    seg_dt = ifelse.(idx .== 1, dt0, zero(dt0))
    seg_n = zero(seg_dt)
    t, dt, k, j = zero(dt0), dt0 * one(dt0), zero(dt0), one(dt0)
    @trace track_numbers = false while t < tend
        Laddie._step_model!(Laddie._stepping_view(model, dt, nu))
        t = t + dt
        k = k + 1
        seg_n = seg_n .+ ifelse.(idx .== j, one(FT), zero(FT))
        dtn = _controller_dt(ts, dt, dt * _cfl_rate(model, cfl))
        dtn = ifelse(rem(k, FT(ts.ncheck)) == 0, dtn, dt)
        changed = dtn != dt
        _rebootstrap_if!(model, changed, dtn, nu)
        j = j + ifelse(changed, one(FT), zero(FT))
        seg_dt = ifelse.(idx .== j, dtn, seg_dt)
        dt = dtn
    end
    return seg_dt, seg_n, j
end

# One compiled pass-1 program per model type, size and controller settings.
const SCHEDULE_PROGRAMS = Dict{Any,Any}()

function Laddie.adaptive_schedule(model::Model, dt; days, stepper = Laddie.AdaptiveDt(),
                                  cfl = Laddie.ExactCFL(), nu = 0.8, maxsegments = 10_000)
    FT = model.FT
    ts = Laddie._promote_param(stepper, FT)
    nuv = FT(nu)
    key = (typeof(model), size(model.melt), ts, cfl, nuv, maxsegments)
    dt0, tend = ConcreteRNumber(FT(dt)), ConcreteRNumber(FT(days * 86400))
    prog = get!(SCHEDULE_PROGRAMS, key) do
        _compile_with((m, dt0, tend) -> _schedule!(m, dt0, tend, ts, cfl, nuv, maxsegments),
                      _fusion(:auto), (model, dt0, tend), (;))
    end
    seg_dt, seg_n, j = prog(model, dt0, tend)
    nseg = round(Int, Reactant.to_number(j))
    nseg <= maxsegments || throw(ArgumentError(
        "the run changed dt $(nseg - 1) times, more than `maxsegments` = $maxsegments allows"))
    steps = round.(Int, Array(seg_n))
    return Laddie.DtSchedule(seg_dt, Reactant.to_rarray(steps), ConcreteRNumber(nseg))
end

# Pass 2: replay, differentiable.  The re-bootstrap at each segment start is computed
# unconditionally and kept with `ifelse` except for the first segment: Enzyme cannot
# reverse a traced `if` that updates arrays in place.  It writes `past` (collapsed on
# `present`) and `future` (the Euler step); the next step recomputes the cache.
function _rebootstrap_blend!(model, keep, dt, nu)
    vars = (model.D, model.U, model.V, model.T, model.S)
    old = map(v -> (copy(v.past), copy(v.future)), vars)
    Laddie._collapse_and_bootstrap!(Laddie._stepping_view(model, dt, nu))
    for (v, (p, f)) in zip(vars, old)
        v.past .= ifelse.(keep, v.past, p)
        v.future .= ifelse.(keep, v.future, f)
    end
    return nothing
end

_field(m, name) = (x = getproperty(m, name); x isa Laddie.Var ? x.present : x)

function _segment!(model, accs, dt, n, nu, names, checkpoints)
    @trace track_numbers = false checkpointing = Reactant.Binomial(checkpoints) for _ = 1:n
        Laddie._step_model!(Laddie._stepping_view(model, dt, nu))
        for (acc, name) in zip(accs, names)
            acc .+= _field(model, name) .* dt
        end
    end
    return nothing
end

function Laddie.integrate!(model, sched::Laddie.DtSchedule{<:Reactant.AnyTracedRArray};
                           means = (), nu = 0.8, checkpoints = DEFAULT_CHECKPOINTS)
    FT = model.FT
    nuv = FT(nu)
    names = Tuple(means)
    accs = map(name -> zero(_field(model, name)), names)
    cap = length(sched.dt)
    idx = Reactant.Ops.iota(Int, [cap]; iota_dimension = 1) .+ 1
    # Few segments: a small checkpoint budget for the outer loop.
    @trace track_numbers = false checkpointing = Reactant.Binomial(checkpoints) for j = 1:sched.nseg
        dt = sum(ifelse.(idx .== j, sched.dt, zero(FT)))
        n = sum(ifelse.(idx .== j, sched.steps, 0))
        _rebootstrap_blend!(model, j > 1, dt, nuv)
        _segment!(model, accs, dt, n, nuv, names, checkpoints)
    end
    isempty(names) && return model
    total = sum(sched.dt .* sched.steps)
    return NamedTuple{names}(map(acc -> acc ./ total, accs))
end

function _program(build, exec, name)
    get!(exec.programs, name) do
        build()
    end
end

_dt(sim) = ConcreteRNumber(sim.model.FT(Laddie._primal(sim.clock.dt)))

# The output accumulators, and where each reads its field (`_accum_field!`).
_accumulators(sim) = values(sim.io.acc)
_sources(sim) = map(name -> Laddie._OUTPUT_FIELDS[name].src, keys(sim.io.acc))

# One leapfrog step, with the output accumulation when `accs` is not empty:
# `time_step!` without the clock, then `_accum!` without its host counters.  The
# clock of the step view carries only `dt`, traced so that an adaptive dt change does
# not recompile; the clock time stays on the host.
function _step!(rs, accs, dt, srcs)
    Laddie._step_model!(rs)
    for (acc, src) in zip(accs, srcs)
        Laddie._accum_field!(acc, src, rs.model, dt)
    end
    return nothing
end

# Steps per iteration of the traced loop.  On the GPU, XLA evaluates a while loop's
# condition on the device and reads it back each iteration, which leaves the GPU idle
# between steps; unrolling amortises that over several steps (4: 10 % faster than 1
# for native kernels, and 10 is no better).  Raised kernels are not unrolled, since
# every copy of the step is raised and optimised again (compile time).
const NATIVE_UNROLL = Ref(4)
_unroll(fusion) = fusion === :native ? NATIVE_UNROLL[] : 1

# `checkpointing` only shapes the reverse pass of Enzyme; see `integrate!`.
function _steps!(model, accs, dt, n, nu, srcs, unroll = 1; checkpointing = false)
    rs = Laddie._stepping_view(model, dt, nu)
    if unroll > 1
        @trace track_numbers = false for _ = 1:(n ÷ unroll)
            for _ = 1:unroll
                _step!(rs, accs, dt, srcs)
            end
        end
        @trace track_numbers = false for _ = 1:(n % unroll)
            _step!(rs, accs, dt, srcs)
        end
    else
        @trace track_numbers = false checkpointing = checkpointing for _ = 1:n
            _step!(rs, accs, dt, srcs)
        end
    end
    return nothing
end

function Laddie._batch_length(::ReactantExecution, sim, args...)
    return Laddie._steps_to_next_event(sim, args...)
end

function Laddie._advance_batch!(exec::ReactantExecution, sim, n, io_on)
    srcs = io_on ? _sources(sim) : ()
    accs = io_on ? _accumulators(sim) : ()
    dt = _dt(sim)
    nn = ConcreteRNumber(n)
    nu = sim.nu   # captured by value: Reactant traces a closure's captured variables
    unroll = _unroll(exec.fusion)
    prog = _program(exec, io_on ? :steps_io : :steps) do
        _compile(
            (model, accs, dt, n) -> _steps!(model, accs, dt, n, nu, srcs, unroll),
            exec,
            sim.model,
            accs,
            dt,
            nn,
        )
    end
    prog(sim.model, accs, dt, nn)
    exec.diag = nothing
    # The host side of `time_step!` and `_accum!`, step by step as they would run.
    c = sim.clock
    for _ = 1:n
        c.time += Laddie._primal(c.dt)
        c.iteration += 1
        if io_on
            sim.io.count += 1
            sim.io.t_accum += Laddie._primal(c.dt)
        end
    end
    return
end

# Re-bootstrap after a dt change (`_rebootstrap_leapfrog!` on the native path).
_rebootstrap!(model, dt, nu) =
    Laddie._collapse_and_bootstrap!(Laddie._stepping_view(model, dt, nu))

function Laddie._rebootstrap_leapfrog!(exec::ReactantExecution, sim)
    dt = _dt(sim)
    nu = sim.nu
    prog = _program(exec, :rebootstrap) do
        _compile((model, dt) -> _rebootstrap!(model, dt, nu), exec, sim.model, dt)
    end
    prog(sim.model, dt)
    exec.diag = nothing
    return
end

# ============================================================================
# Sync-point diagnostics, compiled into one program
# ============================================================================

# The sync-point diagnostics as one flat tuple of numbers: finiteness, the largest
# δρ·D, the four melt statistics, the largest active D, then the CFL reductions,
# which the host combines as the native path does (`_cfl_rate`).
function _diagnostics(m, cfl)
    stats = Laddie.meltstats(m)
    return (Laddie._prognostics_finite(m), Laddie._max_Ddrho(m), stats...,
            Laddie._max_active_D(m), Laddie._cfl_reductions(m, cfl)...)
end

_host_number(x::Reactant.RNumber) = Reactant.to_number(x)
_host_number(x) = x

function _diag(exec::ReactantExecution, sim)
    exec.diag === nothing || return exec.diag
    cfl = sim.cfl
    prog = _program(exec, :diagnostics) do
        _compile(m -> _diagnostics(m, cfl), exec, sim.model)
    end
    r = map(_host_number, prog(sim.model))
    stats = NamedTuple{(:max_meltrate, :mean_meltrate, :max_speed, :total_melt)}(r[3:6])
    exec.diag = (finite = Bool(r[1]), Ddrho = r[2], stats, Dmax = r[7],
                 rate = Laddie._cfl_rate(sim.model, cfl, r[8:end]))
    return exec.diag
end

Laddie._prognostics_finite(exec::ReactantExecution, sim) = _diag(exec, sim).finite
Laddie._sync_cfl_number(exec::ReactantExecution, sim) =
    Laddie._float64(sim.clock.dt) * _diag(exec, sim).rate
Laddie._meltstats(exec::ReactantExecution, sim) = _diag(exec, sim).stats
Laddie._max_Ddrho(exec::ReactantExecution, sim) = _diag(exec, sim).Ddrho
Laddie._max_active_D(exec::ReactantExecution, sim) = _diag(exec, sim).Dmax

# ============================================================================
# Log diagnostics on a CPU mirror
# ============================================================================

# `printdiags` makes about 20 reductions; eager Reactant operations compile on each
# call, so they run on a CPU copy of the model instead, refreshed from the device.
function Laddie._diag_model(exec::ReactantExecution, sim)
    if exec.mirror === nothing
        exec.mirror = to_backend(sim.model, CPU())
    else
        _copy_arrays!(exec.mirror.state, sim.model.state)
        _copy_arrays!(exec.mirror.cache, sim.model.cache)
    end
    return exec.mirror
end

function _copy_arrays!(dst, src)
    for f in fieldnames(typeof(src))
        d, s = getfield(dst, f), getfield(src, f)
        if d isa AbstractArray
            copyto!(d, Array(s))
        elseif fieldcount(typeof(d)) > 0 && !(d isa Number)
            _copy_arrays!(d, s)
        end
    end
    return
end

end
