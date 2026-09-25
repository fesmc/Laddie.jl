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
Laddie.integrate!(model, dt, n::Reactant.TracedRNumber; nu = 0.8,
                  checkpoints = Laddie.DEFAULT_CHECKPOINTS) =
    (_steps!(model, (), dt, n, model.FT(nu), ();
             checkpointing = Reactant.Binomial(checkpoints)); model)

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
