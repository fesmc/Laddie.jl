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
#   :stencil  a barrier on the arrays of each stencil kernel before it runs, so the
#             fields it reads at neighbour offsets are in memory
#   :xla      no barriers: XLA's heuristics decide.  The only strategy Enzyme can
#             differentiate (barriers have no derivative rule).
const FUSION_STRATEGIES = (:native, :kernel, :stencil, :xla)
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
function _launch_traced!(kernel!, A, ndrange, stencil, args)
    f = TRACING_FUSION[]
    f === :stencil && stencil && _barrier!(args...)
    backend = KA.get_backend(A)
    k = f === :native ? kernel!(backend, Laddie._workgroup(backend)) : kernel!(backend)
    k(args...; ndrange)
    f === :kernel && _barrier!(args...)
    return nothing
end
Laddie.launch!(kernel!, A::Reactant.AnyTracedRArray, args...) =
    _launch_traced!(kernel!, A, size(A), false, args)
Laddie.launch_interior!(kernel!, A::Reactant.AnyTracedRArray, args...) =
    _launch_traced!(kernel!, A, size(A) .- 2, true, args)

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

# What the step functions read from a Simulation (they take `sim` untyped).  The
# clock carries only `dt`, traced so that an adaptive dt change does not recompile;
# the clock time stays on the host.  `check_nans` is a data-dependent error and
# cannot be traced.
struct TracedSim{M,C,N,D}
    model::M
    clock::C
    nu::N
    debug::D
end
_traced_sim(model, dt, nu) = TracedSim(model, (; dt), nu, (; check_nans = false))

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

# `integrate!` inside a trace: the steps become a traced loop.  (Written out rather
# than through a closure: `@trace` finds the loop state in the body's variables.)
function Laddie.integrate!(model, dt, n::Reactant.TracedRNumber; nu = 0.8)
    s = Laddie._stepping_view(model, dt, nu)
    @trace track_numbers = false for _ = 1:n
        Laddie.advance_leapfrog!(s)
        Laddie.leapfrog_step!(s, 2)
        Laddie.apply_robert_asselin_filter!(s)
    end
    return model
end

function _program(build, exec, name)
    get!(exec.programs, name) do
        build()
    end
end

_dt(sim) = ConcreteRNumber(sim.model.FT(Laddie._primal(sim.clock.dt)))

# The output accumulators that are switched on (the others are 0×0).
_accumulators(sim) = Tuple(
    getfield(sim.io, f.acc) for f in Laddie._OUTPUT_FIELDS if getfield(sim.output, f.flag)
)
_accumulated_fields(sim) =
    Tuple(f for f in Laddie._OUTPUT_FIELDS if getfield(sim.output, f.flag))

# n leapfrog steps, with the output accumulation after each when `fields` is not
# empty: `time_step!` without the clock, then `_accum!` without its host counters.
function _step!(rs, accs, dt, fields)
    Laddie.advance_leapfrog!(rs)
    Laddie.leapfrog_step!(rs, 2)
    Laddie.apply_robert_asselin_filter!(rs)
    for (acc, f) in zip(accs, fields)
        Laddie._accum_field!(acc, f.src, rs.model, dt)
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

function _steps!(model, accs, dt, n, nu, fields, unroll = 1)
    rs = _traced_sim(model, dt, nu)
    if unroll > 1
        @trace track_numbers = false for _ = 1:(n ÷ unroll)
            for _ = 1:unroll
                _step!(rs, accs, dt, fields)
            end
        end
        @trace track_numbers = false for _ = 1:(n % unroll)
            _step!(rs, accs, dt, fields)
        end
    else
        @trace track_numbers = false for _ = 1:n
            _step!(rs, accs, dt, fields)
        end
    end
    return nothing
end

function Laddie._batch_length(::ReactantExecution, sim, args...)
    return Laddie._steps_to_next_event(sim, args...)
end

function Laddie._advance_batch!(exec::ReactantExecution, sim, n, io_on)
    fields = io_on ? _accumulated_fields(sim) : ()
    accs = io_on ? _accumulators(sim) : ()
    dt = _dt(sim)
    nn = ConcreteRNumber(n)
    nu = sim.nu   # captured by value: Reactant traces a closure's captured variables
    unroll = _unroll(exec.fusion)
    prog = _program(exec, io_on ? :steps_io : :steps) do
        _compile(
            (model, accs, dt, n) -> _steps!(model, accs, dt, n, nu, fields, unroll),
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
function _rebootstrap!(model, dt, nu)
    for var in (model.D, model.U, model.V, model.T, model.S)
        var.past .= var.present
    end
    rs = _traced_sim(model, dt, nu)
    Laddie.update_secondary_fields!(model, dt)
    Laddie.leapfrog_step!(rs, 1)
    return nothing
end

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

_cfl_rates(m, ::Laddie.ExactCFL) =
    (Laddie._launch_tpoint_diag!(Laddie._cfl_rate_kernel!, m); maximum(m.diag))
function _cfl_rates(m, ::Laddie.ConservativeCFL)
    FT = m.FT
    d = m.diag
    @. d = abs(m.U.present) * m.umask
    umax = maximum(d)
    @. d = abs(m.V.present) * m.vmask
    vmax = maximum(d)
    c = sqrt(m.g * max(zero(FT), Laddie._max_Ddrho(m)))
    return (umax + c) / m.dx + (vmax + c) / m.dy
end

function _diagnostics(m, cfl)
    finite = all(isfinite, m.D.present) & all(isfinite, m.U.present) &
             all(isfinite, m.V.present)
    rate = _cfl_rates(m, cfl)
    Ddrho = Laddie._max_Ddrho(m)
    stats = Laddie.meltstats(m)
    @. m.diag = ifelse(m.tmask > 0, m.D.present, m.FT(-Inf))
    Dmax = maximum(m.diag)
    return (finite, rate, Ddrho, stats.max_meltrate, stats.mean_meltrate,
            stats.max_speed, stats.total_melt, Dmax)
end

function _diag(exec::ReactantExecution, sim)
    exec.diag === nothing || return exec.diag
    cfl = sim.cfl
    prog = _program(exec, :diagnostics) do
        _compile(m -> _diagnostics(m, cfl), exec, sim.model)
    end
    r = map(x -> x isa Number && !(x isa Reactant.RNumber) ? x : Reactant.to_number(x),
            prog(sim.model))
    exec.diag = (finite = Bool(r[1]), rate = Float64(r[2]), Ddrho = r[3],
                 stats = (max_meltrate = r[4], mean_meltrate = r[5], max_speed = r[6],
                          total_melt = r[7]), Dmax = r[8])
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
