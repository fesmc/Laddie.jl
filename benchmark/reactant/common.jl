# Shared setup for the Reactant scripts: a Reactant-compiled leapfrog step for a
# Laddie model, and helpers to move a CPU simulation onto Reactant and compare
# its fields against the KernelAbstractions path.
#
# Tested with Reactant 0.2.286, CUDA.jl 6.2.2, KernelAbstractions 0.9.42 on
# Julia 1.12.6 (see laddie-roadmap/reactant.md).  CUDA.jl must be loaded even for
# the CPU target: Reactant compiles KA kernels through CUDA and raises them to
# StableHLO.

using Laddie
using Reactant, CUDA
using KernelAbstractions
import Adapt

const KA = KernelAbstractions
const RB = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend
const RCUDA = Base.get_extension(Reactant, :ReactantCUDAExt)

# Workaround (Reactant 0.2.286): `ReactantCUDAExt.Const{T,N,AS}` stores a
# `CuTracedArray{T,N,AS}` without the 4th (Size) parameter, so the field is
# abstract and every @Const read becomes a dynamic call that fails to compile.
# @Const is only a read-only hint: pass the array through unwrapped.
Adapt.adapt_storage(::KA.ConstAdaptor, a::RCUDA.CuTracedArray) = a

# Launch traced kernels with no workgroup, so Reactant takes the whole ndrange as
# one block.  With Laddie's (32, 8) GPU workgroup the tiling leaks into the
# raised program (index space padded to whole tiles → gathers + transposes), and
# the step runs ~2× slower.  WG=1 keeps Laddie's workgroup, to reproduce that.
#
# BARRIER=1 puts an optimization_barrier on every traced array argument after each
# launch; BARRIER=interior only after the stencil (`launch_interior!`) kernels;
# BARRIER=pre on the arguments of each stencil kernel *before* it runs, so the
# arrays it reads at neighbour offsets are in memory.  BARRIER_AFTER is a
# comma-separated list of kernel names to put a barrier after, on top of those.
# Without it XLA fuses the whole step into a handful of kernels that recompute
# intermediate fields per output cell (compute-bound, register spills); the
# barrier makes it materialise each kernel's outputs, as KA does.
const BARRIER = get(ENV, "BARRIER", "0") == "1"
const BARRIER_INTERIOR = BARRIER || get(ENV, "BARRIER", "0") == "interior"
const BARRIER_PRE = get(ENV, "BARRIER", "0") == "pre"
const BARRIER_AFTER = Set(Symbol.(filter(!isempty, split(get(ENV, "BARRIER_AFTER", ""), ","))))
barrier_after(kernel!) = !isempty(BARRIER_AFTER) && nameof(kernel!) in BARRIER_AFTER

function barrier!(args...)
    arrs = [a for a in args if a isa Reactant.TracedRArray]
    isempty(arrs) && return nothing
    for (a, b) in zip(arrs, Reactant.Ops.optimization_barrier(arrs...))
        a.mlir_data = b.mlir_data
    end
    return nothing
end

# Since 2026-09-24 `launch!(kernel!, out, args...)` passes `out` to the kernel too;
# before, the caller repeated it (`launch!(kernel!, out, out, args...)`).
_kernel_args(A, args) = isdefined(Laddie, :_launch!) ? (A, args...) : args

if get(ENV, "WG", "0") != "1"
    function Laddie.launch!(kernel!, A::Reactant.AnyTracedRArray, args...)
        args = _kernel_args(A, args)
        kernel!(RB())(args...; ndrange = size(A))
        (BARRIER || barrier_after(kernel!)) && barrier!(args...)
        return nothing
    end
    # (Laddie versions before the interior launch have no `launch_interior!`.)
    if isdefined(Laddie, :launch_interior!)
        @eval function Laddie.launch_interior!(kernel!, A::Reactant.AnyTracedRArray, args...)
            args = _kernel_args(A, args)
            BARRIER_PRE && barrier!(args...)
            kernel!(RB())(args...; ndrange = size(A) .- 2)
            (BARRIER_INTERIOR || barrier_after(kernel!)) && barrier!(args...)
            return nothing
        end
    end
end

# `track_numbers` of the traced loop.  `false` bakes every scalar (Params, dt, …)
# in as a constant; `true` (Reactant's default) traces them.
const TRACK_NUMBERS = get(ENV, "TRACK_NUMBERS", "0") == "1"

# Physics of one leapfrog step: `time_step!` without the clock update (the
# Clock's fields are plain numbers, so a traced step would bake them in).
function rstep!(sim)
    Laddie.advance_leapfrog!(sim)
    Laddie.leapfrog_step!(sim, 2)
    Laddie.apply_robert_asselin_filter!(sim)
    return nothing
end

# n steps in one compiled program.
function rstep_for!(sim, n)
    if TRACK_NUMBERS
        @trace for _ = 1:n
            rstep!(sim)
        end
    else
        @trace track_numbers = false for _ = 1:n
            rstep!(sim)
        end
    end
    return nothing
end

# What the step functions read from a Simulation (they take `sim` untyped).  The
# full Simulation also carries IOState, whose 0×0 accumulators Reactant cannot
# export from a traced loop (`tensor.empty`), and host-only I/O config.
struct RSim{M,C,FT,D}
    model::M
    clock::C
    nu::FT
    debug::D
end

raw_sim(m, like::Simulation) = RSim(m, deepcopy(like.clock), like.nu, like.debug)

to_reactant(sim) = raw_sim(to_backend(sim.model, RB()), sim)

prognostics(sim) = (m = sim.model;
[Array(getfield(v, l)) for v in (m.D, m.U, m.V, m.T, m.S) for l in (:past, :present, :future)])

# Largest error over the prognostic fields, each relative to the field's max |value|.
function maxrelerr(a, b)
    maximum(zip(a, b)) do (x, y)
        s = max(maximum(abs, y), eps())
        maximum(abs, x .- y) / s
    end
end

# ISOMIP warm, bootstrapped on the CPU and 20 steps in, so the fields are not trivial.
function initial_sim(nx, ny; FT = Float64)
    sim = build_isomip(CPU(); nx, ny, FT, isomipcond = :warm)
    for _ = 1:20
        time_step!(sim)
    end
    return sim
end
