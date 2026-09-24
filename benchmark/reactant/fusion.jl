# Per-step GPU time of the Reactant extension's compiled step batch, for each fusion
# strategy, against the KernelAbstractions CUDA path.  One grid per process:
#
#   NX=1280 NY=640 julia --project=benchmark benchmark/reactant/fusion.jl
#
# FT=Float64 switches precision (default Float32); FUSIONS=xla,kernel picks the
# strategies (default all).  Prints one CSV row per variant:
#   variant,FT,nx,ny,ms_per_step,compile_s
using Laddie, Reactant, CUDA, KernelAbstractions, Statistics, Printf
Reactant.set_default_backend("gpu")
const Ext = Base.get_extension(Laddie, :LaddieReactantExt)

const NX = parse(Int, get(ENV, "NX", "640"))
const NY = parse(Int, get(ENV, "NY", "320"))
const FT = get(ENV, "FT", "Float32") == "Float64" ? Float64 : Float32
const FUSIONS = Symbol.(split(get(ENV, "FUSIONS", join(Ext.FUSION_STRATEGIES, ",")), ","))
# Extra `Reactant.compile` options as a Julia expression, e.g.
#   XLA_OPTS='(; xla_debug_options = (; xla_gpu_enable_fast_min_max = true))'
const XLA_OPTS = get(ENV, "XLA_OPTS", "")
isempty(XLA_OPTS) || (Ext.EXTRA_COMPILE_OPTIONS[] = eval(Meta.parse(XLA_OPTS)))
# NATIVE_UNROLL=k takes k steps per loop iteration with native kernels (default 4).
haskey(ENV, "NATIVE_UNROLL") && (Ext.NATIVE_UNROLL[] = parse(Int, ENV["NATIVE_UNROLL"]))
const NSTEPS = 100
const NSAMPLES = 5

initial_sim() = build_isomip(CPU(); nx = NX, ny = NY, FT, isomipcond = :warm)

function ka_cuda()
    sim = to_backend(initial_sim(), CUDABackend())
    run!() = (for _ = 1:NSTEPS; time_step!(sim); end; KernelAbstractions.synchronize(CUDABackend()))
    run!()
    return median([@elapsed(run!()) for _ = 1:NSAMPLES]) / NSTEPS, 0.0
end

function reactant(fusion)
    sim = to_backend(initial_sim(), ReactantBackend(; fusion))
    sync() = Reactant.synchronize(sim.model.D.present)
    tc = @elapsed (Laddie._advance_batch!(sim.exec, sim, 1, false); sync())
    run!() = (Laddie._advance_batch!(sim.exec, sim, NSTEPS, false); sync())
    run!()
    return median([@elapsed(run!()) for _ = 1:NSAMPLES]) / NSTEPS, tc
end

println("variant,FT,nx,ny,ms_per_step,compile_s")
if get(ENV, "KA", "1") == "1"
    t, c = ka_cuda()
    @printf("ka-cuda,%s,%d,%d,%.3f,%.1f\n", FT, NX, NY, 1e3t, c)
end
for f in FUSIONS
    t, c = reactant(f)
    @printf("reactant-%s,%s,%d,%d,%.3f,%.1f\n", f, FT, NX, NY, 1e3t, c)
    GC.gc()
end
