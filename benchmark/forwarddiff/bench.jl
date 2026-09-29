# Per-step cost of a ForwardDiff dual pass against the Float64 primal, on ISOMIP warm
# grids, CPU with pinned threads:
#
#   julia -t 8 --project=benchmark benchmark/forwarddiff/bench.jl
#
# The model is built at `FT = Dual{Tag,Float64,N}`, as a gradient over N parameters
# (one chunk) would build it: grid, geometry, state, caches and parameters are all
# Dual, so every array is N + 1 times wider.  The partials are zero here, which costs
# the same arithmetic as seeded ones.
#
# Prints one CSV row per (grid, N):
#   ms_per_step     median over NSAMPLES batches of NSTEPS steps
#   ratio           ms_per_step / primal ms_per_step
#   per_param       ratio / N: the dual cost of one derivative, in primal runs
#                   (central differences cost 2, one-sided 1, per parameter)
#   model_GiB       Base.summarysize of the model
#
# Results of 2026-09-23 (Xeon W-2245, 8 pinned cores) in results-2026-09-23.csv.
using Laddie, ForwardDiff, KernelAbstractions
using ThreadPinning, Statistics, Printf

pinthreads(:cores)

const GRIDS = [(320, 160), (640, 320), (1280, 640)]
const CHUNKS = [1, 4, 10]
const NSTEPS = 50
const NSAMPLES = 5

struct BenchTag end
dualtype(N) = ForwardDiff.Dual{ForwardDiff.Tag{BenchTag,Float64},Float64,N}

function bench(nx, ny, FT)
    sim = build_isomip(CPU(); nx, ny, FT, isomipcond = :warm)
    for _ = 1:20   # past the startup transient
        time_step!(sim)
    end
    ts = map(1:NSAMPLES) do _
        @elapsed for _ = 1:NSTEPS
            time_step!(sim)
        end
    end
    return (; t = median(ts) / NSTEPS, gib = Base.summarysize(sim.model) / 2^30)
end

println("nx,ny,N,ms_per_step,ratio,per_param,model_GiB")
for (nx, ny) in GRIDS
    p = bench(nx, ny, Float64)
    @printf("%d,%d,0,%.2f,1.00,,%.2f\n", nx, ny, 1e3p.t, p.gib)
    for N in CHUNKS
        d = bench(nx, ny, dualtype(N))
        r = d.t / p.t
        @printf("%d,%d,%d,%.2f,%.2f,%.2f,%.2f\n", nx, ny, N, 1e3d.t, r, r / N, d.gib)
        GC.gc()
    end
end
