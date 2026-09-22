# Per-step cost of the KernelAbstractions path vs a Reactant-compiled step, on
# ISOMIP warm grids.  One target per process (run_sweep.sh runs them all):
#
#   TARGET=ka-cpu  julia -t 8 --project=benchmark benchmark/reactant/bench.jl
#   TARGET=r-cpu   taskset -c 0-7 julia --project=benchmark benchmark/reactant/bench.jl
#   TARGET=ka-cuda julia --project=benchmark benchmark/reactant/bench.jl
#   TARGET=r-gpu   julia --project=benchmark benchmark/reactant/bench.jl
#
# FT=Float32 switches precision (default Float64).  Prints one CSV row per grid:
#   ms_per_step           median over NSAMPLES batches of NSTEPS steps
#                         (KA: a host loop of steps; Reactant: one compiled
#                         `@trace` loop of NSTEPS steps)
#   ms_per_step_hostloop  Reactant only: one compiled call per step
#   compile_s             Reactant only: compile time of both programs
#   maxrelerr             max relative difference from KA-CPU after NCHECK steps
#
# ka-cpu pins its threads to cores with ThreadPinning; give r-cpu the same cores
# with taskset (XLA:CPU manages its own threads).
include("common.jl")
using Statistics, Printf

const TARGET = ENV["TARGET"]
const FT = get(ENV, "FT", "Float64") == "Float32" ? Float32 : Float64
const GRIDS = [(80, 40), (320, 160), (640, 320), (1280, 640)]
const NSTEPS = 100
const NSAMPLES = 5
const NCHECK = 50

if TARGET == "ka-cpu"
    using ThreadPinning
    pinthreads(:cores)
elseif TARGET == "r-cpu"
    Reactant.set_default_backend("cpu")
elseif TARGET == "r-gpu"
    Reactant.set_default_backend("gpu")
end

median_per_step(f) = (f(); median([(@elapsed f()) for _ = 1:NSAMPLES]) / NSTEPS)

function run_ka(sim, backend)
    s = backend isa CPU ? sim : raw_sim(to_backend(sim.model, backend), sim)
    step! = backend isa CPU ? time_step! : rstep!
    sync = backend isa CPU ? () -> nothing : () -> KA.synchronize(backend)
    t = median_per_step() do
        for _ = 1:NSTEPS
            step!(s)
        end
        sync()
    end
    return (; t, compile = 0.0, t_call = NaN)
end

function run_reactant(sim)
    rs = to_reactant(sim)
    n = ConcreteRNumber(NSTEPS)
    c1 = @elapsed (f1 = @compile raise = true rstep!(rs))
    cN = @elapsed (fN = @compile raise = true rstep_for!(rs, n))
    sync() = Reactant.synchronize(rs.model.D.present)
    t = median_per_step() do
        fN(rs, n)
        sync()
    end
    t_call = median_per_step() do
        for _ = 1:NSTEPS
            f1(rs)
        end
        sync()
    end
    return (; t, compile = c1 + cN, t_call)
end

# Advance a fresh copy NCHECK steps on this target and on KA-CPU; compare.
function check_error(nx, ny)
    TARGET == "ka-cpu" && return 0.0
    ref = initial_sim(nx, ny; FT)
    chk = initial_sim(nx, ny; FT)
    if TARGET == "ka-cuda"
        cs = raw_sim(to_backend(chk.model, CUDABackend()), chk)
        for _ = 1:NCHECK
            rstep!(cs)
        end
    else
        cs = to_reactant(chk)
        n = ConcreteRNumber(NCHECK)
        @compile(raise = true, rstep_for!(cs, n))(cs, n)
    end
    for _ = 1:NCHECK
        time_step!(ref)
    end
    return maxrelerr(prognostics(cs), prognostics(ref))
end

label = TARGET * (get(ENV, "WG", "0") == "1" ? "-wg" : "")
println("target,FT,nx,ny,ms_per_step,ms_per_step_hostloop,compile_s,maxrelerr")
for (nx, ny) in GRIDS
    sim = initial_sim(nx, ny; FT)
    r = TARGET == "ka-cpu" ? run_ka(sim, CPU()) :
        TARGET == "ka-cuda" ? run_ka(sim, CUDABackend()) : run_reactant(sim)
    err = check_error(nx, ny)
    @printf("%s,%s,%d,%d,%.4f,%.4f,%.1f,%.2e\n", label, FT, nx, ny, 1e3 * r.t,
        1e3 * r.t_call, r.compile, err)
    flush(stdout)
end
