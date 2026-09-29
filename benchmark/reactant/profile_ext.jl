# GPU kernel breakdown of the Reactant extension's compiled step batch (xprof over
# CUPTI), per fusion strategy:
#
#   FUSION=kernel NX=640 NY=320 julia --project=benchmark benchmark/reactant/profile_ext.jl
#
# FT=Float64 switches precision (default Float32).  Compare with the KA CUDA kernels
# from `PART=ka julia --project=benchmark benchmark/reactant/profile.jl`.
using Laddie, Reactant, CUDA, KernelAbstractions, Printf
const RP = Reactant.Profiler
Reactant.set_default_backend("gpu")

const FT = get(ENV, "FT", "Float32") == "Float64" ? Float64 : Float32
const NX, NY = parse(Int, get(ENV, "NX", "640")), parse(Int, get(ENV, "NY", "320"))
const FUSION = Symbol(get(ENV, "FUSION", "kernel"))
const NSTEPS, NREPEAT = 20, 5

sim = to_backend(build_isomip(CPU(); nx = NX, ny = NY, FT, isomipcond = :warm),
                 ReactantBackend(; fusion = FUSION))
const Ext = Base.get_extension(Laddie, :LaddieReactantExt)
dt = ConcreteRNumber(FT(sim.clock.dt))
n = ConcreteRNumber(NSTEPS)
# The extension's step batch, compiled with `sync = true` as the profiler requires.
Ext._const_workaround!()
Ext.TRACING_FUSION[] = FUSION
nu = sim.nu
prog = Base.invokelatest(Reactant.compile,
    (model, accs, dt, n) -> Ext._steps!(model, accs, dt, n, nu, ()),
    (sim.model, (), dt, n); raise = get(ENV, "RAISE", "1") == "1", sync = true)
(; xplane_file) = RP.profile_and_get_xplane_file(prog, sim.model, (), dt, n;
                                                  nrepeat = NREPEAT, warmup = 2)
reports = RP.get_kernel_stats(xplane_file).reports
nsteps = NSTEPS * NREPEAT
per_step(x) = x / nsteps
tot = sum(r -> Int(r.total_duration_ns), reports)
@printf("\n=== Reactant :%s %s %d×%d ===\nlaunches/step %.1f   GPU time/step %.3f ms   distinct kernels %d\n",
        FUSION, FT, NX, NY, per_step(sum(r -> Int(r.occurrences), reports)), per_step(tot) / 1e6,
        length(reports))
println("\ntop kernels:  launches/step  µs/step  µs/launch  regs  occ%")
for r in first(sort(reports; by = r -> -Int(r.total_duration_ns)), 30)
    @printf("  %-40s %6.1f %8.1f %8.2f  %3d  %4.0f\n", RP._clip_str(r.name, 40),
            per_step(Int(r.occurrences)), per_step(Int(r.total_duration_ns)) / 1e3,
            Int(r.total_duration_ns) / Int(r.occurrences) / 1e3,
            Int(r.registers_per_thread), r.occupancy_pct)
end
