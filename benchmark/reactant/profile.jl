# Which GPU kernels does one leapfrog step run, and where does the time go?
# Reactant's step is profiled with its built-in XLA profiler (xprof over CUPTI,
# no Nsight install needed), the KA CUDA step with CUDA.@profile.
#
#   julia --project=benchmark benchmark/reactant/profile.jl
#
# FT=Float64 switches precision (default Float32), NX/NY the grid (640×320).
# PART=r or PART=ka profiles one side only (CUDA.@profile cannot run in a process
# that has already started Reactant's profiler, so run them separately).
# Writes the XLA-optimised module to step_xla.txt in the working directory.
include("common.jl")
using Printf
const RP = Reactant.Profiler

Reactant.set_default_backend("gpu")
const FT = get(ENV, "FT", "Float32") == "Float64" ? Float64 : Float32
const NX, NY = parse(Int, get(ENV, "NX", "640")), parse(Int, get(ENV, "NY", "320"))
const NSTEPS, NREPEAT = 20, 5

const PART = get(ENV, "PART", "r")
sim = initial_sim(NX, NY; FT)

# --- Reactant --------------------------------------------------------------
if PART == "r"
rs = to_reactant(sim)
n = ConcreteRNumber(NSTEPS)
fN = @compile raise = true sync = true rstep_for!(rs, n)
(; xplane_file) = RP.profile_and_get_xplane_file(fN, rs, n; nrepeat = NREPEAT, warmup = 2)

reports = RP.get_kernel_stats(xplane_file).reports
nsteps = NSTEPS * NREPEAT
per_step(x) = x / nsteps
tot = sum(r -> Int(r.total_duration_ns), reports)
launches = sum(r -> Int(r.occurrences), reports)
println("\n=== Reactant GPU $FT $(NX)×$(NY) ===")
@printf("kernel launches / step: %.1f   GPU kernel time / step: %.3f ms   distinct kernels: %d\n",
    per_step(launches), per_step(tot) / 1e6, length(reports))

# Kernels grouped by the kind XLA names them after (loop_fusion, input_fusion,
# dynamic_update_slice, copy, …): the name up to its numeric suffix.
kind(name) = replace(name, r"[_.]?\d+$" => "", r"^(wrapped_)" => "")
groups = Dict{String,Tuple{Int,Int,Int}}()
for r in reports
    k = kind(r.name)
    c, o, t = get(groups, k, (0, 0, 0))
    groups[k] = (c + 1, o + Int(r.occurrences), t + Int(r.total_duration_ns))
end
println("\nby kind:  kernels  launches/step  ms/step  share")
for (k, (c, o, t)) in sort(collect(groups); by = x -> -x[2][3])
    @printf("  %-34s %4d  %8.1f  %8.4f  %5.1f%%\n", k, c, per_step(o), per_step(t) / 1e6, 100t / tot)
end
println("\ntop 25 kernels:  launches/step  µs/step  µs/launch  grid×block  regs  occ%")
for r in first(sort(reports; by = r -> -Int(r.total_duration_ns)), 25)
    @printf("  %-40s %6.1f %8.1f %8.2f  %s×%s  %3d  %4.0f\n", RP._clip_str(r.name, 40),
        per_step(Int(r.occurrences)), per_step(Int(r.total_duration_ns)) / 1e3,
        Int(r.total_duration_ns) / Int(r.occurrences) / 1e3,
        join(Int.(r.grid_dim), ","), join(Int.(r.block_dim), ","), Int(r.registers_per_thread), r.occupancy_pct)
end
println("\nstep time from the trace (host view, 20 steps): ",
    round(RP.extract_mean_step_time(xplane_file, NREPEAT) / NSTEPS / 1e6; digits = 3), " ms/step")

try
    write("step_xla.txt", string(@code_xla raise = true rstep!(rs)))
    println("XLA-optimised module written to step_xla.txt")
catch e
    println("@code_xla failed: ", sprint(showerror, e))
end

end

# --- KA CUDA ----------------------------------------------------------------
if PART == "ka"
cs = raw_sim(to_backend(sim.model, CUDABackend()), sim)
for _ = 1:5
    rstep!(cs)
end
CUDA.synchronize()
println("\n=== KA CUDA $FT $(NX)×$(NY): $NSTEPS steps ===")
show(stdout, CUDA.@profile trace = false begin
    for _ = 1:NSTEPS
        rstep!(cs)
    end
    CUDA.synchronize()
end)
println()
end
