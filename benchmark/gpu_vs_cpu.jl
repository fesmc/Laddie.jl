using Laddie
using Chairmarks
using KernelAbstractions
using Printf
using Statistics

# ============================================================================
# GPU vs CPU performance comparison
#
# Usage:
#   julia --project=benchmark benchmark/gpu_vs_cpu.jl
#
# With CUDA available, also reports GPU timings and speedup ratios.
# ============================================================================

function timestep!(m)
    Laddie.advance_leapfrog!(m)
    Laddie.leapfrog_step!(m, 2)
    Laddie.clamp_velocities!(m)
    Laddie.apply_robert_asselin_filter!(m)
end

# Try to load a GPU backend
gpu_backend = nothing
try
    using CUDA
    if CUDA.functional()
        gpu_backend = CUDABackend()
        @info "CUDA backend detected"
    end
catch
end
if gpu_backend === nothing
    try
        using Metal
        if Metal.functional()
            gpu_backend = MetalBackend()
            @info "Metal backend detected"
        end
    catch
    end
end
if gpu_backend === nothing
    @info "No GPU backend found — only CPU benchmarks will run"
end

# Grid sizes to sweep: (nx, ny, label)
const GRIDS = [
    # (20,  10,  "very small  (20×10)"),
    (80,  40,  "small (80×40)"),
    (320, 160, "medium (320×160)"),
    (640, 320, "large (640×320)"),
]

# ============================================================================
# Benchmark helper
# ============================================================================

function bench_backend(backend, nx, ny; fused=true)
    m = build_isomip(backend; nx=nx, ny=ny, isomipcond=:warm)
    m.fused = fused
    if backend isa CPU
        b = @be (deepcopy(m)) timestep!(_) evals=1 samples=20
    else
        b = @be (deepcopy(m)) begin timestep!(_); CUDA.synchronize() end evals=1 samples=20
    end
    return b
end

# ============================================================================
# Run and report
# ============================================================================

header = @sprintf("%-20s  %12s  %12s  %8s",
    "Grid", "CPU median", "GPU median", "Speedup")
println("\n", header)
println("-" ^ length(header))

for (nx, ny, label) in GRIDS
    b_cpu = bench_backend(CPU(), nx, ny)
    t_cpu_ms = median(b_cpu).time * 1e3   # seconds → ms

    if gpu_backend !== nothing
        b_gpu = bench_backend(gpu_backend, nx, ny)
        t_gpu_ms = median(b_gpu).time * 1e3
        speedup  = t_cpu_ms / t_gpu_ms
        @printf("%-20s  %9.3f ms  %9.3f ms  %6.1f×\n",
            label, t_cpu_ms, t_gpu_ms, speedup)
    else
        @printf("%-20s  %9.3f ms  %12s  %8s\n",
            label, t_cpu_ms, "N/A", "N/A")
    end
end