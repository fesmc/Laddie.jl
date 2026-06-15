using Laddie
using KernelAbstractions
using Printf

# ============================================================================
# Adaptive vs fixed time stepping — wall-time to simulate a fixed duration.
#
# In a slow (cold) cavity the CFL-limited controller grows dt, so the same
# physical time is reached in fewer steps.  This reports the step count and
# wall time for FixedDt vs AdaptiveDt on GPU/Float32 (CPU fallback).
#
# Usage:
#   julia --project=benchmark benchmark/adaptive_dt.jl
# ============================================================================

global backend = CPU()
global label = "CPU"
try
    using CUDA
    if CUDA.functional()
        global backend = CUDABackend()
        global label = "CUDA"
    end
catch
end
gpu = !(backend isa CPU)
sync() = gpu ? CUDA.synchronize() : nothing

const FT = Float32
const NX, NY = 320, 160
const COND = :cold        # slow flow → controller grows dt → biggest win
const DAYS = 2.0

function timed_run(tstep)
    m = build_isomip(backend; FT, nx = NX, ny = NY, isomipcond = COND,
                     params = Params(; FT, tstep))
    t = @elapsed begin
        run!(m; days = DAYS, verbose = false)
        sync()
    end
    return (; wall = t, steps = m.t, dt = Float64(m.dt), dx = Float64(m.dx))
end

# Warm up (compile all kernels, both stepper paths) on a tiny grid.
for ts in (FixedDt(), AdaptiveDt())
    mw = build_isomip(backend; FT, nx = 20, ny = 10, isomipcond = COND,
                      params = Params(; FT, tstep = ts))
    run!(mw; days = 0.05, verbose = false); sync()
end

f = timed_run(FixedDt())
a = timed_run(AdaptiveDt())

@printf("\nAdaptive dt benchmark — %s, Float32, %s cavity, %d×%d (dx = %.0f m), %.1f days\n",
        label, COND, NX, NY, f.dx, DAYS)
@printf("%-12s %10s %12s %14s\n", "stepper", "steps", "final dt", "wall time")
@printf("%-12s %10d %11.1fs %13.3fs\n", "FixedDt",    f.steps, f.dt, f.wall)
@printf("%-12s %10d %11.1fs %13.3fs\n", "AdaptiveDt", a.steps, a.dt, a.wall)
@printf("→ %.2f× fewer steps, %.2f× wall-time speedup\n", f.steps / a.steps, f.wall / a.wall)
