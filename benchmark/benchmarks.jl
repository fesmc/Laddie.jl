using Laddie
using Chairmarks
using KernelAbstractions: CPU

# ============================================================================
# Single-timestep benchmarks (broadcast vs fused, CPU)
# ============================================================================

function _timestep!(m)
    Laddie.advance_leapfrog!(m)
    Laddie.leapfrog_step!(m, 2)
    Laddie.clamp_velocities!(m)
    Laddie.apply_robert_asselin_filter!(m)
end

cpu_broadcast = @be (m = build_isomip(CPU(); isomipcond = :warm); m.fused = false; m) _timestep!(_)
cpu_fused     = @be (m = build_isomip(CPU(); isomipcond = :warm); m.fused = true;  m) _timestep!(_)

# ============================================================================
# GPU benchmarks (optional — only run when CUDA is available)
# ============================================================================
# Uncomment and add CUDA to deps to enable:
#
using CUDA
if CUDA.functional()
    gpu_fused = @be (m = build_isomip(CUDABackend(); isomipcond = :warm); m.fused = true; m) begin
        _timestep!(_); CUDA.synchronize()
    end
end
