using BenchmarkTools
using Laddie
using KernelAbstractions

function timestep!(m)
    Laddie.advance_leapfrog!(m)
    Laddie.leapfrog_step!(m, 2)
    Laddie.clamp_velocities!(m)
    Laddie.apply_robert_asselin_filter!(m)
end

const GRIDS = [
    (80,  40,  "small_80x40"),
    (320, 160, "medium_320x160"),
    (640, 320, "large_640x320"),
]

const SUITE = BenchmarkGroup()
SUITE["cpu"] = BenchmarkGroup()
for (nx, ny, label) in GRIDS
    m = build_isomip(CPU(); nx=nx, ny=ny, isomipcond=:warm)
    SUITE["cpu"][label] = @benchmarkable timestep!(m_) setup=(m_=deepcopy($m)) evals=1
end
