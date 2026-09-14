using BenchmarkTools
using Laddie
using KernelAbstractions

# One leapfrog step of a simulation, no I/O (what run! does per iteration).
timestep!(sim) = time_step!(sim)

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
