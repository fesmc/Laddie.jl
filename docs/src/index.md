# Laddie.jl

A Julia port of **LADDIE** (Lambert et al. 2023, *The Cryosphere* 17:3203;
Python at [github.com/erwinlambert/laddie](https://github.com/erwinlambert/laddie))
made GPU-capable via
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl).

LADDIE computes the **basal melt rate beneath an ice shelf** by modelling the
thin, buoyant meltwater layer in the cavity with a depth-integrated
("one-layer") representation of the circulation.

See the worked [ISOMIP+ run](generated/isomip_run.md) for plots of the melt
rate, layer thickness, temperature, and flow speed, and the
[Python validation](generated/python_comparison.md) page for a cell-by-cell
comparison against the original Python code.

## Quick start

CPU example:
```julia
using Laddie

# Build and run the idealised ISOMIP+ warm cavity (CPU)
m = build_isomip(; isomipcond = :warm)
run!(m; days = 5.0)
mx, mn, sp = meltstats(m)
```

GPU (CUDA example):
```julia
using CUDA, Laddie
m = build_isomip(CUDABackend(); isomipcond = :warm)
m.fused = true     # enable fused kernel path
run!(m; days = 30.0)
```

Config-file driven run:
```julia
m = build_from_config("config.toml")
run!(m)
```

## Performance

Laddie.jl has the same physics as the pure-CPU python implementation, but arrays are
allocated on a chosen KernelAbstractions backend (CPU / CUDA / ROCm / Metal)
so the entire time step executes on-device.

In particular, kernels were fused to eliminate the ~15-20 intermediate arrays the broadcast path allocates. This provides results that are bit-identical to the broadcast path,but offer a significant speedup.

## Documentation map

| page | contents |
|------|----------|
| [Physics](physics.md) | what the model represents and the governing balances |
| [Numerics](numerics.md) | grid, time stepping, boundaries, stability |
| [Implementation](implementation.md) | how the Julia/GPU port was built and verified |
| [Configuration](configuration.md) | complete TOML config-file reference |
| [ISOMIP+ forcing](generated/forcing.md) | the warm/cold ambient profiles |
| [ISOMIP+ run](generated/isomip_run.md) | a full simulation with result plots |
| [Python validation](generated/python_comparison.md) | end-state comparison against the Python reference |
