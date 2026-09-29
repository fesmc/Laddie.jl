# Laddie.jl

A Julia port of **LADDIE** (Lambert et al. 2023, *The Cryosphere* 17:3203;
Python at [github.com/erwinlambert/laddie](https://github.com/erwinlambert/laddie))
made GPU-capable via
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl).

LADDIE computes the **basal melt rate beneath an ice shelf** by modelling the
thin, buoyant meltwater layer in the cavity with a depth-integrated
("one-layer") representation of the circulation.

See the [ISOMIP+ example](generated/isomip.md) for plots of the melt rate, layer
thickness, temperature and flow speed, and a cell-by-cell comparison against the
original Python code; the [Crosson–Dotson example](generated/crosson-dotson.md)
reproduces a published realistic-cavity run.

## Quick start

CPU example:
```julia
using Laddie

# Build and run the idealised ISOMIP+ warm cavity (CPU)
sim = build_isomip(; isomipcond = :warm)
run!(sim; days = 5.0)
mx, mn, sp = meltstats(sim)
```

GPU (CUDA example):
```julia
using CUDA, Laddie
sim = build_isomip(CUDABackend(); isomipcond = :warm)
run!(sim; days = 30.0)
```

## Performance

Laddie.jl has the physics of the Python implementation, but its arrays are
allocated on a chosen KernelAbstractions backend (CPU / CUDA / ROCm / Metal), so
the entire time step executes on-device. Every term is a fused kernel — one pass
per term, with no intermediate arrays — so a time step allocates almost nothing,
and the same code runs multi-threaded on the CPU (`julia -t N`).

## Documentation map

| page | contents |
|------|----------|
| [Physics](physics.md) | what the model represents and the governing balances |
| [Numerics](numerics.md) | grid, time stepping, boundaries, stability |
| [ISOMIP+](generated/isomip.md) | forcing, a warm run, Python validation, spin-up, warm vs cold |
| [Crosson–Dotson](generated/crosson-dotson.md) | reproduction of Lambert et al. (2023) |
