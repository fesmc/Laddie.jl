# Laddie.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fesmc.github.io/Laddie.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fesmc.github.io/Laddie.jl/dev/)
[![Build Status](https://github.com/fesmc/Laddie.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/fesmc/Laddie.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/fesmc/Laddie.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/fesmc/Laddie.jl)

A Julia implementation of **LADDIE** — the one-**L**ayer **A**ntarctic model for
**D**ynamical **D**ownscaling of **I**ce-ocean **E**xchanges
([Lambert et al., 2023](https://doi.org/10.5194/tc-17-3203-2023)).
LADDIE solves depth-integrated conservation equations for a buoyant meltwater
plume under an ice shelf and returns high-resolution basal melt-rate fields
from an ambient temperature/salinity profile and the cavity geometry.

Laddie.jl is a from-scratch port of the
[Python reference implementation](https://github.com/erwinlambert/laddie) with:

- **CPU and GPU execution** from the same code via
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
  (`CUDABackend`, `ROCBackend`, `MetalBackend`),
- **`Float64` or `Float32`** precision throughout,
- a **verification test** against the Python LADDIE end state on the warm
  ISOMIP+ configuration, and a reproduction of the published Crosson–Dotson run,
- NetCDF output, JLD2 restarts, and a TOML provenance record for every run.

**We'd like to acknowledge** that Laddie.jl would not have been possible
without the excellent model description paper and the clear, open-source
Python implementation!

The authors of LADDIE (not us!) have worked hard on v2.0, including another [great description paper](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-930/), which is currently in revision, and [an open-source code](https://github.com/UPSY-group/UPSY-models). This unlocked:
1. parallelization and performance improvements
2. unstructured grids and more flexible domain geometries
3. an improved time stepping scheme
4. modified boundary conditions at the grounding line (better match with observation)
5. running pan-Antarctic domains with evolving geometry

Laddie.jl is a port of the original LADDIE.py v1. A few v2.0 options are available
(no-slip walls, a shear-scaled lateral viscosity, the truncated ice-front pressure
gradient), but a full port of v2.0 is not a target: the Julia version focuses on
developing other capabilities, such as GPU execution and connected ice-shelf gaps.

## Installation

The package is not registered yet:

```julia
using Pkg
Pkg.add(url = "https://github.com/fesmc/Laddie.jl")
```

## Quickstart: ISOMIP+ cavity

```julia
using Laddie

sim = build_isomip(; isomipcond = :warm)   # 240×40 idealised channel, 2 km cells
run!(sim; days = 30)
stats = meltstats(sim)   # max/mean melt (m/yr), max speed (m/s), total melt (Gt/yr)
```

`build_isomip` returns a `Simulation`: a `Model` (geometry, physics, state —
`sim.model`) plus its time integration (`sim.clock`, the time stepper, output).

## Realistic geometry

Build the domain mask and ice draft from BedMachine-style arrays, and the
ambient forcing from any T/S profile data:

```julia
using Laddie, NCDatasets

ds  = NCDataset("BedMachineAntarctica-v3.nc")
bed = Float64.(Array(ds["bed"][i1:i2, j1:j2]))
h   = Float64.(Array(ds["thickness"][i1:i2, j1:j2]))
x, y = ds["x"][i1-1:i2+1], ds["y"][j1-1:j2+1]   # cell centres, border ring included
close(ds)

mask    = build_laddie_mask(bed, h)         # 0 ocean / 1 land / 2 grounded / 3 shelf
zb      = ice_base_depth(bed, h)            # ice-base depth (m, negative)
ocean   = OceanForcing1D(Tz, Sz, z)         # T (°C), S (psu), z (m) vectors

grid  = Grid(mask, zb; x, y)                # geometry: where the cells are (dx, dy from x, y)
model = Model(grid; forcing = ocean)        # physics on that grid
sim   = Simulation(model; dt = 120.0)       # time integration and output
run!(sim; days = 90)
```

All 2D fields — mask, draft, bed, and every diagnostic — are stored `[x, y]`: the
first index runs along x, the second along y, matching what NCDatasets hands back
when reading a NetCDF file and what `heatmap` expects. Output files are written in
the CF layout (`melt(time, y, x)` in `ncdump`), so ncview renders them the usual way
round. Note that x and y are grid axes, not compass directions: a projected polar
domain rotates them relative to true east/north.

This is the grid → model → simulation → `run!` split shared by Oceananigans,
SpeedyWeather and FastIsostasy. The `Grid` holds only what is independent of any
modelling choice (mask, draft, bed, spacing); the `Model` derives the active-cell
masks, ice-base slope and Coriolis field from it, so one grid can drive several
models — e.g. both gap treatments.

A model is driven by a `CavityForcing` — an ocean forcing plus an ice forcing.
Passing the ocean forcing alone, as above, pairs it with a uniform basal ice
temperature of −25 °C. Supply the ice explicitly to vary it in space:

```julia
forcing = CavityForcing(ocean, PrescribedIceForcing(T_ice_base))  # same size as mask
```

Physical parameters and parameterization choices live in a single typed
`Params` object, boundary conditions in a `BoundaryConditions`, and everything
about time integration is a `Simulation` option:

```julia
params   = Params(; A_h = 25.0,
                  melting = TurbulentGamTMelting(13.8, 2432.0, 1.95e-6),
                  convection_scheme = RelaxToAmbient(10000.0))
boundary = BoundaryConditions(; land = FreeSlipLand(), gaps = ConnectedGapsBC())
model    = Model(grid; forcing = ocean, params, boundary)
sim      = Simulation(model; dt = 120.0, tstep = AdaptiveDt(),
                      stop = FixedSimulationEnd(t_end = 90.0))
run!(sim)
```

The defaults follow Python LADDIE v1 except at the walls, which are no-slip; its
single partial-slip factor is
`BoundaryConditions(; grounding_line = PartialSlipGL(1.0), land = PartialSlipLand(1.0))`.

## GPU

```julia
using CUDA
sim = build_isomip(CUDABackend(); FT = Float32, isomipcond = :warm)
run!(sim; days = 30)
```

## Output and restarts

```julia
output = OutputConfig(name = "warm0", saveday = 1.0, restday = 30.0)
sim = build_isomip(; isomipcond = :warm, output)
run!(sim)
```

This writes time-averaged NetCDF fields, JLD2 restart files, a log, and a
`run_metadata.toml` provenance record (parameters, time integration, forcing,
grid, versions) to `./output/warm0/`. Successive `run!` calls continue the same
clock. Continue from a restart file with
`Simulation(model; restart = ".../restart_latest.jld2", ...)`.

## Citing

If you use Laddie.jl, please cite the model description paper:

> Lambert, E., Jüling, A., van de Wal, R. S. W., and Holland, P. R. (2023):
> Modelling Antarctic ice shelf basal melt patterns using the one-layer
> ocean model LADDIE, *The Cryosphere*, 17, 3203–3228,
> [doi:10.5194/tc-17-3203-2023](https://doi.org/10.5194/tc-17-3203-2023).

and the software itself via [CITATION.bib](CITATION.bib).

## License

GPL-3.0 — see [LICENSE](LICENSE).
