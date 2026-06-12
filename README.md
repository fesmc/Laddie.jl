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
  ISOMIP+ configuration,
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

Laddie.jl is currently a port of the original LADDIE.py v1.0. As of now, the improvements of v2.0 are not a target for the Julia version.

## Installation

The package is not registered yet:

```julia
using Pkg
Pkg.add(url = "https://github.com/fesmc/Laddie.jl")
```

## Quickstart: ISOMIP+ cavity

```julia
using Laddie

m = build_isomip(; isomipcond = :warm)   # 240×40 idealised channel, 2 km cells
run!(m; days = 30)
max_melt, mean_melt, max_speed = meltstats(m)   # m/yr, m/yr, m/s
```

## Realistic geometry

Build the domain mask and ice draft from BedMachine-style arrays, and the
ambient forcing from any T/S profile data:

```julia
using Laddie, NCDatasets

ds  = NCDataset("BedMachineAntarctica-v3.nc")
bed = Float64.(Array(ds["bed"][i1:i2, j1:j2]))
h   = Float64.(Array(ds["thickness"][i1:i2, j1:j2]))
close(ds)

mask    = build_laddie_mask(bed, h)         # 0 ocean / 1 land / 2 grounded / 3 shelf
zb      = ice_base_depth(bed, h)            # ice-base depth (m, negative)
forcing = ProfileForcing(Tz, Sz, z)         # T (°C), S (psu), z (m) vectors

m = build_model(mask, zb, 500.0, 500.0, forcing, Params())
run!(m; days = 90)
```

Physical parameters and parameterization choices live in a single typed
`Params` object:

```julia
params = Params(; dt = 120.0, Ah = 25.0,
                meltpar = TurbulentGamT(13.8, 2432.0, 1.95e-6),
                convpar = RelaxToAmbient(10000.0))
```

## GPU

```julia
using CUDA
m = build_isomip(CUDABackend(); FT = Float32, isomipcond = :warm)
run!(m; days = 30)
```

## Output and restarts

```julia
rc = RunConfig(name = "warm0", saveday = 1.0, restday = 30.0)
m  = build_isomip(; isomipcond = :warm, rc)
run!(m)
```

This writes time-averaged NetCDF fields, JLD2 restart files, a log, and a
`run_metadata.toml` provenance record (parameters, forcing, grid, versions)
to `./output/warm0/`. Continue a run by passing
`RunConfig(fromrestart = true, restartfile = ".../restart_latest.jld2", ...)`.

## Citing

If you use Laddie.jl, please cite the model description paper:

> Lambert, E., Jüling, A., van de Wal, R. S. W., and Holland, P. R. (2023):
> Modelling Antarctic ice shelf basal melt patterns using the one-layer
> ocean model LADDIE, *The Cryosphere*, 17, 3203–3228,
> [doi:10.5194/tc-17-3203-2023](https://doi.org/10.5194/tc-17-3203-2023).

and the software itself via [CITATION.bib](CITATION.bib).

## License

GPL-3.0 — see [LICENSE](LICENSE).
