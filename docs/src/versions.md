# LADDIE versions

Three codes share the name LADDIE. **v1** ([Lambert et al. (2023)](@cite lambert_modelling_2023))
runs on a structured Arakawa C-grid. **v2** ([Lambert et al. (2026)](@cite lambert_one-layer_2026))
runs on an unstructured mesh, rewritten and integrated with the UFEMISM ice-sheet model. Laddie.jl
implements the **v2 physics on the v1 C-grid**: it keeps v1's structured grid and stencils, and takes
the physical choices that v2 made. Each choice that differs between the two versions is an option of
Laddie.jl; the [physics](physics/index.md) and [numerics](numerics/index.md) overviews list them, with
the default a plain run uses.

| aspect | LADDIE v1 | LADDIE v2 | Laddie.jl |
|---|---|---|---|
| language | Python | Fortran, parallel | Julia, [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl): CPU threads and GPUs |
| grid | structured Arakawa C-grid | unstructured triangular mesh, UFEMISM-integrated | structured C-grid, as v1 |
| time stepping | leapfrog with Robert–Asselin filter | forward–backward Runge–Kutta (FB-RK3) | leapfrog with Robert–Asselin filter, as v1; fixed or adaptive step (`AdaptiveDt`) |
| typical run time\* | 40 min | 5 min | 30 s |
| automatic differentiation | no | no | yes |

\* Best time achieved by each code: LADDIE v1 on one CPU, LADDIE v2 on 64 CPUs, Laddie.jl on one GPU.

## Using the versions

`Params<Setup>v1` and `Params<Setup>v2` return the parameters each published run used, and
`Simulation<Setup>v1` and `Simulation<Setup>v2` build the run itself. The v2 aliases match v2's
*parameters*: they select the v2 options of the [physics](physics/index.md) and [numerics](numerics/index.md) overviews, but they run on Laddie.jl's C-grid
with leapfrog stepping, so they are not bit-for-bit reproductions. The
[reproduction page](generated/reproductions.md) compares both against the published output, and
the [published setups](API_published.md) list every alias.
