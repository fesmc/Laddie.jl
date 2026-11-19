# Kernel forms that Reactant and Enzyme accept

- **Type:** improvement
- **Status:** in main since 2026-09-28; rules for every kernel
- **Commits:** 78526d1 (`unsafe_trunc` in the ambient sampling), b205d17 (`_face_avg_ring0`), 2c312a1 (wrapped reads as comparisons, sharding), 5f6864d (`@inbounds` in helpers, walls as separate arguments), 3488073 (`MAX_RAISED_BLOCK`)
- **Code:** `_xp1`/`_xm1`/`_yp1`/`_ym1`, `_face_avg_ring0`, `_face_mass_x`, `_v_half_kernel!` in `src/numerics/stencils.jl`; `_depth_levels` in `src/bcs/forcing.jl`; `_u_walls` in `src/bcs/walls.jl`; `_launch_traced!` in `ext/LaddieReactantExt.jl`
- **Docs:** `docs/src/reactant.md`

## Summary

Reactant raises the KernelAbstractions kernels to XLA operations, and Enzyme differentiates the raised program.
Some forms of the same arithmetic raise to slow or non-differentiable operations, or fail to compile.
The kernels follow the rules below; AGENTS.md lists the general ones (no throw path, `ifelse` selects an index).

## Rules

| form | do | because |
|---|---|---|
| periodic neighbour index | compare the shifted index, `ifelse(i + 1 > N, i + 1 - N, i + 1)` | the equality form `i == N ? 1 : i + 1` raises to a gather, which shards as an all-gather of the whole field; the comparison raises to a rotation, which shards as a halo exchange |
| read wrapped along both axes | two passes, one axis each (`_v_half_kernel!`, then `_other_x_kernel!`) | one pass raises to a gather |
| wrapped read in a branch | form the value and select it with `ifelse` (`_face_mass_x`) | a branch around a wrapped read raises to a gather |
| read near the border ring | always form the value, select with `ifelse` (`_face_avg_ring0`) | a branch raises to a reduction Enzyme cannot differentiate |
| float to index | clamp and make finite in float, then `unsafe_trunc` (`_depth_levels`) | the checked `trunc` keeps an `InexactError` trap that Reactant cannot raise |
| interpolation weight | subtract `floor(x)`, not the index converted back | Enzyme on raised kernels (Reactant 0.2.286) gives `x - FT(idx)` a zero tangent |
| array reads in `@inline` helpers | `@inbounds` inside the helper | a kernel's `@inbounds` does not reach into called functions; the leftover bounds checks stop the CPU kernels from vectorising (3–4× slower) |
| wall indicators | separate kernel arguments (`_u_walls`), not one NamedTuple | a NamedTuple of arrays unpacked in a kernel costs the CPU kernels up to a third of their time |

## Caveats

- Reactant 0.2.286 mis-lowers a raised kernel whose single block has 2²² iterations or more (pan-Antarctic grids finer than 3 km): it writes a sliver of the ndrange.
  From that size the raised kernels keep the GPU workgroup (`MAX_RAISED_BLOCK`): slower, but correct.
- A fusion barrier before each stencil kernel (`:stencil`, removed 2026-09-24) was slower than one after every kernel (`:kernel`) on every grid: 2.2× at 2000 × 2000.
