# Cache design notes

## Size

The `Cache` struct holds **~50 pre-allocated `ny×nx` matrices** plus a handful of
scalars (`gamT/gamS/conv2` for fixed-coefficient parameterisations).  At
512×512 Float64 each matrix is ~2 MB, so the full cache is on the order of
100 MB at that resolution.

## Potential reductions

A usage audit identified two tiers of buffer sharing that would reduce the
matrix count without changing numerical results.

### Tier 1 — zero-code-change buffer sharing (−5 matrices)

The following pairs are written and fully consumed in strictly non-overlapping
phases of the leapfrog step, so they could safely share a single buffer:

| Merge | Rationale |
|-------|-----------|
| `cU` ↔ `cV` | U-momentum step completes before V-momentum starts |
| `lU` ↔ `lV` | same ordering |
| `cT` ↔ `cS` | temperature step completes before salinity starts |
| `lT` ↔ `lS` | same ordering |
| `DT` ↔ `DS` | `D·T` product is consumed before `D·S` is computed |

Implementation would be a rename at ~5 call sites per pair.

### Tier 2 — kernel refactoring (−7 matrices)

The D-shift group `Dym1`, `Dyp1`, `Dxm1`, `Dxp1`, `Dxm1ym1`, `Dxp1ym1`,
`Dxm1yp1` is pre-computed in `_precompute_D_shifts_kernel!` and then consumed
inside `_upwind_advection_U_kernel!` / `_upwind_advection_V_kernel!`.  These
intermediates could be eliminated by fusing the precompute step into the
advection kernels (compute-and-consume inline).  The stencil file even carries
a `# TODO can I rm Dxm1, Dym1…?` note at that site.  The saving is real but
requires substantial kernel restructuring.

### Fields that cannot be merged

- **`Vip/Vim/Uip/Uim/Vjp/Vjm/Ujp/Ujm`** — all 8 live simultaneously; each
  represents a distinct stagger location and interpolation direction.
- **`Upos/Uneg/Vpos/Vneg/Vyp1pos/Vyp1neg/Uxp1pos/Uxp1neg`** — all consumed
  together in the same tracer-advection kernel pass.
- **`dDdt`, `Ddrho`** — cross-step state: written once per leapfrog step and
  read by all four prognostic kernels (U, V, T, S).
- **`convD`** — written by `update_entrainment!`, read by `step_thickness!`
  and the PSI diagnostic in `printdiags`.
- All physics output fields (`melt`, `Tb`, `drho`, `ustar`, …) — read at
  output time and consumed by downstream physics routines.

## Decision

We do not address these optimisations for now.  Merging buffers would require
introducing shared aliases whose names no longer describe the physical quantity
they currently hold at any given moment, making the code harder to read and
debug.  The cache size is acceptable at current target resolutions, and clarity
of the physics is the higher priority.
