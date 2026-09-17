# Cache design notes

## Size

The `Cache` struct holds **~47 pre-allocated `nx×ny` matrices** plus a handful of
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

### Tier 2 — kernel refactoring (done)

The D-shift group (`Dym1`, `Dxm1`, …), the upwind splits (`Upos`, `Vyp1neg`, …),
the shifted velocities (`Vyp1`, `Uxp1`) and `signU`/`signV` used to be
pre-computed each step and then consumed by the advection kernels.  They are now
formed inline inside `_upwind_advection_{T,U,V}_kernel!` and the momentum kernels,
which removed 19 matrices and two kernel passes per step, bit-identically.

### Fields that cannot be merged

- **`Vip/Vim/Uip/Uim/Vjp/Vjm/Ujp/Ujm`** — all 8 live simultaneously; each
  represents a distinct stagger location and interpolation direction.
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
