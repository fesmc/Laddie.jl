# Wall slip in NEMO's convention

- **Type:** improvement
- **Status:** default since 2026-11-04
- **Commits:** d33fe0d (feature)
- **Code:** `PartialSlipGL`, `PartialSlipLand`, `NoSlipGL`, `NoSlipLand`, `NoWallAdvection`, `V1WallAdvection` in `src/boundary_conditions.jl`
- **Docs:** `docs/src/numerics/walls.md`

## Summary

The slip factor s of a grounding-line or land wall now sets the ghost velocity (1 − s)·u behind the wall, as NEMO's `rn_shlat`.
s = 0 is free slip (no stress), s = 1 is u = 0 one grid spacing behind the wall (LADDIE v2), and s = 2 is no slip at the wall face.
The defaults are now v2's walls: `PartialSlipGL(1.0)`, `PartialSlipLand(1.0)`, `NoWallAdvection()`.

## Why

LADDIE v1 adds `slip` to a viscous stencil that already holds the stored zero velocity (`physics.py:lapU`), so its viscous wall is one step stronger than labelled:

| v1 `slip` | label | viscous wall actually applied |
|---|---|---|
| 0 | free slip | u = 0 one Δ behind the wall (v2's wall) |
| 1 | partial slip (published) | no slip, u = 0 at the wall face |
| 2 | no slip | beyond no slip, u = 0 at Δ/3 inside the fluid |

Its momentum advection uses the ghost (1 − slip)·u as labelled, so the two terms disagree for the same parameter.
The old Laddie default, `NoSlipGL` with s = 2 in the old convention, was beyond no slip.

## Evidence

- v1's published `slip = 1` is spelled `NoSlipGL()` + `NoSlipLand()` + `V1WallAdvection()`. That is bit-identical to before, and the Python verification test passes.
- `SimulationASEv2` mean melt moves from 20.46 to 20.49 m/yr. The v2 reproductions move by at most 0.2 percentage points.

## Caveats

- Runs on the old defaults are not comparable to new ones.
- Do not call v2's wall "no slip" in Laddie's terms: it is s = 1.
