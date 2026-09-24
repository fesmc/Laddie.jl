# Cache design notes

## Size

The `Cache` struct holds **18 pre-allocated `nx×ny` matrices**, plus
`gamT/gamS`, `conv2` and `melt_prescribed`, which are matrices only for the melt and
convection schemes that need them (scalars otherwise).  At 2000×2000 Float64 each
matrix is 32 MB, so the default cache is about 550 MiB at that resolution.

## Reductions made

Two usage audits found five ways to reduce the matrix count without changing
numerical results.  All five are done, and each was checked to be bit-identical.

### Shared work buffers (−7 matrices)

U, V, T and S are stepped one after another, and each step consumes its
advection and Laplacian terms before the next step writes its own.  So one set of
work buffers serves all four equations:

| Buffer | Replaces |
|--------|----------|
| `adv` | `cU`, `cV`, `cT`, `cS` |
| `lap` | `lU`, `lV`, `lT`, `lS` |
| `Dq`  | `DT`, `DS` (the advected tracer content D·q) |

The reference equation terms in `physics.jl` (`u_advection`, `u_diffusion`, …)
return copies of these buffers, because the next term to be evaluated would
overwrite them.

### Shifted fields formed in the kernels (−19 matrices)

The D-shift group (`Dym1`, `Dxm1`, …), the upwind splits (`Upos`, `Vyp1neg`, …),
the shifted velocities (`Vyp1`, `Uxp1`) and `signU`/`signV` used to be
pre-computed each step and then consumed by the advection kernels.  They are now
formed inline inside `_upwind_advection_{T,U,V}_kernel!` and the momentum kernels,
which also removed two kernel passes per step.

### Write-only and short-lived fields (−8 matrices)

A kernel-by-kernel trace of one time step (September 2026) found four fields that
were written every step and never read: the plume freezing point `Tf` (with its
kernel; the melt kernel forms its own), and `Sb`, `drhob` and the signed entrainment
rate `ent`, which are now locals of `_buoyancy_entrainment_kernel!`.  Four more live
only between two consecutive kernels and now use the shared work buffers: the speed
limiter's scale factors go in `adv`/`lap` (free between the momentum and tracer
steps), and the collocated cross-velocities of `NonlinearLateralViscosity` in `Dq`
(unused during the momentum steps).

### Staggered averages formed in the kernels (−14 matrices)

The eight velocity face averages (`Vip`, `Vim`, `Uip`, `Uim`, `Vjp`, `Vjm`, `Ujp`,
`Ujm`), the Laplacian thickness on the u- and v-points (`D_on_ugrid`, `D_on_vgrid`)
and its four face averages (`D0ip`, `D0im`, `D0jp`, `D0jm`) used to be computed at the
start of each step by two pre-compute kernels.  All 14 are live together through the
momentum steps, so no buffer sharing could remove them.  The kernels that read them
now form them from U, V and D with `_face_avg`.  Where a stencil reads an average at
a neighbouring cell, `_face_avg_ring0` returns zero on the border ring, as the
interior-only pre-compute left it.  Results are bit-identical.

The trade is fewer bytes moved for more arithmetic (about 30 masked divisions per
cell per step instead of 14).  Measured on ISOMIP+ grids (September 2026):

| | 1280×640 | 2000×2000 |
|---|---|---|
| CPU, 8 pinned cores (Xeon W-2245), Float64 | 47 → 42 ms | 232 → 214 ms |
| GPU (RTX A4000), Float32 | 2.6 → 2.4 ms | 12.1 → 10.6 ms |
| GPU (RTX A4000), Float64 | 13.3 → 15.4 ms | 65.6 → 76.3 ms |

Float64 on this GPU runs at 1/32 of the Float32 rate, so there the extra divisions
cost more than the saved bandwidth.  Production GPU runs use Float32.

### Geometry (−20 matrices, outside the Cache)

The same applies to the static `Geometry`: the shifted masks (`ocnxm1`,
`tmaskym1`, …) and the stagger counts (`tmask_ip`, `umask_jm`, …) are no longer
stored.  The kernels read the neighbouring cell instead, and the reference
equation terms use `ip_count(mask)` and its siblings (`utils.jl`).

Together with the shared buffers, this cut the memory of a 1000×1000 model from
827 MiB to 621 MiB.  A time step became 7–9 % faster on one thread and 10–20 %
faster on 16 threads (500×500 and 1000×1000), because the kernels move less
data.

## Fields that cannot be merged

- **`dDdt`, `Ddrho`, `nentr`** — written once per step and read by several of the
  prognostic kernels (U, V, T, S).
- **`convD`** — written by `update_entrainment!`, read by `step_thickness!`
  and the PSI diagnostic in `printdiags`.
- All physics output fields (`melt`, `Tb`, `Ta`, `Sa`, `drho`, `ustar`, `entr`,
  `detr`, `ent2`, `convection`, `gamT`/`gamS`) — read at output time, by the
  diagnostics, or by the density recomputation after the Robert–Asselin filter.
