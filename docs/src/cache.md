# Cache design notes

## Size

The `Cache` struct holds **40 pre-allocated `nx×ny` matrices** plus a handful of
scalars (`gamT/gamS/conv2` for fixed-coefficient parameterisations).  At
512×512 Float64 each matrix is ~2 MB, so the full cache is about 80 MB at that
resolution.

## Reductions made

A usage audit found three ways to reduce the matrix count without changing
numerical results.  All three are done, and each was checked to be bit-identical.

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

- **`Vip/Vim/Uip/Uim/Vjp/Vjm/Ujp/Ujm`** — all 8 live simultaneously; each
  represents a distinct stagger location and interpolation direction.
- **`dDdt`, `Ddrho`** — cross-step state: written once per leapfrog step and
  read by all four prognostic kernels (U, V, T, S).
- **`convD`** — written by `update_entrainment!`, read by `step_thickness!`
  and the PSI diagnostic in `printdiags`.
- All physics output fields (`melt`, `Tb`, `drho`, `ustar`, …) — read at
  output time and consumed by downstream physics routines.
