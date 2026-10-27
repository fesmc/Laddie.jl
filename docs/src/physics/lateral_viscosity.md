# Lateral viscosity

The momentum equations carry a lateral viscous term, a thickness-weighted Laplacian of
the velocity. Its coefficient is selected with `Params(; lateral_viscosity = ...)`:

- [`PrescribedLateralViscosity`](@ref), the default: one global coefficient
  `Params.A_h` (6 m² s⁻¹ by default), as Python LADDIE v1.x does.
- [`NonlinearLateralViscosity`](@ref): a Smagorinsky-type coefficient, as LADDIE v2
  does, which is near zero in smooth flow and large where the shear is strong. At each
  interior face it is
  ```math
  A_h^{\mathrm{eff}} = \frac{C_{\mathrm{visc}}}{100}\,\Delta\,\lVert\delta\mathbf{u}\rVert,
  ```
  with ``\Delta`` the grid spacing across the face and ``\lVert\delta\mathbf{u}\rVert`` the
  magnitude of the velocity difference across it. `C_visc` is dimensionless and
  resolution-independent, so it is directly comparable to the v2 configuration value, but
  not to `A_h`. `C_visc/100` plays the role of a squared Smagorinsky constant; the default
  10 corresponds to ``C_s \approx 0.32``.

Three things are *not* affected by this choice:

- **Wall drag.** The extra drag at grounding-line and land walls always scales with the
  global `Params.A_h`, never with `C_visc`, as in the v2 reference. The slip at the walls
  is set separately; see [Wall slip](../numerics/walls.md).
- **Tracer diffusion.** The tracer diffusivity `Params.K_h` is always a constant.
- **Shear entrainment.** [`HollandEntrainment`](@ref) uses `A_h` and `K_h` as its length
  scale either way.

The time level of the thickness that weights the Laplacian is a separate, numerical
choice: [Laplacian weights](../numerics/laplacian_weights.md). There is no viscous
constraint on the time step, so a large viscosity on a fine grid is not caught by the
[CFL controller](../numerics/cfl.md).

## API

```@docs
AbstractLateralViscosity
PrescribedLateralViscosity
NonlinearLateralViscosity
```
