# Lateral viscosity

The momentum equations carry a lateral viscous term, a thickness-weighted Laplacian of
the velocity. Its coefficient is selected with `Params(; lateral_viscosity = ...)`:

- [`PrescribedLateralViscosity`](@ref), the default: one global coefficient
  `Params.A_h` (6 m² s⁻¹ by default), as [LADDIE v1](../versions.md) does.
- [`NonlinearLateralViscosity`](@ref): a Smagorinsky-type coefficient, as [LADDIE v2](../versions.md)
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

The grounding-line and land walls carry the scheme's own coefficient, with the neighbour
replaced by the ghost velocity ``(1 - s)\,u`` of the slip factor ``s``. With the default
``s = 1`` the ghost is zero, as in the v2 reference, which treats grounded and land neighbours
as ordinary cells at rest (`laddie_velocity.f90`); only at the border of its mesh does v2 apply
the bare `C%laddie_viscosity` as a constant, which Laddie.jl has no counterpart of. How strongly the walls brake the flow is the slip
factor, set separately (see
[Wall slip](../bcs/walls.md)). Two things are *not* affected by this choice:

- **Tracer diffusion.** The tracer diffusivity `Params.K_h` is always a constant.
- **Shear entrainment.** [`HollandEntrainment`](@ref) uses `A_h` and `K_h` as its length
  scale either way.

!!! note "C-grid staggering"
    The v2 reference collocates `U` and `V` at triangle centres, so its velocity difference
    `dUabs` is direct. On the staggered C-grid the cross-component is first interpolated onto
    this component's points, with the 4-point average the quadratic drag uses for the speed
    (`ip_half(jm_half(V))` for the `U` equation, `jp_half(im_half(U))` for `V`). The difference
    is the full vector ``\lVert\delta\mathbf{u}\rVert = \sqrt{\delta U^2 + \delta V^2}``, so one
    isotropic coefficient serves both components of a face.

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
