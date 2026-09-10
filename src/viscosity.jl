#############################
# Lateral momentum viscosity
#############################

abstract type AbstractLateralViscosity end

"""
$(TYPEDSIGNATURES)

Constant-coefficient Laplacian lateral viscosity (the default): the diffusive
flux at every interior face is `Params.A_h * ΔU / dy` (or `ΔV / dx`), i.e. a
plain Laplacian with a single global coefficient. This is the scheme
Laddie.jl has always used, matching Python LADDIE v1.x.

Grounding-line/land wall friction also scales with `Params.A_h`, unaffected
by the choice of `lateral_viscosity` — see [`NonlinearLateralViscosity`](@ref).

Select via `Params(; lateral_viscosity = PrescribedLateralViscosity())` (the default).
"""
struct PrescribedLateralViscosity <: AbstractLateralViscosity end

"""
$(TYPEDSIGNATURES)

Shear-scaled lateral viscosity, as used by the reference LADDIE v2 Fortran
implementation (`laddie_velocity.f90:260`). Instead of a constant coefficient,
the viscosity at each interior face is

```math
A_h^{\\mathrm{eff}} = \\frac{C_{\\mathrm{visc}}}{100}\\,\\Delta\\,|\\delta u|,
```

with `Δ` the grid spacing across that face (`dx` for east/west faces, `dy` for
north/south) and `δu` the velocity difference across it, so the diffusive flux
goes as `|δu|·δu`. This is a Smagorinsky-type closure — near-zero viscosity in
smooth flow, large viscosity where shear is strong — rather than the plain
Laplacian of [`PrescribedLateralViscosity`](@ref).

- `C_visc`: **dimensionless** coefficient, the reference's `C%laddie_viscosity`
  (default `10.0`, its MISMIP+ configuration value; the Antarctic test config
  uses `0.1`). Note `C_visc/100` plays the role of a squared Smagorinsky
  constant, so the default corresponds to `Cs ≈ 0.32`.

Because `Δ` is supplied internally, `C_visc` is resolution-independent and
directly comparable to the reference's config value. It is *not* comparable to
`Params.A_h`, which is a diffusivity in m² s⁻¹.

Grounding-line/land wall friction is **not** affected by this choice: the
reference keeps its border term linear in the viscosity coefficient even
under this scheme (`laddie_velocity.f90:249-254`), and Laddie.jl mirrors that
by always scaling the `grline_bc`/`land_bc` wall-drag terms with the global
`Params.A_h`, never with `C_visc`. Wall slip itself (free-slip vs no-slip) is
controlled independently by `Params.grline_bc` / `Params.land_bc`.

!!! note "Deviation from the reference"
    The reference builds its coefficient from the magnitude of the full
    velocity-difference *vector*, `√(ΔU² + ΔV²)`, so one isotropic coefficient
    serves both momentum components on a face. Laddie.jl uses the difference of
    the component being diffused (`|ΔU|` in the `U` equation, `|ΔV|` in `V`),
    because on a staggered C-grid the cross-component difference is not
    collocated with the face and would need extra interpolation. The closure is
    therefore slightly anisotropic compared with the reference.

Select via `Params(; lateral_viscosity = NonlinearLateralViscosity(10.0))`.
"""
@kwdef struct NonlinearLateralViscosity{FT} <: AbstractLateralViscosity
    C_visc::FT = 10.0
end
