#############################
# Lateral momentum viscosity
#############################

"""
Abstract supertype for lateral (horizontal) momentum viscosity parameterisations.
Pass a concrete instance as `Params(; lateral_viscosity = ...)`:
[`NonlinearLateralViscosity`](@ref) (shear-scaled, as in LADDIE v2, the default) or
[`PrescribedLateralViscosity`](@ref) (constant coefficient).

The momentum equations carry the lateral friction `∇·(A_h H ∇u)`. Tracer diffusivity
`Params.K_h` is always a constant.
"""
abstract type AbstractLateralViscosity end

"""
$(TYPEDEF)

Constant-coefficient Laplacian lateral viscosity, `Params.A_h` everywhere, as in
LADDIE v1 ([Lambert et al. 2023](@cite lambert_modelling_2023), Eqs. 2–3). Wall stress relative to an interior face is set
by the slip factor of [`BoundaryConditions`](@ref).

Select via `Params(; lateral_viscosity = PrescribedLateralViscosity())`.
"""
struct PrescribedLateralViscosity <: AbstractLateralViscosity end

"""
$(TYPEDEF)

Shear-scaled (Smagorinsky-type) lateral viscosity of the LADDIE v2 code
([Lambert et al. 2026](@cite lambert_one-layer_2026)): at each face

```
A_h^eff = C_visc/100 Δ ‖δu‖,
```

with `Δ` the grid spacing across the face and `‖δu‖` the magnitude of the velocity-difference
vector across it. `C_visc` is dimensionless and resolution-independent, comparable to v2's
configuration value but not to `Params.A_h`, which plays no part here. Walls take the ghost
velocity of the slip factor. See [Lateral viscosity](@ref).

Select via `Params(; lateral_viscosity = NonlinearLateralViscosity(0.1))` (the default).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct NonlinearLateralViscosity{FT} <: AbstractLateralViscosity
    "dimensionless coefficient, the reference `C%laddie_viscosity` (default `0.1`, its Antarctic value; its MISMIP+ config uses `10`)"
    C_visc::FT = 0.1
end

#############################
# Laplacian thickness weights
#############################

"""
Abstract supertype for the time level of the layer thickness `H` that weights all
lateral Laplacians (tracer diffusion and momentum viscosity). Pass a concrete instance
as `Params(; laplacian_weights = ...)`: [`PresentLaplacianWeights`](@ref) (the default)
or [`PastLaplacianWeights`](@ref).

The diffused field itself is always taken at the lagged level `past`, as leapfrog
stability requires; this choice only concerns the thickness that multiplies it. On the
Crosson–Dotson validation case the two options give the same mean melt to 0.01 % and
differ locally by at most 1.4 m in `H`.
"""
abstract type AbstractLaplacianWeights end

"""
$(TYPEDEF)

Weight the Laplacians with `H.present` (the default).
"""
struct PresentLaplacianWeights <: AbstractLaplacianWeights end

"""
$(TYPEDEF)

Weight the Laplacians with the lagged, filtered `H.past`, so the thickness weight sits on
the same time level as the diffused field. This is what the LADDIE v1 code does.
"""
struct PastLaplacianWeights <: AbstractLaplacianWeights end

"""
    laplacian_thickness(m, weights)

The thickness that weights the viscous and diffusive Laplacians under the
[`AbstractLaplacianWeights`](@ref) `weights`: `m.H.present` or `m.H.past`.
"""
function laplacian_thickness end

laplacian_thickness(m) = laplacian_thickness(m, m.laplacian_weights)
laplacian_thickness(m, ::PresentLaplacianWeights) = m.H.present
laplacian_thickness(m, ::PastLaplacianWeights) = m.H.past
