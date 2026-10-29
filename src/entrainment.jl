"""
Abstract supertype for the entrainment parameterisation — the rate at which
ambient water is mixed into the layer.  Pass a concrete instance as
`Params(; entrainment = ...)`: [`GasparEntrainment`](@ref) (the default) or
[`HollandEntrainment`](@ref).
"""
abstract type AbstractEntrainment end

"""
$(TYPEDEF)

Buoyancy-flux-driven entrainment after [Gaspar (1988)](@cite gaspar_modeling_1988), following [Gladish et al. (2012)](@cite gladish_ice-shelf_2012), as in [Lambert et al. (2023, Eq. 14)](@cite lambert_modelling_2023) and LADDIE v2 ([Lambert et al. 2026](@cite lambert_one-layer_2026),
Eq. 13). The layer mechanical-energy balance

```math
\\frac{H}{2}\\, g_b' \\dot{m} + \\frac{H}{2}\\, g_a' \\dot{e} = \\mu u_\\star^3
```

is solved for the net entrainment ``\\dot{e}``:

```math
\\dot{e} = \\frac{2\\mu}{g}\\,\\frac{u_\\star^3}{H\\, \\delta \\rho} - \\frac{\\delta \\rho_b}{\\delta \\rho}\\,\\dot{m}.
```

This is the scheme LADDIE v1 calls `'Gaspar'` (`physics.py`) and
v2 calls `'Gaspar1988'` (`laddie_physics.f90`, `compute_entrainment`); it is
verified bit-for-bit against the LADDIE v1 code and is the **default**.

Select via `Params(; entrainment = GasparEntrainment(2.5))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct GasparEntrainment{FT} <: AbstractEntrainment
    "dimensionless efficiency parameter (ISOMIP+ default `2.5`)"
    mu::FT = 2.5
end

"""
$(TYPEDEF)

Shear-driven entrainment following [Holland & Jenkins (1999)](@cite holland_modeling_1999):

```math
\\dot{e} = c_l \\, \\frac{K_h}{A_h^2} \\sqrt{\\max\\left(0,\\; |\\mathbf{u}|^2 - g\\,\\delta\\rho\\,\\frac{K_h}{A_h}\\,H\\right)},
```

with ``|\\mathbf{u}|`` the speed at the T-point, ``\\delta\\rho`` the reduced density
contrast and ``K_h``, ``A_h`` the lateral diffusivity and viscosity of `Params`.
There is no detrainment.

Select via `Params(; entrainment = HollandEntrainment(0.01775))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct HollandEntrainment{FT} <: AbstractEntrainment
    "entrainment coefficient (default `0.01775`)"
    cl::FT = 0.01775
end
