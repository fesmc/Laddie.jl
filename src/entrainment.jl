"""
Abstract supertype for the entrainment parameterisation — the rate at which
ambient water is mixed into the layer.  Pass a concrete instance as
`Params(; entrainment = ...)`: [`LambertEntrainment`](@ref) (the default),
[`GasparEntrainment`](@ref) or [`HollandEntrainment`](@ref).
"""
abstract type AbstractEntrainment end

"""
$(TYPEDEF)

Buoyancy-flux-driven entrainment as implemented in the reference LADDIE model
(Lambert et al. 2023; Gaspar 1988; Gladish et al. 2012). This is the form
verified bit-for-bit against the Python reference and is the **default**.

The net entrainment solves the layer mechanical-energy balance for ``\\dot{e}``,

```math
\\dot{e} = \\frac{2\\mu}{g}\\,\\frac{u_\\star^3}{D\\, \\delta \\rho} - \\frac{\\delta \\rho_b}{\\delta \\rho}\\,\\dot{m},
```

with a single power of `D` and a factor `2 \\mu` in the production term. This
differs from a literal reading of Eq. 14 (see [`GasparEntrainment`](@ref) and
`docs/src/equations.md`).

This is also what LADDIE v2 calls `'Gaspar1988'` (`laddie_physics.f90`,
`compute_entrainment`); Laddie.jl's [`GasparEntrainment`](@ref) is a different form.

Select via `Params(; entrainment = LambertEntrainment(2.5))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LambertEntrainment{FT} <: AbstractEntrainment
    "dimensionless efficiency parameter (ISOMIP+ default `2.5`)"
    mu::FT = 2.5
end

"""
$(TYPEDEF)

Buoyancy-flux-driven entrainment as the **literal reading of Eq. 14** of
Lambert et al. (2023), ``D^2 g_b' \\dot{m} + D^2 g_a' \\dot{e} = \\mu u_\\star^3``, solved for ``\\dot{e}``:

```math
\\dot{e} = \\frac{\\mu}{g}\\,\\frac{u_\\star^3}{D^2\\,\\delta\\rho} - \\frac{\\delta\\rho_b}{\\delta\\rho}\\,\\dot{m}.
```

Note the ``D^2`` in the denominator and the factor ``\\mu`` (not ``2\\mu``). This is **not**
the form the reference LADDIE actually integrates — for that, use the default
[`LambertEntrainment`](@ref). The melt-buoyancy/detrainment term
``-(\\delta\\rho_b/\\delta\\rho)\\,\\dot{m}`` is identical in both.

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

Shear-driven entrainment following Holland & Jenkins (1999):

```math
\\dot{e} = c_l \\, \\frac{K_h}{A_h^2} \\sqrt{\\max\\left(0,\\; |\\mathbf{u}|^2 - g\\,\\delta\\rho\\,\\frac{K_h}{A_h}\\,D\\right)},
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
