
abstract type AbstractEntrainment end

"""
$(TYPEDSIGNATURES)

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

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `Params(; entrainment = LambertEntrainment(2.5))`.
"""
@kwdef struct LambertEntrainment{FT} <: AbstractEntrainment
    mu::FT = 2.5
end

"""
$(TYPEDSIGNATURES)

Buoyancy-flux-driven entrainment as the **literal reading of Eq. 14** of
Lambert et al. (2023), ``D^2 g_b' \\dot{m} + D^2 g_a' \\dot{e} = \\mu u_\\star^3``, solved for ``\\dot{e}``:

```math
\\dot{e} = \\frac{\\mu}{g}\\,\\frac{u_\\star^3}{D^2\\,\\delta\\rho} - \\frac{\\delta\\rho_b}{\\delta\\rho}\\,\\dot{m}.
```

Note the ``D^2`` in the denominator and the factor ``\\mu`` (not ``2\\mu``). This is **not**
the form the reference LADDIE actually integrates — for that, use the default
[`LambertEntrainment`](@ref). The melt-buoyancy/detrainment term
``-(\\delta\\rho_b/\\delta\\rho)\\,\\dot{m}`` is identical in both.

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `Params(; entrainment = GasparEntrainment(2.5))`.
"""
@kwdef struct GasparEntrainment{FT} <: AbstractEntrainment
    mu::FT = 2.5
end

"""
$(TYPEDSIGNATURES)

Shear-driven entrainment following Holland & Jenkins (1999).

- `cl`: drag coefficient for entrainment velocity (default: `0.01775`).

Select via `Params(; entrainment = HollandEntrainment(0.01775))`.
"""
@kwdef struct HollandEntrainment{FT} <: AbstractEntrainment
    cl::FT = 0.01775
end
