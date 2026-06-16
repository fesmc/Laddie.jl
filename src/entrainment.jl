
abstract type AbstractEntrainment end

"""
$(TYPEDSIGNATURES)

Buoyancy-flux-driven entrainment as implemented in the reference LADDIE model
(Lambert et al. 2023; Gaspar 1988; Gladish et al. 2012). This is the form
verified bit-for-bit against the Python reference and is the **default**.

The net entrainment solves the layer mechanical-energy balance for `ė`,

```math
ė = \\frac{2μ}{g}\\,\\frac{u_⋆^3}{D\\,δρ} - \\frac{δρ_b}{δρ}\\,ṁ,
```

with a single power of `D` and a factor `2μ` in the production term. This
differs from a literal reading of Eq. 14 (see [`GasparEntrainment`](@ref) and
`docs/src/equations.md`).

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `Params(; entpar = LambertEntrainment(2.5))`.
"""
struct LambertEntrainment{FT} <: AbstractEntrainment
    ;
    mu::FT;
end

"""
$(TYPEDSIGNATURES)

Buoyancy-flux-driven entrainment as the **literal reading of Eq. 14** of
Lambert et al. (2023), `D² g_b' ṁ + D² g_a' ė = μ u_⋆³`, solved for `ė`:

```math
ė = \\frac{μ}{g}\\,\\frac{u_⋆^3}{D^2\\,δρ} - \\frac{δρ_b}{δρ}\\,ṁ.
```

Note the `D²` in the denominator and the factor `μ` (not `2μ`). This is **not**
the form the reference LADDIE actually integrates — for that, use the default
[`LambertEntrainment`](@ref). The melt-buoyancy/detrainment term
`−(δρ_b/δρ)·ṁ` is identical in both.

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `Params(; entpar = GasparEntrainment(2.5))`.
"""
struct GasparEntrainment{FT} <: AbstractEntrainment
    ;
    mu::FT;
end

"""
$(TYPEDSIGNATURES)

Shear-driven entrainment following Holland & Jenkins (1999).

- `cl`: drag coefficient for entrainment velocity (default: `0.01775`).

Select via `Params(; entpar = HollandEntrainment(0.01775))`.
"""
struct HollandEntrainment{FT} <: AbstractEntrainment
    ;
    cl::FT;
end
