
abstract type AbstractEntrainmentParam end

"""
$(TYPEDSIGNATURES)

Buoyancy-flux-driven entrainment following Gaspar (1988) as used in
Gladish et al. (2012) and Lambert et al. (2023, Eq. 11).

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `entpar = "Gaspar"` and `mu = 2.5` in the `[Parameters]` config section.
"""
struct GasparEntrainment{FT} <: AbstractEntrainmentParam
    ;
    mu::FT;
end

"""
$(TYPEDSIGNATURES)

Shear-driven entrainment following Holland & Jenkins (1999).

- `cl`: drag coefficient for entrainment velocity (default: `0.01775`).

Select via `entpar = "Holland"` in the `[Parameters]` config section.
"""
struct HollandEntrainment{FT} <: AbstractEntrainmentParam
    ;
    cl::FT;
end