
abstract type AbstractEntrainmentParam end

"""
$(TYPEDSIGNATURES)

Buoyancy-flux-driven entrainment following Gaspar (1988) as used in
Gladish et al. (2012) and Lambert et al. (2023, Eq. 11).

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `Params(; entpar = GasparEntrainment(2.5))`.
"""
struct GasparEntrainment{FT} <: AbstractEntrainmentParam
    ;
    mu::FT;
end

"""
$(TYPEDSIGNATURES)

Shear-driven entrainment following Holland & Jenkins (1999).

- `cl`: drag coefficient for entrainment velocity (default: `0.01775`).

Select via `Params(; entpar = HollandEntrainment(0.01775))`.
"""
struct HollandEntrainment{FT} <: AbstractEntrainmentParam
    ;
    cl::FT;
end