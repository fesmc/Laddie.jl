
abstract type AbstractOpenBoundary end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `Params(; openbc = ZeroGradientInflow())` (the default).
"""
struct ZeroGradientInflow <: AbstractOpenBoundary end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: outflow only — inflow velocities are
clipped to zero so ambient water cannot advect into the domain.

Select via `Params(; openbc = NoInflow())`.
"""
struct NoInflow <: AbstractOpenBoundary end
