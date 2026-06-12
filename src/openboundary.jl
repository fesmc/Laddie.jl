
abstract type AbstractOpenBoundary end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `boundop = 1` in `[Options]`.
"""
struct ZeroGradientInflow <: AbstractOpenBoundary end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: outflow only — inflow velocities are
clipped to zero so ambient water cannot advect into the domain.

Select via `boundop = 0` (or any value other than `1`) in `[Options]`.
"""
struct NoInflow <: AbstractOpenBoundary end
