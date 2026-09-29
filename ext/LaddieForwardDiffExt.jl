module LaddieForwardDiffExt

using Laddie
using ForwardDiff: ForwardDiff, Dual

# Strip nested duals too, so the dt controller and the output see plain floats.
@inline Laddie._primal(x::Dual) = Laddie._primal(ForwardDiff.value(x))

# The derivative of sqrt is infinite at zero, and ForwardDiff propagates the NaN of
# `0 × Inf` even when the argument's own partials vanish (a fluid at rest).  Every
# such zero in Laddie is a speed or a clipped discriminant whose product with the
# rest of its term has a zero derivative, so return an exact zero there.
@inline Laddie._safe_sqrt(x::Dual) = iszero(Laddie._primal(x)) ? zero(x) : sqrt(x)

end
