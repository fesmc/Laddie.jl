"""
Abstract supertype for the equation of state: the buoyancy contrast
`δρ = (ρ_a − ρ)/ρ₀` of the ambient water against the layer, which sets the pressure
gradient, the entrainment and the convective stability.  Pass a concrete instance as
`Params(; equation_of_state = ...)`:

 - [`LinearEOS`](@ref) (the default)
 - [`RoquetEOS`](@ref)

All contrasts have the form

```
δρ = β (S_a − S) − (a₀ + a_T T̄ + a_Z Z) (T_a − T),
T̄ = ½ (T_a + T),
```

with ``Z`` the depth of the layer mid-point (positive downward), so that the thermal expansion
`α = a₀ + a_T T̄ + a_Z Z` is linear in temperature (cabbeling) and depth
(thermobaricity).  The parameterisations differ only in the four coefficients, which are scalars
computed where the kernels are launched.
"""
abstract type AbstractEquationOfState end

"""
$(TYPEDEF)

Linear equation of state, `δρ = β (S_a − S) − α (T_a − T)`, with the
constants `Params.alpha` and `Params.beta`.  The default, and the behaviour of LADDIE v1 and v2.
"""
struct LinearEOS <: AbstractEquationOfState end

"""
$(TYPEDEF)

The simplified polynomial equation of state of [Roquet et al. (2015, their Eq. 17)](@cite roquet_defining_2015),

```
ρ′ = −(C_b/2) (Θ − Θ₀)² − T_h Z Θ + b₀ S_A,
```

the simplest form that has both **cabbeling** (the thermal expansion falls with temperature) and
**thermobaricity** (it rises with depth) and whose parameters are directly interpretable:
`C_b` the sensitivity of `α` to `Θ`, `Θ₀` the temperature at which the surface
`α` vanishes, `T_h` its sensitivity to depth, `b₀` the (constant) haline contraction.  The
thermal expansion is `α = C_b(Θ − Θ₀) + T_h Z`, i.e. 0.029 kg m⁻³ K⁻¹ at the
surface freezing point (−1.9 °C), 0.041 at 500 m, and 0.068 at 0.5 °C and 500 m, where the linear
default has 0.038 throughout.  The defaults are the values of the paper.  The constants are
rescaled by `ρ₀` (`Params.rho0_seawater`) to the reduced contrast of the model.

`Params.alpha` and `Params.beta` are not used by this equation of state.

!!! warning "Experimental"
    Constructing a `RoquetEOS` logs a warning.  [Roquet et al. (2015)](@cite roquet_defining_2015) fit it to the open ocean
    (`Θ` down to −1.9 °C, `S_A` ≈ 34–35 g kg⁻¹, `Z` to 5000 m) and in Conservative
    Temperature and Absolute Salinity, which this model takes as its temperature and salinity.
    Since ``b_0`` is constant, only salinity differences enter and the offset between Absolute and
    Practical Salinity cancels; the difference between Conservative and potential temperature is
    below 0.01 K in cold water.  Supercooled cavity water is an extrapolation.  They stress that it serves understanding rather than realistic simulation,
    for which they recommend TEOS-10; what it adds here is a thermal expansion that is right
    near the freezing point, which their simulations show matters most for polar stratification.

# Example

```julia
Params(; equation_of_state = RoquetEOS())
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RoquetEOS{FT} <: AbstractEquationOfState
    "sensitivity of the thermal expansion to temperature ``C_b`` (kg m⁻³ K⁻², default `0.011`); non-negative"
    Cb::FT = 0.011
    "temperature at which the surface thermal expansion vanishes `Θ₀` (°C, default `-4.5`)"
    Theta0::FT = -4.5
    "sensitivity of the thermal expansion to depth ``T_h`` (kg m⁻⁴ K⁻¹, default `2.5e-5`)"
    Th::FT = 2.5e-5
    "haline contraction ``b_0`` (kg m⁻³ (g kg⁻¹)⁻¹, default `0.77`); positive"
    b0::FT = 0.77
    RoquetEOS{FT}(Cb, Theta0, Th, b0) where {FT} = new{FT}(Cb, Theta0, Th, b0)
end

function RoquetEOS(Cb::FT, Theta0::FT, Th::FT, b0::FT) where {FT<:Real}
    Cb >= 0 || throw(ArgumentError("RoquetEOS: Cb must be non-negative, got $Cb"))
    b0 > 0 || throw(ArgumentError("RoquetEOS: b0 must be positive, got $b0"))
    _warn_experimental("RoquetEOS", "equation of state", "LinearEOS()")
    return RoquetEOS{FT}(Cb, Theta0, Th, b0)
end
RoquetEOS(Cb::Real, Theta0::Real, Th::Real, b0::Real) = RoquetEOS(promote(Cb, Theta0, Th, b0)...)

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:RoquetEOS}, Cb, Theta0, Th, b0) =
    RoquetEOS{typeof(Cb)}(Cb, Theta0, Th, b0)

# The coefficients of the contrast `δρ = beta ΔS − (a0 + aT T̄ + aZ Z) ΔT` as the kernels
# take them, a NamedTuple of scalars (which `_val` unwraps) built where a kernel is
# launched.  For `LinearEOS` the thermal expansion is `a0 = alpha` with `aT = aZ = 0`,
# so that the contrast is that of the constant coefficients to the last bit.
_eos_coefficients(m) = _eos_coefficients(m, m.params.equation_of_state)
_eos_coefficients(m, ::LinearEOS) = (; beta = m.beta, a0 = m.alpha, aT = zero(m.FT), aZ = zero(m.FT))
function _eos_coefficients(m, eos::RoquetEOS)
    rho0 = m.rho0_seawater
    return (; beta = eos.b0 / rho0, a0 = -eos.Cb * eos.Theta0 / rho0, aT = eos.Cb / rho0, aZ = eos.Th / rho0)
end

Base.show(io::IO, ::LinearEOS) = print(io, "LinearEOS()")
Base.show(io::IO, e::RoquetEOS) =
    print(io, "RoquetEOS(Cb = $(e.Cb), Theta0 = $(e.Theta0), Th = $(e.Th), b0 = $(e.b0))")

# ============================================================================
# Density contrast: the pointwise pieces, `update_density!` and its kernel
# ============================================================================

# Point-wise equation-of-state and freezing-point pieces shared by several kernels.
# Each spells out one formula in one operation order, so every caller rounds alike.

# Buoyancy contrast (ρ_a − ρ)/ρ₀ of water (S, T) against (Sa, Ta) at depth `Z` (positive
# downward), δρ = β ΔS − (a0 + aT T̄ + aZ Z) ΔT, with the coefficients `c` of
# `_eos_coefficients`.  For the linear equation of state aT = aZ = 0 and a0 = α, which
# leaves `β ΔS − α ΔT` to the last bit.
@inline _drho(Sa, S, Ta, T, Z, c) = c.beta * (Sa - S) - (c.a0 + c.aT * (Ta + T) / 2 + c.aZ * Z) * (Ta - T)

# Depth (positive downward) of the mid-point of the layer of thickness `H` under the ice
# base at `z_draft`.
@inline _layer_depth(z_draft, H) = -z_draft + H / 2

# Liquidus: freezing temperature at salinity `S` and depth `z`, and its inverse, the
# salinity whose freezing temperature at `z` is `Tf`.  `_three_eq_melt_kernel!`
# groups `l2 + l3*z` separately and so keeps its own copy.
@inline _freezing_point(S, z, l1, l2, l3) = l1 * S + l2 + l3 * z
@inline _freezing_salinity(Tf, z, l1, l2, l3) = (Tf - l2 - l3 * z) / l1

@kernel function _density_kernel!(
    drho,
    @Const(Sa),
    @Const(S),
    @Const(Ta),
    @Const(T),
    @Const(tmask),
    @Const(z_draft),
    @Const(H),
    eos,
)
    eos = _val(eos)
    i, j = @index(Global, NTuple)
    @inbounds drho[i, j] =
        _drho(Sa[i, j], S[i, j], Ta[i, j], T[i, j], _layer_depth(z_draft[i, j], H[i, j]), eos) * tmask[i, j]
end

"Reduced (dimensionless) density `δρ = Δρ_a/ρ₀` of the ambient water against the layer, `β(S_a − S) − α(T_a − T)` for [`LinearEOS`](@ref) (Lambert et al. 2023, Eqs. 6–7)."
update_density!(m) = launch!(
    _density_kernel!,
    m.drho,
    m.Sa,
    m.S.present,
    m.Ta,
    m.T.present,
    m.tmask,
    m.z_draft,
    m.H.present,
    _eos_coefficients(m),
)
