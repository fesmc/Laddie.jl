"""
Abstract supertype for the equation of state: the buoyancy contrast `δρ = (ρ_a − ρ)/ρ₀` of the
ambient water against the layer. Pass a concrete instance as `Params(; equation_of_state = ...)`:
[`RoquetEOS`](@ref) (the default) or [`LinearEOS`](@ref). All have the form

```
δρ = β (S_a − S) − (a₀ + a_T T̄ + a_Z Z) (T_a − T),    T̄ = ½ (T_a + T),
```

with `Z` the depth of the layer mid-point; a scheme supplies the four coefficients
([`eos_coefficients`](@ref)).
"""
abstract type AbstractEOS end

"""
$(TYPEDEF)

Linear equation of state, `δρ = β (S_a − S) − α (T_a − T)`, with constant thermal
expansion `alpha` and haline contraction `beta`, the behaviour of LADDIE v1 and v2 (whose
values are the defaults).

Select via `Params(; equation_of_state = LinearEOS())`, or `LinearEOS(; alpha = 3.8e-5)`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LinearEOS{FT} <: AbstractEOS
    "thermal expansion coefficient α (K⁻¹, default `3.733e-5`)"
    alpha::FT = 3.733e-5
    "haline contraction coefficient β (psu⁻¹, default `7.843e-4`)"
    beta::FT = 7.843e-4
end
LinearEOS(alpha, beta) = LinearEOS(promote(alpha, beta)...)

"""
$(TYPEDEF)

The simplified polynomial equation of state of [Roquet et al. (2015, their Eq. 17)](@cite roquet_defining_2015),

```
ρ′ = −(C_b/2) (Θ − Θ₀)² − T_h Z Θ + b₀ S_A,
```

the simplest form with cabbeling (the thermal expansion `α = C_b(Θ − Θ₀) + T_h Z` falls with
temperature) and thermobaricity (it rises with depth). The defaults are the paper's values,
rescaled by `ρ₀`. The default; its range of validity is discussed on the
[Equation of state](@ref) page.

Select via `Params(; equation_of_state = RoquetEOS())` (the default).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RoquetEOS{FT} <: AbstractEOS
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
    return RoquetEOS{FT}(Cb, Theta0, Th, b0)
end
RoquetEOS(Cb::Real, Theta0::Real, Th::Real, b0::Real) = RoquetEOS(promote(Cb, Theta0, Th, b0)...)

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:RoquetEOS}, Cb, Theta0, Th, b0) =
    RoquetEOS{typeof(Cb)}(Cb, Theta0, Th, b0)

"""
    eos_coefficients(m, eos) -> (; beta, a0, aT, aZ)

The coefficients of `δρ = β ΔS − (a₀ + a_T T̄ + a_Z Z) ΔT` (see [`AbstractEOS`](@ref)) under the
equation of state `eos`, as a NamedTuple of scalars built where a kernel is launched.
"""
function eos_coefficients end

# LinearEOS: a0 = alpha, aT = aZ = 0, so the contrast is the constant-coefficient one to the last bit.
eos_coefficients(m) = eos_coefficients(m, m.params.equation_of_state)
eos_coefficients(m, eos::LinearEOS) = (; beta = eos.beta, a0 = eos.alpha, aT = zero(m.FT), aZ = zero(m.FT))
function eos_coefficients(m, eos::RoquetEOS)
    rho0 = m.rho0_seawater
    return (; beta = eos.b0 / rho0, a0 = -eos.Cb * eos.Theta0 / rho0, aT = eos.Cb / rho0, aZ = eos.Th / rho0)
end

Base.show(io::IO, ::LinearEOS) = print(io, "LinearEOS()")
Base.show(io::IO, e::RoquetEOS) =
    print(io, "RoquetEOS(Cb = $(e.Cb), Theta0 = $(e.Theta0), Th = $(e.Th), b0 = $(e.b0))")

# ============================================================================
# Density contrast: the pointwise pieces, `density!` and its kernel
# ============================================================================

# Point-wise equation-of-state and freezing-point pieces shared by several kernels.
# Each spells out one formula in one operation order, so every caller rounds alike.

# Buoyancy contrast (ρ_a − ρ)/ρ₀ of water (S, T) against (Sa, Ta) at depth `Z` (positive
# downward), δρ = β ΔS − (a0 + aT T̄ + aZ Z) ΔT, with the coefficients `c` of
# `eos_coefficients`.  For the linear equation of state aT = aZ = 0 and a0 = α, which
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
density!(m) = launch!(
    _density_kernel!,
    m.drho,
    m.Sa,
    m.S.present,
    m.Ta,
    m.T.present,
    m.tmask,
    m.z_draft,
    m.H.present,
    eos_coefficients(m),
)
