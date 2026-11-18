"""
Abstract supertype for the heat conducted into the ice at the ice–ocean interface,
the ice-side term of the interface heat budget ([Wiskandt & Jourdain 2025](@cite wiskandt_brief_2025), Eqs. 1–2):

```
ṁ L = c_p γ_T (T − T_b) + q_cond,
```

with `q_cond` (per unit mass of water, like the other terms) the conduction
into the ice, negative for a surface colder than the base.  Pass a concrete instance
as `Params(; ice_conduction = ...)`:

 - [`MeltOnlyConduction`](@ref) (the default)
 - [`HollandJenkinsConduction`](@ref), the parameterisation of LADDIE v1 and v2
 - [`NoConduction`](@ref)
 - [`LinearProfileConduction`](@ref)

`NoConduction` and `LinearProfileConduction` are experimental and warn when constructed.  The ice
temperature ``T_i`` they use is the `T_ice_farfield` of the ice forcing
([`PrescribedIceForcing`](@ref)), which plays the role of the ice surface
temperature ``T_s`` of the paper.
"""
abstract type AbstractIceConduction end

_warn_experimental_conduction(name) =
    _warn_experimental(name, "ice-conduction parameterisation", "MeltOnlyConduction()")

"""
$(TYPEDEF)

[Holland & Jenkins (1999)](@cite holland_modeling_1999) linearisation of the heat advected by the ice, their melt-rate
dependent gradient: `q_cond = ṁ c_i (T_b − T_i)`, so the latent heat
becomes `L_eff = L − c_i (T_i − T_b)`.  Applied for melting and refreezing
alike.  The behaviour of LADDIE v1 and v2.

It is Eq. (2C) of [Wiskandt & Jourdain (2025)](@cite wiskandt_brief_2025) without the cut-off at `ṁ = 0`, see
[`MeltOnlyConduction`](@ref).
"""
struct HollandJenkinsConduction <: AbstractIceConduction end

"""
$(TYPEDEF)

The parameterisation of [`HollandJenkinsConduction`](@ref), but with no conduction into the ice
where the layer is not melting it: `q_cond = ṁ c_i (T_b − T_i)` for
`ṁ > 0`, zero otherwise ([Wiskandt & Jourdain 2025](@cite wiskandt_brief_2025), Eq. 2C with
`max(ṁ, 0)`).  Their argument is that a refreezing ice base carries marine
ice close to the freezing point throughout, so there is no gradient to conduct along.

The switch is the sign of the thermal forcing ``T - T_f(S, z_b)``, which has the sign
of `ṁ` and needs no iteration of the melt equations.  Where it is not positive,
the latent heat is the plain ``L``.

Select via `Params(; ice_conduction = MeltOnlyConduction())` (the default).
"""
struct MeltOnlyConduction <: AbstractIceConduction end

"""
$(TYPEDEF)

No conduction into the ice, `q_cond = 0` ([Wiskandt & Jourdain 2025](@cite wiskandt_brief_2025), Eq. 2A),
so `L_eff = L` and `T_ice_farfield` has no effect.  It is a good approximation only
for a base of marine ice at the freezing point.  Overestimates the melt by
``c_i (T_s - T_f)/L``, about 11 % for ``T_s = -20`` °C, compared with
[`MeltOnlyConduction`](@ref).

Select via `Params(; ice_conduction = NoConduction())`; warns when constructed.
"""
struct NoConduction <: AbstractIceConduction
    function NoConduction()
        _warn_experimental_conduction("NoConduction")
        return new()
    end
end

"""
$(TYPEDEF)

Steady linear temperature profile through the ice, with no advection ([Wiskandt & Jourdain
2025](@cite wiskandt_brief_2025), Eq. 2B; used by Losch 2008, Holland et al. 2008):

```
q_cond = (k_i (T_f − T_s))/(ρ_fw H_i),
```

with `T_f` the freezing point at the layer salinity (so that `q_cond` does not
depend on `ṁ` and the melt equation stays quadratic), `T_s` the `T_ice_farfield` of the
ice forcing, and ``H_i`` the ice thickness, taken from the draft by hydrostatic
equilibrium, `H_i = −z_b ρ₀/ρ_ice` with `Params.rho0_seawater` and
`Params.rho_ice`.  The flux acts for melting and refreezing alike.  A thermal steady
state between base and surface takes of order `(H_i/2)²/κ_i` (about 1000 years for
400 m), which only the large, slow ice shelves reach.

Select via `Params(; ice_conduction = LinearProfileConduction())`; warns when constructed.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LinearProfileConduction{FT} <: AbstractIceConduction
    "thermal conductivity of ice `k_i = ρ_i c_i κ_i` (W m⁻¹ K⁻¹, default `2.1`; Jenkins et al. 2010)"
    conductivity::FT = 2.1
    function LinearProfileConduction{FT}(conductivity) where {FT}
        _warn_experimental_conduction("LinearProfileConduction")
        return new{FT}(conductivity)
    end
end

LinearProfileConduction(conductivity::FT) where {FT} = LinearProfileConduction{FT}(conductivity)

# What the melt kernels need of a parameterisation, as three scalars so that no object has to
# cross into a kernel: the factors on `c_i` where the thermal forcing T − T_f is
# positive (melting) and where it is not, and the conductivity of the fixed flux.
_conduction_coefficients(::HollandJenkinsConduction, FT) = (one(FT), one(FT), zero(FT))
_conduction_coefficients(::MeltOnlyConduction, FT) = (one(FT), zero(FT), zero(FT))
_conduction_coefficients(::NoConduction, FT) = (zero(FT), zero(FT), zero(FT))
_conduction_coefficients(c::LinearProfileConduction, FT) = (zero(FT), zero(FT), c.conductivity)

Base.show(io::IO, ::HollandJenkinsConduction) = print(io, "HollandJenkinsConduction()")
Base.show(io::IO, ::MeltOnlyConduction) = print(io, "MeltOnlyConduction()")
Base.show(io::IO, ::NoConduction) = print(io, "NoConduction()")
Base.show(io::IO, c::LinearProfileConduction) =
    print(io, "LinearProfileConduction(conductivity = $(c.conductivity) W/m/K)")

# ============================================================================
# Conduction at one cell, for the melt kernels
# ============================================================================

# Conduction into the ice at one cell (see `AbstractIceConduction`): the effective
# `c_i` of the melt-proportional term, and the fixed flux `q` (J kg⁻¹ m s⁻¹, per unit
# mass of freshwater like the melt) that the heat balance carries on top of it.  `Tf`
# is the freezing point at the layer salinity, `T - Tf` having the sign of the melt,
# `Ts` the ice temperature of the forcing.  The ice thickness is the hydrostatic one
# from the draft; `_safe_div` keeps cells without ice at zero.  The switch of c_i at
# the freezing point is blended over ε_T under `SmoothKinks`.
@inline function _ice_conduction(conduction, c_i, T, Tf, Ts, z_draft)
    k = conduction.kinks
    ci_eff = c_i * _sswitch(k, T - Tf, conduction.ci_melt, conduction.ci_freeze, _kink_width(k, :T))
    Hi = -z_draft * conduction.thick_ratio
    q = _safe_div(conduction.k_ice * (Tf - Ts), Hi * conduction.rho_fw)
    return ci_eff, q
end

# The scalars `_ice_conduction` reads, for the kernel launches, and the kink treatment,
# which the melt kernels take from here.
function _conduction_args(m)
    ci_melt, ci_freeze, k_ice = _conduction_coefficients(m.params.ice_conduction, m.FT)
    return (; ci_melt, ci_freeze, k_ice, thick_ratio = m.rho0_seawater / m.rho_ice, rho_fw = m.rho_freshwater,
            kinks = m.params.kinks)
end
