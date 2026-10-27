"""
Abstract supertype for the heat conducted into the ice at the ice–ocean interface,
the ice-side term of the interface heat budget (Wiskandt & Jourdain 2025, Eqs. 1–2):

```math
\\dot m\\,L = c_p\\,\\gamma_T\\,(T - T_b) + q_\\text{cond},
```

with ``q_\\text{cond}`` (per unit mass of water, like the other terms) the conduction
into the ice, negative for a surface colder than the base.  Pass a concrete instance
as `Params(; ice_conduction = ...)`:

 - [`HollandJenkinsConduction`](@ref) (the default)
 - [`MeltOnlyConduction`](@ref)
 - [`NoConduction`](@ref)
 - [`LinearProfileConduction`](@ref)

All except the default are experimental and warn when constructed.  The ice
temperature ``T_i`` they use is the `T_ice_base` of the ice forcing
([`PrescribedIceForcing`](@ref)), which plays the role of the ice surface
temperature ``T_s`` of the paper.
"""
abstract type AbstractIceConduction end

_warn_experimental_conduction(name) =
    _warn_experimental(name, "ice-conduction closure", "HollandJenkinsConduction()")

"""
$(TYPEDEF)

Holland & Jenkins (1999) linearisation of the heat advected by the ice, their melt-rate
dependent gradient: ``q_\\text{cond} = \\dot m\\,c_i\\,(T_b - T_i)``, so the latent heat
becomes ``L_\\text{eff} = L - c_i (T_i - T_b)``.  Applied for melting and refreezing
alike.  The default, and the behaviour of LADDIE v1 and v2.

It is Eq. (2C) of Wiskandt & Jourdain (2025) without the cut-off at ``\\dot m = 0``, see
[`MeltOnlyConduction`](@ref).
"""
struct HollandJenkinsConduction <: AbstractIceConduction end

"""
$(TYPEDEF)

The closure of [`HollandJenkinsConduction`](@ref), but with no conduction into the ice
where the layer is not melting it: ``q_\\text{cond} = \\dot m\\,c_i\\,(T_b - T_i)`` for
``\\dot m > 0``, zero otherwise (Wiskandt & Jourdain 2025, Eq. 2C with
``\\max\\{\\dot m, 0\\}``).  Their argument is that a refreezing ice base carries marine
ice close to the freezing point throughout, so there is no gradient to conduct along.

The switch is the sign of the thermal forcing ``T - T_f(S, z_b)``, which has the sign
of ``\\dot m`` and needs no iteration of the melt equations.  Where it is not positive,
the latent heat is the plain ``L``.

Select via `Params(; ice_conduction = MeltOnlyConduction())`; warns when constructed.
"""
struct MeltOnlyConduction <: AbstractIceConduction
    function MeltOnlyConduction()
        _warn_experimental_conduction("MeltOnlyConduction")
        return new()
    end
end

"""
$(TYPEDEF)

No conduction into the ice, ``q_\\text{cond} = 0`` (Wiskandt & Jourdain 2025, Eq. 2A),
so ``L_\\text{eff} = L`` and `T_ice_base` has no effect.  It is a good approximation only
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

Steady linear temperature profile through the ice, with no advection (Wiskandt & Jourdain
2025, Eq. 2B; used by Losch 2008, Holland et al. 2008):

```math
q_\\text{cond} = \\frac{k_i\\,(T_f - T_s)}{\\rho_\\text{fw}\\,H_i},
```

with ``T_f`` the freezing point at the layer salinity (so that ``q_\\text{cond}`` does not
depend on ``\\dot m`` and the melt equation stays quadratic), ``T_s`` the `T_ice_base` of the
ice forcing, and ``H_i`` the ice thickness, taken from the draft by hydrostatic
equilibrium, ``H_i = -z_b\\,\\rho_0/\\rho_\\text{ice}`` with `Params.rho0_seawater` and
`Params.rho_ice`.  The flux acts for melting and refreezing alike.  A thermal steady
state between base and surface takes of order ``(H_i/2)^2/\\kappa_i`` (about 1000 years for
400 m), which only the large, slow ice shelves reach.

Select via `Params(; ice_conduction = LinearProfileConduction())`; warns when constructed.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LinearProfileConduction{FT} <: AbstractIceConduction
    "thermal conductivity of ice ``k_i = \\rho_i c_i \\kappa_i`` (W m⁻¹ K⁻¹, default `2.1`; Jenkins et al. 2010)"
    conductivity::FT = 2.1
    function LinearProfileConduction{FT}(conductivity) where {FT}
        _warn_experimental_conduction("LinearProfileConduction")
        return new{FT}(conductivity)
    end
end

LinearProfileConduction(conductivity::FT) where {FT} = LinearProfileConduction{FT}(conductivity)

# What the melt kernels need of a closure, as three scalars so that no object has to
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
