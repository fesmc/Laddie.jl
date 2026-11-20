"""
Abstract supertype for frazil-ice formation: ice crystals that grow in a supercooled layer,
``T < T_f(S, z_b)``, and are deposited on the ice base. Pass a concrete instance as
`Params(; frazil_ice = ...)`: [`NoFrazilIce`](@ref) (the default) or [`RelaxedFrazilIce`](@ref).
"""
abstract type AbstractFrazilIce end

"""
$(TYPEDEF)

No frazil ice: supercooling is released only by basal freezing in the melt
parameterization.  The default, and the behaviour of LADDIE v1 and v2.
"""
struct NoFrazilIce <: AbstractFrazilIce end

"""
$(TYPEDEF)

Relax the supercooling of the layer toward zero over `relaxation_time` τ by forming frazil ice,
deposited on the ice base at once. Under the ice cover the formation rate is

```
ḟ = (c_p/L) H max(T_f − T, 0)/τ,    T_f = l₁ S + l₂ + l₃ z_b
```

(m s⁻¹ freshwater, in `model.frazil`). The frozen water leaves the layer, its salt stays and
its latent heat warms the rest. Read at the past level, so stable for `τ > dt`. The default
τ = 1 h is an order-of-magnitude choice.

Select via `Params(; frazil_ice = RelaxedFrazilIce(3600.0))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelaxedFrazilIce{FT} <: AbstractFrazilIce
    "relaxation time of the supercooling (s, default `3600`)"
    relaxation_time::FT = 3600.0
end

# ============================================================================
# `frazil!` and its kernel
# ============================================================================

"""
$(TYPEDSIGNATURES)

Frazil-ice formation rate `m.frazil` (m s⁻¹, freshwater equivalent) under
[`RelaxedFrazilIce`](@ref), on ice-covered cells only (a gap forms sea ice, not marine ice);
nothing under [`NoFrazilIce`](@ref). Called before `step_entrainment!`, whose preventive
bounds on `H` include it.
"""
frazil!(m) = frazil!(m, m.frazil_ice)
frazil!(m, ::NoFrazilIce) = nothing
function frazil!(m, fz::RelaxedFrazilIce)
    launch!(
        _frazil_kernel!,
        m.frazil,
        m.T.past,
        m.S.past,
        m.H.present,
        m.z_draft,
        m.imask,
        fz.relaxation_time,
        m.c_p,
        m.L,
        m.l1,
        m.l2,
        m.l3,
        m.params.kinks,
    )
end

@kernel function _frazil_kernel!(
    frazil,
    @Const(T),
    @Const(S),
    @Const(H),
    @Const(z_draft),
    @Const(imask),
    relaxation_time,
    c_p,
    L,
    l1,
    l2,
    l3,
    kinks,
)
    relaxation_time, c_p, L, l1, l2, l3 =
        _val(relaxation_time), _val(c_p), _val(L), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(c_p)
        supercooling = _smax0(kinks, l1 * S[i, j] + l2 + l3 * z_draft[i, j] - T[i, j], kink_width(kinks, :T))
        frazil[i, j] = imask[i, j] * c_p / L * H[i, j] * supercooling / relaxation_time
    end
end
