"""
Abstract supertype for frazil-ice formation: ice crystals that grow in a
supercooled layer, i.e. where ``T < T_f(S, z_b)``, and are deposited on the ice
base.  The three-equation melt can already freeze directly at the interface, but in
real Ice Shelf Water plumes most of the supercooling is released by frazil instead
([Jenkins & Bombosch 1995](@cite jenkins_modeling_1995); [Smedsrud & Jenkins 2004](@cite smedsrud_frazil_2004)).  Pass a concrete instance as
`Params(; frazil_ice = ...)`: [`NoFrazilIce`](@ref) (the default) or
[`RelaxedFrazilIce`](@ref).
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

Relax the supercooling of the layer toward zero over `relaxation_time` by forming
frazil ice, which is deposited on the ice base at once (no suspended crystals, no
new prognostic).  Under the ice cover (`imask`), the formation rate is

```
ḟ = (c_p/L) H max(T_f − T, 0)/τ,
T_f = l₁ S + l₂ + l₃ z_b,
```

a freshwater-equivalent rate in m s⁻¹ like `melt` (available as `model.frazil`).
The water that freezes leaves the layer at its own T and velocity, its salt stays
behind, and its latent heat `L ḟ` warms the rest.  So T relaxes toward
`T_f` at the rate `1/τ`, S rises by brine rejection, and momentum per unit
mass is unchanged.  ``L`` is used, not the ``L - c_i T_i`` of the melt kernel: the
crystals form at the freezing point inside the water, and cooling the deposited
ice to the shelf temperature is a matter for the ice, not the layer.

The supercooling is read at the past leapfrog level, like the convective
relaxation of [`RelaxToAmbient`](@ref), so the step is stable for
`relaxation_time > dt` and does not overshoot ``T_f`` for `relaxation_time > 2dt`.

The default `τ` = 1 h is an order-of-magnitude choice.  The literature has no
relaxation time for a depth-integrated plume.  [Rees Jones & Wells (2018, The
Cryosphere 12, 25–38)](@cite rees_jones_frazil-ice_2018) find that frazil adjusts to its forcing over a few hundred
seconds in a 1 m layer, a time proportional to the layer depth, which gives about an
hour for a 10 m layer.  [Smedsrud & Jenkins (2004)](@cite smedsrud_frazil_2004) find a residual supercooling of
~0.02 °C that frazil growth leaves in place for up to a day.  Ocean models often
remove the supercooling within one step (`τ → Δt`).

Select via `Params(; frazil_ice = RelaxedFrazilIce(3600.0))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelaxedFrazilIce{FT} <: AbstractFrazilIce
    "relaxation time of the supercooling (s, default `3600`)"
    relaxation_time::FT = 3600.0
end

# ============================================================================
# `update_frazil!` and its kernel
# ============================================================================

"""
$(TYPEDSIGNATURES)

Frazil-ice formation rate `m.frazil` (m s⁻¹, freshwater equivalent) under
[`RelaxedFrazilIce`](@ref): the rate whose latent heat relaxes the supercooling of
the past level, `max(T_f − T, 0)`, over `relaxation_time`.  Ice-covered cells
only (`imask`): a gap cell is open water, and the ice it would form is sea ice, not
marine ice.  Nothing to do under [`NoFrazilIce`](@ref), where `m.frazil` is a scalar
zero.  Computed before `update_entrainment!`, whose preventive bounds on H include
it.
"""
update_frazil!(m) = update_frazil!(m, m.frazil_ice)
update_frazil!(m, ::NoFrazilIce) = nothing
function update_frazil!(m, fz::RelaxedFrazilIce)
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
        supercooling = _smax0(kinks, l1 * S[i, j] + l2 + l3 * z_draft[i, j] - T[i, j], _kink_width(kinks, :T))
        frazil[i, j] = imask[i, j] * c_p / L * H[i, j] * supercooling / relaxation_time
    end
end
