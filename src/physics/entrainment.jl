"""
Abstract supertype for the entrainment parameterisation — the rate at which
ambient water is mixed into the layer.  Pass a concrete instance as
`Params(; entrainment = ...)`: [`GasparEntrainment`](@ref) (the default) or
[`HollandEntrainment`](@ref).
"""
abstract type AbstractEntrainment end

"""
$(TYPEDEF)

Buoyancy-flux-driven entrainment after [Gaspar (1988)](@cite gaspar_modeling_1988), following [Gladish et al. (2012)](@cite gladish_ice-shelf_2012), as in [Lambert et al. (2023, Eq. 14)](@cite lambert_modelling_2023) and LADDIE v2 ([Lambert et al. 2026](@cite lambert_one-layer_2026),
Eq. 13). The layer mechanical-energy balance

```
(H/2) g_b′ ṁ + (H/2) g_a′ ė = μ u⋆³
```

is solved for the net entrainment `ė`:

```
ė = (2μ/g) u⋆³/(H δρ) − (δρ_b/δρ) ṁ.
```

LADDIE v1's `'Gaspar'` and v2's `'Gaspar1988'`, verified bit for bit against the v1 code; the
default.

Select via `Params(; entrainment = GasparEntrainment(; mu = 2.5))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct GasparEntrainment{FT} <: AbstractEntrainment
    "dimensionless efficiency parameter (ISOMIP+ default `2.5`)"
    mu::FT = 2.5
    "lower bound on δρ in the production term, which divides by it (–, default `1e-4`)"
    drho_floor::FT = 1e-4
    "upper bound on the detrainment rate (m s⁻¹, default `1e6`, i.e. unbounded)"
    max_detrainment::FT = 1e6
end
GasparEntrainment(mu, drho_floor, max_detrainment) =
    GasparEntrainment(promote(mu, drho_floor, max_detrainment)...)
GasparEntrainment(mu) = GasparEntrainment(; mu)

"""
$(TYPEDEF)

Shear-driven entrainment following [Holland & Jenkins (1999)](@cite holland_modeling_1999):

```
ė = c_l (K_h/A_h²) √(max(0, |u|² − g δρ (K_h/A_h) H)),
```

with `|u|` the speed at the T-point, `δρ` the reduced density
contrast and ``K_h``, ``A_h`` the lateral diffusivity and viscosity of `Params`.
There is no detrainment.

Select via `Params(; entrainment = HollandEntrainment(0.01775))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct HollandEntrainment{FT} <: AbstractEntrainment
    "entrainment coefficient (default `0.01775`)"
    cl::FT = 0.01775
end

# ============================================================================
# `step_entrainment!` and its kernels
# ============================================================================

"""
$(TYPEDSIGNATURES)

Holland–Jenkins shear entrainment (see [`HollandEntrainment`](@ref)), an
alternative to the buoyancy-flux form of [Lambert et al. (2023, Eq. 14)](@cite lambert_modelling_2023).
"""
function entrainment!(m, ep::HollandEntrainment)
    coeff = ep.cl * m.K_h / m.A_h^2
    drho_coeff = m.g * m.K_h / m.A_h
    level = m.params.closure_level
    launch_interior!(
        _holland_entrainment_kernel!,
        m.entr,
        m.detr,
        closure_field(m.U, level),
        closure_field(m.V, level),
        m.drho,
        closure_field(m.H, level),
        m.tmask,
        coeff,
        drho_coeff,
        m.params.kinks,
    )
    return nothing
end

"""
$(TYPEDSIGNATURES)

[Gaspar (1988)](@cite gaspar_modeling_1988) mechanical-energy entrainment (see [`GasparEntrainment`](@ref)):
`e = 2μ u⋆³ / (g H δρ)` minus a melt detrainment correction
([Lambert et al. 2023](@cite lambert_modelling_2023), Eq. 14; see `docs/src/physics/equations.md`).
"""
function entrainment!(m, ep::GasparEntrainment)
    launch!(
        _buoyancy_entrainment_kernel!,
        m.entr,
        m.detr,
        m.T.present,
        m.S.present,
        m.Tb,
        m.z_draft,
        m.ustar,
        closure_field(m.H, m.params.closure_level),
        m.drho,
        m.melt,
        m.tmask,
        (ep.mu + ep.mu) / m.g,
        ep.max_detrainment,
        ep.drho_floor,
        eos_coefficients(m),
        m.l1,
        m.l2,
        m.l3,
        m.params.kinks,
    )
end

"""
$(TYPEDSIGNATURES)

Compute entrainment/detrainment rates and assemble the net entrainment
`m.nentr = entr + ent2 - detr` that enters the thickness and tracer equations.
`ent2` is the extra entrainment that keeps the first guess of `H` above `H_min`, and `detr_cap`
the detrainment that brings it down to `Params.max_layer_thickness`, both from `H.past` before
the step (LADDIE v1 and v2). `detr_cap` leaves at the layer's own T, S and velocity.
"""
function step_entrainment!(m, dt)
    entrainment!(m, m.entrainment)
    upwind_advection_T(m.convH, m, m.H.present)
    FT = m.FT
    launch!(
        _net_entrainment_kernel!,
        m.ent2,
        m.nentr,
        m.detr_cap,
        m.H.past,
        m.convH,
        m.melt,
        m.entr,
        m.detr,
        m.z_draft,
        m.z_bed,
        m.tmask,
        m.frazil,
        m.params.max_layer_thickness,
        m.params.kinks,
        m.H_min,
        FT(2) * dt,
    )
    return
end

@kernel function _net_entrainment_kernel!(
    ent2,
    nentr,
    detr_cap,
    @Const(H_past),
    @Const(convH),
    @Const(melt),
    @Const(entr),
    @Const(detr),
    @Const(z_draft),
    @Const(z_bed),
    @Const(tmask),
    frazil,
    max_layer_thickness,
    kinks,
    H_min,
    dt2,
)
    H_min, dt2, frazil = _val(H_min), _val(dt2), _val(frazil)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(H_min)
        eH = kink_width(kinks, :H)
        # Frazil formation removes layer water like a sink, so the floor tops it up.
        ent2[i, j] =
            _smax0(
                kinks,
                (H_min - H_past[i, j]) / dt2 -
                (convH[i, j] + melt[i, j] - _at(frazil, i, j) + entr[i, j] - detr[i, j]),
                eH / dt2,
            ) * tmask[i, j]
        nentr[i, j] = entr[i, j] + ent2[i, j] - detr[i, j]
        # First guess of the stepped H, then the excess over the cap.  The floor
        # wins where the cap is below H_min, as in `_clamp_thickness_kernel!`.
        # Masked cells may sit under a bed of -Inf, hence the `ifelse`.
        H_star = H_past[i, j] + (convH[i, j] + melt[i, j] - _at(frazil, i, j) + nentr[i, j]) * dt2
        H_top = smooth_max(
            kinks,
            cap_layer_thickness(max_layer_thickness, kinks, H_star, z_draft[i, j], z_bed[i, j], one(FT)),
            H_min,
            eH,
        )
        # Not smoothed: H_star - H_top is zero wherever the cap does not bind.
        detr_cap[i, j] =
            ifelse(tmask[i, j] > 0, max(zero(FT), (H_star - H_top) / dt2), zero(FT))
    end
end

# Buoyancy-flux entrainment, fusing S_b, δρ_b, drho_pos, the signed rate, entr and
# detr into one pass; only entr and detr are stored.  Production term =
# prefactor·u★³/(H·δρ) with prefactor = 2μ/g (GasparEntrainment).
# drho_pos = max(drho_floor, drho) is never zero, so the δρ_b/drho_pos division is
# safe; H can be zero outside the domain, hence _safe_div for the production term.
@kernel function _buoyancy_entrainment_kernel!(
    entr,
    detr,
    @Const(T),
    @Const(S),
    @Const(Tb),
    @Const(z_draft),
    @Const(ustar),
    @Const(H),
    @Const(drho),
    @Const(melt),
    @Const(tmask),
    prefactor,
    max_detrainment,
    drho_floor,
    eos,
    l1,
    l2,
    l3,
    kinks,
)
    prefactor, max_detrainment, drho_floor, eos, l1, l2, l3 =
        _val(prefactor), _val(max_detrainment), _val(drho_floor), _val(eos), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(prefactor)
        drho_pos = smooth_max(kinks, drho_floor, drho[i, j], kink_width(kinks, :drho))
        sb = _freezing_salinity(Tb[i, j], z_draft[i, j], l1, l2, l3)
        db_ij = _drho(S[i, j], sb, T[i, j], Tb[i, j], -z_draft[i, j], eos) * tmask[i, j]
        us3 = ustar[i, j]^3
        e_ij =
            prefactor * _safe_div(us3, H[i, j] * drho_pos) -
            db_ij / drho_pos * melt[i, j] * tmask[i, j]
        ew = kink_width(kinks, :w)
        entr[i, j] = _smax0(kinks, e_ij, ew)
        detr[i, j] = smooth_min(kinks, max_detrainment, _smax0(kinks, -e_ij, ew), ew)
    end
end

# Holland–Jenkins entrainment fused with im/jm: avoids two circshift allocations.
@kernel function _holland_entrainment_kernel!(
    entr,
    detr,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(H),
    @Const(tmask),
    coeff,
    drho_coeff,
    kinks,
)
    coeff, drho_coeff = _val(coeff), _val(drho_coeff)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(coeff)
        half = FT(0.5)
        im1 = i - 1
        jm1 = j - 1
        u_im = (U[i, j] + U[im1, j]) * half
        v_jm = (V[i, j] + V[i, jm1]) * half
        speed_sq = u_im * u_im + v_jm * v_jm - drho_coeff * drho[i, j] * H[i, j]
        entr[i, j] = coeff * _ssqrt(kinks, speed_sq, kink_width(kinks, :u)) * tmask[i, j]
        detr[i, j] = zero(FT)
    end
end
