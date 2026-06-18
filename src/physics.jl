@kernel function _freezing_point_kernel!(Tf, @Const(S), @Const(zb), l1, l2, l3)
    i, j = @index(Global, NTuple)
    @inbounds Tf[i, j] = l1 * S[i, j] + l2 + l3 * zb[i, j]
end

@kernel function _density_kernel!(
    drho,
    @Const(Sa),
    @Const(S),
    @Const(Ta),
    @Const(T),
    @Const(tmask),
    beta,
    alpha,
)
    i, j = @index(Global, NTuple)
    @inbounds drho[i, j] =
        (beta * (Sa[i, j] - S[i, j]) - alpha * (Ta[i, j] - T[i, j])) * tmask[i, j]
end

# Three-equation melt parameterisation (Jenkins 1991) + ice-base temperature.
# Fixed-transfer-coefficient (gamT scalar) case only; variable-gamT falls back
# to the broadcast path.
@kernel function _three_eq_melt_kernel!(
    melt,
    Tb,
    @Const(T),
    @Const(S),
    @Const(zb),
    @Const(tmask),
    gamT,
    gamS,
    cp_over_Leff,
    ci_over_cp,
    l1,
    l2,
    l3,
)
    i, j = @index(Global, NTuple)
    FT = typeof(gamT)
    @inbounds begin
        Tf_depth = l2 + l3 * zb[i, j]
        quad_b =
            cp_over_Leff * gamT * (Tf_depth - T[i, j]) +
            gamS * (1 + cp_over_Leff * ci_over_cp * (Tf_depth + l1 * S[i, j]))
        quad_c = cp_over_Leff * gamT * gamS * (Tf_depth - T[i, j] + l1 * S[i, j])
        disc = quad_b * quad_b - FT(4) * quad_c
        disc = ifelse(disc < zero(FT), zero(FT), disc)
        # TODO check / 2
        melt_rate = (-quad_b + sqrt(disc)) / (FT(1) + FT(1))
        melt[i, j] = iszero(tmask[i, j]) ? zero(FT) : melt_rate
        Tb_denom = cp_over_Leff * gamT + cp_over_Leff * ci_over_cp * melt_rate
        Tb[i, j] =
            (iszero(tmask[i, j]) || iszero(Tb_denom)) ? zero(FT) :
            (cp_over_Leff * gamT * T[i, j] - melt_rate) / Tb_denom
    end
end

# Matrix-gamT variant for TurbulentGamT: reads per-element gamT[i,j] / gamS[i,j].
# TODO rm redundancy with previous function?
@kernel function _three_eq_melt_mat_gamT_kernel!(
    melt,
    Tb,
    @Const(T),
    @Const(S),
    @Const(zb),
    @Const(tmask),
    @Const(gamT),
    @Const(gamS),
    cp_over_Leff,
    ci_over_cp,
    l1,
    l2,
    l3,
)
    i, j = @index(Global, NTuple)
    FT = typeof(cp_over_Leff)
    @inbounds begin
        gT = gamT[i, j]
        gS = gamS[i, j]
        Tf_depth = l2 + l3 * zb[i, j]
        quad_b =
            cp_over_Leff * gT * (Tf_depth - T[i, j]) +
            gS * (one(FT) + cp_over_Leff * ci_over_cp * (Tf_depth + l1 * S[i, j]))
        quad_c = cp_over_Leff * gT * gS * (Tf_depth - T[i, j] + l1 * S[i, j])
        disc = quad_b * quad_b - FT(4) * quad_c
        disc = ifelse(disc < zero(FT), zero(FT), disc)
        melt_rate = (-quad_b + sqrt(disc)) / (FT(1) + FT(1))
        melt[i, j] = iszero(tmask[i, j]) ? zero(FT) : melt_rate
        Tb_denom = cp_over_Leff * gT + cp_over_Leff * ci_over_cp * melt_rate
        Tb[i, j] =
            (iszero(tmask[i, j]) || iszero(Tb_denom)) ? zero(FT) :
            (cp_over_Leff * gT * T[i, j] - melt_rate) / Tb_denom
    end
end

@kernel function _ambient_interp_kernel!(
    Ta,
    Sa,
    @Const(zb),
    @Const(D),
    @Const(Tz),
    @Const(Sz),
    z0,
    dz,
    nz,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(z0)
        depth_idx = -z0 + (zb[i, j] - D[i, j]) / dz
        # Guard against non-finite or out-of-range depth_idx before the integer
        # conversion: trunc(Int, x) throws InexactError on CPU and is UB on GPU
        # when x overflows Int64 (~9.2e18).  Clamp to [0, nz-1] in float first
        # so trunc always receives a representable value.
        depth_idx = ifelse(isfinite(depth_idx), depth_idx, zero(FT))
        depth_idx = clamp(depth_idx, zero(FT), FT(nz - 1))
        idx_lo = trunc(Int, depth_idx)
        idx_hi = clamp(idx_lo + 1, 0, nz - 1)
        weight = depth_idx - FT(idx_lo)
        Ta[i, j] = weight * Tz[idx_hi+1] + (one(FT) - weight) * Tz[idx_lo+1]
        Sa[i, j] = weight * Sz[idx_hi+1] + (one(FT) - weight) * Sz[idx_lo+1]
    end
end


"""
$(TYPEDSIGNATURES)

Vertically interpolate the ambient T/S profiles to the depth of each grid cell's
plume base (zb − D), writing results into `m.Ta` and `m.Sa`. This sampling has
no numbered equation in Lambert et al. (2023); see `docs/src/equations.md`.
"""
function update_ambient_fields!(m)
    nz = length(m.z)
    launch!(
        _ambient_interp_kernel!,
        m.Ta,
        m.Ta,
        m.Sa,
        m.zb,
        m.D.present,
        m.Tz,
        m.Sz,
        m.z0,
        m.dz,
        nz,
    )
    return
end

"Linear liquidus: Tf = l1·S + l2 + l3·zb  (Lambert et al. 2023, Eq. 10)."
update_freezing_temperature!(m) =
    launch!(_freezing_point_kernel!, m.Tf, m.Tf, m.S.present, m.zb, m.l1, m.l2, m.l3)

"Reduced (dimensionless) density δρ = Δρₐ/ρ₀ = β·(Sa − S) − α·(Ta − T)  (Lambert et al. 2023, Eqs. 6–7)."
update_density!(m) = launch!(
    _density_kernel!,
    m.drho,
    m.drho,
    m.Sa,
    m.S.present,
    m.Ta,
    m.T.present,
    m.tmask,
    m.beta,
    m.alpha,
)

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells and clamp δρ to a minimum positive value
so the plume remains denser than ambient.
"""
function update_convection!(m, c_p::ClampDensity)
    thr = c_p.mindrho / m.rho0_seawater
    @. m.convection = m.drho < 0
    @. m.drho = max(m.drho, thr)
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells, then instantly reset their T/S to ambient
values so the density remains stable.
"""
function update_convection!(m, c_p::ResetToAmbient)
    thr = c_p.mindrho / m.rho0_seawater
    S_adj = c_p.mindrho / (m.rho0_seawater * m.beta)
    @. m.convection = m.drho < 0
    @. m.T.present = ifelse(m.drho < thr, m.Ta, m.T.present)
    @. m.S.present = ifelse(m.drho < thr, m.Sa - S_adj, m.S.present)
    update_density!(m)
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells; relaxation is applied implicitly during the
tracer time step via `conv2`.
"""
function update_convection!(m, ::RelaxToAmbient)
    m.convection .= m.drho .< 0
end

update_convection!(m) = update_convection!(m, m.convection_scheme)

function _compute_turbulent_transfer_coefficients!(m, mp::TurbulentGamT)
    FT = typeof(mp.Pr)
    PrCorr = FT(12.5) * mp.Pr^(FT(2)/FT(3)) - FT(8.68)
    ScCorr = FT(12.5) * mp.Sc^(FT(2)/FT(3)) - FT(8.68)
    nu0 = mp.nu0
    # @show extrema(m.ustar), extrema(m.D.present)
    # @show nu0, PrCorr, ScCorr
    @. m.gamT = ifelse(
        m.tmask > 0,
        m.ustar / (FT(2.12) * log(m.ustar * m.D.present / nu0 + FT(1e-12)) + PrCorr),
        zero(FT),
    )
    @. m.gamS = ifelse(
        m.tmask > 0,
        m.ustar / (FT(2.12) * log(m.ustar * m.D.present / nu0 + FT(1e-12)) + ScCorr),
        zero(FT),
    )
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with a fixed heat transfer
coefficient γ_T (Jenkins 1991; Lambert et al. 2023, Eqs. 8–10 and 13).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::FixedGamT)
    FT = m.FT
    ny, nx = size(m.ustar)
    launch!(
        _ustar_kernel!,
        m.ustar,
        m.ustar,
        m.U.present,
        m.V.present,
        m.tmask,
        m.C_d_top,
        m.u_tide,
        ny,
        nx,
    )
    cp_over_Leff = m.c_p / (m.L - m.c_i * m.T_i)
    ci_over_cp = m.c_i / m.c_p
    m.gamT = mp.gamTfix
    m.gamS = m.gamT / FT(35)    # TODO could be 35
    launch!(
        _three_eq_melt_kernel!,
        m.melt,
        m.melt,
        m.Tb,
        m.T.present,
        m.S.present,
        m.zb,
        m.tmask,
        m.gamT,
        m.gamS,
        cp_over_Leff,
        ci_over_cp,
        m.l1,
        m.l2,
        m.l3,
    )
end

function update_melt!(m, mp::PrescribedMelt)
    @. m.melt = 0
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with turbulence-dependent
transfer coefficients γ_T, γ_S via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 8–13).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::TurbulentGamT)
    ny, nx = size(m.ustar)
    launch!(
        _ustar_kernel!,
        m.ustar,
        m.ustar,
        m.U.present,
        m.V.present,
        m.tmask,
        m.C_d_top,
        m.u_tide,
        ny,
        nx,
    )
    cp_over_Leff = m.c_p / (m.L - m.c_i * m.T_i)
    ci_over_cp = m.c_i / m.c_p
    _compute_turbulent_transfer_coefficients!(m, mp)
    launch!(
        _three_eq_melt_mat_gamT_kernel!,
        m.melt,
        m.melt,
        m.Tb,
        m.T.present,
        m.S.present,
        m.zb,
        m.tmask,
        m.gamT,
        m.gamS,
        cp_over_Leff,
        ci_over_cp,
        m.l1,
        m.l2,
        m.l3,
    )
end

update_melt!(m) = update_melt!(m, m.melting)

"""
$(TYPEDSIGNATURES)

Holland–Jenkins shear entrainment: e ∝ √max(0, |u|² − gₐ′·Kh/Ah·D)
(Holland & Jenkins 1999). This is an alternative to the Gaspar scheme of
Lambert et al. (2023, Eq. 14); see `docs/src/equations.md`.
"""
function _compute_entrainment!(m, ep::HollandEntrainment)
    coeff = ep.cl * m.K_h / m.A_h^2
    drho_coeff = m.g * m.K_h / m.A_h
    ny, nx = size(m.entr)
    launch!(
        _holland_entrainment_kernel!,
        m.entr,
        m.entr,
        m.detr,
        m.U.present,
        m.V.present,
        m.drho,
        m.D.present,
        m.tmask,
        coeff,
        drho_coeff,
        ny,
        nx,
    )
end

"""
$(TYPEDSIGNATURES)

Reference-LADDIE mechanical-energy entrainment: e = 2μ·u★³/(g·D·δρ) minus a melt
detrainment correction. This is the form verified against the Python reference
(Lambert et al. 2023; see `docs/src/equations.md`). Contrast
[`_compute_entrainment!`](@ref)`(m, ::GasparEntrainment)`, the literal Eq. 14.
"""
function _compute_entrainment!(m, ep::LambertEntrainment)
    mu2_over_g = (ep.mu + ep.mu) / m.g
    launch!(
        _lambert_entrainment_kernel!,
        m.entr,
        m.Sb,
        m.drhob,
        m.ent,
        m.entr,
        m.detr,
        m.T.present,
        m.S.present,
        m.Tb,
        m.zb,
        m.ustar,
        m.D.present,
        m.drho,
        m.melt,
        m.tmask,
        mu2_over_g,
        m.max_detrainment,
        m.alpha,
        m.beta,
        m.l1,
        m.l2,
        m.l3,
    )
end

"""
$(TYPEDSIGNATURES)

Literal Eq. 14 entrainment: e = μ·u★³/(g·D²·δρ) minus the same melt detrainment
correction (Lambert et al. 2023, Eq. 14; see `docs/src/equations.md`). Differs
from the reference [`LambertEntrainment`](@ref) by the `D²` denominator and the
`μ` (not `2μ`) prefactor.
"""
function _compute_entrainment!(m, ep::GasparEntrainment)
    mu_over_g = ep.mu / m.g
    launch!(
        _gaspar_entrainment_kernel!,
        m.entr,
        m.Sb,
        m.drhob,
        m.ent,
        m.entr,
        m.detr,
        m.T.present,
        m.S.present,
        m.Tb,
        m.zb,
        m.ustar,
        m.D.present,
        m.drho,
        m.melt,
        m.tmask,
        mu_over_g,
        m.max_detrainment,
        m.alpha,
        m.beta,
        m.l1,
        m.l2,
        m.l3,
    )
end

"""
$(TYPEDSIGNATURES)

Compute entrainment/detrainment rates and assemble the net entrainment
`m.nentr = entr − detr` that enters the thickness and tracer equations.
`ent2` (the mass rate from saturating D at `D_min`) is computed separately
in `_clamp_thickness!` after the thickness step so that T/S are not
perturbed by the floor correction.
"""
function update_entrainment!(m)
    _compute_entrainment!(m, m.entrainment)
    upwind_advection_T(m.convD, m, m.D.present)
    @. m.nentr = m.entr - m.detr
    return
end

# Friction velocity at the T-point: |u|_T = √(C_d_top · (im_half(U)² + jm_half(V)² + u_tide²))
# im_half(U)[i,j] = (U[i,j] + U[i,j−1]) / 2,  jm_half(V)[i,j] = (V[i,j] + V[i−1,j]) / 2
@kernel function _ustar_kernel!(
    ustar,
    @Const(U),
    @Const(V),
    @Const(tmask),
    C_d_top,
    u_tide,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(C_d_top)
        half = FT(1) / (FT(1) + FT(1))
        w = _west(j, Nx)
        s = _south(i, Ny)
        u_im = (U[i, j] + U[i, w]) * half
        v_jm = (V[i, j] + V[s, j]) * half
        ustar[i, j] =
            sqrt(C_d_top * (u_im * u_im + v_jm * v_jm + u_tide * u_tide)) * tmask[i, j]
    end
end

# Reference-LADDIE entrainment (Lambert et al. 2023): fuses Sb, drhob, drho_pos,
# ent, entr, detr into one pass. Production term = 2μ·u★³/(g·D·δρ).
# drho_pos = max(0.0001, drho) is computed inline — never zero — so the drhob/drho_pos
# division is safe. D*drho_pos can be zero where D=0 (outside domain), so _safe_div is
# used for the ustar³/(D·δρ) term.
@kernel function _lambert_entrainment_kernel!(
    Sb,
    drhob,
    ent,
    entr,
    detr,
    @Const(T),
    @Const(S),
    @Const(Tb),
    @Const(zb),
    @Const(ustar),
    @Const(D),
    @Const(drho),
    @Const(melt),
    @Const(tmask),
    mu2_over_g,
    max_detrainment,
    alpha,
    beta,
    l1,
    l2,
    l3,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(mu2_over_g)
        drho_pos = max(FT(0.0001), drho[i, j])
        sb = (Tb[i, j] - l2 - l3 * zb[i, j]) / l1
        Sb[i, j] = sb
        db_ij = (beta * (S[i, j] - sb) - alpha * (T[i, j] - Tb[i, j])) * tmask[i, j]
        drhob[i, j] = db_ij
        us3 = ustar[i, j]^3
        e_ij =
            mu2_over_g * _safe_div(us3, D[i, j] * drho_pos) -
            db_ij / drho_pos * melt[i, j] * tmask[i, j]
        ent[i, j] = e_ij
        z = zero(FT)
        entr[i, j] = max(e_ij, z)
        detr[i, j] = min(max_detrainment, max(-e_ij, z))
    end
end

# Literal Eq. 14 entrainment: identical to _lambert_entrainment_kernel! except
# the production term is μ·u★³/(g·D²·δρ) — a D² denominator and a μ (not 2μ)
# prefactor. _safe_div guards D²·δρ = 0 outside the domain.
@kernel function _gaspar_entrainment_kernel!(
    Sb,
    drhob,
    ent,
    entr,
    detr,
    @Const(T),
    @Const(S),
    @Const(Tb),
    @Const(zb),
    @Const(ustar),
    @Const(D),
    @Const(drho),
    @Const(melt),
    @Const(tmask),
    mu_over_g,
    max_detrainment,
    alpha,
    beta,
    l1,
    l2,
    l3,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(mu_over_g)
        drho_pos = max(FT(0.0001), drho[i, j])
        sb = (Tb[i, j] - l2 - l3 * zb[i, j]) / l1
        Sb[i, j] = sb
        db_ij = (beta * (S[i, j] - sb) - alpha * (T[i, j] - Tb[i, j])) * tmask[i, j]
        drhob[i, j] = db_ij
        us3 = ustar[i, j]^3
        e_ij =
            mu_over_g * _safe_div(us3, D[i, j] * D[i, j] * drho_pos) -
            db_ij / drho_pos * melt[i, j] * tmask[i, j]
        ent[i, j] = e_ij
        z = zero(FT)
        entr[i, j] = max(e_ij, z)
        detr[i, j] = min(max_detrainment, max(-e_ij, z))
    end
end

# Holland–Jenkins entrainment fused with im/jm: avoids two circshift allocations.
@kernel function _holland_entrainment_kernel!(
    entr,
    detr,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(D),
    @Const(tmask),
    coeff,
    drho_coeff,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(coeff)
        half = FT(1) / (FT(1) + FT(1))
        w = _west(j, Nx)
        s = _south(i, Ny)
        u_im = (U[i, j] + U[i, w]) * half
        v_jm = (V[i, j] + V[s, j]) * half
        speed_sq =
            max(zero(FT), u_im * u_im + v_jm * v_jm - drho_coeff * drho[i, j] * D[i, j])
        entr[i, j] = coeff * sqrt(speed_sq) * tmask[i, j]
        detr[i, j] = zero(FT)
    end
end

# TODO feels like this could be removed altogether if reduced cache
@kernel function _upwind_split_kernel!(
    Upos,
    Uneg,
    Vpos,
    Vneg,
    Vyp1pos,
    Vyp1neg,
    Uxp1pos,
    Uxp1neg,
    @Const(U),
    @Const(V),
    @Const(Vyp1),
    @Const(Uxp1),
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        z = zero(eltype(U))
        u = U[i, j];
        v = V[i, j]
        vy1 = Vyp1[i, j];
        ux1 = Uxp1[i, j]
        Upos[i, j] = max(u, z)
        Uneg[i, j] = min(u, z)
        Vpos[i, j] = max(v, z)
        Vneg[i, j] = min(v, z)
        Vyp1pos[i, j] = max(vy1, z)
        Vyp1neg[i, j] = min(vy1, z)
        Uxp1pos[i, j] = max(ux1, z)
        Uxp1neg[i, j] = min(ux1, z)
    end
end

# TODO don't like this name, consider new one + add docstring
function update_secondary_fields!(m)
    update_ambient_fields!(m)
    update_freezing_temperature!(m)
    update_density!(m)
    update_convection!(m)
    update_melt!(m)
    precompute_advection_stencils!(m)
    precompute_laplacian_stencils!(m)
    update_entrainment!(m)
    return
end

# ============================================================================
# Equation-term functions — one named function per term in each prognostic
# equation. Functions return the term value; the caller applies the sign,
# making the step functions read like the written equations.
#
# These are the readable REFERENCE implementation of the governing equations.
# The time loop runs the fused kernels in numerics.jl instead (one pass per
# prognostic, no intermediate allocations); the test suite asserts that the
# kernels reproduce these terms exactly (testset "Fused kernels match
# reference equation terms"), so the two cannot drift apart silently.
# All equation references are to Lambert et al. (2023), The Cryosphere,
# https://doi.org/10.5194/tc-17-3203-2023.
#
# Notation shared across all five governing equations:
#   D       plume layer thickness [m]
#   U, V    depth-averaged x- and y-velocity components [m s⁻¹]
#   T, S    depth-averaged plume temperature [°C] and salinity [PSU]
#   ṁ       basal melt rate [m s⁻¹]; positive = melting
#   ė       net entrainment rate, ė = entr − detr [m s⁻¹]
#   Tₐ, Sₐ  ambient temperature and salinity interpolated to plume depth
#   Tb      ice–ocean boundary (basal) temperature [°C]
#   δρ      plume–ambient density difference, ρ − ρₐ [kg m⁻³]
#   ρ̄       δρ / ρ₀, reduced density contrast
#   D̄       layer thickness face-interpolated to the velocity node
#   f       Coriolis parameter [s⁻¹]
#   g       gravitational acceleration [m s⁻²]
#   ρ₀      reference seawater density [kg m⁻³]
#   zb      ice-base depth, negative below sea level [m]
#   C_d      quadratic bottom drag coefficient [–]
#   |u|     current speed, √(U² + V²) [m s⁻¹]
#   A_h      horizontal viscosity [m² s⁻¹]
#   K_h      horizontal diffusivity [m² s⁻¹]
#   γT      turbulent heat transfer coefficient [m s⁻¹]
# ============================================================================
# -- U-momentum terms (Eq. 2) -----------------------------------------------

# U·∂D/∂t  (thickness-tendency coupling)
@inline u_thickness_tendency(m) = m.U.present .* ip_t(m, m.dDdt)
# ∇·(DUu)  (momentum advection)
@inline u_advection(m) = upwind_advection_U(m)
# g·D̄·ρ̄·∂D/∂x  (pressure gradient from plume-thickness depth)
@inline u_pressure_depth(m) = m.g .* ip_t(m, m.Ddrho) .* (m.Dxm1 .- m.D.present) ./ m.dx
# g·D̄·ρ̄·∂zb/∂x  (baroclinic pressure via ice-base slope)
@inline u_pressure_slope(m) = m.g .* ip_t(m, m.Ddrho .* m.dzdx)
# ½g·D̄²·∂δρ/∂x  (internal pressure gradient)
@inline u_pressure_density(m) =
    (m.g / 2) .* ip_t(m, m.D.present) .^ 2 .* (xm1(m.drho) .- m.drho) ./ m.dx
# f·D̄·V  (Coriolis)
@inline u_coriolis(m) = m.f .* ip_t(m, m.D.present .* m.Vjm)
# Cd·U·|u|  (quadratic bottom drag)
@inline u_bottom_drag(m) =
    m.C_d .* m.U.present .* sqrt.(m.U.present .^ 2 .+ ip_half(jm_half(m.V.present)) .^ 2)
# Ah·∇²(DU)  (lateral diffusion)
@inline u_diffusion(m) = m.A_h .* laplace_U(m)
# e·U  (detrainment momentum loss)
@inline u_detrainment(m) = m.detr .* m.U.present

# -- V-momentum terms (Eq. 3) -----------------------------------------------

# V·∂D/∂t  (thickness-tendency coupling)
@inline v_thickness_tendency(m) = m.V.present .* jp_t(m, m.dDdt)
# ∇·(DVv)  (momentum advection)
@inline v_advection(m) = upwind_advection_V(m)
# g·D̄·ρ̄·∂D/∂y  (pressure gradient from plume-thickness depth)
@inline v_pressure_depth(m) = m.g .* jp_t(m, m.Ddrho) .* (m.Dym1 .- m.D.present) ./ m.dy
# g·D̄·ρ̄·∂zb/∂y  (baroclinic pressure via ice-base slope)
@inline v_pressure_slope(m) = m.g .* jp_t(m, m.Ddrho .* m.dzdy)
# ½g·D̄²·∂δρ/∂y  (internal pressure gradient)
@inline v_pressure_density(m) =
    (m.g / 2) .* jp_t(m, m.D.present) .^ 2 .* (ym1(m.drho) .- m.drho) ./ m.dy
# f·D̄·U  (Coriolis)
@inline v_coriolis(m) = m.f .* jp_t(m, m.D.present .* m.Uim)
# Cd·V·|u|  (quadratic bottom drag)
@inline v_bottom_drag(m) =
    m.C_d .* m.V.present .* sqrt.(m.V.present .^ 2 .+ jp_half(im_half(m.U.present)) .^ 2)
# Ah·∇²(DV)  (lateral diffusion)
@inline v_diffusion(m) = m.A_h .* laplace_V(m)
# ė·V  (detrainment momentum loss)
@inline v_detrainment(m) = m.detr .* m.V.present

# -- Tracer terms (Eqs. 4–5) -------------------------------------------------

# q·∂D/∂t  (thickness-tendency coupling)
@inline tracer_thickness_tendency(m, q) = q .* m.dDdt
# ∇·(D·u·q)  (horizontal tracer advection)
@inline tracer_advection(m, q) = upwind_advection_T(similar(m.D.present), m, m.D.present .* q)
# e_net·qa  (entrainment of ambient water)
@inline tracer_entrainment(m, qa) = m.nentr .* qa
# Kh·∇²q  (horizontal diffusion)
@inline tracer_diffusion(m, q_past) = m.K_h .* laplace_T(similar(q_past), m, q_past)
# (q_past − qa)·conv2  (convective relaxation, RelaxToAmbient only)
@inline tracer_convection(m, q_past, qa) = (q_past .- qa) .* m.conv2
# ṁ·Tb − γT·(T − Tb)  (ice-ocean heat exchange; temperature equation only)
@inline T_ice_ocean_exchange(m) = m.melt .* m.Tb .- m.gamT .* (m.T.present .- m.Tb)
