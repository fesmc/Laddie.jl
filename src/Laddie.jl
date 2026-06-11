module Laddie
using KernelAbstractions
const KA = KernelAbstractions

include(joinpath(@__DIR__, "CGridProto.jl"))
using .CGridProto: Center, Face

include("parameterizations.jl")
include("geometry.jl")
include("io.jl")

export Model, Grid, State, Cache, Params, RunConfig
export build_isomip, run!, meltstats, to_backend!

export AbstractEntrainmentParam, HollandEntrainment, GasparEntrainment
export AbstractMeltParam, FixedGamT, TurbulentGamT
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient

export AbstractOpenBoundary, ZeroGradientInflow, NoInflow
export AbstractForcing, ISOMIPForcing, LinearForcing, Linear2Forcing
export TanhForcing, FileForcing

const spy = 365.25 * 24 * 3600   # seconds per year

include("variable.jl")
include("utils.jl")
include("structs.jl")

# ============================================================================
# KernelAbstractions kernels — elementwise physics
# ============================================================================

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
        melt_rate = (-quad_b + sqrt(disc)) / (FT(1) + FT(1)) * tmask[i, j]
        melt[i, j] = melt_rate
        Tb_denom = cp_over_Leff * gamT + cp_over_Leff * ci_over_cp * melt_rate
        Tb[i, j] =
            iszero(Tb_denom) ? zero(FT) :
            (cp_over_Leff * gamT * T[i, j] - melt_rate) / Tb_denom * tmask[i, j]
    end
end

# Robert-Asselin time filter applied to one leapfrog level.
@kernel function _robert_asselin_kernel!(
    present,
    @Const(past),
    @Const(future),
    @Const(mask),
    nu,
)
    i, j = @index(Global, NTuple)
    FT = typeof(nu)
    @inbounds present[i, j] +=
        nu / (FT(1) + FT(1)) *
        (past[i, j] + future[i, j] - (FT(1) + FT(1)) * present[i, j]) *
        mask[i, j]
end

@kernel function _precompute_D_shifts_kernel!(
    Dym1, Dyp1, Dxm1, Dxp1, Dxm1ym1, Dxp1ym1, Dxm1yp1,
    @Const(D), @Const(tmask),
    Ny, Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny); s = _south(i, Ny)
        e = _east(j, Nx);  w = _west(j, Nx)
        Dn  = D[n, j] * tmask[n, j]
        Ds  = D[s, j] * tmask[s, j]
        De  = D[i, e] * tmask[i, e]
        Dw  = D[i, w] * tmask[i, w]
        Dym1[i, j]    = Dn
        Dyp1[i, j]    = Ds
        Dxm1[i, j]    = De
        Dxp1[i, j]    = Dw
        Dxm1ym1[i, j] = D[n, e] * tmask[n, e]
        Dxp1ym1[i, j] = D[n, w] * tmask[n, w]
        Dxm1yp1[i, j] = D[s, e] * tmask[s, e]
    end
end

@kernel function _precompute_staggered_kernel!(
    Vip, Vim, Vjp, Vjm,
    Uip, Uim, Ujp, Ujm,
    signU, signV, Vyp1, Uxp1,
    @Const(V), @Const(U),
    @Const(vmask_ip), @Const(vmask_im), @Const(vmask_jp), @Const(vmask_jm),
    @Const(umask_ip), @Const(umask_im), @Const(umask_jp), @Const(umask_jm),
    Ny, Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny); s = _south(i, Ny)
        e = _east(j, Nx);  w = _west(j, Nx)
        Vij = V[i, j]; Uij = U[i, j]
        Vip[i, j]   = _safe_div(Vij + V[i, e], vmask_ip[i, j])
        Vim[i, j]   = _safe_div(Vij + V[i, w], vmask_im[i, j])
        Vjp[i, j]   = _safe_div(Vij + V[n, j], vmask_jp[i, j])
        Vjm[i, j]   = _safe_div(Vij + V[s, j], vmask_jm[i, j])
        Uip[i, j]   = _safe_div(Uij + U[i, e], umask_ip[i, j])
        Uim[i, j]   = _safe_div(Uij + U[i, w], umask_im[i, j])
        Ujp[i, j]   = _safe_div(Uij + U[n, j], umask_jp[i, j])
        Ujm[i, j]   = _safe_div(Uij + U[s, j], umask_jm[i, j])
        signU[i, j] = sign(Uij)
        signV[i, j] = sign(Vij)
        Vyp1[i, j]  = V[s, j]
        Uxp1[i, j]  = U[i, w]
    end
end

@kernel function _precompute_laplacian_kernel!(
    D0ip, D0im, D0jp, D0jm,
    @Const(D),
    @Const(tmask_ip), @Const(tmask_im), @Const(tmask_jp), @Const(tmask_jm),
    Ny, Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny); s = _south(i, Ny)
        e = _east(j, Nx);  w = _west(j, Nx)
        Dij = D[i, j]
        D0ip[i, j] = _safe_div(Dij + D[i, e], tmask_ip[i, j])
        D0im[i, j] = _safe_div(Dij + D[i, w], tmask_im[i, j])
        D0jp[i, j] = _safe_div(Dij + D[n, j], tmask_jp[i, j])
        D0jm[i, j] = _safe_div(Dij + D[s, j], tmask_jm[i, j])
    end
end

@kernel function _ambient_interp_kernel!(
    Ta, Sa,
    @Const(zb), @Const(D),
    @Const(Tz), @Const(Sz),
    z0, dz, nz,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(z0)
        depth_idx = -z0 + (zb[i, j] - D[i, j]) / dz
        idx_lo = clamp(trunc(Int, depth_idx), 0, nz - 1)
        idx_hi = clamp(idx_lo + 1, 0, nz - 1)
        weight = depth_idx - FT(idx_lo)
        Ta[i, j] = weight * Tz[idx_hi + 1] + (one(FT) - weight) * Tz[idx_lo + 1]
        Sa[i, j] = weight * Sz[idx_hi + 1] + (one(FT) - weight) * Sz[idx_lo + 1]
    end
end

@kernel function _clamp_kernel!(a, lo, hi)
    i, j = @index(Global, NTuple)
    @inbounds a[i, j] = clamp(a[i, j], lo, hi)
end

# Infer backend from array `A`, launch `kernel!` over the full array extent.
function launch!(kernel!, A, args...)
    backend = KA.get_backend(A)
    kernel!(backend, (16, 16))(args...; ndrange = size(A))
    return nothing
end

# ============================================================================
# Stencil operators — fused kernels
# ============================================================================

@inline _east(j, N) = ifelse(j == N, 1, j + 1)
@inline _west(j, N) = ifelse(j == 1, N, j - 1)
@inline _north(i, N) = ifelse(i == N, 1, i + 1)
@inline _south(i, N) = ifelse(i == 1, N, i - 1)
@inline _safe_div(a, b) = iszero(b) ? zero(a) : a / b

@kernel function _lapT_kernel!(
    out,
    @Const(var),
    @Const(D0jp),
    @Const(D0jm),
    @Const(D0ip),
    @Const(D0im),
    @Const(tmaskym1),
    @Const(tmaskyp1),
    @Const(tmaskxm1),
    @Const(tmaskxp1),
    dy2,
    dx2,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny);
        s = _south(i, Ny);
        e = _east(j, Nx);
        w = _west(j, Nx)
        flux_N = D0jp[i, j] * (var[n, j] - var[i, j]) * tmaskym1[i, j] / dy2
        flux_S = D0jm[i, j] * (var[s, j] - var[i, j]) * tmaskyp1[i, j] / dy2
        flux_E = D0ip[i, j] * (var[i, e] - var[i, j]) * tmaskxm1[i, j] / dx2
        flux_W = D0im[i, j] * (var[i, w] - var[i, j]) * tmaskxp1[i, j] / dx2
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _convT_kernel!(
    out,
    @Const(var),
    @Const(Vpos),
    @Const(Vneg),
    @Const(Vyp1pos),
    @Const(Vyp1neg),
    @Const(Upos),
    @Const(Uneg),
    @Const(Uxp1pos),
    @Const(Uxp1neg),
    @Const(tmaskym1),
    @Const(ocnym1),
    @Const(tmaskyp1),
    @Const(ocnyp1),
    @Const(tmaskxm1),
    @Const(ocnxm1),
    @Const(tmaskxp1),
    @Const(ocnxp1),
    @Const(vmask),
    @Const(vmaskyp1),
    @Const(umask),
    @Const(umaskxp1),
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny);
        s = _south(i, Ny);
        e = _east(j, Nx);
        w = _west(j, Nx)
        v = var[i, j]
        flux_N =
            -(
                Vpos[i, j] * v +
                Vneg[i, j] * (var[n, j] * tmaskym1[i, j] + v * ocnym1[i, j])
            ) / dy * vmask[i, j]
        flux_S =
            (
                Vyp1pos[i, j] * (var[s, j] * tmaskyp1[i, j] + v * ocnyp1[i, j]) +
                Vyp1neg[i, j] * v
            ) / dy * vmaskyp1[i, j]
        flux_E =
            -(
                Upos[i, j] * v +
                Uneg[i, j] * (var[i, e] * tmaskxm1[i, j] + v * ocnxm1[i, j])
            ) / dx * umask[i, j]
        flux_W =
            (
                Uxp1pos[i, j] * (var[i, w] * tmaskxp1[i, j] + v * ocnxp1[i, j]) +
                Uxp1neg[i, j] * v
            ) / dx * umaskxp1[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _convT_noinflow_kernel!(
    out,
    @Const(var),
    @Const(Vpos),
    @Const(Vneg),
    @Const(Vyp1pos),
    @Const(Vyp1neg),
    @Const(Upos),
    @Const(Uneg),
    @Const(Uxp1pos),
    @Const(Uxp1neg),
    @Const(vmask),
    @Const(vmaskyp1),
    @Const(umask),
    @Const(umaskxp1),
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny);
        s = _south(i, Ny);
        e = _east(j, Nx);
        w = _west(j, Nx)
        v = var[i, j]
        flux_N = -(Vpos[i, j] * v + Vneg[i, j] * var[n, j]) / dy * vmask[i, j]
        flux_S = (Vyp1pos[i, j] * var[s, j] + Vyp1neg[i, j] * v) / dy * vmaskyp1[i, j]
        flux_E = -(Upos[i, j] * v + Uneg[i, j] * var[i, e]) / dx * umask[i, j]
        flux_W = (Uxp1pos[i, j] * var[i, w] + Uxp1neg[i, j] * v) / dx * umaskxp1[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _convU_kernel!(
    out,
    @Const(D),
    @Const(Dxm1),
    @Const(Dym1),
    @Const(Dyp1),
    @Const(Dxm1ym1),
    @Const(Dxm1yp1),
    @Const(tmask),
    @Const(tmaskxm1),
    @Const(tmaskym1),
    @Const(tmaskyp1),
    @Const(tmaskxm1ym1),
    @Const(tmaskxm1yp1),
    @Const(ocn),
    @Const(ocnxm1),
    @Const(Vip),
    @Const(Ujp),
    @Const(Ujm),
    @Const(Uip),
    @Const(Uim),
    @Const(signU),
    @Const(U),
    @Const(grdNu),
    @Const(grdSu),
    slip,
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = _south(i, Ny)
        D_N = _safe_div(
            D[i, j] + Dxm1[i, j] + Dym1[i, j] + Dxm1ym1[i, j],
            tmask[i, j] + tmaskxm1[i, j] + tmaskym1[i, j] + tmaskxm1ym1[i, j],
        )
        D_S = _safe_div(
            D[i, j] + Dxm1[i, j] + Dyp1[i, j] + Dxm1yp1[i, j],
            tmask[i, j] + tmaskxm1[i, j] + tmaskyp1[i, j] + tmaskxm1yp1[i, j],
        )
        FT = typeof(slip)
        D_E = Dxm1[i, j] + ocnxm1[i, j] * D[i, j]
        D_W = D[i, j] + ocn[i, j] * Dxm1[i, j]
        flux_N = -D_N * Vip[i, j] * (Ujp[i, j] - slip * U[i, j] * grdNu[i, j]) / dy
        flux_S = D_S * Vip[s, j] * (Ujm[i, j] - slip * U[i, j] * grdSu[i, j]) / dy
        flux_E =
            -D_E *
            Uip[i, j] *
            (Uip[i, j] - (one(FT) - signU[i, j]) * U[i, j] * ocnxm1[i, j]) / dx
        flux_W = D_W * Uim[i, j] * (Uim[i, j] - signU[i, j] * U[i, j] * ocn[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _convV_kernel!(
    out,
    @Const(D),
    @Const(Dym1),
    @Const(Dxm1),
    @Const(Dxp1),
    @Const(Dxm1ym1),
    @Const(Dxp1ym1),
    @Const(tmask),
    @Const(tmaskym1),
    @Const(tmaskxm1),
    @Const(tmaskxp1),
    @Const(tmaskxm1ym1),
    @Const(tmaskxp1ym1),
    @Const(ocn),
    @Const(ocnym1),
    @Const(Vjp),
    @Const(Vjm),
    @Const(Vip),
    @Const(Vim),
    @Const(Ujp),
    @Const(signV),
    @Const(V),
    @Const(grdEv),
    @Const(grdWv),
    slip,
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w = _west(j, Nx)
        D_E = _safe_div(
            D[i, j] + Dym1[i, j] + Dxm1[i, j] + Dxm1ym1[i, j],
            tmask[i, j] + tmaskym1[i, j] + tmaskxm1[i, j] + tmaskxm1ym1[i, j],
        )
        D_W = _safe_div(
            D[i, j] + Dym1[i, j] + Dxp1[i, j] + Dxp1ym1[i, j],
            tmask[i, j] + tmaskym1[i, j] + tmaskxp1[i, j] + tmaskxp1ym1[i, j],
        )
        FT = typeof(slip)
        D_N = Dym1[i, j] + ocnym1[i, j] * D[i, j]
        D_S = D[i, j] + ocn[i, j] * Dym1[i, j]
        flux_N =
            -D_N *
            Vjp[i, j] *
            (Vjp[i, j] - (one(FT) - signV[i, j]) * V[i, j] * ocnym1[i, j]) / dy
        flux_S = D_S * Vjm[i, j] * (Vjm[i, j] - signV[i, j] * V[i, j] * ocn[i, j]) / dy
        flux_E = -D_E * Ujp[i, j] * (Vip[i, j] - slip * V[i, j] * grdEv[i, j]) / dx
        flux_W = D_W * Ujp[i, w] * (Vim[i, j] - slip * V[i, j] * grdWv[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _lapU_kernel!(
    out,
    @Const(var),
    @Const(D0),
    @Const(D_on_ugrid),
    @Const(tmask_jp),
    @Const(tmask_jm),
    @Const(ocnym1),
    @Const(ocnyp1),
    @Const(ocnxm1),
    @Const(ocn),
    @Const(grdNu),
    @Const(grdSu),
    slip,
    dx2,
    dy2,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    FT = typeof(slip)
    @inbounds begin
        n = _north(i, Ny);
        s = _south(i, Ny);
        e = _east(j, Nx);
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        slip_drag = slip * D_on_ugrid[i, j] * v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[n, j], tmask_jp[i, j])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[s, j], tmask_jm[i, j])
        flux_N =
            jpD * (var[n, j] - v) / dy2 * (o - ocnym1[i, j]) - slip_drag * grdNu[i, j]
        flux_S =
            jmD * (var[s, j] - v) / dy2 * (o - ocnyp1[i, j]) - slip_drag * grdSu[i, j]
        flux_E = D0[i, e] * (var[i, e] - v) / dx2 * (o - ocnxm1[i, j])
        flux_W = D0[i, j] * (var[i, w] - v) / dx2 * (o - ocn[i, j])
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _lapV_kernel!(
    out,
    @Const(var),
    @Const(D0),
    @Const(D_on_vgrid),
    @Const(tmask_ip),
    @Const(tmask_im),
    @Const(ocnym1),
    @Const(ocn),
    @Const(ocnxm1),
    @Const(ocnxp1),
    @Const(grdEv),
    @Const(grdWv),
    slip,
    dx2,
    dy2,
    Ny,
    Nx,
)
    FT = typeof(slip)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny);
        s = _south(i, Ny);
        e = _east(j, Nx);
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        slip_drag = slip * D_on_vgrid[i, j] * v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, e], tmask_ip[i, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, w], tmask_im[i, j])
        flux_N = D0[n, j] * (var[n, j] - v) / dy2 * (o - ocnym1[i, j])
        flux_S = D0[i, j] * (var[s, j] - v) / dy2 * (o - ocn[i, j])
        flux_E =
            ipD * (var[i, e] - v) / dx2 * (o - ocnxm1[i, j]) - slip_drag * grdEv[i, j]
        flux_W =
            imD * (var[i, w] - v) / dx2 * (o - ocnxp1[i, j]) - slip_drag * grdWv[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

function convT(out, m, var)
    ny, nx = size(var)
    if m.openbc isa ZeroGradientInflow
        launch!(
            _convT_kernel!,
            out,
            out,
            var,
            m.Vpos,
            m.Vneg,
            m.Vyp1pos,
            m.Vyp1neg,
            m.Upos,
            m.Uneg,
            m.Uxp1pos,
            m.Uxp1neg,
            m.tmaskym1,
            m.ocnym1,
            m.tmaskyp1,
            m.ocnyp1,
            m.tmaskxm1,
            m.ocnxm1,
            m.tmaskxp1,
            m.ocnxp1,
            m.vmask,
            m.vmaskyp1,
            m.umask,
            m.umaskxp1,
            m.dx,
            m.dy,
            ny,
            nx,
        )
    else  # NoInflow
        launch!(
            _convT_noinflow_kernel!,
            out,
            out,
            var,
            m.Vpos,
            m.Vneg,
            m.Vyp1pos,
            m.Vyp1neg,
            m.Upos,
            m.Uneg,
            m.Uxp1pos,
            m.Uxp1neg,
            m.vmask,
            m.vmaskyp1,
            m.umask,
            m.umaskxp1,
            m.dx,
            m.dy,
            ny,
            nx,
        )
    end
    return out
end
function convU(m)
    ny, nx = size(m.U.present)
    launch!(
        _convU_kernel!,
        m.cU,
        m.cU,
        m.D.present,
        m.Dxm1,
        m.Dym1,
        m.Dyp1,
        m.Dxm1ym1,
        m.Dxm1yp1,
        m.tmask,
        m.tmaskxm1,
        m.tmaskym1,
        m.tmaskyp1,
        m.tmaskxm1ym1,
        m.tmaskxm1yp1,
        m.ocn,
        m.ocnxm1,
        m.Vip,
        m.Ujp,
        m.Ujm,
        m.Uip,
        m.Uim,
        m.signU,
        m.U.present,
        m.grdNu,
        m.grdSu,
        m.slip,
        m.dx,
        m.dy,
        ny,
        nx,
    )
    return m.cU
end
function convV(m)
    ny, nx = size(m.V.present)
    launch!(
        _convV_kernel!,
        m.cV,
        m.cV,
        m.D.present,
        m.Dym1,
        m.Dxm1,
        m.Dxp1,
        m.Dxm1ym1,
        m.Dxp1ym1,
        m.tmask,
        m.tmaskym1,
        m.tmaskxm1,
        m.tmaskxp1,
        m.tmaskxm1ym1,
        m.tmaskxp1ym1,
        m.ocn,
        m.ocnym1,
        m.Vjp,
        m.Vjm,
        m.Vip,
        m.Vim,
        m.Ujp,
        m.signV,
        m.V.present,
        m.grdEv,
        m.grdWv,
        m.slip,
        m.dx,
        m.dy,
        ny,
        nx,
    )
    return m.cV
end
function lapT(out, m, var)
    ny, nx = size(var)
    launch!(
        _lapT_kernel!,
        out,
        out,
        var,
        m.D0jp,
        m.D0jm,
        m.D0ip,
        m.D0im,
        m.tmaskym1,
        m.tmaskyp1,
        m.tmaskxm1,
        m.tmaskxp1,
        m.dy^2,
        m.dx^2,
        ny,
        nx,
    )
    return out
end
function lapU(m)
    ny, nx = size(m.U.past)
    launch!(
        _lapU_kernel!,
        m.lU,
        m.lU,
        m.U.past,
        m.D.past,
        m.D_on_ugrid,
        m.tmask_jp,
        m.tmask_jm,
        m.ocnym1,
        m.ocnyp1,
        m.ocnxm1,
        m.ocn,
        m.grdNu,
        m.grdSu,
        m.slip,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    return m.lU
end
function lapV(m)
    ny, nx = size(m.V.past)
    launch!(
        _lapV_kernel!,
        m.lV,
        m.lV,
        m.V.past,
        m.D.past,
        m.D_on_vgrid,
        m.tmask_ip,
        m.tmask_im,
        m.ocnym1,
        m.ocn,
        m.ocnxm1,
        m.ocnxp1,
        m.grdEv,
        m.grdWv,
        m.slip,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    return m.lV
end


# ============================================================================
# Physics — secondary-field updates
# ============================================================================

"""
    update_ambient_fields!(m)

Vertically interpolate the ambient T/S profiles to the depth of each grid cell's
plume base (zb − D), writing results into `m.Ta` and `m.Sa`
(Lambert et al. 2023, Eqs. A1–A2).
"""
function update_ambient_fields!(m)
    nz = length(m.z)
    launch!(
        _ambient_interp_kernel!, m.Ta,
        m.Ta, m.Sa, m.zb, m.D.present, m.Tz, m.Sz,
        m.z0, m.dz, nz,
    )
    return
end

"Linear liquidus: Tf = l1·S + l2 + l3·zb  (Lambert et al. 2023, Eq. 6)."
update_freezing_temperature!(m) =
    launch!(_freezing_point_kernel!, m.Tf, m.Tf, m.S.present, m.zb, m.l1, m.l2, m.l3)

"Linear EOS: δρ = β·(Sa − S) − α·(Ta − T)  (Lambert et al. 2023, Eq. 7)."
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
    update_convection!(m, cp::ClampDensity)

Flag convectively unstable cells and clamp δρ to a minimum positive value
so the plume remains denser than ambient (Lambert et al. 2023, Sect. 2.4).
"""
function update_convection!(m, cp::ClampDensity)
    m.convection .= m.drho .< 0
    m.drho = max.(m.drho, cp.mindrho / m.rho0)
end

"""
    update_convection!(m, cp::ResetToAmbient)

Flag convectively unstable cells, then instantly reset their T/S to ambient
values so the density remains stable (Lambert et al. 2023, Sect. 2.4).
"""
function update_convection!(m, cp::ResetToAmbient)
    m.convection .= m.drho .< 0
    thr = cp.mindrho / m.rho0
    cond = m.drho .< thr
    m.T.present = ifelse.(cond, m.Ta, m.T.present)
    m.S.present = ifelse.(cond, m.Sa .- cp.mindrho / (m.rho0 * m.beta), m.S.present)
    update_density!(m)
end

"""
    update_convection!(m, ::RelaxToAmbient)

Flag convectively unstable cells; relaxation is applied implicitly during the
tracer time step via `conv2` (Lambert et al. 2023, Sect. 2.4).
"""
function update_convection!(m, ::RelaxToAmbient)
    m.convection .= m.drho .< 0
end

update_convection!(m) = update_convection!(m, m.convpar)

function _compute_turbulent_transfer_coefficients!(m, mp::TurbulentGamT)
    FT = typeof(mp.Pr)
    turb_log = FT(2.12) .* log.(m.ustar .* m.D.present ./ mp.nu0 .+ FT(1e-12))
    m.gamT = m.ustar ./ (turb_log .+ FT(12.5) * mp.Pr^(FT(2)/FT(3)) .- FT(8.68))
    m.gamS = m.ustar ./ (turb_log .+ FT(12.5) * mp.Sc^(FT(2)/FT(3)) .- FT(8.68))
end

"""
    update_melt!(m, mp::FixedGamT)

Three-equation ice-ocean melt parameterisation with a fixed heat transfer
coefficient γ_T (Jenkins 1991; Lambert et al. 2023, Eqs. 8–10).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::FixedGamT)
    FT = m.FT
    m.ustar =
        sqrt.(m.Cdtop .* (im(m.U.present) .^ 2 .+ jm(m.V.present) .^ 2 .+ m.utide^2)) .*
        m.tmask
    cp_over_Leff = m.cp / (m.L - m.ci * m.Ti)
    ci_over_cp = m.ci / m.cp
    m.gamT = mp.gamTfix
    m.gamS = m.gamT / FT(35)
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

"""
    update_melt!(m, mp::TurbulentGamT)

Three-equation ice-ocean melt parameterisation with turbulence-dependent
transfer coefficients γ_T, γ_S via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 8–10).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::TurbulentGamT)
    FT = m.FT
    m.ustar =
        sqrt.(m.Cdtop .* (im(m.U.present) .^ 2 .+ jm(m.V.present) .^ 2 .+ m.utide^2)) .*
        m.tmask
    cp_over_Leff = m.cp / (m.L - m.ci * m.Ti)
    ci_over_cp = m.ci / m.cp
    _compute_turbulent_transfer_coefficients!(m, mp)
    Tf_depth = m.l2 .+ m.l3 .* m.zb
    quad_b =
        cp_over_Leff .* m.gamT .* (Tf_depth .- m.T.present) .+
        m.gamS .*
        (one(FT) .+ cp_over_Leff * ci_over_cp .* (Tf_depth .+ m.l1 .* m.S.present))
    quad_c =
        cp_over_Leff .* m.gamT .* m.gamS .*
        (Tf_depth .- m.T.present .+ m.l1 .* m.S.present)
    m.melt =
        (-quad_b .+ sqrt.(max.(quad_b .^ 2 .- FT(4) .* quad_c, zero(FT)))) ./ FT(2) .*
        m.tmask
    m.Tb =
        div0(
            cp_over_Leff .* m.gamT .* m.T.present .- m.melt,
            cp_over_Leff .* m.gamT .+ cp_over_Leff * ci_over_cp .* m.melt,
        ) .* m.tmask
end

update_melt!(m) = update_melt!(m, m.meltpar)

"""
    _compute_entrainment!(m, ep::HollandEntrainment)

Holland–Jenkins entrainment: e ∝ √max(0, |u|² − g·δρ·Kh/Ah·D)
(Holland & Jenkins 1999; Lambert et al. 2023, Eq. 11).
"""
function _compute_entrainment!(m, ep::HollandEntrainment)
    FT = m.FT
    speed_sq = im(m.U.present) .^ 2 .+ jm(m.V.present) .^ 2
    m.entr =
        ep.cl * m.Kh / m.Ah^2 .*
        sqrt.(max.(zero(FT), speed_sq .- m.g .* m.drho .* m.Kh / m.Ah .* m.D.present)) .*
        m.tmask
    m.detr = zero(m.entr)
end

"""
    _compute_entrainment!(m, ep::GasparEntrainment)

Gaspar (1988) mechanical-energy entrainment: e ∝ u★³ / (D·δρ) minus a melt
detrainment correction (Lambert et al. 2023, Eq. 12).
"""
function _compute_entrainment!(m, ep::GasparEntrainment)
    FT = m.FT
    m.Sb = (m.Tb .- m.l2 .- m.l3 .* m.zb) ./ m.l1
    m.drhob =
        (m.beta .* (m.S.present .- m.Sb) .- m.alpha .* (m.T.present .- m.Tb)) .* m.tmask
    drho_pos = max.(FT(0.0001), m.drho)
    m.ent =
        ((ep.mu + ep.mu) / m.g) .* div0(m.ustar .^ 3, m.D.present .* drho_pos) .-
        div0(m.drhob, drho_pos) .* m.melt .* m.tmask
    m.entr = max.(m.ent, zero(FT))
    m.detr = min.(m.maxdetr, max.(-m.ent, zero(FT)))
end

"""
    update_entrainment!(m)

Compute entrainment/detrainment rates and the minimum-D correction term `ent2`,
then set `m.nentr = entr + ent2 − detr` (Lambert et al. 2023, Sect. 2.3).
"""
function update_entrainment!(m)
    FT = m.FT
    _compute_entrainment!(m, m.entpar)
    convT(m.convD, m, m.D.present)
    m.ent2 =
        max.(
            zero(FT),
            (m.minD .- m.D.past) ./ (m.dt + m.dt) .-
            (m.convD .+ m.melt .+ m.entr .- m.detr),
        ) .* m.tmask
    m.nentr = m.entr .+ m.ent2 .- m.detr
    return
end

function precompute_advection_stencils!(m)
    ny, nx = size(m.D.present)
    # All 7 D-shift fields in one kernel pass, Dt = D.*tmask computed on the fly
    launch!(
        _precompute_D_shifts_kernel!, m.Dym1,
        m.Dym1, m.Dyp1, m.Dxm1, m.Dxp1, m.Dxm1ym1, m.Dxp1ym1, m.Dxm1yp1,
        m.D.present, m.tmask, ny, nx,
    )
    # Staggered interpolations, sign, and velocity shifts in one kernel pass
    launch!(
        _precompute_staggered_kernel!, m.Vip,
        m.Vip, m.Vim, m.Vjp, m.Vjm,
        m.Uip, m.Uim, m.Ujp, m.Ujm,
        m.signU, m.signV, m.Vyp1, m.Uxp1,
        m.V.present, m.U.present,
        m.vmask_ip, m.vmask_im, m.vmask_jp, m.vmask_jm,
        m.umask_ip, m.umask_im, m.umask_jp, m.umask_jm,
        ny, nx,
    )
    # Upwind splits — fused broadcast, no allocation
    z = zero(m.FT)
    @. m.Upos    = max(m.U.present, z)
    @. m.Uneg    = min(m.U.present, z)
    @. m.Vpos    = max(m.V.present, z)
    @. m.Vneg    = min(m.V.present, z)
    @. m.Vyp1pos = max(m.Vyp1, z)
    @. m.Vyp1neg = min(m.Vyp1, z)
    @. m.Uxp1pos = max(m.Uxp1, z)
    @. m.Uxp1neg = min(m.Uxp1, z)
    return
end

function precompute_laplacian_stencils!(m)
    ny, nx = size(m.D.past)
    launch!(
        _precompute_laplacian_kernel!, m.D0ip,
        m.D0ip, m.D0im, m.D0jp, m.D0jm,
        m.D.past,
        m.tmask_ip, m.tmask_im, m.tmask_jp, m.tmask_jm,
        ny, nx,
    )
    @. m.D_on_ugrid = m.D0ip * m.tmask
    @. m.D_on_vgrid = m.D0jp * m.tmask
    return
end

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
#   Cd      quadratic bottom drag coefficient [–]
#   |u|     current speed, √(U² + V²) [m s⁻¹]
#   Ah      horizontal viscosity [m² s⁻¹]
#   Kh      horizontal diffusivity [m² s⁻¹]
#   γT      turbulent heat transfer coefficient [m s⁻¹]
# ============================================================================

# -- U-momentum terms (Eq. 2) -----------------------------------------------

# U·∂D/∂t  (thickness-tendency coupling)
@inline u_thickness_tendency(m) = m.U.present .* ip_t(m, m.dDdt)
# ∇·(DUu)  (momentum advection)
@inline u_advection(m) = convU(m)
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
    m.Cd .* m.U.present .* sqrt.(m.U.present .^ 2 .+ ip(jm(m.V.present)) .^ 2)
# Ah·∇²(DU)  (lateral diffusion)
@inline u_diffusion(m) = m.Ah .* lapU(m)
# e·U  (detrainment momentum loss)
@inline u_detrainment(m) = m.detr .* m.U.present

# -- V-momentum terms (Eq. 3) -----------------------------------------------

# V·∂D/∂t  (thickness-tendency coupling)
@inline v_thickness_tendency(m) = m.V.present .* jp_t(m, m.dDdt)
# ∇·(DVv)  (momentum advection)
@inline v_advection(m) = convV(m)
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
    m.Cd .* m.V.present .* sqrt.(m.V.present .^ 2 .+ jp(im(m.U.present)) .^ 2)
# Ah·∇²(DV)  (lateral diffusion)
@inline v_diffusion(m) = m.Ah .* lapV(m)
# ė·V  (detrainment momentum loss)
@inline v_detrainment(m) = m.detr .* m.V.present

# -- Tracer terms (Eqs. 4–5) -------------------------------------------------

# q·∂D/∂t  (thickness-tendency coupling)
@inline tracer_thickness_tendency(m, q) = q .* m.dDdt
# ∇·(D·u·q)  (horizontal tracer advection)
@inline tracer_advection(m, q) = convT(similar(m.D.present), m, m.D.present .* q)
# e_net·qa  (entrainment of ambient water)
@inline tracer_entrainment(m, qa) = m.nentr .* qa
# Kh·∇²q  (horizontal diffusion)
@inline tracer_diffusion(m, q_past) = m.Kh .* lapT(similar(q_past), m, q_past)
# (q_past − qa)·conv2  (convective relaxation, RelaxToAmbient only)
@inline tracer_convection(m, q_past, qa) = (q_past .- qa) .* m.conv2
# ṁ·Tb − γT·(T − Tb)  (ice-ocean heat exchange; temperature equation only)
@inline T_ice_ocean_exchange(m) = m.melt .* m.Tb .- m.gamT .* (m.T.present .- m.Tb)

# ============================================================================
# Time integration (Lambert et al. 2023)
# ============================================================================

_update_conv2!(::Any, ::ClampDensity)   = nothing
_update_conv2!(::Any, ::ResetToAmbient) = nothing
function _update_conv2!(m, cp::RelaxToAmbient)
    @. m.conv2 = (m.drho < 0) * m.D.present / cp.convtime
end

function precompute_integration_terms!(m)
    @. m.dDdt  = (m.D.future - m.D.past) / (m.dt + m.dt)
    @. m.Ddrho = m.D.present * m.drho
    _update_conv2!(m, m.convpar)
    return
end

@kernel function _step_thickness_kernel!(
    out,
    @Const(D0),
    @Const(convD),
    @Const(melt),
    @Const(nentr),
    @Const(tmask),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] =
        D0[i, j] + (convD[i, j] + melt[i, j] + nentr[i, j]) * tmask[i, j] * dt
end

@kernel function _step_u_momentum_kernel!(
    out,
    @Const(Up),
    @Const(U1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(Dxm1),
    @Const(D1),
    @Const(drho),
    @Const(dzdx),
    @Const(Vjm),
    @Const(V1),
    @Const(detr),
    @Const(cU),
    @Const(lU),
    @Const(tmask_ip),
    @Const(umask),
    g,
    f,
    Cd,
    Ah,
    dx,
    dt,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(g)
        half = FT(1) / (FT(1) + FT(1))
        e = _east(j, Nx);
        s = _south(i, Ny);
        tmip = tmask_ip[i, j]
        ip_dDdt = _safe_div(dDdt[i, j] + dDdt[i, e], tmip)
        ip_D_drho = _safe_div(Ddrho[i, j] + Ddrho[i, e], tmip)
        ip_D_dzdx = _safe_div(Ddrho[i, j] * dzdx[i, j] + Ddrho[i, e] * dzdx[i, e], tmip)
        ip_D = _safe_div(D1[i, j] + D1[i, e], tmip)
        ip_D_Vjm = _safe_div(D1[i, j] * Vjm[i, j] + D1[i, e] * Vjm[i, e], tmip)
        ipjmV = half * (half * (V1[i, j] + V1[s, j]) + half * (V1[i, e] + V1[s, e]))
        rhs =
            -U1[i, j] * ip_dDdt +
            cU[i, j] +
            -g * ip_D_drho * (Dxm1[i, j] - D1[i, j]) / dx +
            g * ip_D_dzdx +
            -half * g * ip_D^2 * (drho[i, e] - drho[i, j]) / dx +
            f * ip_D_Vjm +
            -Cd * U1[i, j] * sqrt(U1[i, j]^2 + ipjmV^2) +
            Ah * lU[i, j] +
            -detr[i, j] * U1[i, j]
        out[i, j] = Up[i, j] + _safe_div(rhs, ip_D) * umask[i, j] * dt
    end
end

@kernel function _step_v_momentum_kernel!(
    out,
    @Const(Vp),
    @Const(V1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(Dym1),
    @Const(D1),
    @Const(drho),
    @Const(dzdy),
    @Const(Uim),
    @Const(U1),
    @Const(detr),
    @Const(cV),
    @Const(lV),
    @Const(tmask_jp),
    @Const(vmask),
    g,
    f,
    Cd,
    Ah,
    dy,
    dt,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(g)
        half = FT(1) / (FT(1) + FT(1))
        n = _north(i, Ny);
        w = _west(j, Nx);
        tmjp = tmask_jp[i, j]
        jp_dDdt = _safe_div(dDdt[i, j] + dDdt[n, j], tmjp)
        jp_D_drho = _safe_div(Ddrho[i, j] + Ddrho[n, j], tmjp)
        jp_D_dzdy = _safe_div(Ddrho[i, j] * dzdy[i, j] + Ddrho[n, j] * dzdy[n, j], tmjp)
        jp_D = _safe_div(D1[i, j] + D1[n, j], tmjp)
        jp_D_Uim = _safe_div(D1[i, j] * Uim[i, j] + D1[n, j] * Uim[n, j], tmjp)
        jpimU = half * (half * (U1[i, j] + U1[i, w]) + half * (U1[n, j] + U1[n, w]))
        rhs =
            -V1[i, j] * jp_dDdt +
            cV[i, j] +
            -g * jp_D_drho * (Dym1[i, j] - D1[i, j]) / dy +
            g * jp_D_dzdy +
            -half * g * jp_D^2 * (drho[n, j] - drho[i, j]) / dy +
            -f * jp_D_Uim +
            -Cd * V1[i, j] * sqrt(V1[i, j]^2 + jpimU^2) +
            Ah * lV[i, j] +
            -detr[i, j] * V1[i, j]
        out[i, j] = Vp[i, j] + _safe_div(rhs, jp_D) * vmask[i, j] * dt
    end
end

@kernel function _step_temperature_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    gamT,
    Kh,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +
            cT[i, j] +
            nentr[i, j] * Ta[i, j] +
            melt[i, j] * Tb[i, j] +
            -gamT * (T_present[i, j] - Tb[i, j]) +
            Kh * lT[i, j] +
            -(T_past[i, j] - Ta[i, j]) * conv2
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_gamT_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    @Const(gamT),
    Kh,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +
            cT[i, j] +
            nentr[i, j] * Ta[i, j] +
            melt[i, j] * Tb[i, j] +
            -gamT[i, j] * (T_present[i, j] - Tb[i, j]) +
            Kh * lT[i, j] +
            -(T_past[i, j] - Ta[i, j]) * conv2
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_conv2_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    gamT,
    Kh,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +
            cT[i, j] +
            nentr[i, j] * Ta[i, j] +
            melt[i, j] * Tb[i, j] +
            -gamT * (T_present[i, j] - Tb[i, j]) +
            Kh * lT[i, j] +
            -(T_past[i, j] - Ta[i, j]) * conv2[i, j]
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_both_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    @Const(gamT),
    Kh,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +
            cT[i, j] +
            nentr[i, j] * Ta[i, j] +
            melt[i, j] * Tb[i, j] +
            -gamT[i, j] * (T_present[i, j] - Tb[i, j]) +
            Kh * lT[i, j] +
            -(T_past[i, j] - Ta[i, j]) * conv2[i, j]
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_salinity_kernel!(
    out,
    @Const(S_past),
    @Const(S_present),
    @Const(dDdt),
    @Const(cS),
    @Const(nentr),
    @Const(Sa),
    @Const(lS),
    @Const(D1),
    @Const(tmask),
    Kh,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -S_present[i, j] * dDdt[i, j] +
            cS[i, j] +
            nentr[i, j] * Sa[i, j] +
            Kh * lS[i, j] +
            -(S_past[i, j] - Sa[i, j]) * conv2
        out[i, j] = S_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_salinity_mat_conv2_kernel!(
    out,
    @Const(S_past),
    @Const(S_present),
    @Const(dDdt),
    @Const(cS),
    @Const(nentr),
    @Const(Sa),
    @Const(lS),
    @Const(D1),
    @Const(tmask),
    Kh,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -S_present[i, j] * dDdt[i, j] +
            cS[i, j] +
            nentr[i, j] * Sa[i, j] +
            Kh * lS[i, j] +
            -(S_past[i, j] - Sa[i, j]) * conv2[i, j]
        out[i, j] = S_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

function step_thickness(m, dt)
    launch!(
        _step_thickness_kernel!,
        m.D.future,
        m.D.future,
        m.D.past,
        m.convD,
        m.melt,
        m.nentr,
        m.tmask,
        dt,
    )
    return
end
function step_u_momentum(m, dt)
    convU(m)
    lapU(m)
    ny, nx = size(m.U.future)
    launch!(
        _step_u_momentum_kernel!,
        m.U.future,
        m.U.future,
        m.U.past,
        m.U.present,
        m.dDdt,
        m.Ddrho,
        m.Dxm1,
        m.D.present,
        m.drho,
        m.dzdx,
        m.Vjm,
        m.V.present,
        m.detr,
        m.cU,
        m.lU,
        m.tmask_ip,
        m.umask,
        m.g,
        m.f,
        m.Cd,
        m.Ah,
        m.dx,
        dt,
        ny,
        nx,
    )
    return
end
function step_v_momentum(m, dt)
    convV(m)
    lapV(m)
    ny, nx = size(m.V.future)
    launch!(
        _step_v_momentum_kernel!,
        m.V.future,
        m.V.future,
        m.V.past,
        m.V.present,
        m.dDdt,
        m.Ddrho,
        m.Dym1,
        m.D.present,
        m.drho,
        m.dzdy,
        m.Uim,
        m.U.present,
        m.detr,
        m.cV,
        m.lV,
        m.tmask_jp,
        m.vmask,
        m.g,
        m.f,
        m.Cd,
        m.Ah,
        m.dy,
        dt,
        ny,
        nx,
    )
    return
end
# Kernel selection dispatches on the concrete types of gamT and conv2, which are
# fixed at model-construction time by the MP and CS type parameters.
_launch_T!(args, gamT::Number,          Kh, conv2::Number,          dt) =
    launch!(_step_temperature_kernel!,          args..., gamT, Kh, conv2, dt)
_launch_T!(args, gamT::Number,          Kh, conv2::AbstractArray,   dt) =
    launch!(_step_temperature_mat_conv2_kernel!, args..., gamT, Kh, conv2, dt)
_launch_T!(args, gamT::AbstractArray,   Kh, conv2::Number,          dt) =
    launch!(_step_temperature_mat_gamT_kernel!, args..., gamT, Kh, conv2, dt)
_launch_T!(args, gamT::AbstractArray,   Kh, conv2::AbstractArray,   dt) =
    launch!(_step_temperature_mat_both_kernel!, args..., gamT, Kh, conv2, dt)

_launch_S!(args, Kh, conv2::Number,        dt) =
    launch!(_step_salinity_kernel!,         args..., Kh, conv2, dt)
_launch_S!(args, Kh, conv2::AbstractArray, dt) =
    launch!(_step_salinity_mat_conv2_kernel!, args..., Kh, conv2, dt)

function step_temperature(m, dt)
    @. m.DT = m.D.present * m.T.present
    convT(m.cT, m, m.DT)
    lapT(m.lT, m, m.T.past)
    args = (
        m.T.future,
        m.T.future,
        m.T.past,
        m.T.present,
        m.dDdt,
        m.cT,
        m.nentr,
        m.Ta,
        m.melt,
        m.Tb,
        m.lT,
        m.D.present,
        m.tmask,
    )
    _launch_T!(args, m.gamT, m.Kh, m.conv2, dt)
    return
end
function step_salinity(m, dt)
    @. m.DS = m.D.present * m.S.present
    convT(m.cS, m, m.DS)
    lapT(m.lS, m, m.S.past)
    args = (
        m.S.future,
        m.S.future,
        m.S.past,
        m.S.present,
        m.dDdt,
        m.cS,
        m.nentr,
        m.Sa,
        m.lS,
        m.D.present,
        m.tmask,
    )
    _launch_S!(args, m.Kh, m.conv2, dt)
    return
end


function leapfrog_step!(m, nsteps)
    dt = nsteps * m.dt
    step_thickness(m, dt)
    precompute_integration_terms!(m)
    step_u_momentum(m, dt)
    step_v_momentum(m, dt)
    step_temperature(m, dt)
    step_salinity(m, dt)
    return
end

# ============================================================================
# Time-stepping orchestration
# ============================================================================

function apply_robert_asselin_filter!(m)
    for (var, mask) in (
        (m.D, m.tmask), (m.U, m.umask), (m.V, m.vmask),
        (m.T, m.tmask), (m.S, m.tmask),
    )
        launch!(_robert_asselin_kernel!, var.present, var.present, var.past, var.future, mask, m.nu)
    end
    update_density!(m)
    update_convection!(m)
    return
end

function advance_leapfrog!(m)
    for var in (m.D, m.U, m.V, m.T, m.S); rotate!(var); end
    update_secondary_fields!(m)
    return
end

function clamp_velocities!(m)
    launch!(_clamp_kernel!, m.U.future, m.U.future, -m.vcut, m.vcut)
    launch!(_clamp_kernel!, m.V.future, m.V.future, -m.vcut, m.vcut)
    return
end

# ============================================================================
# Geometry helpers for builders
# ============================================================================

# Adjust zb before Grid construction: zero out ice-front ocean cells and
# clamp very shallow ice-shelf cells to -1 m (mirrors initialize_from_scratch!
# in the old dict-bag flow, but applied pre-Grid so the immutable Grid is final).
function _adjust_zb(mask::AbstractMatrix{Int}, zb_raw::AbstractMatrix, FT)
    tmask = FT.(mask .== 3)
    ocn = FT.(mask .== 0)
    isf = ocn .* (xp1(tmask) .+ xm1(tmask) .+ yp1(tmask) .+ ym1(tmask))
    zb = FT.(zb_raw)
    zb = ifelse.(isf .> 0, zero(FT), zb)
    zb = ifelse.((tmask .> 0) .& (zb .> FT(-1)), FT(-1), zb)
    return zb
end

# Initialise prognostic fields from scratch (no restart file).
function _initialize_prognostics!(m)
    update_ambient_fields!(m)
    for level in (:past, :present, :future)
        setfield!(m.D, level, m.Dinit .* m.tmask)
        setfield!(m.T, level, (m.Ta .+ m.dTinit) .* m.tmask)
        setfield!(m.S, level, (m.Sa .+ m.dSinit) .* m.tmask)
    end
    update_secondary_fields!(m)
    leapfrog_step!(m, 1)
    return
end

# ============================================================================
# Backend transfer
# ============================================================================

_to_device(backend, a::AbstractArray) =
    (b = KA.allocate(backend, eltype(a), size(a)); copyto!(b, a); b)

# Reconstruct an immutable Grid with all float arrays moved to backend.
# The integer mask is left on CPU (used for host-side branching only).
function _grid_to_backend(g::Grid{FT}, backend) where {FT}
    mv(a::AbstractArray{<:AbstractFloat}) = _to_device(backend, a)
    mv(a) = a
    Grid{FT}(map(fn -> mv(getfield(g, fn)), fieldnames(typeof(g)))...)
end

# Reconstruct a forcing struct with all float vectors moved to backend.
# Required so update_ambient_fields! can index Tz/Sz with GPU index arrays.
function _forcing_to_backend(f::F, backend) where {F<:AbstractForcing}
    fields = map(fieldnames(F)) do fn
        v = getfield(f, fn)
        v isa AbstractVector && eltype(v) <: AbstractFloat ? _to_device(backend, v) : v
    end
    F(fields...)
end

"""
    to_backend!(m, backend) → m

Move all floating-point arrays in model `m` to `backend` in place.

Transfers Grid mask arrays, State leapfrog arrays (all three levels),
Cache scratch arrays, and Forcing profile vectors.  Scalar parameters and
the integer mask are left on the CPU.  Returns `m` for chaining.

```julia
using CUDA
m = build_isomip()
to_backend!(m, CUDABackend())
run!(m; days = 5.0)
```
"""
function to_backend!(m::Model, backend)
    setfield!(m, :grid, _grid_to_backend(getfield(m, :grid), backend))
    setfield!(m, :forcing, _forcing_to_backend(getfield(m, :forcing), backend))
    s = getfield(m, :state)
    for fn in fieldnames(typeof(s))
        var = getfield(s, fn)
        var.past = _to_device(backend, var.past)
        var.present = _to_device(backend, var.present)
        var.future = _to_device(backend, var.future)
    end
    c = getfield(m, :cache)
    for fn in fieldnames(typeof(c))
        v = getfield(c, fn)
        if v isa AbstractArray && eltype(v) <: AbstractFloat
            setfield!(c, fn, _to_device(backend, v))
        end
    end
    return m
end

# ============================================================================
# Driver
# ============================================================================

"""
    meltstats(m) → (max_meltrate, mean_meltrate, max_speed)

Return melt-rate and velocity statistics for the current model state.
All reductions execute on the device (GPU-safe); results are returned as
CPU scalars.

- `max_meltrate`  — maximum basal melt rate, m yr⁻¹
- `mean_meltrate` — area-mean basal melt rate over floating ice, m yr⁻¹
- `max_speed`     — maximum depth-averaged current speed |u|, m s⁻¹
"""
function meltstats(m)
    mask = m.tmask
    n = sum(mask)
    meltyr = m.melt .* spy
    return maximum(meltyr .* mask),
    sum(meltyr .* mask) / n,
    maximum(sqrt.(im(m.U.present) .^ 2 .+ jm(m.V.present) .^ 2) .* mask)
end

"""
    run!(m; days, verbose=true) → m

Advance model `m` forward in time for `days` days (default: `m.rc.days`).

Each step applies `advance_leapfrog!` → `leapfrog_step!` (2×dt) →
`clamp_velocities!` → `apply_robert_asselin_filter!`.  When `m.rc.saveday > 0`,
`savefields!`, `printdiags`, and `saverestart!` are also called.  When
`verbose = true`, progress is printed every 5 % of steps.

Returns `m` for chaining.
"""
function run!(m; days = nothing, verbose = true)
    n_days = days !== nothing ? Float64(days) : m.rc.days
    nt = round(Int, n_days * 86400 / m.dt)
    diagint = max(1, nt ÷ 20)
    io_on = m.rc.saveday > 0
    io_on && (m.nt = nt)
    verbose && println(
        "[KA/",
        nameof(typeof(KA.get_backend(m.tmask))),
        "] $(nt) steps, ",
        size(m.tmask, 1) - 2,
        "×",
        size(m.tmask, 2) - 2,
        " interior",
    )
    for t = 1:nt
        io_on && (m.t = t)
        advance_leapfrog!(m)
        leapfrog_step!(m, 2)
        clamp_velocities!(m)
        apply_robert_asselin_filter!(m)
        if io_on
            savefields!(m)
            printdiags(m)
            saverestart!(m)
        end
        if verbose && (t % diagint == 0 || t == nt)
            mx, mn, sp = meltstats(m)
            Dmax = maximum(ifelse.(m.tmask .> 0, m.D.present, -Inf))
            println(
                "  step ",
                lpad(t, 5),
                "/",
                nt,
                "  melt[m/yr] mean=",
                round(mn, digits = 2),
                " max=",
                round(mx, digits = 2),
                "  Dmax=",
                round(Dmax, digits = 1),
                "  |u|max=",
                round(sp, digits = 3),
            )
        end
    end
    return m
end

# ============================================================================
# ISOMIP+ geometry builder
# ============================================================================

"""
    build_isomip(backend=CPU(); nx, ny, dx, dy, xgl, xfront,
                 zb_gl, zb_front, isomipcond, params, rc) → Model

Construct and initialise a LADDIE model for the idealised ISOMIP+ channel
geometry (Asay-Davis et al. 2016).

# Arguments
- `backend`: KernelAbstractions backend.  Default `CPU()`; use `CUDABackend()`
  for NVIDIA GPU, `ROCBackend()` for AMD, `MetalBackend()` for Apple Silicon.
- `nx`, `ny`: interior cell counts in x and y (default 240 × 40).
- `dx`, `dy`: cell size in metres (default 2 km).
- `xgl`: grounding-line x-position in metres (default 20 km).
- `xfront`: ice-front x-position in metres (default 460 km).
- `zb_gl`, `zb_front`: ice-draft depth at grounding line and ice front in
  metres (default −720 m and −200 m).
- `isomipcond`: `:warm` (1 °C at depth) or `:cold` (nearly freezing).
- `FT`: floating-point precision type (default `Float64`; use `Float32` for GPU).
- `params`: `Params` object (default: ISOMIP+-canonical `Params(; FT)`).
- `rc`: `RunConfig` object (default: `RunConfig()`, I/O disabled).

The returned model is fully initialised and ready for `run!`.
"""
function build_isomip(
    backend = CPU();
    FT = Float64,
    nx = 240,
    ny = 40,
    dx = 2000.0,
    dy = 2000.0,
    xgl = 20_000.0,
    xfront = 460_000.0,
    zb_gl = -720.0,
    zb_front = -200.0,
    isomipcond = :warm,
    params = nothing,
    rc = RunConfig(),
)
    # Build raw mask and draft
    ny_total, nx_total = ny + 2, nx + 2
    mask = zeros(Int, ny_total, nx_total)
    zb_raw = zeros(FT, ny_total, nx_total)
    xgl_ft = FT(xgl);
    xfront_ft = FT(xfront)
    zgl_ft = FT(zb_gl);
    zfr_ft = FT(zb_front)
    for j = 1:ny, i = 1:nx
        x = FT((i - 1) * dx)
        jp, ip = j + 1, i + 1
        if x < xgl_ft
            mask[jp, ip] = 2
        elseif x <= xfront_ft
            mask[jp, ip] = 3
            zb_raw[jp, ip] =
                zgl_ft + (zfr_ft - zgl_ft) * (x - xgl_ft) / (xfront_ft - xgl_ft)
        end
    end
    mask[1, :] .= 1;
    mask[end, :] .= 1
    mask[:, 1] .= 1;
    mask[:, end] .= 1

    # Assemble typed sub-structs
    zb = _adjust_zb(mask, zb_raw, FT)
    grid = Grid(mask, zb, FT(dx), FT(dy); FT)
    forcing = ISOMIPForcing(FT, isomipcond)
    _params = isnothing(params) ? Params(;
        FT,
        entpar  = GasparEntrainment(FT(2.5)),
        meltpar = FixedGamT(FT(0.00018)),
        convpar = ResetToAmbient(FT(0.005)),
        openbc  = ZeroGradientInflow(),
    ) : params
    state = State(FT, ny_total, nx_total)
    cache = Cache(FT, typeof(_params.meltpar), typeof(_params.convpar), ny_total, nx_total)

    m = Model(Dict{Symbol,Any}(), rc, grid, state, cache, _params, forcing)
    m.d[:x] = collect(FT(dx) .* (1:nx))
    m.d[:y] = collect(FT(dy) .* (1:ny))
    if rc.saveday > 0
        create_rundir!(m)
    end
    if rc.fromrestart
        m.drho = zero(grid.tmask)
        m.Tf   = zero(grid.tmask)
        m.melt = zero(grid.tmask)
        m.Tb   = zero(grid.tmask)
        init_from_restart!(m)
    else
        _initialize_prognostics!(m)
    end
    backend === CPU() || to_backend!(m, backend)
    if rc.saveday > 0
        prepare_output!(m)
    end
    return m
end

end # module Laddie
