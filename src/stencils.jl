
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
    @Const(glNu),
    @Const(glSu),
    slip,
    dslip,
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
        # Per-face slip factor: land walls use slip, grounding-line walls
        # slip + dslip (dslip = 0 for FreeSlipGL → bitwise v1 behaviour).
        flux_N = -D_N * Vip[i, j] *
                 (Ujp[i, j] - (slip + dslip * glNu[i, j]) * U[i, j] * grdNu[i, j]) / dy
        flux_S = D_S * Vip[s, j] *
                 (Ujm[i, j] - (slip + dslip * glSu[i, j]) * U[i, j] * grdSu[i, j]) / dy
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
    @Const(glEv),
    @Const(glWv),
    slip,
    dslip,
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
        flux_E = -D_E * Ujp[i, j] *
                 (Vip[i, j] - (slip + dslip * glEv[i, j]) * V[i, j] * grdEv[i, j]) / dx
        flux_W = D_W * Ujp[i, w] *
                 (Vim[i, j] - (slip + dslip * glWv[i, j]) * V[i, j] * grdWv[i, j]) / dx
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
    @Const(glNu),
    @Const(glSu),
    slip,
    dslip,
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
        # Per-face wall drag: grounding-line walls add dslip to the factor
        # (dslip = 0 for FreeSlipGL → bitwise v1 behaviour).
        dragN = (slip + dslip * glNu[i, j]) * D_on_ugrid[i, j] * v / dy2
        dragS = (slip + dslip * glSu[i, j]) * D_on_ugrid[i, j] * v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[n, j], tmask_jp[i, j])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[s, j], tmask_jm[i, j])
        flux_N =
            jpD * (var[n, j] - v) / dy2 * (o - ocnym1[i, j]) - dragN * grdNu[i, j]
        flux_S =
            jmD * (var[s, j] - v) / dy2 * (o - ocnyp1[i, j]) - dragS * grdSu[i, j]
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
    @Const(glEv),
    @Const(glWv),
    slip,
    dslip,
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
        dragE = (slip + dslip * glEv[i, j]) * D_on_vgrid[i, j] * v / dx2
        dragW = (slip + dslip * glWv[i, j]) * D_on_vgrid[i, j] * v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, e], tmask_ip[i, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, w], tmask_im[i, j])
        flux_N = D0[n, j] * (var[n, j] - v) / dy2 * (o - ocnym1[i, j])
        flux_S = D0[i, j] * (var[s, j] - v) / dy2 * (o - ocn[i, j])
        flux_E =
            ipD * (var[i, e] - v) / dx2 * (o - ocnxm1[i, j]) - dragE * grdEv[i, j]
        flux_W =
            imD * (var[i, w] - v) / dx2 * (o - ocnxp1[i, j]) - dragW * grdWv[i, j]
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
    dslip = _gl_slip(m.glbc, m.slip) - m.slip
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
        m.glNu,
        m.glSu,
        m.slip,
        dslip,
        m.dx,
        m.dy,
        ny,
        nx,
    )
    return m.cU
end
function convV(m)
    ny, nx = size(m.V.present)
    dslip = _gl_slip(m.glbc, m.slip) - m.slip
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
        m.glEv,
        m.glWv,
        m.slip,
        dslip,
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
    dslip = _gl_slip(m.glbc, m.slip) - m.slip
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
        m.glNu,
        m.glSu,
        m.slip,
        dslip,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    return m.lU
end
function lapV(m)
    ny, nx = size(m.V.past)
    dslip = _gl_slip(m.glbc, m.slip) - m.slip
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
        m.glEv,
        m.glWv,
        m.slip,
        dslip,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    return m.lV
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
    # Upwind splits — all 8 fields in one kernel pass
    launch!(
        _upwind_split_kernel!, m.Upos,
        m.Upos, m.Uneg, m.Vpos, m.Vneg,
        m.Vyp1pos, m.Vyp1neg, m.Uxp1pos, m.Uxp1neg,
        m.U.present, m.V.present, m.Vyp1, m.Uxp1,
    )
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