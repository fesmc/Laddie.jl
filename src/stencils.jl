
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
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        flux_N = D0jp[i, j] * (var[n, j] - var[i, j]) * tmaskym1[i, j] / dy2
        flux_S = D0jm[i, j] * (var[s, j] - var[i, j]) * tmaskyp1[i, j] / dy2
        flux_E = D0ip[i, j] * (var[i, e] - var[i, j]) * tmaskxm1[i, j] / dx2
        flux_W = D0im[i, j] * (var[i, w] - var[i, j]) * tmaskxp1[i, j] / dx2
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _upwind_advection_T_kernel!(
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
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
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

@kernel function _upwind_advection_T_noinflow_kernel!(
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
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        v = var[i, j]
        flux_N = -(Vpos[i, j] * v + Vneg[i, j] * var[n, j]) / dy * vmask[i, j]
        flux_S = (Vyp1pos[i, j] * var[s, j] + Vyp1neg[i, j] * v) / dy * vmaskyp1[i, j]
        flux_E = -(Upos[i, j] * v + Uneg[i, j] * var[i, e]) / dx * umask[i, j]
        flux_W = (Uxp1pos[i, j] * var[i, w] + Uxp1neg[i, j] * v) / dx * umaskxp1[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# TODO can I rm Dxm1, Dym1... etc? Would free up cache,
# prevent memory access and make function signature legible
@kernel function _upwind_advection_U_kernel!(
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
    @Const(lndNu),
    @Const(lndSu),
    slip,
    dslip_gl,
    dslip_land,
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        s = _south(i, Ny)
        D0 = D[i, j] * tmask[i, j]
        D_N = _safe_div(
            D0 + Dxm1[i, j] + Dym1[i, j] + Dxm1ym1[i, j],
            tmask[i, j] + tmaskxm1[i, j] + tmaskym1[i, j] + tmaskxm1ym1[i, j],
        )
        D_S = _safe_div(
            D0 + Dxm1[i, j] + Dyp1[i, j] + Dxm1yp1[i, j],
            tmask[i, j] + tmaskxm1[i, j] + tmaskyp1[i, j] + tmaskxm1yp1[i, j],
        )
        FT = typeof(slip)
        D_E = Dxm1[i, j] + ocnxm1[i, j] * D0
        D_W = D0 + ocn[i, j] * Dxm1[i, j]
        # Per-face slip factor: `slip` at every wall, plus dslip_gl at grounding-line
        # faces and dslip_land at land faces (both 0 under the free-slip defaults →
        # bitwise v1 behaviour).  gl?? and lnd?? partition the wall faces (Grid gives
        # the grounding line precedence at mixed corners), so at most one applies.
        slipN = slip + dslip_gl * glNu[i, j] + dslip_land * lndNu[i, j]
        slipS = slip + dslip_gl * glSu[i, j] + dslip_land * lndSu[i, j]
        flux_N = -D_N * Vip[i, j] * (Ujp[i, j] - slipN * U[i, j] * grdNu[i, j]) / dy
        flux_S = D_S * Vip[s, j] * (Ujm[i, j] - slipS * U[i, j] * grdSu[i, j]) / dy
        flux_E =
            -D_E *
            Uip[i, j] *
            (Uip[i, j] - (one(FT) - signU[i, j]) * U[i, j] * ocnxm1[i, j]) / dx
        flux_W = D_W * Uim[i, j] * (Uim[i, j] - signU[i, j] * U[i, j] * ocn[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _upwind_advection_V_kernel!(
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
    @Const(lndEv),
    @Const(lndWv),
    slip,
    dslip_gl,
    dslip_land,
    dx,
    dy,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        w = _west(j, Nx)
        D0 = D[i, j] * tmask[i, j]
        D_E = _safe_div(
            D0 + Dym1[i, j] + Dxm1[i, j] + Dxm1ym1[i, j],
            tmask[i, j] + tmaskym1[i, j] + tmaskxm1[i, j] + tmaskxm1ym1[i, j],
        )
        D_W = _safe_div(
            D0 + Dym1[i, j] + Dxp1[i, j] + Dxp1ym1[i, j],
            tmask[i, j] + tmaskym1[i, j] + tmaskxp1[i, j] + tmaskxp1ym1[i, j],
        )
        FT = typeof(slip)
        D_N = Dym1[i, j] + ocnym1[i, j] * D0
        D_S = D0 + ocn[i, j] * Dym1[i, j]
        flux_N =
            -D_N *
            Vjp[i, j] *
            (Vjp[i, j] - (one(FT) - signV[i, j]) * V[i, j] * ocnym1[i, j]) / dy
        flux_S = D_S * Vjm[i, j] * (Vjm[i, j] - signV[i, j] * V[i, j] * ocn[i, j]) / dy
        # Per-face slip factor: see _upwind_advection_U_kernel! for the composition.
        slipE = slip + dslip_gl * glEv[i, j] + dslip_land * lndEv[i, j]
        slipW = slip + dslip_gl * glWv[i, j] + dslip_land * lndWv[i, j]
        flux_E = -D_E * Ujp[i, j] * (Vip[i, j] - slipE * V[i, j] * grdEv[i, j]) / dx
        flux_W = D_W * Ujp[i, w] * (Vim[i, j] - slipW * V[i, j] * grdWv[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _laplace_U_kernel!(
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
    @Const(lndNu),
    @Const(lndSu),
    slip,
    dslip_gl,
    dslip_land,
    dx2,
    dy2,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    FT = typeof(slip)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        # Per-face wall drag: `slip` at every wall, plus dslip_gl at grounding-line
        # faces and dslip_land at land faces (both 0 under the free-slip defaults
        # → bitwise v1 behaviour).
        dragN = (slip + dslip_gl * glNu[i, j] + dslip_land * lndNu[i, j]) * D_on_ugrid[i, j] * v / dy2
        dragS = (slip + dslip_gl * glSu[i, j] + dslip_land * lndSu[i, j]) * D_on_ugrid[i, j] * v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[n, j], tmask_jp[i, j])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[s, j], tmask_jm[i, j])
        flux_N = jpD * (var[n, j] - v) / dy2 * (o - ocnym1[i, j]) - dragN * grdNu[i, j]
        flux_S = jmD * (var[s, j] - v) / dy2 * (o - ocnyp1[i, j]) - dragS * grdSu[i, j]
        flux_E = D0[i, e] * (var[i, e] - v) / dx2 * (o - ocnxm1[i, j])
        flux_W = D0[i, j] * (var[i, w] - v) / dx2 * (o - ocn[i, j])
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _laplace_V_kernel!(
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
    @Const(lndEv),
    @Const(lndWv),
    slip,
    dslip_gl,
    dslip_land,
    dx2,
    dy2,
    Ny,
    Nx,
)
    FT = typeof(slip)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        # See _laplace_U_kernel! for the slip-factor composition.
        dragE = (slip + dslip_gl * glEv[i, j] + dslip_land * lndEv[i, j]) * D_on_vgrid[i, j] * v / dx2
        dragW = (slip + dslip_gl * glWv[i, j] + dslip_land * lndWv[i, j]) * D_on_vgrid[i, j] * v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, e], tmask_ip[i, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, w], tmask_im[i, j])
        flux_N = D0[n, j] * (var[n, j] - v) / dy2 * (o - ocnym1[i, j])
        flux_S = D0[i, j] * (var[s, j] - v) / dy2 * (o - ocn[i, j])
        flux_E = ipD * (var[i, e] - v) / dx2 * (o - ocnxm1[i, j]) - dragE * grdEv[i, j]
        flux_W = imD * (var[i, w] - v) / dx2 * (o - ocnxp1[i, j]) - dragW * grdWv[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Shear-scaled (NonlinearLateralViscosity) counterparts of _laplace_U_kernel!/
# _laplace_V_kernel!: same geometry and masking, but each interior flux term gets
# its own coefficient visc_? * |Δvar| in place of a single constant A_h, where
# visc_x/visc_y = C_visc * dx/100 and C_visc * dy/100 carry the reference's
# dUabs * triCw / 100 scaling (laddie_velocity.f90:260).  The grounding-line/land
# wall-drag terms keep the plain, unscaled A_h_wall — mirroring the reference,
# which never scales its border term by dUabs (laddie_velocity.f90:249-254).
@kernel function _nonlinear_laplace_U_kernel!(
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
    @Const(lndNu),
    @Const(lndSu),
    slip,
    dslip_gl,
    dslip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    FT = typeof(slip)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        dragN =
            A_h_wall *
            (slip + dslip_gl * glNu[i, j] + dslip_land * lndNu[i, j]) *
            D_on_ugrid[i, j] * v / dy2
        dragS =
            A_h_wall *
            (slip + dslip_gl * glSu[i, j] + dslip_land * lndSu[i, j]) *
            D_on_ugrid[i, j] * v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[n, j], tmask_jp[i, j])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[s, j], tmask_jm[i, j])
        dN = var[n, j] - v
        dS = var[s, j] - v
        dE = var[i, e] - v
        dW = var[i, w] - v
        flux_N = visc_y * abs(dN) * jpD * dN / dy2 * (o - ocnym1[i, j]) - dragN * grdNu[i, j]
        flux_S = visc_y * abs(dS) * jmD * dS / dy2 * (o - ocnyp1[i, j]) - dragS * grdSu[i, j]
        flux_E = visc_x * abs(dE) * D0[i, e] * dE / dx2 * (o - ocnxm1[i, j])
        flux_W = visc_x * abs(dW) * D0[i, j] * dW / dx2 * (o - ocn[i, j])
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _nonlinear_laplace_V_kernel!(
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
    @Const(lndEv),
    @Const(lndWv),
    slip,
    dslip_gl,
    dslip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    Ny,
    Nx,
)
    FT = typeof(slip)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        o = one(FT)
        v = var[i, j]
        dragE =
            A_h_wall *
            (slip + dslip_gl * glEv[i, j] + dslip_land * lndEv[i, j]) *
            D_on_vgrid[i, j] * v / dx2
        dragW =
            A_h_wall *
            (slip + dslip_gl * glWv[i, j] + dslip_land * lndWv[i, j]) *
            D_on_vgrid[i, j] * v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, e], tmask_ip[i, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[i, w], tmask_im[i, j])
        dN = var[n, j] - v
        dS = var[s, j] - v
        dE = var[i, e] - v
        dW = var[i, w] - v
        flux_N = visc_y * abs(dN) * D0[n, j] * dN / dy2 * (o - ocnym1[i, j])
        flux_S = visc_y * abs(dS) * D0[i, j] * dS / dy2 * (o - ocn[i, j])
        flux_E = visc_x * abs(dE) * ipD * dE / dx2 * (o - ocnxm1[i, j]) - dragE * grdEv[i, j]
        flux_W = visc_x * abs(dW) * imD * dW / dx2 * (o - ocnxp1[i, j]) - dragW * grdWv[i, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

function upwind_advection_T(out, m, var)
    ny, nx = size(var)
    if m.open_bc isa ZeroGradientInflow
        launch!(
            _upwind_advection_T_kernel!,
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
            _upwind_advection_T_noinflow_kernel!,
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
function upwind_advection_U(m)
    ny, nx = size(m.U.present)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _upwind_advection_U_kernel!,
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
        m.lndNu,
        m.lndSu,
        m.slip,
        dslip_gl,
        dslip_land,
        m.dx,
        m.dy,
        ny,
        nx,
    )
    return m.cU
end
function upwind_advection_V(m)
    ny, nx = size(m.V.present)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _upwind_advection_V_kernel!,
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
        m.lndEv,
        m.lndWv,
        m.slip,
        dslip_gl,
        dslip_land,
        m.dx,
        m.dy,
        ny,
        nx,
    )
    return m.cV
end
function laplace_T(out, m, var)
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
laplace_U(m) = laplace_U(m, m.lateral_viscosity)
laplace_V(m) = laplace_V(m, m.lateral_viscosity)

# PrescribedLateralViscosity: unchanged geometry-only kernel (bit-identical with
# pre-AbstractLateralViscosity Laddie.jl), scaled by the global A_h afterwards —
# exactly what the momentum kernels used to do themselves.
function laplace_U(m, ::PrescribedLateralViscosity)
    ny, nx = size(m.U.past)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _laplace_U_kernel!,
        m.lU,
        m.lU,
        m.U.past,
        m.D.present,
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
        m.lndNu,
        m.lndSu,
        m.slip,
        dslip_gl,
        dslip_land,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    m.lU .*= m.A_h
    return m.lU
end
function laplace_V(m, ::PrescribedLateralViscosity)
    ny, nx = size(m.V.past)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _laplace_V_kernel!,
        m.lV,
        m.lV,
        m.V.past,
        m.D.present,
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
        m.lndEv,
        m.lndWv,
        m.slip,
        dslip_gl,
        dslip_land,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    m.lV .*= m.A_h
    return m.lV
end

# NonlinearLateralViscosity: the shear-scaled coefficient is per-face, so it must
# be applied inside the kernel rather than as a post-hoc scalar multiply; wall
# drag keeps using the plain, unscaled m.A_h (see _nonlinear_laplace_U_kernel!).
function laplace_U(m, lv::NonlinearLateralViscosity)
    ny, nx = size(m.U.past)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _nonlinear_laplace_U_kernel!,
        m.lU,
        m.lU,
        m.U.past,
        m.D.present,
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
        m.lndNu,
        m.lndSu,
        m.slip,
        dslip_gl,
        dslip_land,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
        ny,
        nx,
    )
    return m.lU
end
function laplace_V(m, lv::NonlinearLateralViscosity)
    ny, nx = size(m.V.past)
    dslip_gl = _gl_slip(m.grline_bc, m.slip) - m.slip
    dslip_land = _land_slip(m.land_bc, m.slip) - m.slip
    launch!(
        _nonlinear_laplace_V_kernel!,
        m.lV,
        m.lV,
        m.V.past,
        m.D.present,
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
        m.lndEv,
        m.lndWv,
        m.slip,
        dslip_gl,
        dslip_land,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
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
        _precompute_D_shifts_kernel!,
        m.Dym1,
        m.Dym1,
        m.Dyp1,
        m.Dxm1,
        m.Dxp1,
        m.Dxm1ym1,
        m.Dxp1ym1,
        m.Dxm1yp1,
        m.D.present,
        m.tmask,
        ny,
        nx,
    )
    # Staggered interpolations, sign, and velocity shifts in one kernel pass
    launch!(
        _precompute_staggered_kernel!,
        m.Vip,
        m.Vip,
        m.Vim,
        m.Vjp,
        m.Vjm,
        m.Uip,
        m.Uim,
        m.Ujp,
        m.Ujm,
        m.signU,
        m.signV,
        m.Vyp1,
        m.Uxp1,
        m.V.present,
        m.U.present,
        m.vmask_ip,
        m.vmask_im,
        m.vmask_jp,
        m.vmask_jm,
        m.umask_ip,
        m.umask_im,
        m.umask_jp,
        m.umask_jm,
        ny,
        nx,
    )
    # Upwind splits — all 8 fields in one kernel pass
    launch!(
        _upwind_split_kernel!,
        m.Upos,
        m.Upos,
        m.Uneg,
        m.Vpos,
        m.Vneg,
        m.Vyp1pos,
        m.Vyp1neg,
        m.Uxp1pos,
        m.Uxp1neg,
        m.U.present,
        m.V.present,
        m.Vyp1,
        m.Uxp1,
    )
    return
end

function precompute_laplacian_stencils!(m)
    ny, nx = size(m.D.present)
    launch!(
        _precompute_laplacian_kernel!,
        m.D0ip,
        m.D0ip,
        m.D0im,
        m.D0jp,
        m.D0jm,
        m.D.present,
        m.tmask_ip,
        m.tmask_im,
        m.tmask_jp,
        m.tmask_jm,
        ny,
        nx,
    )
    @. m.D_on_ugrid = m.D0ip * m.tmask
    @. m.D_on_vgrid = m.D0jp * m.tmask
    return
end