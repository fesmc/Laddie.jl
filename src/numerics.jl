# Robert-Asselin time filter applied to one leapfrog level.
@kernel function _robert_asselin_kernel!(
    present,
    @Const(past),
    @Const(future),
    @Const(mask),
    nu,
)
    nu = _val(nu)
    i, j = @index(Global, NTuple)
    @inbounds present[i, j] +=
        nu / 2 * (past[i, j] + future[i, j] - 2 * present[i, j]) * mask[i, j]
end

# Launch `kernel!(out, args...)` over the full extent of `out`, on its backend.  Every
# kernel writes its first argument, and every array has the grid's size.
# CPU: long blocks along the first (column-major inner) index keep the inner loops
# vectorisable, and a grid smaller than one block runs on a single task instead of
# paying the spawn overhead — measured 1.1–1.8× faster than (8, 8) from 80×20 to
# 1000×1000 cells on 1–16 threads.
_workgroup(::CPU) = (128, 32)
_workgroup(::Any) = (32, 8)   # GPU: 256 threads, warp-aligned x-dimension

launch!(kernel!, out, args...) = _launch!(kernel!, size(out), out, args...)

# Launch a stencil kernel over the interior cells only, the full array minus its
# one-cell border ring.  The kernel offsets its index by one past the ring, so
# `i ± 1`, `j ± 1` always stay inside the array and need no periodic wrap: the
# border ring is never active, and a stencil output there was only ever multiplied
# by a zero mask.  Plain `i ± 1` also keeps the indices affine, which lets Reactant
# raise the neighbour reads to slices; a wrapped read along x (the contiguous axis)
# becomes a gather, and a wrapped diagonal read a gather plus a transpose.
launch_interior!(kernel!, out, args...) = _launch!(kernel!, size(out) .- 2, out, args...)

function _launch!(kernel!, ndrange, out, args...)
    backend = _launch_backend(KA.get_backend(out))
    kernel!(backend, _workgroup(backend))(out, args...; ndrange)
    return nothing
end

# CPU: static thread assignment (`@threads :static` instead of one `@spawn` per
# chunk), so the same chunk of the grid lands on the same thread in every kernel
# and its arrays stay in that core's cache.  With dynamic assignment the chunks
# migrate between cores from kernel to kernel; on the 241×466 ASE grid that made
# 2 threads no faster than 1, and 8 threads only 1.7× (3.3× static, with the
# threads pinned).  `:static` cannot run inside another threaded region, so a
# launch from within `@threads` falls back to dynamic assignment.
_launch_backend(backend) = backend
_launch_backend(backend::CPU) =
    ccall(:jl_in_threaded_region, Cint, ()) == 0 ? CPU(; static = true) : backend


# ==================================================================
# Time integration (Lambert et al. 2023)
# ==================================================================

_update_conv2!(::Any, ::ClampDensity) = nothing
_update_conv2!(::Any, ::ResetToAmbient) = nothing
function _update_conv2!(m, cs::RelaxToAmbient)
    # `imask`, not `tmask`: gap cells are never relaxed towards ambient.  See
    # `update_convection!(m, ::RelaxToAmbient)`.
    launch!(
        _relax_conv2_kernel!,
        m.conv2,
        m.drho,
        m.imask,
        m.D.present,
        cs.convection_time,
    )
end

@kernel function _relax_conv2_kernel!(
    conv2,
    @Const(drho),
    @Const(imask),
    @Const(D),
    convection_time,
)
    convection_time = _val(convection_time)
    i, j = @index(Global, NTuple)
    @inbounds conv2[i, j] = (drho[i, j] < 0) * imask[i, j] * D[i, j] / convection_time
end

# The per-step elementwise updates below are kernels rather than broadcasts: on
# the CPU a broadcast runs on the main thread only, both costing its own time
# serially and pulling the arrays the threaded kernels just wrote across cores.
@kernel function _integration_terms_kernel!(
    dDdt,
    Ddrho,
    @Const(D_future),
    @Const(D_past),
    @Const(D),
    @Const(drho),
    dt,
)
    dt = _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        dDdt[i, j] = (D_future[i, j] - D_past[i, j]) / (dt + dt)
        Ddrho[i, j] = D[i, j] * drho[i, j]
    end
end

# `dt` is the base time step.  dD/dt is always taken over 2·dt, as in Python LADDIE
# v1 (`prepare_integrate`), although `D.future` and `D.past` are only dt apart in
# the bootstrap step, which therefore sees half the thickness tendency.  Dividing
# by the true step length instead is not a free fix: the ISOMIP+ spin-up is
# sensitive to that one step, and the 1-day Python verification then fails
# (max |ΔD| 3 → 35 m, mean melt −3 %).
function precompute_integration_terms!(m, dt)
    launch!(
        _integration_terms_kernel!,
        m.dDdt,
        m.Ddrho,
        m.D.future,
        m.D.past,
        m.D.present,
        m.drho,
        dt,
    )
    _update_conv2!(m, m.convection_scheme)
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
    dt = _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = ifelse(
        iszero(tmask[i, j]),
        D0[i, j],
        D0[i, j] + (convD[i, j] + melt[i, j] + nentr[i, j]) * dt,
    )
end

@kernel function _step_u_momentum_kernel!(
    out,
    @Const(Up),
    @Const(U1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(D1),
    @Const(drho),
    @Const(dzdx),
    @Const(V1),
    @Const(detr),
    @Const(cU),
    @Const(lU),
    @Const(tmask),
    @Const(umask),
    @Const(vmask),
    @Const(fu),
    g,
    C_d,
    pgf_w,
    dx,
    dt,
)
    g, C_d, pgf_w, dx, dt = _val(g), _val(C_d), _val(pgf_w), _val(dx), _val(dt)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(g)
        half = FT(1/2)
        ip1 = i + 1
        jm1 = j - 1
        tmip = tmask[i, j] + tmask[ip1, j]
        ip_dDdt = _safe_div(dDdt[i, j] + dDdt[ip1, j], tmip)
        ip_D_drho = _safe_div(Ddrho[i, j] + Ddrho[ip1, j], tmip)
        ip_D_dzdx = _safe_div(Ddrho[i, j] * dzdx[i, j] + Ddrho[ip1, j] * dzdx[ip1, j], tmip)
        ip_D = _safe_div(D1[i, j] + D1[ip1, j], tmip)
        Vjm = _face_avg(V1, vmask, i, j, i, jm1)
        Vjm_ip = _face_avg_ring0(V1, vmask, ip1, j, 0, -1)
        ip_D_Vjm = _safe_div(D1[i, j] * Vjm + D1[ip1, j] * Vjm_ip, tmip)
        ipjmV = half * (half * (V1[i, j] + V1[i, jm1]) + half * (V1[ip1, j] + V1[ip1, jm1]))
        # tmip is 2 at a fully-interior face (both neighbours active) and 1 at a
        # one-sided face (ice front, or a SinkGapsBC gap-sink edge), where the
        # masked neighbour thickness is a zero stand-in rather than a real one.
        # pgf_w selects what happens there: 0 (FullDepthGradient, the Python
        # v1.x behaviour) keeps the term and leaves this an exact multiply by
        # 1.0; 1 (TruncatedDepthGradient) drops it, as LADDIE v2 does at
        # mask_cf_b faces.  See AbstractFrontPressure.
        pgf_gate = one(FT) + pgf_w * (tmip - FT(2))
        D_next = D1[ip1, j] * tmask[ip1, j]
        rhs =
            -U1[i, j] * ip_dDdt +                                      # thickness-tendency correction
            cU[i, j] +                                                  # horizontal advection
            -g * ip_D_drho * (D_next - D1[i, j]) / dx * pgf_gate +     # pressure: D gradient
            g * ip_D_dzdx +                                             # pressure: ice-shelf slope
            -half * g * ip_D^2 * (drho[ip1, j] - drho[i, j]) / dx +     # pressure: density gradient
            fu[i, j] * ip_D_Vjm +                                              # Coriolis
            -C_d * U1[i, j] * _safe_sqrt(U1[i, j]^2 + ipjmV^2) +             # quadratic drag
            lU[i, j] +                                                   # horizontal viscosity
            -detr[i, j] * U1[i, j]                                     # momentum loss by detrainment
        out[i, j] = Up[i, j] + _safe_div(rhs, ip_D) * umask[i, j] * dt
    end
end

@kernel function _step_v_momentum_kernel!(
    out,
    @Const(Vp),
    @Const(V1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(D1),
    @Const(drho),
    @Const(dzdy),
    @Const(U1),
    @Const(detr),
    @Const(cV),
    @Const(lV),
    @Const(tmask),
    @Const(vmask),
    @Const(umask),
    @Const(fv),
    g,
    C_d,
    pgf_w,
    dy,
    dt,
)
    g, C_d, pgf_w, dy, dt = _val(g), _val(C_d), _val(pgf_w), _val(dy), _val(dt)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(g)
        half = FT(0.5)
        jp1 = j + 1
        im1 = i - 1
        tmjp = tmask[i, j] + tmask[i, jp1]
        jp_dDdt = _safe_div(dDdt[i, j] + dDdt[i, jp1], tmjp)
        jp_D_drho = _safe_div(Ddrho[i, j] + Ddrho[i, jp1], tmjp)
        jp_D_dzdy = _safe_div(Ddrho[i, j] * dzdy[i, j] + Ddrho[i, jp1] * dzdy[i, jp1], tmjp)
        jp_D = _safe_div(D1[i, j] + D1[i, jp1], tmjp)
        Uim = _face_avg(U1, umask, i, j, im1, j)
        Uim_jp = _face_avg_ring0(U1, umask, i, jp1, -1, 0)
        jp_D_Uim = _safe_div(D1[i, j] * Uim + D1[i, jp1] * Uim_jp, tmjp)
        jpimU = half * (half * (U1[i, j] + U1[im1, j]) + half * (U1[i, jp1] + U1[im1, jp1]))
        # See _step_u_momentum_kernel! for the ice-front gate.
        pgf_gate = one(FT) + pgf_w * (tmjp - FT(2))
        D_next = D1[i, jp1] * tmask[i, jp1]
        rhs =
            -V1[i, j] * jp_dDdt +                                      # thickness-tendency correction
            cV[i, j] +                                                  # horizontal advection
            -g * jp_D_drho * (D_next - D1[i, j]) / dy * pgf_gate +     # pressure: D gradient
            g * jp_D_dzdy +                                             # pressure: ice-shelf slope
            -half * g * jp_D^2 * (drho[i, jp1] - drho[i, j]) / dy +     # pressure: density gradient
            -fv[i, j] * jp_D_Uim +                                             # Coriolis
            -C_d * V1[i, j] * _safe_sqrt(V1[i, j]^2 + jpimU^2) +             # quadratic drag
            lV[i, j] +                                                   # horizontal viscosity
            -detr[i, j] * V1[i, j]                                     # momentum loss by detrainment
        out[i, j] = Vp[i, j] + _safe_div(rhs, jp_D) * vmask[i, j] * dt
    end
end

# A kernel argument that is either a per-cell field or one value for the whole
# domain (gamT and conv2 are scalars or matrices depending on the melt and
# convection schemes); dispatch picks the right read at compile time.
@inline _at(x::Number, i, j) = x
@inline _at(x::AbstractArray, i, j) = @inbounds x[i, j]

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
    K_h,
    conv2,
    dt,
)
    gamT, K_h, conv2, dt = _val(gamT), _val(K_h), _val(conv2), _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +                 # thickness-tendency correction
            cT[i, j] +                                      # horizontal advection
            nentr[i, j] * Ta[i, j] +                        # entrainment of ambient water at Ta
            melt[i, j] * Tb[i, j] +                         # meltwater input at freezing point
            -_at(gamT, i, j) * (T_present[i, j] - Tb[i, j]) + # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * _at(conv2, i, j)   # convective restoring to ambient
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
    K_h,
    conv2,
    dt,
)
    K_h, conv2, dt = _val(K_h), _val(conv2), _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -S_present[i, j] * dDdt[i, j] +                 # thickness-tendency correction
            cS[i, j] +                                      # horizontal advection
            nentr[i, j] * Sa[i, j] +                        # entrainment of ambient water at Sa
            K_h * lS[i, j] +                                # horizontal diffusion
            -(S_past[i, j] - Sa[i, j]) * _at(conv2, i, j)   # convective restoring to ambient
        out[i, j] = S_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

function step_thickness!(m, dt)
    launch!(
        _step_thickness_kernel!,
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
function step_u_momentum!(m, dt)
    upwind_advection_U(m)
    laplace_U(m)
    launch_interior!(
        _step_u_momentum_kernel!,
        m.U.future,
        m.U.past,
        m.U.present,
        m.dDdt,
        m.Ddrho,
        m.D.present,
        m.drho,
        m.dzdx,
        m.V.present,
        m.detr,
        m.adv,
        m.lap,
        m.tmask,
        m.umask,
        m.vmask,
        m.fu,
        m.g,
        m.C_d,
        _front_pgf_weight(m.front_pressure, m.g),
        m.dx,
        dt,
    )
    return
end
function step_v_momentum!(m, dt)
    upwind_advection_V(m)
    laplace_V(m)
    launch_interior!(
        _step_v_momentum_kernel!,
        m.V.future,
        m.V.past,
        m.V.present,
        m.dDdt,
        m.Ddrho,
        m.D.present,
        m.drho,
        m.dzdy,
        m.U.present,
        m.detr,
        m.adv,
        m.lap,
        m.tmask,
        m.vmask,
        m.umask,
        m.fv,
        m.g,
        m.C_d,
        _front_pgf_weight(m.front_pressure, m.g),
        m.dy,
        dt,
    )
    return
end
function step_temperature!(m, dt)
    launch!(_product_kernel!, m.Dq, m.D.present, m.T.present)
    upwind_advection_T(m.adv, m, m.Dq)
    laplace_T(m.lap, m, m.T.past)
    args = (
        m.T.future,
        m.T.past,
        m.T.present,
        m.dDdt,
        m.adv,
        m.nentr,
        m.Ta,
        m.melt,
        m.Tb,
        m.lap,
        m.D.present,
        m.tmask,
    )
    gamT = first(_exchange_velocities(m))
    launch!(_step_temperature_kernel!, args..., gamT, m.K_h, m.conv2, dt)
    return
end
function step_salinity!(m, dt)
    launch!(_product_kernel!, m.Dq, m.D.present, m.S.present)
    upwind_advection_T(m.adv, m, m.Dq)
    laplace_T(m.lap, m, m.S.past)
    args = (
        m.S.future,
        m.S.past,
        m.S.present,
        m.dDdt,
        m.adv,
        m.nentr,
        m.Sa,
        m.lap,
        m.D.present,
        m.tmask,
    )
    launch!(_step_salinity_kernel!, args..., m.K_h, m.conv2, dt)
    return
end


@kernel function _product_kernel!(out, @Const(a), @Const(b))
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[i, j] * b[i, j]
end

# The upper bound of `max_layer_thickness` (see `_max_layer_thickness`), then the
# D_min floor, in one pass.
function _clamp_thickness!(m)
    launch!(
        _clamp_thickness_kernel!,
        m.D.future,
        m.z_draft,
        m.z_bed,
        m.tmask,
        m.params.max_layer_thickness,
        m.D_min,
    )
    return
end

@kernel function _clamp_thickness_kernel!(
    D,
    @Const(z_draft),
    @Const(z_bed),
    @Const(tmask),
    max_layer_thickness,
    D_min,
)
    D_min = _val(D_min)
    i, j = @index(Global, NTuple)
    @inbounds begin
        D[i, j] = _max_layer_thickness(
            max_layer_thickness,
            D[i, j],
            z_draft[i, j],
            z_bed[i, j],
            tmask[i, j],
        )
        # The D_min floor must respect the domain mask: inactive cells hold D = 0.
        # A non-zero D outside the domain would leak into the interior, because
        # the face averages of the Laplacian stencils (`_face_avg`) divide the
        # *sum* over a cell pair by the number of active cells in it, biasing the
        # diffusion of every boundary cell.
        D[i, j] = max(D[i, j], D_min) * tmask[i, j]
    end
end

# Tracer bounds inside the domain only; see `leapfrog_step!`.
@kernel function _clamp_tracer_kernel!(q, @Const(tmask), q_min, q_max)
    q_min, q_max = _val(q_min), _val(q_max)
    i, j = @index(Global, NTuple)
    @inbounds q[i, j] = ifelse(tmask[i, j] > 0, clamp(q[i, j], q_min, q_max), q[i, j])
end

@kernel function _nan_on_shelf_kernel!(flag, @Const(arr), @Const(tmask))
    i, j = @index(Global, NTuple)
    @inbounds flag[i, j] = isnan(arr[i, j]) & (tmask[i, j] > 0)
end

function _check_nans_shelf!(sim, varname, arr)
    m = sim.model
    launch!(_nan_on_shelf_kernel!, m.diag, arr, m.tmask)
    any(>(0), m.diag) && error(
        "NaN in $varname at t = $(round(_t_days(sim), digits=4)) days " *
        "(iteration $(sim.clock.iteration))",
    )
end

# One leapfrog integration over `nsteps × dt`: `nsteps = 2` is the centred step,
# `nsteps = 1` the first-order bootstrap.  dD/dt keeps the base `dt` (see
# `precompute_integration_terms!`).
function leapfrog_step!(sim, nsteps)
    m = sim.model
    dt = nsteps * sim.clock.dt
    check_nans = sim.debug.check_nans
    step_thickness!(m, dt)
    _clamp_thickness!(m)
    precompute_integration_terms!(m, sim.clock.dt)
    check_nans && _check_nans_shelf!(sim, "D", m.D.future)

    # Both momentum components are stepped before the limiter, because it caps the
    # speed and so needs U and V together (see `clamp_velocities!`).
    step_u_momentum!(m, dt)
    step_v_momentum!(m, dt)
    clamp_velocities!(m)
    check_nans && _check_nans_shelf!(sim, "U", m.U.future)
    check_nans && _check_nans_shelf!(sim, "V", m.V.future)

    # Tracer bounds (`Params.T_min` … `S_max`).  LADDIE v2 has no counterpart — it
    # assigns T and S only from the flux-form integration and never bounds them —
    # and they carry a real cost:
    # with the tracers pinned the melt rate is bounded too, so an unstable run stays
    # finite and `_check_blowup` cannot see it.  They are kept because they hold
    # real-world domains together.
    #
    # Applied inside the domain only: unmasked, the salinity floor would raise every
    # inactive cell from S = 0 to S_min, breaking the invariant that prognostics are
    # zero outside `tmask` (see `_clamp_thickness!`).
    step_temperature!(m, dt)
    launch!(_clamp_tracer_kernel!, m.T.future, m.tmask, m.T_min, m.T_max)
    check_nans && _check_nans_shelf!(sim, "T", m.T.future)

    step_salinity!(m, dt)
    launch!(_clamp_tracer_kernel!, m.S.future, m.tmask, m.S_min, m.S_max)
    check_nans && _check_nans_shelf!(sim, "S", m.S.future)
    return
end

# Start the leapfrog: refresh secondary fields at the current dt, then take one
# first-order step.  Run when a Simulation is constructed on a freshly
# initialised model (whose three time levels are identical), and by
# `init_from_restart!` after loading the saved levels.
function _bootstrap_leapfrog!(sim)
    update_secondary_fields!(sim.model, sim.clock.dt)
    leapfrog_step!(sim, 1)
    return
end

# Re-initialise the leapfrog after a dt change: collapse the `past` level onto
# `present` so the two are co-located in time, then bootstrap at the new dt.  The
# next `advance_leapfrog!` rotation leaves a past/present pair separated by the
# new dt, so the following centred `leapfrog_step!(sim, 2)` is consistent.  The
# anchor is the Robert–Asselin-filtered `present`, exactly as at startup.
_rebootstrap_leapfrog!(sim) = _rebootstrap_leapfrog!(sim.exec, sim)
_rebootstrap_leapfrog!(::NativeExecution, sim) = _collapse_and_bootstrap!(sim)
function _collapse_and_bootstrap!(sim)
    m = sim.model
    for var in (m.D, m.U, m.V, m.T, m.S)
        var.past .= var.present
    end
    _bootstrap_leapfrog!(sim)
    return
end

# ============================================================================
# Time-stepping orchestration
# ============================================================================

function apply_robert_asselin_filter!(sim)
    m = sim.model
    for (var, mask) in
        ((m.D, m.tmask), (m.U, m.umask), (m.V, m.vmask), (m.T, m.tmask), (m.S, m.tmask))
        launch!(
            _robert_asselin_kernel!,
            var.present,
            var.past,
            var.future,
            mask,
            sim.nu,
        )
    end
    # Refresh density and convection on the filtered level, as Python LADDIE v1
    # does.  The next `advance_leapfrog!` recomputes both on the rotated level, but
    # this call is not redundant: under ResetToAmbient it resets the filtered T/S,
    # which become `past`, and for every scheme it is the `drho` and `convection`
    # that the output, the diagnostics and the CFL number read.
    update_density!(m)
    update_convection!(m)
    return
end

function advance_leapfrog!(sim)
    m = sim.model
    for var in (m.D, m.U, m.V, m.T, m.S)
        rotate!(var)
    end
    update_secondary_fields!(m, sim.clock.dt)
    return
end

# Per-point scale factors for the speed limiter.  Reads U and V, writes neither,
# so the factors are all computed from the unlimited field and the result cannot
# depend on the order the components are written.
#
# On the C-grid U and V are not co-located, so the partner component is averaged
# onto the point being limited — the same four-point stencil the bottom-drag
# terms use (`u_bottom_drag` / `v_bottom_drag` in physics.jl).
@kernel function _speed_scale_kernel!(sU, sV, @Const(U), @Const(V), v_cut)
    v_cut = _val(v_cut)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(v_cut)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        Vbar = (V[i, j] + V[i, jm1] + V[ip1, j] + V[ip1, jm1]) / FT(4)   # V at the U-point
        Ubar = (U[i, j] + U[im1, j] + U[i, jp1] + U[im1, jp1]) / FT(4)   # U at the V-point
        spdU = _safe_sqrt(U[i, j] * U[i, j] + Vbar * Vbar)
        spdV = _safe_sqrt(V[i, j] * V[i, j] + Ubar * Ubar)
        sU[i, j] = spdU > v_cut ? v_cut / spdU : one(FT)
        sV[i, j] = spdV > v_cut ? v_cut / spdV : one(FT)
    end
end

@kernel function _apply_scale_kernel!(U, V, @Const(sU), @Const(sV))
    i, j = @index(Global, NTuple)
    @inbounds begin
        U[i, j] = U[i, j] * sU[i, j]
        V[i, j] = V[i, j] * sV[i, j]
    end
end

"""
$(TYPEDSIGNATURES)

Cap the flow **speed** at `Params.v_cut` by scaling both velocity components with
one factor, `(U, V) *= min(1, v_cut / |u|)`, so the flow direction is preserved.

This mirrors LADDIE v2 (`laddie_velocity.f90`, "Cutoff velocities to ensure
Uabs <= Uabs_max").  Clamping each component independently would rotate the
velocity vector whenever one component saturates and the other does not, and
admit speeds up to `√2 · v_cut` along the diagonal.

On the C-grid each component is scaled with the speed at its own point, formed
with a 4-point average of the other component; all factors are computed from the
unlimited field before either component is scaled.
"""
function clamp_velocities!(m)
    # The scale factors live in the `adv`/`lap` work buffers: both momentum steps
    # have consumed them, and the tracer steps overwrite them.
    scaleU, scaleV = m.adv, m.lap
    launch_interior!(
        _speed_scale_kernel!,
        scaleU,
        scaleV,
        m.U.future,
        m.V.future,
        m.v_cut,
    )
    launch!(_apply_scale_kernel!, m.U.future, m.V.future, scaleU, scaleV)
    return
end
