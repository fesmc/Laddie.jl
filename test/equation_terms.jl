# Equation-term functions — one named function per term in each prognostic
# equation. Functions return the term value; the caller applies the sign,
# making the step functions read like the written equations.  Included by setup.jl.
#
# These are the readable REFERENCE implementation of the governing equations,
# written with the whole-array shift and interpolation primitives of src/utils.jl.
# The momentum advection and diffusion terms are copied out of the shared work
# buffers (`Cache.adv`, `Cache.lap`), which the next term would overwrite.
# The time loop runs the fused kernels in src/numerics.jl instead (one pass per
# prognostic, no intermediate allocations); the testset "Fused kernels match
# reference equation terms" (numerics.jl) asserts that the kernels reproduce these
# terms exactly, so the two cannot drift apart silently.
# All equation references are to Lambert et al. (2023), The Cryosphere,
# https://doi.org/10.5194/tc-17-3203-2023.
#
# Notation shared across all five governing equations:
#   H       plume layer thickness [m]
#   U, V    depth-averaged x- and y-velocity components [m s⁻¹]
#   T, S    depth-averaged plume temperature [°C] and salinity [PSU]
#   ṁ       basal melt rate [m s⁻¹]; positive = melting
#   ė       net entrainment rate, ė = entr − detr [m s⁻¹]
#   Tₐ, Sₐ  ambient temperature and salinity interpolated to plume depth
#   Tb      ice–ocean boundary (basal) temperature [°C]
#   δρ      reduced density contrast with ambient, (ρₐ − ρ)/ρ₀ [–]
#   H̄       layer thickness face-interpolated to the velocity node
#   f       Coriolis parameter [s⁻¹]; `fu`/`fv` are its u- and v-face averages
#   g       gravitational acceleration [m s⁻²]
#   ρ₀      reference seawater density [kg m⁻³]
#   z_draft      ice-base depth, negative below sea level [m]
#   C_d     quadratic drag coefficient at the ice base [–]
#   |u|     current speed, √(U² + V²) [m s⁻¹]
#   A_h      horizontal viscosity [m² s⁻¹]
#   K_h      horizontal diffusivity [m² s⁻¹]
#   γT      turbulent heat transfer coefficient [m s⁻¹]
# ============================================================================
using Laddie: xm1, ym1, xp1, yp1, ip_half, jp_half, im_half, jm_half, ip_count, jp_count, ip_t, jp_t,
    im_u, jm_v, _safe_sqrt, _front_pgf_weight, _front_drho_weight, momentum_advection_U, momentum_advection_V,
    laplace_U, laplace_V, laplace_T, upwind_advection_T

# -- U-momentum terms (Eq. 2) -----------------------------------------------

# U·∂H/∂t  (thickness-tendency coupling)
@inline u_thickness_tendency(m) = m.U.present .* ip_t(m, m.dHdt)
# ∇·(HUu)  (momentum advection)
@inline u_advection(m) = copy(momentum_advection_U(m))
# Per-face weight on the depth-gradient PGF term: always 1 on a fully-interior
# face; at a one-sided face (ice front, SinkGapsBC gap-sink edge) 1 under
# FullDepthGradient and 0 under TruncatedDepthGradient.  See AbstractFrontPressure.
@inline _pgf_gate(m, tmask_stag) =
    one(m.FT) .+ _front_pgf_weight(m.front_pressure, m.g) .* (tmask_stag .- 2)
# The same gate on the density-gradient PGF term: 0 at a one-sided face only
# under ZeroGradientFront.
@inline _drho_gate(m, tmask_stag) =
    one(m.FT) .+ _front_drho_weight(m.front_pressure, m.g) .* (tmask_stag .- 2)

# g·H̄·ρ̄·∂H/∂x  (pressure gradient from plume-thickness depth)
@inline u_pressure_depth(m) =
    m.g .* ip_t(m, m.Hdrho) .* (xm1(m.H.present .* m.tmask) .- m.H.present) ./ m.dx .*
    _pgf_gate(m, ip_count(m.tmask))
# g·H̄·ρ̄·∂z_draft/∂x  (baroclinic pressure via ice-base slope)
@inline u_pressure_slope(m) = m.g .* ip_t(m, m.Hdrho .* m.dzdx)
# ½g·H̄²·∂δρ/∂x  (internal pressure gradient)
@inline u_pressure_density(m) =
    (m.g / 2) .* ip_t(m, m.H.present) .^ 2 .* (xm1(m.drho) .- m.drho) ./ m.dx .*
    _drho_gate(m, ip_count(m.tmask))
# f·H̄·V  (Coriolis)
@inline u_coriolis(m) = m.fu .* ip_t(m, m.H.present .* jm_v(m, m.V.present))
# |u| on the U points, at the level of `Params.closure_level`
@inline u_drag_speed(m) =
    _safe_sqrt.(closure_level(m, m.U) .^ 2 .+ ip_half(jm_half(closure_level(m, m.V))) .^ 2)
# Cd·U·|u|  (quadratic bottom drag)
@inline u_bottom_drag(m) = m.C_d .* m.U.present .* u_drag_speed(m)
# Ah·∇²(HU)  (lateral diffusion; the A_h/shear scaling is applied inside
# laplace_U, since it dispatches on `Params.lateral_viscosity`)
@inline u_diffusion(m) = copy(laplace_U(m))
# e·U  (detrainment momentum loss)
@inline u_detrainment(m) = (m.detr .+ ip_t(m, m.detr_cap) .+ ip_t(m, frazil_field(m))) .* m.U.present

# -- V-momentum terms (Eq. 3) -----------------------------------------------

# V·∂H/∂t  (thickness-tendency coupling)
@inline v_thickness_tendency(m) = m.V.present .* jp_t(m, m.dHdt)
# ∇·(HVv)  (momentum advection)
@inline v_advection(m) = copy(momentum_advection_V(m))
# g·H̄·ρ̄·∂H/∂y  (pressure gradient from plume-thickness depth; see u_pressure_depth)
@inline v_pressure_depth(m) =
    m.g .* jp_t(m, m.Hdrho) .* (ym1(m.H.present .* m.tmask) .- m.H.present) ./ m.dy .*
    _pgf_gate(m, jp_count(m.tmask))
# g·H̄·ρ̄·∂z_draft/∂y  (baroclinic pressure via ice-base slope)
@inline v_pressure_slope(m) = m.g .* jp_t(m, m.Hdrho .* m.dzdy)
# ½g·H̄²·∂δρ/∂y  (internal pressure gradient)
@inline v_pressure_density(m) =
    (m.g / 2) .* jp_t(m, m.H.present) .^ 2 .* (ym1(m.drho) .- m.drho) ./ m.dy .*
    _drho_gate(m, jp_count(m.tmask))
# f·H̄·U  (Coriolis)
@inline v_coriolis(m) = m.fv .* jp_t(m, m.H.present .* im_u(m, m.U.present))
# |u| on the V points, at the level of `Params.closure_level`
@inline v_drag_speed(m) =
    _safe_sqrt.(closure_level(m, m.V) .^ 2 .+ jp_half(im_half(closure_level(m, m.U))) .^ 2)
# Cd·V·|u|  (quadratic bottom drag)
@inline v_bottom_drag(m) = m.C_d .* m.V.present .* v_drag_speed(m)
# Ah·∇²(HV)  (lateral diffusion; the A_h/shear scaling is applied inside
# laplace_V, since it dispatches on `Params.lateral_viscosity`)
@inline v_diffusion(m) = copy(laplace_V(m))
# ė·V  (detrainment momentum loss)
@inline v_detrainment(m) = (m.detr .+ jp_t(m, m.detr_cap) .+ jp_t(m, frazil_field(m))) .* m.V.present

# The prognostic level the closures read (`Params.closure_level`).
closure_level(m, v) = m.params.closure_level isa PastClosures ? v.past : v.present

# One momentum step of U from `rhs`, the momentum terms without U·∂H/∂t, drag and
# detrainment, in the model's `momentum_form`: the advective form adds those terms
# at the present level and divides by H̄; the flux form steps H̄U, divides by the new
# face thickness, and takes the detrainments, frazil and drag at the new level.
u_momentum_step(m, rhs, dt) = momentum_step(m.params.momentum_form, m, m.U, rhs, dt, ip_t, m.umask,
    u_thickness_tendency(m), u_bottom_drag(m), u_detrainment(m), u_drag_speed(m))
v_momentum_step(m, rhs, dt) = momentum_step(m.params.momentum_form, m, m.V, rhs, dt, jp_t, m.vmask,
    v_thickness_tendency(m), v_bottom_drag(m), v_detrainment(m), v_drag_speed(m))
momentum_step(::AdvectiveMomentumForm, m, U, rhs, dt, face, mask, tendency, drag, detrainment, _) =
    U.past .+ Laddie.div0(rhs .- tendency .- drag .- detrainment, face(m, m.H.present)) .* mask .* dt
# The outflow rate of the upstream advection, which the flux form takes at the new
# level; zero under centred advection.  (Its formula is tested on its own.)
function momentum_outflow(m, advect)
    m.momentum_advection isa UpstreamMomentumAdvection || return zero(m.U.present)
    buf = zero(m.U.present)
    advect(m, buf)
    return buf
end
# Divides by the stepped thickness, before `_clamp_thickness!`; the kernel's
# max(H⁺, H*) is H⁺ here.
function momentum_step(::FluxMomentumForm, m, U, rhs, dt, face, mask, _, _, _, speed)
    outflow = momentum_outflow(m, U === m.U ? Laddie.momentum_advection_U : Laddie.momentum_advection_V)
    sink = face(m, m.detr) .+ face(m, m.detr_cap) .+ face(m, frazil_field(m)) .+ m.C_d .* speed .+ outflow
    transport = face(m, m.H.past) .* U.past .+ (rhs .+ outflow .* U.present) .* dt
    return ifelse.(mask .> 0, Laddie.div0(transport, face(m, m.H.future) .+ sink .* dt), U.past)
end

# -- Tracer terms (Eqs. 4–5) -------------------------------------------------

# q·∂H/∂t  (thickness-tendency coupling)
@inline tracer_thickness_tendency(m, q) = q .* m.dHdt
# ∇·(H·u·q)  (horizontal tracer advection)
@inline tracer_advection(m, q) =
    upwind_advection_T(similar(m.H.present), m, m.H.present .* q)
# e_net·qa  (entrainment of ambient water)
@inline tracer_entrainment(m, qa) = m.nentr .* qa
# d_cap·q  (cap detrainment of layer water; see `detr_cap`)
@inline tracer_cap_detrainment(m, q) = m.detr_cap .* q

# One tracer step of the Var `q` from `rhs`, the tracer terms without q·∂H/∂t, in
# the model's `tracer_form`: the advective form adds that term and divides by
# H, the flux form steps the content H·q and divides by the new thickness.
# `extra_sink` is any further removal of layer water at q (frazil, for T).
tracer_step(m, q, rhs, dt; extra_sink = 0) =
    tracer_step(m.params.tracer_form, m, q, rhs, dt, extra_sink)
tracer_step(::AdvectiveTracerForm, m, q, rhs, dt, _) =
    q.past .+ Laddie.div0(rhs .- tracer_thickness_tendency(m, q.present), m.H.present) .*
              m.tmask .* dt
# The flux form divides by the stepped thickness, before `_clamp_thickness!`, and
# takes the terms that remove water at the layer's own q at the new level: the
# outflow through the faces, the cap detrainment and `extra_sink`.
function tracer_step(::FluxTracerForm, m, q, rhs, dt, extra_sink)
    z = zero(eltype(m.U.present))
    U, V = m.U.present, m.V.present
    outflow = m.H.present .* (
        (max.(V, z) .* m.vmask .- min.(yp1(V), z) .* yp1(m.vmask)) ./ m.dy .+
        (max.(U, z) .* m.umask .- min.(xp1(U), z) .* xp1(m.umask)) ./ m.dx)
    sink = outflow .+ m.detr_cap .+ extra_sink
    content = m.H.past .* q.past .+ (rhs .+ sink .* q.present) .* dt
    return ifelse.(m.tmask .> 0, Laddie.div0(content, m.H.future .+ sink .* dt), q.past)
end
# Kh·∇²q  (horizontal diffusion)
@inline tracer_diffusion(m, q_past) = m.K_h .* laplace_T(similar(q_past), m, q_past)
# (q_past − qa)·conv2  (convective relaxation, RelaxToAmbient only)
@inline tracer_convection(m, q_past, qa) = (q_past .- qa) .* m.conv2
# ḟ·(L/c_p − T)  (frazil: latent heat released, water removed at T; temperature only,
# since the salt stays behind)
@inline T_frazil(m) = frazil_field(m) .* (m.L / m.c_p .- m.T.present)
# ṁ·Tb − γT·(T − Tb)  (ice-ocean heat exchange; temperature equation only)
@inline T_ice_ocean_exchange(m) = m.melt .* m.Tb .- m.gamT .* (m.T.present .- m.Tb)

# The frazil formation rate as a field: `model.frazil` is a scalar zero under
# NoFrazilIce.
frazil_field(m) = m.frazil .+ zero(m.H.present)

# Set T at the present and future levels (past and present after the next
# `advance_leapfrog!`) to `dT` below the freezing point at the ice base, under the
# ice cover only.  ISOMIP+ never becomes supercooled on its own.
function supercool!(m, dT)
    for (T, S) in ((m.T.present, m.S.present), (m.T.future, m.S.future))
        Tf = m.l1 .* S .+ m.l2 .+ m.l3 .* m.z_draft
        T .= ifelse.(m.imask .> 0, Tf .- dT, T)
    end
    return m
end
