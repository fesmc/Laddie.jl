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
#   D       plume layer thickness [m]
#   U, V    depth-averaged x- and y-velocity components [m s⁻¹]
#   T, S    depth-averaged plume temperature [°C] and salinity [PSU]
#   ṁ       basal melt rate [m s⁻¹]; positive = melting
#   ė       net entrainment rate, ė = entr − detr [m s⁻¹]
#   Tₐ, Sₐ  ambient temperature and salinity interpolated to plume depth
#   Tb      ice–ocean boundary (basal) temperature [°C]
#   δρ      reduced density contrast with ambient, (ρₐ − ρ)/ρ₀ [–]
#   D̄       layer thickness face-interpolated to the velocity node
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
using Laddie: xm1, ym1, ip_half, jp_half, im_half, jm_half, ip_count, jp_count, ip_t, jp_t,
    im_u, jm_v, _safe_sqrt, _front_pgf_weight, momentum_advection_U, momentum_advection_V,
    laplace_U, laplace_V, laplace_T, upwind_advection_T

# -- U-momentum terms (Eq. 2) -----------------------------------------------

# U·∂D/∂t  (thickness-tendency coupling)
@inline u_thickness_tendency(m) = m.U.present .* ip_t(m, m.dDdt)
# ∇·(DUu)  (momentum advection)
@inline u_advection(m) = copy(momentum_advection_U(m))
# Per-face weight on the depth-gradient PGF term: always 1 on a fully-interior
# face; at a one-sided face (ice front, SinkGapsBC gap-sink edge) 1 under
# FullDepthGradient and 0 under TruncatedDepthGradient.  See AbstractFrontPressure.
@inline _pgf_gate(m, tmask_stag) =
    one(m.FT) .+ _front_pgf_weight(m.front_pressure, m.g) .* (tmask_stag .- 2)

# g·D̄·ρ̄·∂D/∂x  (pressure gradient from plume-thickness depth)
@inline u_pressure_depth(m) =
    m.g .* ip_t(m, m.Ddrho) .* (xm1(m.D.present .* m.tmask) .- m.D.present) ./ m.dx .*
    _pgf_gate(m, ip_count(m.tmask))
# g·D̄·ρ̄·∂z_draft/∂x  (baroclinic pressure via ice-base slope)
@inline u_pressure_slope(m) = m.g .* ip_t(m, m.Ddrho .* m.dzdx)
# ½g·D̄²·∂δρ/∂x  (internal pressure gradient)
@inline u_pressure_density(m) =
    (m.g / 2) .* ip_t(m, m.D.present) .^ 2 .* (xm1(m.drho) .- m.drho) ./ m.dx
# f·D̄·V  (Coriolis)
@inline u_coriolis(m) = m.fu .* ip_t(m, m.D.present .* jm_v(m, m.V.present))
# Cd·U·|u|  (quadratic bottom drag)
@inline u_bottom_drag(m) =
    m.C_d .* m.U.present .*
    _safe_sqrt.(m.U.present .^ 2 .+ ip_half(jm_half(m.V.present)) .^ 2)
# Ah·∇²(DU)  (lateral diffusion; the A_h/shear scaling is applied inside
# laplace_U, since it dispatches on `Params.lateral_viscosity`)
@inline u_diffusion(m) = copy(laplace_U(m))
# e·U  (detrainment momentum loss)
@inline u_detrainment(m) = m.detr .* m.U.present

# -- V-momentum terms (Eq. 3) -----------------------------------------------

# V·∂D/∂t  (thickness-tendency coupling)
@inline v_thickness_tendency(m) = m.V.present .* jp_t(m, m.dDdt)
# ∇·(DVv)  (momentum advection)
@inline v_advection(m) = copy(momentum_advection_V(m))
# g·D̄·ρ̄·∂D/∂y  (pressure gradient from plume-thickness depth; see u_pressure_depth)
@inline v_pressure_depth(m) =
    m.g .* jp_t(m, m.Ddrho) .* (ym1(m.D.present .* m.tmask) .- m.D.present) ./ m.dy .*
    _pgf_gate(m, jp_count(m.tmask))
# g·D̄·ρ̄·∂z_draft/∂y  (baroclinic pressure via ice-base slope)
@inline v_pressure_slope(m) = m.g .* jp_t(m, m.Ddrho .* m.dzdy)
# ½g·D̄²·∂δρ/∂y  (internal pressure gradient)
@inline v_pressure_density(m) =
    (m.g / 2) .* jp_t(m, m.D.present) .^ 2 .* (ym1(m.drho) .- m.drho) ./ m.dy
# f·D̄·U  (Coriolis)
@inline v_coriolis(m) = m.fv .* jp_t(m, m.D.present .* im_u(m, m.U.present))
# Cd·V·|u|  (quadratic bottom drag)
@inline v_bottom_drag(m) =
    m.C_d .* m.V.present .*
    _safe_sqrt.(m.V.present .^ 2 .+ jp_half(im_half(m.U.present)) .^ 2)
# Ah·∇²(DV)  (lateral diffusion; the A_h/shear scaling is applied inside
# laplace_V, since it dispatches on `Params.lateral_viscosity`)
@inline v_diffusion(m) = copy(laplace_V(m))
# ė·V  (detrainment momentum loss)
@inline v_detrainment(m) = m.detr .* m.V.present

# -- Tracer terms (Eqs. 4–5) -------------------------------------------------

# q·∂D/∂t  (thickness-tendency coupling)
@inline tracer_thickness_tendency(m, q) = q .* m.dDdt
# ∇·(D·u·q)  (horizontal tracer advection)
@inline tracer_advection(m, q) =
    upwind_advection_T(similar(m.D.present), m, m.D.present .* q)
# e_net·qa  (entrainment of ambient water)
@inline tracer_entrainment(m, qa) = m.nentr .* qa
# Kh·∇²q  (horizontal diffusion)
@inline tracer_diffusion(m, q_past) = m.K_h .* laplace_T(similar(q_past), m, q_past)
# (q_past − qa)·conv2  (convective relaxation, RelaxToAmbient only)
@inline tracer_convection(m, q_past, qa) = (q_past .- qa) .* m.conv2
# ṁ·Tb − γT·(T − Tb)  (ice-ocean heat exchange; temperature equation only)
@inline T_ice_ocean_exchange(m) = m.melt .* m.Tb .- m.gamT .* (m.T.present .- m.Tb)
