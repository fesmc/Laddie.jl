# The secondary fields of the layer, updated from the prognostic levels before each
# step, in the order of their inputs.  Each process lives with its scheme types:
# `ambient_fields!` (forcing.jl), `density!` (eos.jl),
# `convection!` (convection.jl), `melt!` (melting.jl), `frazil!`
# (frazil.jl), `step_entrainment!` (entrainment.jl).  The friction velocity, which
# the melt and entrainment schemes share, is here.

# `span` is the time the next step covers (2·dt in the leapfrog, the stage span in
# FB-RK3); only the preventive bounds on H in `step_entrainment!` need it.
function step_secondary_fields!(m, span)
    ambient_fields!(m)
    density!(m)
    convection!(m)
    melt!(m)
    frazil!(m)
    step_entrainment!(m, span)
    return
end

"""
    ustar!(m)

Friction velocity on T-points, `m.ustar = √(C_d_top (Ū² + V̄² + u_tide²))`, with the velocities
averaged onto the T-point at the [`closure_level`](@ref) of the integrator. A melting scheme calls it,
since the entrainment needs it too.
"""
function ustar! end

function ustar!(m)
    launch_interior!(
        _ustar_kernel!,
        m.ustar,
        closure_field(m.U, closure_level(m.params.integrator)),
        closure_field(m.V, closure_level(m.params.integrator)),
        m.tmask,
        m.C_d_top,
        m.u_tide,
    )
end

@kernel function _ustar_kernel!(ustar, @Const(U), @Const(V), @Const(tmask), C_d_top, u_tide)
    C_d_top, u_tide = _val(C_d_top), _val(u_tide)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(C_d_top)
        half = FT(0.5)
        im1 = i - 1
        jm1 = j - 1
        u_im = (U[i, j] + U[im1, j]) * half
        v_jm = (V[i, j] + V[i, jm1]) * half
        ustar[i, j] =
            sqrt(C_d_top * (u_im * u_im + v_jm * v_jm + u_tide * u_tide)) * tmask[i, j]
    end
end
