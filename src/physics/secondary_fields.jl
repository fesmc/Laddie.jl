# The secondary fields of the layer, updated from the prognostic levels before each
# step, in the order of their inputs.  Each process lives with its scheme types:
# `update_ambient_fields!` (forcing.jl), `update_density!` (eos.jl),
# `update_convection!` (convection.jl), `update_melt!` (melting.jl), `update_frazil!`
# (frazil.jl), `update_entrainment!` (entrainment.jl).  The friction velocity, which
# the melt and entrainment schemes share, is here.

# `dt` is the base time step; only the `ent2` top-up in `update_entrainment!` needs it.
function update_secondary_fields!(m, dt)
    update_ambient_fields!(m)
    update_density!(m)
    update_convection!(m)
    update_melt!(m)
    update_frazil!(m)
    update_entrainment!(m, dt)
    return
end

# Friction velocity at the T-point: u★ = √(C_d_top · (im_half(U)² + jm_half(V)² + u_tide²))
# im_half(U)[i,j] = (U[i,j] + U[i−1,j]) / 2,  jm_half(V)[i,j] = (V[i,j] + V[i,j−1]) / 2,
# at the level of `Params.closure_level`.
function update_ustar!(m)
    launch_interior!(
        _ustar_kernel!,
        m.ustar,
        _closure_level(m.U, m.params.closure_level),
        _closure_level(m.V, m.params.closure_level),
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
