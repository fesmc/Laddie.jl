abstract type AbstractMaxLayerThickness end

struct TopographicMaxLayerThickness <: AbstractMaxLayerThickness end
@kwdef struct AbsoluteMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    D_max::FT = 100
end

@kwdef struct RelativeMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    f_D_max::FT = 4/5
end

function max_layer_thickness!(m, ::TopographicMaxLayerThickness)
    @. m.D.future = min(m.D.future, m.zb - m.z_bed) .* m.tmask
end
function max_layer_thickness!(m, c::AbsoluteMaxLayerThickness)
    @. m.D.future = min(m.D.future, c.D_max, m.zb - m.z_bed) .* m.tmask
end
function max_layer_thickness!(m, c::RelativeMaxLayerThickness)
    @. m.D.future = min(m.D.future, c.f_D_max * (m.zb - m.z_bed)) .* m.tmask
end

# ============================================================================
# Shift / interpolation primitives  (≡ np.roll & tools.py, GPU-capable)
# ============================================================================

# Periodic one-cell shifts along each axis.  The domain is wrapped in a
# grounded border so periodic wrap is harmless (masked off everywhere).
@inline xm1(a) = circshift(a, (0, -1))   # east neighbour  : np.roll(a, -1, axis=1)
@inline xp1(a) = circshift(a, (0, 1))   # west neighbour  : np.roll(a,  1, axis=1)
@inline ym1(a) = circshift(a, (-1, 0))   # north neighbour : np.roll(a, -1, axis=0)
@inline yp1(a) = circshift(a, (1, 0))   # south neighbour : np.roll(a,  1, axis=0)

# Arithmetic-mean interpolation to cell-face midpoints.
# Naming convention: `im` = value at i−½, `ip` = i+½, `jm` = j−½, `jp` = j+½.
im_half(a) = (a .+ xp1(a)) ./ 2
ip_half(a) = (a .+ xm1(a)) ./ 2
jm_half(a) = (a .+ yp1(a)) ./ 2
jp_half(a) = (a .+ ym1(a)) ./ 2

# Safe division: returns 0 where the denominator is zero.
div0(a, b) = ifelse.(b .== 0, zero(eltype(a)), a ./ b)

# Masked staggered interpolation — normalises by the count of live neighbours
# to avoid gradient artefacts across boundaries (tools.py in the reference).
im_t(m, a) = div0(a .+ xp1(a), m.tmask_im)
ip_t(m, a) = div0(a .+ xm1(a), m.tmask_ip)
jm_t(m, a) = div0(a .+ yp1(a), m.tmask_jm)
jp_t(m, a) = div0(a .+ ym1(a), m.tmask_jp)
im_u(m, a) = div0(a .+ xp1(a), m.umask_im)
ip_u(m, a) = div0(a .+ xm1(a), m.umask_ip)
jm_u(m, a) = div0(a .+ yp1(a), m.umask_jm)
jp_u(m, a) = div0(a .+ ym1(a), m.umask_jp)
im_v(m, a) = div0(a .+ xp1(a), m.vmask_im)
ip_v(m, a) = div0(a .+ xm1(a), m.vmask_ip)
jm_v(m, a) = div0(a .+ yp1(a), m.vmask_jm)
jp_v(m, a) = div0(a .+ ym1(a), m.vmask_jp)

# Numpy-style gradient: second-order central differences on the interior,
# first-order one-sided at the two boundary rows/columns.
function gradient_x(a, dx)
    g = similar(a)
    n = size(a, 2)
    @views g[:, 2:(n-1)] .= (a[:, 3:n] .- a[:, 1:(n-2)]) ./ (2dx)
    @views g[:, 1] .= (a[:, 2] .- a[:, 1]) ./ dx
    @views g[:, n] .= (a[:, n] .- a[:, n-1]) ./ dx
    return g
end
function gradient_y(a, dy)
    g = similar(a)
    n = size(a, 1)
    @views g[2:(n-1), :] .= (a[3:n, :] .- a[1:(n-2), :]) ./ (2dy)
    @views g[1, :] .= (a[2, :] .- a[1, :]) ./ dy
    @views g[n, :] .= (a[n, :] .- a[n-1, :]) ./ dy
    return g
end
