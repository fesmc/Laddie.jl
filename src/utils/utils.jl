# ============================================================================
# Shift / interpolation primitives  (≡ np.roll & tools.py, GPU-capable)
# ============================================================================

# Periodic one-cell shifts along each axis.  Arrays are stored as [ix, iy]: the
# first index runs along x, the second along y.  The domain is wrapped in a
# grounded border so periodic wrap is harmless (masked off everywhere).
#
# The names are historical and refer to the *index shift*, not to a compass
# direction: `xm1(a)[i, j] == a[i+1, j]` is the neighbour at the next x index.
# Grid axes need not align with east/north — a polar stereographic projection
# rotates them by an arbitrary angle — so nothing here means "east" or "north".
@inline xm1(a) = circshift(a, (-1, 0))   # next x neighbour : a[i+1, j]
@inline xp1(a) = circshift(a, (1, 0))    # prev x neighbour : a[i-1, j]
@inline ym1(a) = circshift(a, (0, -1))   # next y neighbour : a[i, j+1]
@inline yp1(a) = circshift(a, (0, 1))    # prev y neighbour : a[i, j-1]

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
# The counts (0, 1 or 2 active cells) are formed like the kernels form them inline.
im_count(mask) = mask .+ xp1(mask)
ip_count(mask) = mask .+ xm1(mask)
jm_count(mask) = mask .+ yp1(mask)
jp_count(mask) = mask .+ ym1(mask)
ip_t(m, a) = div0(a .+ xm1(a), ip_count(m.tmask))
jp_t(m, a) = div0(a .+ ym1(a), jp_count(m.tmask))
im_u(m, a) = div0(a .+ xp1(a), im_count(m.umask))
jm_v(m, a) = div0(a .+ yp1(a), jm_count(m.vmask))

# Cells with at least one open-ocean neighbour.
next_to_ocean(ocn) = xm1(ocn) .+ xp1(ocn) .+ ym1(ocn) .+ yp1(ocn) .> 0

# Numpy-style gradient: second-order central differences on the interior,
# first-order one-sided at the two boundary rows/columns.
function gradient_x(a, dx)
    g = similar(a)
    n = size(a, 1)
    @views g[2:(n-1), :] .= (a[3:n, :] .- a[1:(n-2), :]) ./ (2dx)
    @views g[1, :] .= (a[2, :] .- a[1, :]) ./ dx
    @views g[n, :] .= (a[n, :] .- a[n-1, :]) ./ dx
    return g
end
function gradient_y(a, dy)
    g = similar(a)
    n = size(a, 2)
    @views g[:, 2:(n-1)] .= (a[:, 3:n] .- a[:, 1:(n-2)]) ./ (2dy)
    @views g[:, 1] .= (a[:, 2] .- a[:, 1]) ./ dy
    @views g[:, n] .= (a[:, n] .- a[:, n-1]) ./ dy
    return g
end

# ============================================================================
# Rebuilding structs field by field
# ============================================================================

# `x` rebuilt with `f` applied to each of its fields: how a struct is moved to another
# backend, promoted to another precision or traced.  Every field type of a Laddie
# struct is a type parameter, so the constructor re-infers the parameters from the new
# fields.  Field-less singletons are returned as they are.
function _mapfields(f, x)
    T = typeof(x)
    fieldcount(T) == 0 && return x
    return _rebuild(T, ntuple(i -> f(getfield(x, i)), fieldcount(T))...)
end

# A struct of type `T` from its fields, unchecked: the type's own name, unless the type
# has a method of its own (when a type parameter does not show in the fields, or the
# constructor validates what a rebuild must let through, such as a zero AD tangent).
_rebuild(::Type{T}, fields...) where {T} = Base.typename(T).wrapper(fields...)

# ============================================================================
# Scalar helpers for kernels
# ============================================================================

# Zero where `b` is zero.  The divisor of the unused branch is swapped for one as well:
# reverse-mode AD (Enzyme) still differentiates that branch, and its zero adjoint
# times 1/0 would give NaN.
@inline _safe_div(a, b) = ifelse(iszero(b), zero(a), a / ifelse(iszero(b), one(b), b))
# The plain value of a number, without derivative parts.  Used where a value
# leaves the differentiable computation: integer indices, the dt controller,
# display and file output.  The ForwardDiff extension strips `Dual` numbers.
@inline _primal(x) = x
_float64(x) = Float64(_primal(x))
# `sqrt` for arguments that reach exactly zero (a speed at rest, a clipped
# discriminant), with a zero derivative there instead of the NaN of `0 × Inf`.  The
# same double `ifelse` as `_safe_div`, for Enzyme; the ForwardDiff extension has its
# own method for `Dual`s.
@inline _safe_sqrt(x) = ifelse(iszero(x), zero(x), sqrt(ifelse(iszero(x), one(x), x)))
# A kernel's scalar argument as a plain value.  The kernels unwrap their scalars
# with it on entry.  `getindex(x::Number)` is `x` for floats and `Dual`s; a scalar
# traced by Reactant arrives as a device reference, which `x[]` loads, so that the
# kernel body computes with a plain float (`FT(…)`, `clamp`, `promote` do not work
# on the reference).  Arrays pass through: some slots are scalars or fields.
@inline _val(x::Number) = x[]
@inline _val(x::NamedTuple) = map(_val, x)
@inline _val(x) = x

# Elementwise product `out = a·b`.
@kernel function _product_kernel!(out, @Const(a), @Const(b))
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[i, j] * b[i, j]
end

# `out = a` everywhere.
@kernel function _copy_kernel!(out, @Const(a))
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[i, j]
end

# ============================================================================
# Host-side interpolation
# ============================================================================

# Linear interpolation of the points (x, y), `x` sorted ascending, at `xi`; constant
# beyond the end points.
function _interp1d(x, y, xi)
    out = similar(xi)
    for (k, xk) in enumerate(xi)
        if xk <= x[1]
            out[k] = y[1]
        elseif xk >= x[end]
            out[k] = y[end]
        else
            j = searchsortedlast(x, xk)
            t = (xk - x[j]) / (x[j+1] - x[j])
            out[k] = y[j] + t * (y[j+1] - y[j])
        end
    end
    return out
end
