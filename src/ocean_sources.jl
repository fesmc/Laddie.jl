# ============================================================================
# Building ocean forcings from a 3D ocean field.
#
# Ocean models that force ice-sheet models mostly have no ice-shelf cavities, so
# the meltwater layer should see only water from outside the ice fronts.
# `front_ocean_forcing` gives every cavity cell the water column of the open-ocean
# cell at the front that feeds it, as a `PerBasinOceanForcing1D` with one profile
# per such column.  `regrid_ocean` and `fill_ocean!` bring a coarse ocean grid onto
# the model's cells, for this or for an `OceanForcing2D`.
# All of this is host-side preprocessing on CPU arrays.
# ============================================================================

const _NEIGHBOURS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

"""
    nearest_seed(labels) -> (label, d2)

For every cell of the matrix `labels`, the label of the nearest seed (a cell with
`labels > 0`) and the squared distance to it in cells², by exact Euclidean distance
(Felzenszwalb & Huttenlocher 2012: the 1D lower-envelope transform along the
columns, then along the rows).  Seeds keep their own label.  With no seed at all,
every label is `0` and every distance `Inf`.
"""
function nearest_seed(labels::AbstractMatrix{<:Integer})
    nx, ny = size(labels)
    f = ifelse.(labels .> 0, 0.0, Inf)
    g, glab = similar(f), similar(labels, Int)
    Threads.@threads for j = 1:ny
        _edt1d!(view(g, :, j), view(glab, :, j), view(f, :, j), view(labels, :, j))
    end
    d2, lab = similar(f), similar(glab)
    Threads.@threads for i = 1:nx
        _edt1d!(view(d2, i, :), view(lab, i, :), collect(view(g, i, :)), collect(view(glab, i, :)))
    end
    return lab, d2
end

# The 1D transform d(q) = min_p ((q − p)² + f(p)), and the label of the minimising
# p, over the finite f(p) only (all Inf gives Inf and label 0), so Inf never
# enters the arithmetic.
function _edt1d!(d, lab, f, flab)
    n = length(f)
    v = Vector{Int}(undef, n)
    z = Vector{Float64}(undef, n + 1)
    k = 0
    for q = 1:n
        isfinite(f[q]) || continue
        if k == 0
            k = 1
            v[1], z[1], z[2] = q, -Inf, Inf
            continue
        end
        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2q - 2v[k])
        while s <= z[k]
            k -= 1
            s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2q - 2v[k])
        end
        k += 1
        v[k], z[k], z[k+1] = q, s, Inf
    end
    if k == 0
        fill!(d, Inf)
        fill!(lab, 0)
        return d
    end
    k = 1
    for q = 1:n
        while z[k+1] < q
            k += 1
        end
        d[q] = (q - v[k])^2 + f[v[k]]
        lab[q] = flab[v[k]]
    end
    return d
end

"""
    front_sources(mask; wet = trues(size(mask)), depth = nothing) -> (source, lost)

The open-ocean cell whose water column forces each cavity cell, on a Laddie `mask`
(border ring included, as [`Grid`](@ref) takes it):

  * open ocean: the `0` cells connected (4-neighbour) to the edge of the domain;
  * cavity: floating ice (`3`), gaps (`4`) and enclosed water (`0` cells not
    connected to the open ocean);
  * a cavity cell next to open ocean takes the deepest such neighbour that is
    `wet` (has water data) as its source: the one with the lowest `depth` (a matrix
    the size of `mask`, e.g. the bed elevation), or the first when `depth` is not
    given;
  * every other cavity cell takes the source of the front nearest *through the
    cavity* (8-neighbour chamfer distance), so that no front reaches across
    grounded ice or land;
  * cavity cells that no front reaches, the source of the nearest front in a
    straight line.

Returns the linear index into `mask` of the source of every cavity cell (0
elsewhere), and the number of cavity cells that took a straight line.
"""
function front_sources(mask::AbstractMatrix{<:Integer}; wet = trues(size(mask)), depth = nothing)
    nx, ny = size(mask)
    size(wet) == size(mask) || throw(ArgumentError("wet is $(size(wet)) but mask is $(size(mask))"))
    depth === nothing || size(depth) == size(mask) ||
        throw(ArgumentError("depth is $(size(depth)) but mask is $(size(mask))"))
    edge(I) = I[1] in (1, 2, nx - 1, nx) || I[2] in (1, 2, ny - 1, ny)
    open = falses(nx, ny)
    stack = [I for I in CartesianIndices(mask) if mask[I] == 0 && edge(I)]
    foreach(I -> open[I] = true, stack)
    while !isempty(stack)
        I = pop!(stack)
        for d in _NEIGHBOURS4
            J = CartesianIndex(I[1] + d[1], I[2] + d[2])
            checkbounds(Bool, mask, J) && mask[J] == 0 && !open[J] && (open[J] = true; push!(stack, J))
        end
    end
    cavity = ((mask .== 3) .| (mask .== 4) .| (mask .== 0)) .& .!open
    L = LinearIndices(mask)
    dist = fill(Inf, nx, ny)
    src = zeros(Int, nx, ny)
    for I in findall(cavity)
        best = Inf
        for d in _NEIGHBOURS4
            J = CartesianIndex(I[1] + d[1], I[2] + d[2])
            checkbounds(Bool, mask, J) && open[J] && wet[J] || continue
            score = depth === nothing ? 0.0 : Float64(depth[J])
            if src[I] == 0 || score < best
                best = score
                src[I], dist[I] = L[J], 1.0
            end
        end
    end
    any(>(0), src) || throw(ArgumentError("no cavity cell borders wet open ocean: the mask has no ice front"))
    fwd = ((-1, -1, sqrt(2)), (0, -1, 1.0), (1, -1, sqrt(2)), (-1, 0, 1.0))
    bwd = ((1, 1, sqrt(2)), (0, 1, 1.0), (-1, 1, sqrt(2)), (1, 0, 1.0))
    function sweep!(order, steps)
        changed = false
        for I in order
            cavity[I] || continue
            for (a, b, w) in steps
                J = CartesianIndex(I[1] + a, I[2] + b)
                if checkbounds(Bool, mask, J) && cavity[J] && dist[J] + w < dist[I]
                    dist[I], src[I] = dist[J] + w, src[J]
                    changed = true
                end
            end
        end
        return changed
    end
    # Each pair of sweeps settles every path that turns at most once; a cavity
    # winding back and forth needs more, so repeat until nothing changes.
    while true
        c1 = sweep!(CartesianIndices(mask), fwd)
        c2 = sweep!(Iterators.reverse(CartesianIndices(mask)), bwd)
        c1 || c2 || break
    end
    lost = cavity .& (src .== 0)
    if any(lost)
        # Seeds are the front cells (dist == 1), labelled with their source.
        nearest, _ = nearest_seed(ifelse.(dist .== 1, src, 0))
        src[lost] .= nearest[lost]
    end
    return src, count(lost)
end

# Give every level of the column `col` without a value (NaN) that of the nearest
# level with one, so the column extends flat beyond its water.  False, leaving
# `col` as it is, if no level has a value.
function _extend_flat!(col)
    wet = findall(isfinite, col)
    isempty(wet) && return false
    for k in eachindex(col)
        isfinite(col[k]) || (col[k] = col[wet[argmin(abs.(wet .- k))]])
    end
    return true
end

"""
    front_ocean_forcing(T, S, z, mask; depth = nothing, FT = Float64)

Ambient forcing from outside the ice fronts only, as an ocean model without
cavities would give it: every cavity cell of `mask` takes the water column of its
open-ocean source ([`front_sources`](@ref)), extended flat below that column's
water.  `T`, `S` are `(nx, ny, nz)` on the depths `z` (uniform) for the full domain
of `mask`, border ring included, with `NaN` where there is no water; only cells
with water in both can be sources.  `depth` ranks the open-ocean neighbours of a
front cell, the lowest winning (pass the bed elevation, size of `mask`); without
it, the column with the most levels of water wins.

Returns a [`PerBasinOceanForcing1D`](@ref) with one profile per source column:
`n_sources × nz` numbers instead of the full 3D field.  Cells outside the cavities
take the profile of the nearest cavity cell, so every cell has one.
"""
function front_ocean_forcing(T::AbstractArray{<:Real,3}, S::AbstractArray{<:Real,3}, z, mask; depth = nothing, FT = Float64)
    size(T) == size(S) || throw(ArgumentError("T is $(size(T)) but S is $(size(S))"))
    size(T)[1:2] == size(mask) ||
        throw(ArgumentError("T is $(size(T)) but the mask is $(size(mask)); give the full domain"))
    size(T, 3) == length(z) || throw(ArgumentError("T has $(size(T, 3)) levels but z has $(length(z))"))
    wet = dropdims(any(isfinite, T; dims = 3) .& any(isfinite, S; dims = 3); dims = 3)
    rank = depth === nothing ? -Float64.(dropdims(count(isfinite, T; dims = 3); dims = 3)) : depth
    src, lost = front_sources(mask; wet, depth = rank)
    sources = unique(filter(>(0), src))
    C = CartesianIndices(mask)
    Tz = Matrix{Float64}(undef, length(sources), length(z))
    Sz = similar(Tz)
    for (n, l) in enumerate(sources)
        Tz[n, :] .= T[C[l], :]
        Sz[n, :] .= S[C[l], :]
        _extend_flat!(view(Tz, n, :))
        _extend_flat!(view(Sz, n, :))
    end
    profile = Dict(l => n for (n, l) in enumerate(sources))
    index = [s > 0 ? profile[s] : 0 for s in src]
    index, _ = nearest_seed(index)
    lost > 0 && @info "front_ocean_forcing: $lost cavity cells reach no front through the cavity; they take the nearest in a straight line"
    return PerBasinOceanForcing1D(Tz, Sz, z, index; FT)
end

"""
    regrid_ocean(T, S, x, y, px, py) -> (; T, S)

Bilinear interpolation of the 3D ocean fields `T`, `S` (`(length(x), length(y),
nz)` on the ascending axes `x`, `y`, `NaN` or `missing` where there is no water)
to the points `px` × `py`, level by level, flat beyond the axes.  Missing values
are left out, the weights renormalised over the valid source cells; a point with
no valid neighbour is `NaN`.  For the cell centres of a full domain, pass its
axes with the border ring (`Grid` takes `x`, `y` that way).

Use the result in [`front_ocean_forcing`](@ref), or fill it with
[`fill_ocean!`](@ref) for an `OceanForcing2D`.
"""
function regrid_ocean(T::AbstractArray{<:Any,3}, S::AbstractArray{<:Any,3}, x, y, px, py)
    size(T) == size(S) || throw(ArgumentError("T is $(size(T)) but S is $(size(S))"))
    size(T)[1:2] == (length(x), length(y)) ||
        throw(ArgumentError("T is $(size(T)) but the axes are $((length(x), length(y)))"))
    issorted(x) && issorted(y) || throw(ArgumentError("x and y must be ascending"))
    Wx, Wy = _linear_weights(x, px), _linear_weights(y, py)
    function level(A, k)
        a = Float64[ismissing(v) ? NaN : Float64(v) for v in view(A, :, :, k)]
        valid = isfinite.(a)
        num = Wx * ifelse.(valid, a, 0.0) * Wy'
        den = Wx * Float64.(valid) * Wy'
        return ifelse.(den .> 1e-6, num ./ max.(den, 1e-6), NaN)
    end
    nz = size(T, 3)
    return (; T = stack(level(T, k) for k = 1:nz), S = stack(level(S, k) for k = 1:nz))
end

# Linear interpolation weights from the ascending axis `c` to the points `p`, flat
# beyond its ends: `W[n, k]` is the weight of `c[k]` at `p[n]`.
function _linear_weights(c, p)
    W = zeros(length(p), length(c))
    length(c) == 1 && (W .= 1; return W)
    for (n, v) in enumerate(p)
        k = clamp(searchsortedlast(c, v), 1, length(c) - 1)
        w = clamp((v - c[k]) / (c[k+1] - c[k]), 0, 1)
        W[n, k], W[n, k+1] = 1 - w, w
    end
    return W
end

"""
    fill_ocean!(T, S) -> (T, S)

Fill the `NaN`s of the 3D ocean fields `T`, `S` (e.g. from [`regrid_ocean`](@ref))
so that every column is complete: each column extends flat beyond its own water
(a level takes the value of the nearest level with one), and a column with no
water takes the nearest column with water in both, by exact Euclidean distance.
"""
function fill_ocean!(T::AbstractArray{<:AbstractFloat,3}, S::AbstractArray{<:AbstractFloat,3})
    size(T) == size(S) || throw(ArgumentError("T is $(size(T)) but S is $(size(S))"))
    nx, ny = size(T, 1), size(T, 2)
    full = falses(nx, ny)
    Threads.@threads for j = 1:ny
        for i = 1:nx
            full[i, j] = _extend_flat!(view(T, i, j, :)) & _extend_flat!(view(S, i, j, :))
        end
    end
    any(full) || throw(ArgumentError("no column has water in both T and S"))
    L = LinearIndices(full)
    nearest, _ = nearest_seed(ifelse.(full, L, 0))
    C = CartesianIndices(full)
    for I in findall(.!full)
        J = C[nearest[I]]
        T[I, :] .= T[J, :]
        S[I, :] .= S[J, :]
    end
    return T, S
end
