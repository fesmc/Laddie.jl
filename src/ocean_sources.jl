# ============================================================================
# Building ocean forcings from a 3D ocean field.
#
# Ocean models that force ice-sheet models mostly have no ice-shelf cavities, so
# the meltwater layer should see only water from outside the ice fronts.
# `front_ocean_forcing` gives every cavity cell the water column of the open-ocean
# cell at the front that feeds it, or a blend of those of nearby fronts
# (`FrontBlend`), as a `PerBasinOceanForcing1D` with one profile per such column
# (and per blended cell).  `regrid_ocean` and `fill_ocean!` bring a coarse ocean grid onto
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
    cavity, src, dist = _front_seeds(mask, wet, depth)
    seeds = ifelse.(dist .== 1, src, 0)
    fwd, bwd = _CHAMFER
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
        nearest, _ = nearest_seed(seeds)
        src[lost] .= nearest[lost]
    end
    return src, count(lost)
end

# The 8-neighbour chamfer steps of the forward and the backward sweep.
const _CHAMFER = (
    ((-1, -1, sqrt(2)), (0, -1, 1.0), (1, -1, sqrt(2)), (-1, 0, 1.0)),
    ((1, 1, sqrt(2)), (0, 1, 1.0), (-1, 1, sqrt(2)), (1, 0, 1.0)),
)

# The start of `front_sources`: the cavity (a `BitMatrix`), and the source (linear
# index of its open-ocean cell, 0 elsewhere) and distance (1, `Inf` elsewhere) of
# every front cell.
function _front_seeds(mask, wet, depth)
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
    return cavity, src, dist
end

"""
    FrontBlend(width; dx, kmax = 64)

Blend the front columns that force a cavity cell, for [`front_ocean_forcing`](@ref):
every cavity cell sees the weighted mean of up to `kmax` open-ocean source columns,
those whose fronts lie within `width` (m) of its nearest, distances measured through
the cavity as in [`front_sources`](@ref). `dx` is the cell size (m).

Source ``k`` at distance ``d_k`` takes the weight ``d^* - d_k``, normalised, with
``d^*`` the distance of the first source left out: the `kmax + 1`-th, or ``d_1 +``
`width`. A source therefore enters and leaves a cell's blend at zero weight, and the
forcing has no steps where the nearest front changes. Such steps make one-cell-wide
lines of convergence and runaway layer thickness in the cavity.

Where fronts are dense, `kmax` rather than `width` sets the blend radius; that radius
shrinks with the cell size, so raise `kmax` at fine resolution to keep it. At
`width = 0` every cell takes one nearest front, as without a blend; where two are
exactly as near, it may be the other one.
"""
struct FrontBlend
    "blend width, in cells"
    width::Float64
    "most sources per cell"
    kmax::Int
    function FrontBlend(width::Real; dx::Real, kmax::Integer = 64)
        width >= 0 || throw(ArgumentError("width must be ≥ 0, got $width"))
        dx > 0 || throw(ArgumentError("dx must be > 0, got $dx"))
        kmax >= 1 || throw(ArgumentError("kmax must be ≥ 1, got $kmax"))
        new(Float64(width / dx), Int(kmax))
    end
end

Base.show(io::IO, b::FrontBlend) = print(io, "FrontBlend(", _r(b.width, 3), " cells, kmax = ", b.kmax, ")")

# The sources and weights of every cavity cell for `blend`: as the sweep of
# `front_sources`, but each cell keeps the `kmax + 1` nearest distinct sources
# within `blend.width` of its nearest, nearest first. Returns `(; cells, src, w,
# lost)`: the cavity cells, their sources (linear indices, `(kmax, ncells)`, 0 where
# unused) and weights (summing to 1), and the number of cells no front reaches
# through the cavity (their nearest front in a straight line, weight 1).
function _front_blend(mask, cavity, src0, dist0, blend::FrontBlend)
    cells = findall(cavity)
    nc, kmax = length(cells), blend.kmax
    K1 = kmax + 1
    pos = zeros(Int32, size(mask))
    foreach(((i, I),) -> pos[I] = i, enumerate(cells))
    W = Float32(blend.width)
    D = fill(Inf32, K1, nc)
    S = zeros(Int32, K1, nc)
    n = zeros(Int, nc)
    for (i, I) in enumerate(cells)
        src0[I] > 0 && (D[1, i] = dist0[I]; S[1, i] = src0[I]; n[i] = 1)
    end
    # Merge the list of cell j, `step` away, into that of cell i; true if it changed.
    # Lists stay sorted; a list's nearest only gets nearer, so what lies beyond its
    # width now stays beyond it.
    function merge!(i, j, step)
        changed = false
        for e = 1:n[j]
            c = D[e, j] + step
            (c > D[1, i] + W || (n[i] == K1 && c >= D[K1, i])) && break
            s = S[e, j]
            p = findfirst(==(s), view(S, 1:n[i], i))
            if !isnothing(p)
                c < D[p, i] || continue
                for q = p:(n[i]-1)
                    D[q, i], S[q, i] = D[q+1, i], S[q+1, i]
                end
                n[i] -= 1
            end
            q = n[i] < K1 ? n[i] + 1 : K1
            while q > 1 && D[q-1, i] > c
                D[q, i], S[q, i] = D[q-1, i], S[q-1, i]
                q -= 1
            end
            D[q, i], S[q, i] = c, s
            n[i] = min(n[i] + 1, K1)
            changed = true
        end
        return changed
    end
    fwd, bwd = map(st -> map(((a, b, w),) -> (a, b, Float32(w)), st), _CHAMFER)
    function sweep!(order, steps)
        changed = false
        for i in order
            I = cells[i]
            for (a, b, w) in steps
                J = CartesianIndex(I[1] + a, I[2] + b)
                checkbounds(Bool, mask, J) && pos[J] > 0 && merge!(i, pos[J], w) && (changed = true)
            end
        end
        return changed
    end
    while true
        c1 = sweep!(1:nc, fwd)
        c2 = sweep!(nc:-1:1, bwd)
        c1 || c2 || break
    end
    src, w = zeros(Int32, kmax, nc), zeros(Float32, kmax, nc)
    lost = findall(==(0), n)
    if !isempty(lost)
        nearest, _ = nearest_seed(ifelse.(dist0 .== 1, src0, 0))
        for i in lost
            src[1, i], w[1, i] = nearest[cells[i]], 1
        end
    end
    for i = 1:nc
        n[i] == 0 && continue
        m = count(e -> D[e, i] <= D[1, i] + W, 1:n[i])
        k = min(m, kmax)
        dstar = m > kmax ? D[K1, i] : D[1, i] + W
        for e = 1:k
            src[e, i], w[e, i] = S[e, i], dstar - D[e, i]
        end
        tot = sum(view(w, 1:k, i))
        if tot > 0
            w[1:k, i] ./= tot
        else                     # width 0 and no tie: the nearest alone
            w[1:k, i] .= 0
            w[1, i] = 1
        end
    end
    return (; cells, src, w, lost = length(lost))
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
    front_ocean_forcing(T, S, z, mask; depth = nothing, blend = nothing, FT = Float64)

Ambient forcing from outside the ice fronts only, as an ocean model without
cavities would give it: every cavity cell of `mask` takes the water column of its
open-ocean source ([`front_sources`](@ref)), extended flat below that column's
water.  `T`, `S` are `(nx, ny, nz)` on the depths `z` (uniform) for the full domain
of `mask`, border ring included, with `NaN` where there is no water; only cells
with water in both can be sources.  `depth` ranks the open-ocean neighbours of a
front cell, the lowest winning (pass the bed elevation, size of `mask`); without
it, the column with the most levels of water wins.

With `blend = FrontBlend(width; dx)` ([`FrontBlend`](@ref)), every cavity cell
takes instead the weighted mean of the source columns of the fronts within `width`
of its nearest, so the forcing has no steps where the nearest front changes.

Returns a [`PerBasinOceanForcing1D`](@ref) with one profile per source column and,
blended, one more per cavity cell that blends several: `n_rows × nz` numbers
instead of the full 3D field.  Cells outside the cavities take the profile of the
nearest cavity cell, so every cell has one.
"""
function front_ocean_forcing(
    T::AbstractArray{<:Real,3},
    S::AbstractArray{<:Real,3},
    z,
    mask;
    depth = nothing,
    blend::Union{Nothing,FrontBlend} = nothing,
    FT = Float64,
)
    size(T) == size(S) || throw(ArgumentError("T is $(size(T)) but S is $(size(S))"))
    size(T)[1:2] == size(mask) ||
        throw(ArgumentError("T is $(size(T)) but the mask is $(size(mask)); give the full domain"))
    size(T, 3) == length(z) || throw(ArgumentError("T has $(size(T, 3)) levels but z has $(length(z))"))
    wet = dropdims(any(isfinite, T; dims = 3) .& any(isfinite, S; dims = 3); dims = 3)
    rank = depth === nothing ? -Float64.(dropdims(count(isfinite, T; dims = 3); dims = 3)) : depth
    if blend === nothing
        src, lost = front_sources(mask; wet, depth = rank)
        sources = unique(filter(>(0), src))
    else
        b = _front_blend(mask, _front_seeds(mask, wet, rank)..., blend)
        lost = b.lost
        sources = Int.(unique(filter(>(0), b.src)))
    end
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
    if blend === nothing
        index = [s > 0 ? profile[s] : 0 for s in src]
    else
        index, Tb, Sb = _blend_rows(b, profile, Tz, Sz, size(mask))
        mixed = size(Tb, 1)
        nsrc = count(>(0), b.w; dims = 1)
        @info "front_ocean_forcing: $(length(b.cells)) cavity cells from $(length(sources)) source columns, " *
              "$mixed blended, $(_r(sum(nsrc) / length(nsrc), 3)) sources per cell (max $(maximum(nsrc)))"
        Tz, Sz = vcat(Tz, Tb), vcat(Sz, Sb)
    end
    index, _ = nearest_seed(index)
    lost > 0 && @info "front_ocean_forcing: $lost cavity cells reach no front through the cavity; they take the nearest in a straight line"
    return PerBasinOceanForcing1D(Tz, Sz, z, index; FT)
end

# The rows of the cells of the blend `b` that blend several sources, a row each,
# from the source rows `Tz`, `Sz` (row `profile[l]` for the source of linear index
# `l`), and the row index of every cavity cell: its own, or that of its one source.
function _blend_rows(b, profile, Tz, Sz, sz)
    nz = size(Tz, 2)
    index = zeros(Int, sz)
    mixed = [i for i in eachindex(b.cells) if count(>(0), view(b.w, :, i)) > 1]
    Tb, Sb = zeros(length(mixed), nz), zeros(length(mixed), nz)
    Threads.@threads for r in eachindex(mixed)
        i = mixed[r]
        for e in axes(b.src, 1)
            w = b.w[e, i]
            w > 0 || continue
            n = profile[b.src[e, i]]
            for k = 1:nz
                Tb[r, k] += w * Tz[n, k]
                Sb[r, k] += w * Sz[n, k]
            end
        end
        index[b.cells[i]] = size(Tz, 1) + r
    end
    for (i, I) in enumerate(b.cells)
        index[I] == 0 && (index[I] = profile[b.src[argmax(view(b.w, :, i)), i]])
    end
    return index, Tb, Sb
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
