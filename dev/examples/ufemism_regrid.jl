# Regridding of UFEMISM output onto a regular grid, for docs/src/examples/jesse-gaps.jl.
#
# UFEMISM stores its fields on the vertices of an unstructured mesh (`Hi`, `Hib`, `Hb`, the
# `mask`, and LADDIE's `melt`, `T_lad`, `S_lad`, `H_lad`) or on its triangles (`U_lad`,
# `V_lad`).  Every cell centre of the target grid is located in the triangle that contains
# it; vertex fields are then interpolated with barycentric weights, the mask is taken from
# the nearest of the three vertices, and triangle fields from the containing triangle.
#
# `regrid_snapshot` returns everything the example needs, on the target grid:
#   mask       Laddie.jl mask (0 ocean, 1 land border, 2 grounded, 3 floating)
#   footprint  where the reference geometry (year 0) had ice — what marks a cell a gap
#   z_draft, z_bed
#   melt, speed, D, T, S   LADDIE v2's own fields at that time
#   area_float             floating area on the mesh (m²), to check the regridding

using NCDatasets

# Containing triangle and barycentric weights of every point (xc[i], yc[j]).  A uniform
# bucket grid over the triangles' bounding boxes keeps the search local.
function _locate(V, Tri, xc, yc; nbx = 800, nby = 80)
    (x0, x1), (y0, y1) = extrema(V[:, 1]), extrema(V[:, 2])
    bx(x) = clamp(floor(Int, (x - x0) / (x1 - x0) * nbx) + 1, 1, nbx)
    by(y) = clamp(floor(Int, (y - y0) / (y1 - y0) * nby) + 1, 1, nby)
    buckets = [Int[] for _ in 1:nbx, _ in 1:nby]
    for t in axes(Tri, 1)
        xs, ys = V[Tri[t, :], 1], V[Tri[t, :], 2]
        for i in bx(minimum(xs)):bx(maximum(xs)), j in by(minimum(ys)):by(maximum(ys))
            push!(buckets[i, j], t)
        end
    end
    tri = zeros(Int, length(xc), length(yc))
    w = zeros(3, length(xc), length(yc))
    for j in eachindex(yc), i in eachindex(xc)
        x, y = xc[i], yc[j]
        (x0 <= x <= x1 && y0 <= y <= y1) || continue      # the border ring lies outside
        for t in buckets[bx(x), by(y)]
            a, b, c = Tri[t, 1], Tri[t, 2], Tri[t, 3]
            xa, ya, xb, yb, xq, yq = V[a, 1], V[a, 2], V[b, 1], V[b, 2], V[c, 1], V[c, 2]
            det = (yb - yq) * (xa - xq) + (xq - xb) * (ya - yq)
            la = ((yb - yq) * (x - xq) + (xq - xb) * (y - yq)) / det
            lb = ((yq - ya) * (x - xq) + (xa - xq) * (y - yq)) / det
            lc = 1 - la - lb
            if min(la, lb, lc) >= -1e-9
                tri[i, j] = t
                w[:, i, j] .= (la, lb, lc)
                break
            end
        end
    end
    return (; tri, w, Tri)
end

function _interp(f, loc)                          # vertex field, linear
    out = fill(NaN, size(loc.tri))
    for I in CartesianIndices(out)
        t = loc.tri[I]
        t == 0 && continue
        out[I] = sum(loc.w[k, I] * f[loc.Tri[t, k]] for k in 1:3)
    end
    return out
end

function _nearest(f, loc, fill_value)             # vertex field, nearest vertex
    out = fill(fill_value, size(loc.tri))
    for I in CartesianIndices(out)
        t = loc.tri[I]
        t == 0 && continue
        out[I] = f[loc.Tri[t, argmax(loc.w[:, I])]]
    end
    return out
end

function _per_triangle(f, loc)                    # triangle field
    out = fill(NaN, size(loc.tri))
    for I in CartesianIndices(out)
        loc.tri[I] == 0 || (out[I] = f[loc.tri[I]])
    end
    return out
end

# UFEMISM's mask codes: 2 is ice-free ocean, 3 and 5 are grounded ice (5 at the grounding
# line), and 4, 6 and 8 are floating ice (6 at the grounding line, 8 at the calving front).
_laddie_code(k) = k in (3, 5) ? 2 : k in (4, 6, 8) ? 3 : 0
_is_ice(k) = k in (3, 4, 5, 6, 8)

function regrid_snapshot(path, year, xc, yc)
    ds = Dataset(path)
    it = findfirst(==(year), ds["time"][:])
    it === nothing && error("year $year not in $path")
    loc = _locate(Array(ds["V"][:, :]), Array(ds["Tri"][:, :]), xc, yc)
    at(v, i = it) = _interp(Float64.(coalesce.(ds[v][:, i], NaN)), loc)
    mask = _laddie_code.(_nearest(Int.(ds["mask"][:, it]), loc, 1))
    mask[[1, end], :] .= 1                          # border ring: land
    mask[:, [1, end]] .= 1
    snap = (;
        mask,
        footprint = _is_ice.(_nearest(Int.(ds["mask"][:, 1]), loc, 1)),
        z_draft = replace(at("Hib"), NaN => 0.0),
        z_bed = replace(at("Hb"), NaN => 0.0),
        melt = at("melt"),                          # m s⁻¹
        speed = hypot.(_per_triangle(Float64.(coalesce.(ds["U_lad"][:, it], NaN)), loc),
                       _per_triangle(Float64.(coalesce.(ds["V_lad"][:, it], NaN)), loc)),
        D = at("H_lad"), T = at("T_lad"), S = at("S_lad"),
        area_float = sum(ds["A"][:][map(k -> k in (4, 6, 8), ds["mask"][:, it])]),
    )
    close(ds)
    return snap
end
