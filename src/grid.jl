
# ============================================================================
# Ice-base slope gradient variants
# ============================================================================

"""
Abstract supertype for ice-base slope (dzdx/dzdy) computation strategies.
Pass a concrete instance as the `gradient` keyword to `Model` or
`build_isomip`.  The slope is part of the model's `Geometry`, not of the
[`Grid`](@ref): it depends on which cells are active.
"""
abstract type AbstractIceSlopeGradient end

"""
    PyGradient()

Python-ported centred-difference gradient: `(zb[i+1] − zb[i−1]) / (2 dx)` applied
uniformly over all cells with no mask awareness.  Matches the Python LADDIE
v1.1 stencil exactly and should be used when comparing against the Python
reference output.
"""
struct PyGradient <: AbstractIceSlopeGradient end

"""
    JlGradient()

Mask-aware gradient (default): the ice-base slope at each shelf cell is
computed using only shelf-cell neighbours, falling back to one-sided
differences at ice fronts / grounding lines, and zero when no shelf neighbour
exists.  This avoids spurious slopes from the physically-incompatible `z_draft`
values stored at ocean cells (`z_draft=0`) and grounded cells (`z_draft=bed`).
"""
struct JlGradient <: AbstractIceSlopeGradient end

function _icebase_slope(::PyGradient, tmask, z_draft_ft, dx_ft, dy_ft, FT)
    return gradient_x(z_draft_ft, dx_ft), gradient_y(z_draft_ft, dy_ft)
end

function _icebase_slope(::JlGradient, tmask, z_draft_ft, dx_ft, dy_ft, FT)
    # Only read shelf-cell neighbours to avoid incorporating the
    # physically-incompatible z_draft values at ocean (z_draft=0) and grounded
    # (z_draft=bed) cells, which would produce O(0.2) spurious slopes.
    #
    # Stencil at each shelf cell, along each axis ("next" = index + 1, "prev" =
    # index − 1; see the shift primitives in utils.jl):
    #   both neighbours shelf → centred  (zb_next − zb_prev) / (2 dx)
    #   next only             → forward  (zb_next − zb) / dx
    #   prev only             → backward (zb − zb_prev) / dx
    #   neither               → 0
    _tm_e, _tm_w = xm1(tmask), xp1(tmask)
    _tm_n, _tm_s = ym1(tmask), yp1(tmask)
    _zb_e, _zb_w = xm1(z_draft_ft), xp1(z_draft_ft)
    _zb_n, _zb_s = ym1(z_draft_ft), yp1(z_draft_ft)
    dzdx = ifelse.(
        tmask .> 0,
        ifelse.(
            _tm_e .* _tm_w .> 0,
            (_zb_e .- _zb_w) ./ (FT(2) .* dx_ft),
            ifelse.(
                _tm_e .> 0,
                (_zb_e .- z_draft_ft) ./ dx_ft,
                ifelse.(_tm_w .> 0, (z_draft_ft .- _zb_w) ./ dx_ft, zero(FT)),
            ),
        ),
        gradient_x(z_draft_ft, dx_ft),
    )
    dzdy = ifelse.(
        tmask .> 0,
        ifelse.(
            _tm_n .* _tm_s .> 0,
            (_zb_n .- _zb_s) ./ (FT(2) .* dy_ft),
            ifelse.(
                _tm_n .> 0,
                (_zb_n .- z_draft_ft) ./ dy_ft,
                ifelse.(_tm_s .> 0, (z_draft_ft .- _zb_s) ./ dy_ft, zero(FT)),
            ),
        ),
        gradient_y(z_draft_ft, dy_ft),
    )
    return dzdx, dzdy
end

# ============================================================================
# Geometry{FT, A} — everything derived from the grid once a model has decided
# which cells are active: the gap-resolved mask, the masks and wall indicators,
# stagger-count denominators, the ice-base slope and the Coriolis field.  Owned by
# the Model; the Grid itself carries only what is independent of any modelling
# choice.
# A is the concrete matrix type (Matrix{FT} on CPU, CuArray{FT,2} on GPU).
# resolved_mask is always kept as Matrix{Int} on the CPU for host-side branching.
# ============================================================================

struct Geometry{FT,A<:AbstractMatrix{FT}}
    # The grid's mask after the gaps boundary condition: under SinkGapsBC gap cells
    # (4) are open ocean (0), under ConnectedGapsBC they stay 4.
    resolved_mask::Matrix{Int}
    dzdx::A
    dzdy::A

    # Coriolis parameter.  `f` is the T-point field (diagnostic, and what the
    # reference writes out); `fu`/`fv` are its face averages, which is what the
    # momentum kernels need — on a C-grid the two components live on different
    # faces, so one staggered copy each.
    f::A
    fu::A
    fv::A

    tmask::A
    imask::A
    grd::A
    lnd::A
    ocn::A

    ocnym1::A
    ocnyp1::A
    ocnxm1::A
    ocnxp1::A

    tmaskym1::A
    tmaskyp1::A
    tmaskxm1::A
    tmaskxp1::A

    glNu::A
    glSu::A
    glEv::A
    glWv::A
    lndNu::A
    lndSu::A
    lndEv::A
    lndWv::A
    isf::A

    umask::A
    vmask::A

    tmask_im::A
    tmask_ip::A
    tmask_jm::A
    tmask_jp::A
    umask_im::A
    umask_ip::A
    umask_jm::A
    umask_jp::A
    vmask_im::A
    vmask_ip::A
    vmask_jm::A
    vmask_jp::A
end

# Build all masks and stagger-count denominators from the gap-resolved integer `mask`
# and the (adjusted) ice draft `z_draft`, plus the ice-base slope and the Coriolis
# field staggered onto the velocity faces.  CPU arrays; the Model moves them.
function Geometry(
    mask::AbstractMatrix{Int},
    z_draft::AbstractMatrix,
    f_t::AbstractMatrix,
    dx,
    dy;
    FT = Float64,
    gradient = JlGradient(),
)
    dx_ft = FT(dx)
    dy_ft = FT(dy)
    z_draft_ft = FT.(z_draft)

    # Plain arithmetic face averages, not the masked `ip_t`/`jp_t` used for
    # prognostics: `f` is a property of position on Earth and is defined in every
    # cell, including the ones outside the domain, so there is nothing to mask out.
    f = FT.(f_t)
    fu = ip_half(f)
    fv = jp_half(f)

    # Primary classification.  Shelf (3) and gap (4) cells are both dynamically
    # active — every prognostic is stepped there — so they share `tmask`.  Only
    # shelf cells carry ice, so `imask` is the subset that can melt; see
    # `AbstractGapsBC` in boundary_conditions.jl.
    tmask = FT.((mask .== 3) .| (mask .== 4))
    imask = FT.(mask .== 3)
    # `grd` is the wall mask — everything the plume cannot flow into — so it unions
    # land and grounded ice.  The two are kept separately as well: `gl` (below) is
    # the grounding line proper and `lnd` is rock, and only the former takes the
    # grounding-line slip condition.
    grd = FT.((mask .== 2) .| (mask .== 1))
    lnd = FT.(mask .== 1)
    ocn = FT.(mask .== 0)

    ocnym1 = ym1(ocn)
    ocnyp1 = yp1(ocn)
    ocnxm1 = xm1(ocn)
    ocnxp1 = xp1(ocn)

    dzdx, dzdy = _icebase_slope(gradient, tmask, z_draft_ft, dx_ft, dy_ft, FT)

    tmaskym1 = ym1(tmask)
    tmaskyp1 = yp1(tmask)
    tmaskxm1 = xm1(tmask)
    tmaskxp1 = xp1(tmask)

    # Boundary geometry
    o = one(FT)
    # Wall-face indicators, one per face orientation: a u-face (N/S walls) or a
    # v-face (E/W walls) is a wall face when its two-cell stencil touches a wall.
    # Grounding-line indicators come from grounded ice (mask == 2) only, so the
    # momentum kernels can apply the grounding-line slip factor there
    # (AbstractGroundingLineBC).
    gl = FT.(mask .== 2)
    glNu = o .- ym1((o .- gl) .* (o .- xm1(gl)))
    glSu = o .- yp1((o .- gl) .* (o .- xm1(gl)))
    glEv = o .- xm1((o .- gl) .* (o .- ym1(gl)))
    glWv = o .- xp1((o .- gl) .* (o .- ym1(gl)))
    # Land-only wall indicators: the exact same construction as gl??, but from
    # `lnd` (mask == 1) instead of `gl` (mask == 2), so AbstractLandBC can apply
    # its own slip factor at walls bordering exposed bedrock/border, independent
    # of AbstractGroundingLineBC.
    #
    # The momentum kernels compose the two additively, as
    # `slip_gl*gl?? + slip_land*lnd??`, so the indicators must *partition* the wall
    # faces rather than overlap: a face whose two-cell stencil touches both
    # grounded ice and exposed rock (a coastline corner, ubiquitous in real
    # geometry) would otherwise receive both factors and end up at slip 4 under
    # NoSlipGL + NoSlipLand.  Grounding line takes precedence there, so that
    # gl?? + lnd?? is exactly the wall-face indicator of the whole wall `grd`.
    lndNu = (o .- ym1((o .- lnd) .* (o .- xm1(lnd)))) .* (o .- glNu)
    lndSu = (o .- yp1((o .- lnd) .* (o .- xm1(lnd)))) .* (o .- glSu)
    lndEv = (o .- xm1((o .- lnd) .* (o .- ym1(lnd)))) .* (o .- glEv)
    lndWv = (o .- xp1((o .- lnd) .* (o .- ym1(lnd)))) .* (o .- glWv)
    # Ice-front cells: ocean cells with an active neighbour, and the count of
    # such neighbours.
    isfW = ocn .* tmaskxm1
    isfS = ocn .* tmaskym1
    isf = ocn .* tmaskxp1 .+ ocn .* tmaskyp1 .+ isfW .+ isfS

    # Velocity masks: a u-point is active between two active cells, or between an
    # active cell and the ocean beyond it, but not facing a wall.
    umask = (tmask .+ isfW) .* (o .- xm1(grd .* tmaskxp1))
    vmask = (tmask .+ isfS) .* (o .- ym1(grd .* tmaskyp1))

    # Stagger-count denominators: active cells in each two-point average.
    tmask_im = tmask .+ tmaskxp1
    tmask_ip = tmask .+ tmaskxm1
    tmask_jm = tmask .+ tmaskyp1
    tmask_jp = tmask .+ tmaskym1
    umask_im = umask .+ xp1(umask)
    umask_ip = umask .+ xm1(umask)
    umask_jm = umask .+ yp1(umask)
    umask_jp = umask .+ ym1(umask)
    vmask_im = vmask .+ xp1(vmask)
    vmask_ip = vmask .+ xm1(vmask)
    vmask_jm = vmask .+ yp1(vmask)
    vmask_jp = vmask .+ ym1(vmask)

    resolved_mask = Matrix{Int}(mask)
    # Every field is a local of the same name.
    vars = Base.@locals
    return Geometry{FT,Matrix{FT}}((vars[fn] for fn in fieldnames(Geometry))...)
end

# ============================================================================
# Grid{FT, A} — where the cells are and what is under them: the cell layout and
# spacing, the (preprocessed, cropped) mask with gaps still marked 4, the ice draft
# and bed, and the cell-centre coordinates.  Nothing here depends on a modelling
# choice; everything that does lives in the Model's `Geometry`.
# ============================================================================

"""
$(TYPEDEF)

The model grid: a cropped rectangle of cells with its mask, ice draft and bed.
Construct with [`Grid(mask, z_draft, dx, dy; ...)`](@ref Grid(::AbstractMatrix{Int}, ::AbstractMatrix, ::Real, ::Real))
and pass it to [`Model`](@ref).

The grid knows nothing about boundary conditions or physics: a gap cell (`4`) is
kept as such whatever the gaps treatment, the masks the solver uses are derived
by the model, and so are the ice-base slope and the Coriolis field.

# Fields
$(TYPEDFIELDS)
"""
struct Grid{FT,A<:AbstractMatrix{FT}}
    "total cells in x, including the one-cell border ring"
    Nx::Int
    "total cells in y, including the one-cell border ring"
    Ny::Int
    "cell spacing in x (m)"
    dx::FT
    "cell spacing in y (m)"
    dy::FT
    "cell classification after preprocessing and cropping; gaps still marked `4` (CPU)"
    mask::Matrix{Int}
    "ice-base depth (m, ≤ 0), zeroed outside grounded ice and shelf"
    z_draft::A
    "bed elevation (m); `-Inf` when none was given (no cap on the layer thickness)"
    z_bed::A
    "interior cell-centre x coordinates (m, CPU)"
    x::Vector{FT}
    "interior cell-centre y coordinates (m, CPU)"
    y::Vector{FT}
    "row and column ranges of the input arrays that the grid keeps"
    crop::Tuple{UnitRange{Int},UnitRange{Int}}
    "size of the input arrays, before cropping"
    input_size::Tuple{Int,Int}
end

"""
    Grid(mask, z_draft, dx, dy; x, y, kwargs...)
    Grid(mask, z_draft; x, y, kwargs...)

Build a grid from a domain mask and ice draft with cell spacing `dx`, `dy` (m).
The spacing may be left out when coordinate vectors `x` and `y` are given, in which
case it is taken from them.

# Mask convention
| Value | Meaning |
|-------|---------|
| `0`   | open ocean (outside domain, passive) |
| `1`   | land — exposed bedrock, and the one-cell border ring |
| `2`   | grounded ice (sets inflow boundary for the plume) |
| `3`   | floating ice shelf (active plume cells) |
| `4`   | ice-shelf gap (ice-free; treated as the model's [`AbstractGapsBC`](@ref) decides) |

`mask` and `z_draft` must include the one-cell border ring, i.e. have size
`(nx+2, ny+2)` where `nx × ny` are the interior cells: **the first index runs along
x, the second along y**, the order NetCDF readers such as NCDatasets hand you.
`z_draft` is the ice-base depth in metres (negative downward); values outside
grounded ice and shelf are ignored and zeroed, and shallow shelf drafts are clamped
to −1 m.

# Keywords
- `x`, `y`: cell-centre coordinates (m) of the input arrays, of length `size(mask, 1)`
  and `size(mask, 2)` — border ring included, like every other input — e.g. the
  projection coordinates of a BedMachine subset.  They must be uniformly spaced
  (ascending or descending); their spacing sets `dx`/`dy`, or must match them when
  both are given.  They are cropped with the mask and written to the NetCDF output.
  By default `x = dx · (0:size(mask, 1) - 1)`, and likewise for `y`.
- `z_bed`: bed elevation (same size), or `nothing` (default) for none; only the
  topographic caps of [`AbstractMaxLayerThickness`](@ref) use it.
- `preprocess`: mask preprocessing steps applied in order before cropping, e.g.
  [`MarkGapsPreprocess`](@ref) or `FillSmallShelfPatchesPreprocess()`.  The
  caller's `mask` is not modified.
- `domain_cropping`: `MinRectangleDomainCropping()` (default) or `NoDomainCropping()`.
  Gap cells count as active, so a gap is never cropped away.
- `backend`: KernelAbstractions backend of the arrays (default `CPU()`).
- `FT`: floating-point precision type (default `Float64`).

Full-domain fields a model is given later — a 2D latitude or basal ice
temperature — must match the size of the arrays passed here; the model crops
them with `grid.crop`.

```julia
grid = Grid(mask, z_draft, 500.0, 500.0)
grid = Grid(mask, z_draft; x = ds["x"][i1:i2], y = ds["y"][j1:j2])
```
"""
Grid(mask::AbstractMatrix{Int}, z_draft::AbstractMatrix, dx::Real, dy::Real; kwargs...) =
    _grid(mask, z_draft, dx, dy; kwargs...)
Grid(mask::AbstractMatrix{Int}, z_draft::AbstractMatrix; kwargs...) =
    _grid(mask, z_draft, nothing, nothing; kwargs...)

function _grid(
    mask,
    z_draft,
    dx,
    dy;
    x = nothing,
    y = nothing,
    z_bed = nothing,
    preprocess = AbstractPreprocess[],
    domain_cropping = MinRectangleDomainCropping(),
    backend = CPU(),
    FT = Float64,
)
    _validate_input_shapes(mask, z_draft, z_bed)
    dx, x = _resolve_axis(x, dx, size(mask, 1), "x", FT)
    dy, y = _resolve_axis(y, dy, size(mask, 2), "y", FT)
    input_size = size(mask)
    mask = Matrix{Int}(mask)   # a copy: preprocessing never touches the caller's array
    for p in preprocess
        preprocess!(mask, p)
    end
    r, c = _crop_ranges(mask, domain_cropping)
    mask = mask[r, c]
    _validate_grid_mask(mask)
    nx_total, ny_total = size(mask)
    z_bed_ft = z_bed === nothing ? fill(FT(-Inf), nx_total, ny_total) : FT.(z_bed[r, c])
    grid = Grid{FT,Matrix{FT}}(
        nx_total,
        ny_total,
        FT(dx),
        FT(dy),
        mask,
        _adjust_z_draft(mask, z_draft[r, c], FT),
        z_bed_ft,
        x[r][2:(end-1)],
        y[c][2:(end-1)],
        (r, c),
        input_size,
    )
    return backend isa CPU ? grid : _grid_to_backend(grid, backend)
end

# Spacing and full-domain coordinates along one axis, from the spacing, the
# coordinates, or both (which must then agree).
function _resolve_axis(coord, d, n, name, FT)
    dname = "d" * name
    if coord === nothing
        d === nothing &&
            throw(ArgumentError("give the spacing `$dname` or the coordinates `$name`"))
        d > 0 || throw(ArgumentError("$dname must be positive, got $d"))
        return d, FT(d) .* (0:(n-1))
    end
    length(coord) == n || throw(
        ArgumentError(
            "`$name` has length $(length(coord)) but the mask has $n cells along " *
            "$name (the border ring included)",
        ),
    )
    steps = diff(Float64.(coord))
    s = steps[1]
    (s != 0 && all(st -> isapprox(st, s; rtol = 1e-6), steps)) ||
        throw(ArgumentError("`$name` must be uniformly spaced"))
    spacing = abs(s)
    d === nothing ||
        isapprox(d, spacing; rtol = 1e-6) ||
        throw(
            ArgumentError(
                "$dname = $d does not match the spacing $spacing of the `$name` coordinates",
            ),
        )
    return spacing, FT.(coord)
end
