
# ============================================================================
# Ice-base slope gradient variants
# ============================================================================

"""
Abstract supertype for ice-base slope (dzdx/dzdy) computation strategies.
Pass a concrete instance as the `gradient` keyword to `build_model` or
`build_isomip`.
"""
abstract type AbstractIceSlopeGradient end

"""
    PyGradient()

Python-ported centred-difference gradient: `(zb_E − zb_W) / (2 dx)` applied
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
exists.  This avoids spurious slopes from the physically-incompatible `zb`
values stored at ocean cells (`zb=0`) and grounded cells (`zb=bed`).
"""
struct JlGradient <: AbstractIceSlopeGradient end

function _icebase_slope(::PyGradient, tmask, zb_ft, dx_ft, dy_ft, FT)
    return gradient_x(zb_ft, dx_ft), gradient_y(zb_ft, dy_ft)
end

function _icebase_slope(::JlGradient, tmask, zb_ft, dx_ft, dy_ft, FT)
    # Only read shelf-cell neighbours to avoid incorporating the
    # physically-incompatible zb values at ocean (zb=0) and grounded
    # (zb=bed) cells, which would produce O(0.2) spurious slopes.
    #
    # Shift convention: xm1=east, xp1=west, ym1=north, yp1=south.
    # Stencil at each shelf cell:
    #   both neighbours shelf → centred  (zb_E − zb_W) / (2 dx)
    #   east only             → forward  (zb_E − zb_C) / dx
    #   west only             → backward (zb_C − zb_W) / dx
    #   neither               → 0
    _tm_e = xm1(tmask); _tm_w = xp1(tmask)
    _tm_n = ym1(tmask); _tm_s = yp1(tmask)
    _zb_e = xm1(zb_ft); _zb_w = xp1(zb_ft)
    _zb_n = ym1(zb_ft); _zb_s = yp1(zb_ft)
    dzdx = ifelse.(tmask .> 0,
        ifelse.(_tm_e .* _tm_w .> 0,
            (_zb_e .- _zb_w) ./ (FT(2) .* dx_ft),
            ifelse.(_tm_e .> 0,
                (_zb_e .- zb_ft) ./ dx_ft,
                ifelse.(_tm_w .> 0,
                    (zb_ft .- _zb_w) ./ dx_ft,
                    zero(FT)))),
        gradient_x(zb_ft, dx_ft))
    dzdy = ifelse.(tmask .> 0,
        ifelse.(_tm_n .* _tm_s .> 0,
            (_zb_n .- _zb_s) ./ (FT(2) .* dy_ft),
            ifelse.(_tm_n .> 0,
                (_zb_n .- zb_ft) ./ dy_ft,
                ifelse.(_tm_s .> 0,
                    (zb_ft .- _zb_s) ./ dy_ft,
                    zero(FT)))),
        gradient_y(zb_ft, dy_ft))
    return dzdx, dzdy
end

# ============================================================================
# Grid{FT, A} — static geometry, masks, stagger-count denominators.
# A is the concrete matrix type (Matrix{FT} on CPU, CuArray{FT,2} on GPU).
# mask is always kept as Matrix{Int} on the CPU for host-side branching.
# ============================================================================

struct Grid{FT,A<:AbstractMatrix{FT}}
    Nx::Int
    Ny::Int
    dx::FT
    dy::FT

    mask::Matrix{Int}
    zb::A
    z_bed::A
    dzdx::A
    dzdy::A

    tmask::A
    grd::A
    ocn::A

    ocnym1::A
    ocnyp1::A
    ocnxm1::A
    ocnxp1::A

    tmaskym1::A
    tmaskyp1::A
    tmaskxm1::A
    tmaskxp1::A
    tmaskxm1ym1::A
    tmaskxm1yp1::A
    tmaskxp1ym1::A

    grdNu::A
    grdSu::A
    grdEv::A
    grdWv::A
    glNu::A
    glSu::A
    glEv::A
    glWv::A
    isfE::A
    isfW::A
    isfN::A
    isfS::A
    isf::A
    grlE::A
    grlW::A
    grlN::A
    grlS::A
    grl::A

    umask::A
    vmask::A
    umaskym1::A
    umaskyp1::A
    umaskxm1::A
    umaskxp1::A
    vmaskym1::A
    vmaskyp1::A
    vmaskxm1::A
    vmaskxp1::A

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

"""
$(TYPEDSIGNATURES)

Build all masks and stagger-count denominators from the raw integer `mask` and
ice-draft array `zb`.  Returns an immutable typed struct.
"""
function Grid(mask::AbstractMatrix{Int}, zb::AbstractMatrix, z_bed_raw::AbstractMatrix, dx, dy; FT = Float64, gradient = JlGradient())
    dx_ft = FT(dx)
    dy_ft = FT(dy)
    zb_ft = FT.(zb)
    z_bed_ft = FT.(z_bed_raw)

    # Primary classification
    tmask = FT.(mask .== 3)
    grd = FT.((mask .== 2) .| (mask .== 1))
    ocn = FT.(mask .== 0)

    ocnym1 = ym1(ocn)
    ocnyp1 = yp1(ocn)
    ocnxm1 = xm1(ocn)
    ocnxp1 = xp1(ocn)

    dzdx, dzdy = _icebase_slope(gradient, tmask, zb_ft, dx_ft, dy_ft, FT)

    tmaskym1 = ym1(tmask)
    tmaskyp1 = yp1(tmask)
    tmaskxm1 = xm1(tmask)
    tmaskxp1 = xp1(tmask)
    tmaskxm1ym1 = ym1(xm1(tmask))
    tmaskxm1yp1 = yp1(xm1(tmask))
    tmaskxp1ym1 = ym1(xp1(tmask))

    # Boundary geometry
    o = one(FT)
    grdNu = o .- ym1((o .- grd) .* (o .- xm1(grd)))
    grdSu = o .- yp1((o .- grd) .* (o .- xm1(grd)))
    grdEv = o .- xm1((o .- grd) .* (o .- ym1(grd)))
    grdWv = o .- xp1((o .- grd) .* (o .- ym1(grd)))
    # Grounding-line-only wall indicators: same stencil as grdNu..grdWv but
    # restricted to grounded ice (mask == 2, excluding land/border walls),
    # so the momentum kernels can apply a different slip factor at the
    # grounding line (AbstractGroundingLineBC).  Subset of grd?? pointwise.
    gl = FT.(mask .== 2)
    glNu = o .- ym1((o .- gl) .* (o .- xm1(gl)))
    glSu = o .- yp1((o .- gl) .* (o .- xm1(gl)))
    glEv = o .- xm1((o .- gl) .* (o .- ym1(gl)))
    glWv = o .- xp1((o .- gl) .* (o .- ym1(gl)))
    isfE = ocn .* tmaskxp1
    isfW = ocn .* tmaskxm1
    isfN = ocn .* tmaskyp1
    isfS = ocn .* tmaskym1
    isf = isfE .+ isfN .+ isfW .+ isfS
    grlE = grd .* tmaskxp1
    grlW = grd .* tmaskxm1
    grlN = grd .* tmaskyp1
    grlS = grd .* tmaskym1
    grl = grlE .+ grlN .+ grlW .+ grlS

    # Velocity masks
    umask = (tmask .+ isfW) .* (o .- xm1(grlE))
    vmask = (tmask .+ isfS) .* (o .- ym1(grlN))
    umaskym1 = ym1(umask)
    umaskyp1 = yp1(umask)
    umaskxm1 = xm1(umask)
    umaskxp1 = xp1(umask)
    vmaskym1 = ym1(vmask)
    vmaskyp1 = yp1(vmask)
    vmaskxm1 = xm1(vmask)
    vmaskxp1 = xp1(vmask)

    # Stagger-count denominators
    tmask_im = tmask .+ tmaskxp1
    tmask_ip = tmask .+ tmaskxm1
    tmask_jm = tmask .+ tmaskyp1
    tmask_jp = tmask .+ tmaskym1
    umask_im = umask .+ umaskxp1
    umask_ip = umask .+ umaskxm1
    umask_jm = umask .+ umaskyp1
    umask_jp = umask .+ umaskym1
    vmask_im = vmask .+ vmaskxp1
    vmask_ip = vmask .+ vmaskxm1
    vmask_jm = vmask .+ vmaskyp1
    vmask_jp = vmask .+ vmaskym1

    ny, nx = size(mask)
    Grid{FT,Matrix{FT}}(
        nx,
        ny,
        dx_ft,
        dy_ft,
        Matrix{Int}(mask),
        zb_ft,
        z_bed_ft,
        dzdx,
        dzdy,
        tmask,
        grd,
        ocn,
        ocnym1,
        ocnyp1,
        ocnxm1,
        ocnxp1,
        tmaskym1,
        tmaskyp1,
        tmaskxm1,
        tmaskxp1,
        tmaskxm1ym1,
        tmaskxm1yp1,
        tmaskxp1ym1,
        grdNu,
        grdSu,
        grdEv,
        grdWv,
        glNu,
        glSu,
        glEv,
        glWv,
        isfE,
        isfW,
        isfN,
        isfS,
        isf,
        grlE,
        grlW,
        grlN,
        grlS,
        grl,
        umask,
        vmask,
        umaskym1,
        umaskyp1,
        umaskxm1,
        umaskxp1,
        vmaskym1,
        vmaskyp1,
        vmaskxm1,
        vmaskxp1,
        tmask_im,
        tmask_ip,
        tmask_jm,
        tmask_jp,
        umask_im,
        umask_ip,
        umask_jm,
        umask_jp,
        vmask_im,
        vmask_ip,
        vmask_jm,
        vmask_jp,
    )
end
