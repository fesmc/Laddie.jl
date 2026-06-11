"""
    CGridProto

Standalone, Oceananigans-inspired prototype of the grid/field layer.  The
single idea on display: a `Field` carries its Arakawa C-grid location in its
*type* (`Center`/`Face` per axis), so the staggered operators
`∂x/∂y/ℑx/ℑy` dispatch to the correct stencil at compile time.

Everything is backend-agnostic via KernelAbstractions: the grid/fields
allocate on whatever backend you pass (here `CPU()`), and the same `@kernel`s
run on GPU backends unchanged.

This module is included inside `Laddie` for its `Center`/`Face` location
types and the leapfrog `Prognostic` container.
"""
module CGridProto

using KernelAbstractions
using DocStringExtensions
const KA = KernelAbstractions

export Center, Face, Periodic, Bounded, Flat
export CGrid, Field, Prognostic, State
export ∂x, ∂y, ℑx, ℑy
export fill_halo!, set!, set_constant!, interior, rotate!
export advect_euler!, advect_leapfrog
export CPU

# ----------------------------------------------------------------------------
# 1. Locations and topology — type-level vocabulary
# ----------------------------------------------------------------------------
abstract type AbstractLoc end
struct Center <: AbstractLoc end
struct Face <: AbstractLoc end

abstract type AbstractTopo end
struct Periodic <: AbstractTopo end
struct Bounded <: AbstractTopo end
struct Flat <: AbstractTopo end

# ----------------------------------------------------------------------------
# 2. Grid — float type + topology in the type; geometry/masks + backend in data
# ----------------------------------------------------------------------------
"""
$(TYPEDSIGNATURES)

Arakawa C-grid with float type `FT` and topology `(TX, TY)` encoded in the
type. Geometry scalars and mask/draft arrays are allocated on the
KernelAbstractions backend supplied at construction time.
"""
struct CGrid{FT,TX,TY,M}
    Nx::Int;
    Ny::Int
    Hx::Int;
    Hy::Int
    dx::FT;
    dy::FT
    draft::M
    tmask::M
    umask::M
    vmask::M
end

topology(::CGrid{FT,TX,TY}) where {FT,TX,TY} = (TX, TY)

"""
$(TYPEDSIGNATURES)

Allocate a `CGrid` on `backend` with float type `FT`, `Nx×Ny` interior cells,
physical size `Lx×Ly`, halo widths `Hx`/`Hy`, and a topology tuple
`(TX, TY)` drawn from `Periodic`, `Bounded`, or `Flat`.
"""
function CGrid(
    backend,
    ::Type{FT},
    Nx,
    Ny,
    Lx,
    Ly;
    Hx = 1,
    Hy = 1,
    topology = (Periodic, Periodic),
) where {FT}
    dims = (Nx + 2Hx, Ny + 2Hy)
    z() = KA.zeros(backend, FT, dims...)
    tmask = z();
    fill!(tmask, one(FT))
    umask = z();
    fill!(umask, one(FT))
    vmask = z();
    fill!(vmask, one(FT))
    draft = z()
    CGrid{FT,topology[1],topology[2],typeof(tmask)}(
        Nx,
        Ny,
        Hx,
        Hy,
        FT(Lx / Nx),
        FT(Ly / Ny),
        draft,
        tmask,
        umask,
        vmask,
    )
end

# ----------------------------------------------------------------------------
# 3. Field — location encoded as type parameters LX, LY
# ----------------------------------------------------------------------------
"""
    Field{LX, LY, G, D}

Grid-located array whose Arakawa C-grid position (`Center`/`Face` per axis) is
encoded in type parameters `LX` and `LY`, enabling compile-time dispatch to
the correct stencil in `∂x`, `∂y`, `ℑx`, and `ℑy`.
"""
struct Field{LX,LY,G,D}
    grid::G
    data::D
end

"""
$(TYPEDSIGNATURES)

Allocate a zero-initialised `Field` at location `(LX, LY)` on `grid`.
"""
function Field(::Type{LX}, ::Type{LY}, grid::CGrid{FT}) where {LX,LY,FT}
    data = KA.zeros(KA.get_backend(grid.tmask), FT, size(grid.tmask)...)
    Field{LX,LY,typeof(grid),typeof(data)}(grid, data)
end

location(::Field{LX,LY}) where {LX,LY} = (LX, LY)

"""
$(TYPEDSIGNATURES)

Return a view of the interior (non-halo) region of `f`.
"""
@inline interior(f::Field) = @view f.data[
    (f.grid.Hx+1):(f.grid.Hx+f.grid.Nx),
    (f.grid.Hy+1):(f.grid.Hy+f.grid.Ny),
]

# ----------------------------------------------------------------------------
# 4. Operators — low-level indexed stencils + location-dispatched front ends
# ----------------------------------------------------------------------------
@inline ∂xᶠ(i, j, dx, c) = (c[i, j] - c[i-1, j]) / dx
@inline ∂yᶠ(i, j, dy, c) = (c[i, j] - c[i, j-1]) / dy
@inline ∂xᶜ(i, j, dx, u) = (u[i+1, j] - u[i, j]) / dx
@inline ∂yᶜ(i, j, dy, v) = (v[i, j+1] - v[i, j]) / dy
@inline ℑxᶠ(i, j, c) = (c[i, j] + c[i-1, j]) / 2
@inline ℑyᶠ(i, j, c) = (c[i, j] + c[i, j-1]) / 2
@inline ℑxᶜ(i, j, u) = (u[i+1, j] + u[i, j]) / 2
@inline ℑyᶜ(i, j, v) = (v[i, j+1] + v[i, j]) / 2

@inline ∂x(i, j, f::Field{Center}) = ∂xᶠ(i, j, f.grid.dx, f.data)
@inline ∂x(i, j, f::Field{Face}) = ∂xᶜ(i, j, f.grid.dx, f.data)
@inline ∂y(i, j, f::Field{LX,Center}) where {LX} = ∂yᶠ(i, j, f.grid.dy, f.data)
@inline ∂y(i, j, f::Field{LX,Face}) where {LX} = ∂yᶜ(i, j, f.grid.dy, f.data)

@inline ℑx(i, j, f::Field{Center}) = ℑxᶠ(i, j, f.data)
@inline ℑx(i, j, f::Field{Face}) = ℑxᶜ(i, j, f.data)
@inline ℑy(i, j, f::Field{LX,Center}) where {LX} = ℑyᶠ(i, j, f.data)
@inline ℑy(i, j, f::Field{LX,Face}) where {LX} = ℑyᶜ(i, j, f.data)

# ----------------------------------------------------------------------------
# 5. Prognostic — 3-level leapfrog state, advanced by reference rotation
# ----------------------------------------------------------------------------
"""
    Prognostic{LX, LY, F}

Three-level leapfrog container (`past`, `present`, `future`) for a `Field`
located at `(LX, LY)`. Advance the time levels in-place with `rotate!`.
"""
mutable struct Prognostic{LX,LY,F}
    past::F
    present::F
    future::F
end

"""
$(TYPEDSIGNATURES)

Allocate three zero-initialised `Field`s at location `(LX, LY)` and wrap them
in a `Prognostic` container.
"""
function Prognostic(::Type{LX}, ::Type{LY}, grid) where {LX,LY}
    past, present, future =
        Field(LX, LY, grid), Field(LX, LY, grid), Field(LX, LY, grid)
    Prognostic{LX,LY,typeof(past)}(past, present, future)
end

"""
$(TYPEDSIGNATURES)

Cycle the time levels of `p` in-place: `past ← present`, `present ← future`,
`future ← past` (old past buffer is reused for the next future step).
"""
function rotate!(p::Prognostic)
    p.past, p.present, p.future = p.present, p.future, p.past
    return p
end

"""
    State{D,U,V,T,S}

Bundle of `Prognostic` fields for layer thickness `D`, zonal velocity `U`,
meridional velocity `V`, temperature `T`, and salinity `S`.
"""
struct State{D,U,V,T,S}
    D::D;
    U::U;
    V::V;
    T::T;
    S::S
end
State(g) = State(
    Prognostic(Center, Center, g),
    Prognostic(Face, Center, g),
    Prognostic(Center, Face, g),
    Prognostic(Center, Center, g),
    Prognostic(Center, Center, g),
)

# ----------------------------------------------------------------------------
# 6. Halo filling — dispatched per axis on topology
# ----------------------------------------------------------------------------
fill_x!(a, ::Type{Periodic}, Nx, Hx) = begin
    for h = 1:Hx
        @views a[h, :] .= a[h+Nx, :]
        @views a[Hx+Nx+h, :] .= a[Hx+h, :]
    end
end
fill_x!(a, ::Type{Bounded}, Nx, Hx) = begin
    for h = 1:Hx
        @views a[h, :] .= a[Hx+1, :]
        @views a[Hx+Nx+h, :] .= a[Hx+Nx, :]
    end
end
fill_x!(a, ::Type{Flat}, Nx, Hx) = nothing

fill_y!(a, ::Type{Periodic}, Ny, Hy) = begin
    for h = 1:Hy
        @views a[:, h] .= a[:, h+Ny]
        @views a[:, Hy+Ny+h] .= a[:, Hy+h]
    end
end
fill_y!(a, ::Type{Bounded}, Ny, Hy) = begin
    for h = 1:Hy
        @views a[:, h] .= a[:, Hy+1]
        @views a[:, Hy+Ny+h] .= a[:, Hy+Ny]
    end
end
fill_y!(a, ::Type{Flat}, Ny, Hy) = nothing

"""
$(TYPEDSIGNATURES)

Fill the halo cells of `f` according to the grid topology (`Periodic`,
`Bounded`, or `Flat`) along each axis.
"""
function fill_halo!(f::Field)
    g = f.grid
    tx, ty = topology(g)
    fill_x!(f.data, tx, g.Nx, g.Hx)
    fill_y!(f.data, ty, g.Ny, g.Hy)
    return f
end

# ----------------------------------------------------------------------------
# 7. Initialization helpers
# ----------------------------------------------------------------------------
"""
$(TYPEDSIGNATURES)

Set the interior of `f` by evaluating `fun(x, y)` at each cell center, then
fill halos.
"""
function set!(f::Field, fun)
    g = f.grid
    for j = 1:g.Ny, i = 1:g.Nx
        x = (i - 0.5) * g.dx
        y = (j - 0.5) * g.dy
        f.data[g.Hx+i, g.Hy+j] = fun(x, y)
    end
    fill_halo!(f)
    return f
end

"""
$(TYPEDSIGNATURES)

Fill all of `f.data` (including halos) with the scalar `val`.
"""
set_constant!(f::Field, val) = (fill!(f.data, val); f)

# ----------------------------------------------------------------------------
# 8. Kernels — mass-conserving upstream advection + leapfrog/RA update
# ----------------------------------------------------------------------------
@kernel function _advect_tendency!(
    dc,
    @Const(c),
    @Const(u),
    @Const(v),
    @Const(tmask),
    dx,
    dy,
    Hx,
    Hy,
)
    I, J = @index(Global, NTuple)
    i = I + Hx;
    j = J + Hy
    @inbounds begin
        Fxe = u[i+1, j] >= 0 ? u[i+1, j]*c[i, j] : u[i+1, j]*c[i+1, j]
        Fxw = u[i, j] >= 0 ? u[i, j] * c[i-1, j] : u[i, j] * c[i, j]
        Fyn = v[i, j+1] >= 0 ? v[i, j+1]*c[i, j] : v[i, j+1]*c[i, j+1]
        Fys = v[i, j] >= 0 ? v[i, j] * c[i, j-1] : v[i, j] * c[i, j]
        dc[i, j] = (-(Fxe - Fxw)/dx - (Fyn - Fys)/dy) * tmask[i, j]
    end
end

@kernel function _advect_centered_tendency!(
    dc,
    @Const(c),
    @Const(u),
    @Const(v),
    @Const(tmask),
    dx,
    dy,
    Hx,
    Hy,
)
    I, J = @index(Global, NTuple)
    i = I + Hx;
    j = J + Hy
    @inbounds begin
        Fxe = u[i+1, j] * (c[i+1, j] + c[i, j]) / 2
        Fxw = u[i, j] * (c[i, j] + c[i-1, j]) / 2
        Fyn = v[i, j+1] * (c[i, j+1] + c[i, j]) / 2
        Fys = v[i, j] * (c[i, j] + c[i, j-1]) / 2
        dc[i, j] = (-(Fxe - Fxw)/dx - (Fyn - Fys)/dy) * tmask[i, j]
    end
end

@kernel function _euler!(c, @Const(dc), dt, Hx, Hy)
    I, J = @index(Global, NTuple)
    i = I + Hx;
    j = J + Hy
    @inbounds c[i, j] += dt * dc[i, j]
end

@kernel function _leap!(future, @Const(base), @Const(dc), @Const(tmask), Δt, Hx, Hy)
    I, J = @index(Global, NTuple)
    i = I + Hx;
    j = J + Hy
    @inbounds future[i, j] =
        (base[i, j] + Δt * dc[i, j]) * tmask[i, j] + base[i, j] * (1 - tmask[i, j])
end

@kernel function _ra!(present, @Const(past), @Const(future), ν, Hx, Hy)
    I, J = @index(Global, NTuple)
    i = I + Hx;
    j = J + Hy
    FT = typeof(ν)
    @inbounds present[i, j] +=
        ν / (FT(1) + FT(1)) *
        (past[i, j] + future[i, j] - (FT(1) + FT(1)) * present[i, j])
end

# ----------------------------------------------------------------------------
# 9. Drivers
# ----------------------------------------------------------------------------
function _tendency!(dc, c::Field, u::Field, v::Field; kernel = _advect_tendency!)
    g = c.grid;
    b = KA.get_backend(c.data)
    kernel(b, (16, 16))(
        dc,
        c.data,
        u.data,
        v.data,
        g.tmask,
        g.dx,
        g.dy,
        g.Hx,
        g.Hy;
        ndrange = (g.Nx, g.Ny),
    )
    KA.synchronize(b)
    return dc
end

"""
$(TYPEDSIGNATURES)

Forward-Euler upstream advection (strictly mass-conserving on periodic grids).
"""
function advect_euler!(c::Field, u::Field, v::Field, dt, nsteps)
    g = c.grid;
    b = KA.get_backend(c.data)
    dc = similar(c.data);
    fill!(dc, 0)
    fill_halo!(u);
    fill_halo!(v)
    for _ = 1:nsteps
        fill_halo!(c)
        _tendency!(dc, c, u, v)
        _euler!(b, (16, 16))(c.data, dc, dt, g.Hx, g.Hy; ndrange = (g.Nx, g.Ny))
        KA.synchronize(b)
    end
    return c
end

"""
$(TYPEDSIGNATURES)

Leapfrog + Robert–Asselin advection; returns the present field after `nsteps`.
"""
function advect_leapfrog(c0::Field, u::Field, v::Field, dt, nsteps; ν = 0.05)
    g = c0.grid;
    b = KA.get_backend(c0.data)
    p = Prognostic(Center, Center, g)
    copyto!(p.past.data, c0.data)
    copyto!(p.present.data, c0.data)
    dc = similar(c0.data);
    fill!(dc, 0)
    fill_halo!(u);
    fill_halo!(v)
    for n = 1:nsteps
        fill_halo!(p.present)
        _tendency!(dc, p.present, u, v; kernel = _advect_centered_tendency!)
        base = n == 1 ? p.present.data : p.past.data
        Δt = n == 1 ? dt : 2dt
        _leap!(b, (16, 16))(
            p.future.data,
            base,
            dc,
            g.tmask,
            Δt,
            g.Hx,
            g.Hy;
            ndrange = (g.Nx, g.Ny),
        )
        KA.synchronize(b)
        if n > 1
            _ra!(b, (16, 16))(
                p.present.data,
                p.past.data,
                p.future.data,
                ν,
                g.Hx,
                g.Hy;
                ndrange = (g.Nx, g.Ny),
            )
            KA.synchronize(b)
        end
        rotate!(p)
    end
    return p.present
end

end # module CGridProto
