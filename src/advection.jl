"""
$(TYPEDEF)

Abstract supertype for the momentum-advection scheme — how the momentum carried
through a face of a velocity control volume is evaluated. 

To use this within a simulation, pass a concrete
instance as `Params(; momentum_advection = ...)`:
[`CentredMomentumAdvection`](@ref) (the default) or
[`UpstreamMomentumAdvection`](@ref).

Tracer advection is always upwind and is unaffected.
"""
abstract type AbstractMomentumAdvection end

"""
$(TYPEDEF)

Evaluate the advected momentum as the average of the two velocities either side
of the face (the default), as Python LADDIE v1.x does: the east face of a
U-control volume carries `D̄·ū·ū` with `ū = (U[i] + U[i+1])/2`, and likewise for
the other faces.  Centred differencing adds no numerical viscosity of its own.

Select via `Params(; momentum_advection = CentredMomentumAdvection())` (the default).
"""
struct CentredMomentumAdvection <: AbstractMomentumAdvection end

"""
$(TYPEDEF)

Flux-form donor-cell (first-order upwind) momentum advection, as in LADDIE v2.

Each face of a velocity control volume carries the mass flux of the thickness
equation — the same donor-cell `D·u` the tracer advection uses, gated by
`umask`/`vmask` — averaged onto that face from the two adjacent T-cell faces.
The momentum it carries is the velocity on the upstream side of the face.  Both
the thickness and the velocity are therefore taken from the donor side, as v2
does (`Hstar_b(ti)` vs `Hstar_b(tj)` in `compute_divQUV_upstream` /
`compute_divQUV_fesom`, `laddie_velocity.f90`).

v2's two schemes, `'upstream'` and `'fesom'`, differ only in how the face-normal
velocity is built from the mesh's staggered fields — a triangle-vs-vertex
degree-of-freedom question that does not arise on a C-grid, where the face
velocity is the one used here.

Two consequences follow from building the fluxes this way:

  * Walls carry no momentum, because they carry no mass (`umask`/`vmask` are
    zero there).  That is v2's "No flux across grounding line", so
    [`AbstractWallAdvection`](@ref) has no effect under this scheme and the wall
    slip factor acts through the viscosity alone, again as in v2.
  * An inflow face at the ice front carries the cell's own thickness and
    velocity (zero gradient), matching the thickness equation's
    [`ZeroGradientInflow`](@ref).

First-order upwinding is dissipative: its implicit numerical viscosity is of
order `|u|·Δ/2`, which at 1 km and 0.1 m s⁻¹ is ≈ 50 m² s⁻¹, far above a typical
[`Params.A_h`](@ref Params).  Expect a less energetic layer than under
[`CentredMomentumAdvection`](@ref).

Select via `Params(; momentum_advection = UpstreamMomentumAdvection())`.
"""
struct UpstreamMomentumAdvection <: AbstractMomentumAdvection end
