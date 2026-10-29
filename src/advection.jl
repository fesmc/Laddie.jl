"""
$(TYPEDEF)

Abstract supertype for the momentum-advection scheme — how the momentum carried
through a face of a velocity control volume is evaluated. 

To use this within a simulation, pass a concrete
instance as `Params(; momentum_advection = ...)`:
[`CentredMomentumAdvection`](@ref) (the default) or
[`UpstreamMomentumAdvection`](@ref).

Tracer advection is always upwind and is unaffected; for the form of the tracer
equations see [`AbstractTracerForm`](@ref).
"""
abstract type AbstractMomentumAdvection end

"""
$(TYPEDEF)

Evaluate the advected momentum as the average of the two velocities either side
of the face (the default), as Python LADDIE v1.x does: the east face of a
U-control volume carries `H̄·ū·ū` with `ū = (U[i] + U[i+1])/2`, and likewise for
the other faces.  Centred differencing adds no numerical viscosity of its own.

Select via `Params(; momentum_advection = CentredMomentumAdvection())` (the default).
"""
struct CentredMomentumAdvection <: AbstractMomentumAdvection end

"""
$(TYPEDEF)

Flux-form donor-cell (first-order upwind) momentum advection, as in LADDIE v2.

Each face of a velocity control volume carries the mass flux of the thickness
equation — the same donor-cell `H·u` the tracer advection uses, gated by
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

"""
$(TYPEDEF)

Abstract supertype for the form in which the temperature and salinity equations
are stepped.  Pass a concrete instance as `Params(; tracer_form = ...)`:
[`FluxTracerForm`](@ref) (the default) or [`AdvectiveTracerForm`](@ref).

Both integrate the same flux-form budget `∂(Hq)/∂t = -∇·(H u q) + …`
and differ only in what they step: the tracer `q` itself, or its content `H q`.
"""
abstract type AbstractTracerForm end

"""
$(TYPEDEF)

Step the tracer itself, as Python LADDIE v1.x does: the budget is
expanded by the product rule into `H ∂q/∂t = … - q ∂H/∂t`, with `∂H/∂t`
taken from the thickness step just made, and

```
q⁺ = q⁻ + 2Δt · rhs / Hⁿ.
```

The `- q ∂H/∂t` term cancels the volume sources only as far as the discrete
`∂H/∂t` matches them in the same step.  It does not wherever the thickness
step is clamped or restarted, and the mismatch is then divided by `H`: in thin
layers this can drive `q` far outside the range of the waters that make it up.

Select it explicitly, `Params(; tracer_form = AdvectiveTracerForm())`, to reproduce
Python LADDIE v1.x, as its verification test does.
"""
struct AdvectiveTracerForm <: AbstractTracerForm end

"""
$(TYPEDEF)

Step the tracer content (the default), as LADDIE v2 does (`laddie_tracers.f90`), and recover the
tracer by dividing by the new thickness:

```
(Hq)⁺ = H⁻ q⁻ + 2Δt · rhs,     q⁺ = (Hq)⁺ / H⁺,
```

where `rhs` is the advective tracer `rhs` without its `- q ∂H/∂t` term.  The
sources then enter the tracer only in proportion to the volume they bring, so a
mismatch between the thickness and tracer budgets cannot be amplified by a thin
layer.

Where the clamp after the thickness step moves `H⁺` off the stepped value `H*`,
the moved water carries its matching tracer, as LADDIE v2's preventive bounds do:
volume cut at the upper bound leaves at the layer's own `q`, volume added by the
`H_min` floor comes in at the ambient `q_a`,

```
q⁺ = ((Hq)⁺ + max(H⁺ - H*, 0) q_a) / max(H⁺, H*).
```

The clamp binds mostly in the bootstrap step, where `ent2` and the cap
detrainment, sized for a 2Δt step, do only half their work.

Select via `Params(; tracer_form = FluxTracerForm())` (the default).
"""
struct FluxTracerForm <: AbstractTracerForm end

"""
$(TYPEDEF)

Abstract supertype for the form in which the momentum equations are stepped.  Pass a
concrete instance as `Params(; momentum_form = ...)`: [`AdvectiveMomentumForm`](@ref)
(the default) or [`FluxMomentumForm`](@ref).

Both integrate the same budget `∂(HU)/∂t = -∇·(H U u) + …`, the momentum counterpart
of [`AbstractTracerForm`](@ref), and differ in what they step: the velocity `U`
itself, or the transport `H U`.
"""
abstract type AbstractMomentumForm end

"""
$(TYPEDEF)

Step the velocity itself, as Python LADDIE v1.x does (the default): the budget is
expanded by the product rule into `H ∂U/∂t = … - U ∂H/∂t`, with all terms at the
present level, and

```
U⁺ = U⁻ + 2Δt · rhs / H̄ⁿ.
```

Water entrained at rest dilutes the momentum through `- U ∂H/∂t`, a damping at the
rate `e/H` evaluated at the centred level.  The leapfrog with the Robert–Asselin filter
(ν = 0.8) holds such a term only up to a rate of about `0.57/Δt`, and Gaspar's
rate grows with `|U|³`, which makes the linearised rate `4e/H`: thin, fast,
strongly entraining layers exceed it, and the computational mode then flips `U` every
step.  [`FluxMomentumForm`](@ref) takes these terms at the new level.
"""
struct AdvectiveMomentumForm <: AbstractMomentumForm end

"""
$(TYPEDEF)

Step the transport `H̄U` and recover the velocity by dividing by the new face
thickness, as LADDIE v2 does (`laddie_velocity.f90`):

```
U⁺ = (H̄⁻ U⁻ + 2Δt · rhs) / (H̄⁺ + 2Δt · s),     s = d + d_cap + f + C_d |u| + o,
```

where `rhs` is the advective `rhs` without its `- U ∂H/∂t` term and without the terms
that remove momentum at the layer's own `U`: the detrainment `d`, the cap
detrainment `d_cap`, the frazil formation `f` (all averaged onto the face), the
quadratic drag and, under [`UpstreamMomentumAdvection`](@ref), the outflow `o` of the
control volume's own velocity through its faces.  Those enter as the rate `s` at the
new level, with the drag speed `|u|` at the level of [`AbstractClosureLevel`](@ref).
The outflow is the momentum counterpart of the tracer outflow of
[`FluxTracerForm`](@ref): explicit, a thin face next to thick donors loses more
momentum per step than it holds, and the leapfrog flips its sign every step.

Water entrained at rest then dilutes `U` through `H̄⁺` alone, by the factor
`H̄⁻/H̄⁺ ≤ 1` whatever the rate, and water leaving at `U` leaves it unchanged.
Under upstream advection `U⁺` is a weighted mean of `U⁻` and the momentum coming in.  All
of these are unconditionally stable, where the advective form becomes unstable at a
damping rate of about `0.57/Δt` (see [`AdvectiveMomentumForm`](@ref)).

Where the clamp after the thickness step moves `H⁺` off the stepped value `H*`, the
moved water carries its matching momentum, as in [`FluxTracerForm`](@ref): volume cut
at the upper bound leaves at `U`, volume added by the `H_min` floor comes in at rest,
so the denominator takes the face average of `max(H⁺, H*)`.

Select via `Params(; momentum_form = FluxMomentumForm())`.
"""
struct FluxMomentumForm <: AbstractMomentumForm end

"""
$(TYPEDEF)

Abstract supertype for the time level the closures see: the friction velocity `u★`
(hence the melt and entrainment rates), the thickness in the entrainment rate, and
the speed in the quadratic drag.  Pass a concrete instance as
`Params(; closure_level = ...)`: [`PresentClosures`](@ref) (the default) or
[`PastClosures`](@ref).

A closure that damps the state it is computed from (entrainment ∝ `u★³/H` dilutes
`U` and thickens `H`; drag ∝ `|u|` slows `u`) is unstable in the leapfrog at the
centred level once its rate exceeds about `0.57/Δt`; from the lagged level it stays
stable to well beyond that.  Python LADDIE v1.x lags the diffusion and the convective
relaxation for the same reason.  The density contrast `δρ` stays at the present level
under both, since it also drives the pressure gradient.
"""
abstract type AbstractClosureLevel end

"""
$(TYPEDEF)

Compute the closures from the present level (the default), as Python LADDIE v1.x does.
"""
struct PresentClosures <: AbstractClosureLevel end

"""
$(TYPEDEF)

Compute `u★`, the thickness in the entrainment rate, and the drag speed from the
lagged, filtered `past` level.  First order in time for these terms; combined with
[`FluxMomentumForm`](@ref) it removes the step-to-step oscillation of thin, strongly
entraining layers.

Select via `Params(; closure_level = PastClosures())`.
"""
struct PastClosures <: AbstractClosureLevel end

# The prognostic levels the closures read.
_closure_level(v, ::PresentClosures) = v.present
_closure_level(v, ::PastClosures) = v.past
