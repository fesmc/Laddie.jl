"""
$(TYPEDEF)

Abstract supertype for the momentum-advection scheme: how the momentum carried through a face
of a velocity control volume is evaluated. Pass a concrete instance as
`Params(; momentum_advection = ...)`: [`CentredMomentumAdvection`](@ref) (the default) or
[`UpstreamMomentumAdvection`](@ref). Tracer advection is always upwind.
"""
abstract type AbstractMomentumAdvection end

"""
$(TYPEDEF)

Evaluate the advected momentum as the average of the two velocities either side
of the face (the default), as LADDIE v1 does: the east face of a
U-control volume carries `H̄·ū·ū` with `ū = (U[i] + U[i+1])/2`, and likewise for
the other faces.  Centred differencing adds no numerical viscosity of its own.

Select via `Params(; momentum_advection = CentredMomentumAdvection())` (the default).
"""
struct CentredMomentumAdvection <: AbstractMomentumAdvection end

"""
$(TYPEDEF)

Flux-form donor-cell (first-order upwind) momentum advection, as in LADDIE v2.

Each face of a velocity control volume carries the donor-cell mass flux `H·u` of the thickness
equation, averaged onto the face, and the velocity of its upstream side. Walls therefore carry
no momentum (so [`AbstractWallAdvection`](@ref) has no effect), and the scheme is dissipative,
of order `|u|·Δ/2`; see [Momentum advection](@ref).

Select via `Params(; momentum_advection = UpstreamMomentumAdvection())`.
"""
struct UpstreamMomentumAdvection <: AbstractMomentumAdvection end

"""
$(TYPEDEF)

Abstract supertype for the form in which the temperature and salinity equations
are stepped.  Pass a concrete instance as `Params(; tracer_form = ...)`:
[`FluxTracerForm`](@ref) (the default) or [`AdvectiveTracerForm`](@ref).

Both integrate the budget `∂(Hq)/∂t = -∇·(H u q) + …`, stepping either `q` or `H q`.
"""
abstract type AbstractTracerForm end

"""
$(TYPEDEF)

Step the tracer itself, as LADDIE v1 does: the budget is
expanded by the product rule into `H ∂q/∂t = … - q ∂H/∂t`, with `∂H/∂t`
taken from the thickness step just made, and

```
q⁺ = q⁻ + 2Δt · rhs / Hⁿ.
```

In thin layers a mismatch between the thickness and tracer steps, divided by `H`, can drive `q`
outside the range of its source waters; see [Tracer form](@ref).

Select it explicitly, `Params(; tracer_form = AdvectiveTracerForm())`, to reproduce
LADDIE v1, as its verification test does.
"""
struct AdvectiveTracerForm <: AbstractTracerForm end

"""
$(TYPEDEF)

Step the tracer content (the default), as LADDIE v2 does (`laddie_tracers.f90`), and recover the
tracer by dividing by the new thickness:

```
(Hq)⁺ = H⁻ q⁻ + 2Δt · rhs,     q⁺ = (Hq)⁺ / H⁺,
```

where `rhs` is the advective `rhs` without its `- q ∂H/∂t` term, so a thin layer cannot amplify
a mismatch of the budgets. Where the clamp after the thickness step moves `H⁺` off the stepped
`H*`, the moved water carries its matching tracer (the layer's `q` out, ambient `q_a` in):

```
q⁺ = ((Hq)⁺ + max(H⁺ - H*, 0) q_a) / max(H⁺, H*).
```

Select via `Params(; tracer_form = FluxTracerForm())` (the default).
"""
struct FluxTracerForm <: AbstractTracerForm end

"""
$(TYPEDEF)

Abstract supertype for the form in which the momentum equations are stepped. Pass a concrete
instance as `Params(; momentum_form = ...)`: [`FluxMomentumForm`](@ref) (the default) or
[`AdvectiveMomentumForm`](@ref). Both integrate `∂(HU)/∂t = -∇·(H U u) + …`, stepping either
`U` or `H U`.
"""
abstract type AbstractMomentumForm end

"""
$(TYPEDEF)

Step the velocity itself, as LADDIE v1 does: the budget is expanded into
`H ∂U/∂t = … - U ∂H/∂t`, all terms at the present level, and

```
U⁺ = U⁻ + 2Δt · rhs / H̄ⁿ.
```

The dilution `- U ∂H/∂t` is a damping that the filtered leapfrog holds only up to about
`0.57/Δt`; thin, fast, strongly entraining layers exceed it (see [Momentum form and closure level](@ref)).

Select via `Params(; momentum_form = AdvectiveMomentumForm())`.
"""
struct AdvectiveMomentumForm <: AbstractMomentumForm end

"""
$(TYPEDEF)

Step the transport `H̄U` and recover the velocity by dividing by the new face
thickness, as LADDIE v2 does (`laddie_velocity.f90`):

```
U⁺ = (H̄⁻ U⁻ + 2Δt · rhs) / (H̄⁺ + 2Δt · s),     s = d + d_cap + f + C_d |u| + o,
```

where `rhs` is the advective `rhs` without `- U ∂H/∂t` and without the terms that remove
momentum at the layer's own `U`: detrainment `d`, cap detrainment `d_cap`, frazil `f`, the
quadratic drag and, under [`UpstreamMomentumAdvection`](@ref), the outflow `o`. These enter as
the rate `s` at the new level, which keeps them unconditionally stable. Water moved by the
clamp of `H` carries its matching momentum, as in [`FluxTracerForm`](@ref). See
[Momentum form and closure level](@ref).

Select via `Params(; momentum_form = FluxMomentumForm())` (the default).
"""
struct FluxMomentumForm <: AbstractMomentumForm end

"""
$(TYPEDEF)

Abstract supertype for the time level the closures see: the friction velocity `u★`
(hence the melt and entrainment rates), the thickness in the entrainment rate, and
the speed in the quadratic drag.  Pass a concrete instance as
`Params(; closure_level = ...)`: [`PastClosureLevel`](@ref) (the default) or
[`PresentClosureLevel`](@ref).

A closure that damps the state it is computed from is unstable in the leapfrog at the centred
level once its rate exceeds about `0.57/Δt`, and stable well beyond that from the lagged
level. `δρ` stays at the present level under both.
"""
abstract type AbstractClosureLevel end

"""
$(TYPEDEF)

Compute the closures from the present level, as LADDIE v1 does.

Select via `Params(; closure_level = PresentClosureLevel())`.
"""
struct PresentClosureLevel <: AbstractClosureLevel end

"""
$(TYPEDEF)

Compute `u★`, the thickness in the entrainment rate, and the drag speed from the
lagged, filtered `past` level (the default). First order in time for these terms; with
[`FluxMomentumForm`](@ref) it removes the step-to-step oscillation of thin, strongly entraining
layers.

Select via `Params(; closure_level = PastClosureLevel())` (the default).
"""
struct PastClosureLevel <: AbstractClosureLevel end

"""
    closure_field(v, level)

The level of the prognostic `v` that the closures read under the
[`AbstractClosureLevel`](@ref) `level`: `v.present` or `v.past`.
"""
function closure_field end

closure_field(v, ::PresentClosureLevel) = v.present
closure_field(v, ::PastClosureLevel) = v.past
