"""
Abstract supertype for the time integration of the prognostic fields, the scheme that
`time_step!` and `integrate!` advance a [`Model`](@ref) with. Pass a concrete instance as
`Params(; integrator = ...)`: [`LeapfrogIntegrator`](@ref) (the default) or
[`FBRK3Integrator`](@ref). See [Time stepping](@ref).
"""
abstract type AbstractIntegrator end

"""
$(TYPEDEF)

Leapfrog with a Robert–Asselin filter, as LADDIE v1:

```
qⁿ⁺¹ = qⁿ⁻¹ + 2Δt F(qⁿ, qⁿ⁻¹),    qⁿ ← qⁿ + ν/2 (qⁿ⁻¹ + qⁿ⁺¹ − 2qⁿ)
```

The filter damps the computational (2Δt) mode. The first step, and the first after a change
of Δt, is a forward step over Δt.

Select via `Params(; integrator = LeapfrogIntegrator(; nu = 0.8))` (the default).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LeapfrogIntegrator{FT} <: AbstractIntegrator
    "Robert–Asselin filter coefficient ν (–, default `0.8`)"
    nu::FT = 0.8
end

"""
$(TYPEDEF)

Forward–backward Runge–Kutta in three stages ([Lilly et al. 2023](@cite lilly_cfl_2023)), as
LADDIE v2 ([Lambert et al. 2026](@cite lambert_one-layer_2026), eqs. 16–24):

```
Hⁿ⁺¹ᐟ³ = Hⁿ + Δt/3 Ψ(Hⁿ, Uⁿ),             Uⁿ⁺¹ᐟ³ = Uⁿ + Δt/3 Φ(H*, Uⁿ),       H* = β₁Hⁿ⁺¹ᐟ³ + (1−β₁)Hⁿ
Hⁿ⁺¹ᐟ² = Hⁿ + Δt/2 Ψ(Hⁿ⁺¹ᐟ³, Uⁿ⁺¹ᐟ³),     Uⁿ⁺¹ᐟ² = Uⁿ + Δt/2 Φ(H**, Uⁿ⁺¹ᐟ³),   H** = β₂Hⁿ⁺¹ᐟ² + (1−β₂)Hⁿ
Hⁿ⁺¹ = Hⁿ + Δt Ψ(Hⁿ⁺¹ᐟ², Uⁿ⁺¹ᐟ²),         Uⁿ⁺¹ = Uⁿ + Δt Φ(H***, Uⁿ⁺¹ᐟ²),     H*** = β₃Hⁿ⁺¹ + (1−2β₃)Hⁿ⁺¹ᐟ² + β₃Hⁿ
```

T and S are stepped as U. Lateral viscosity and diffusion act only in the last stage, from
level n. No computational mode, no filter, and no bootstrap step. Needs
`closure_level = PresentClosureLevel()` and the flux forms. Defaults: v2's configurations.

Select via `Params(; integrator = FBRK3Integrator(), closure_level = PresentClosureLevel())`.

# Fields
$(TYPEDFIELDS)
"""
struct FBRK3Integrator{FT} <: AbstractIntegrator
    "weight of Hⁿ⁺¹ᐟ³ in the thickness of the first momentum stage (–, default `0.5`)"
    beta1::FT
    "weight of Hⁿ⁺¹ᐟ² in the thickness of the second momentum stage (–, default `0.5`)"
    beta2::FT
    "weight of Hⁿ⁺¹ and of Hⁿ in the thickness of the last momentum stage (–, default `0.344`)"
    beta3::FT
end

function FBRK3Integrator(; beta1 = 0.5, beta2 = 0.5, beta3 = 0.344)
    for (name, β) in ((:beta1, beta1), (:beta2, beta2), (:beta3, beta3))
        0 <= β <= 1 || throw(ArgumentError("FBRK3Integrator: $name must lie in [0, 1], got $β"))
    end
    return FBRK3Integrator(promote(beta1, beta2, beta3)...)
end

"""
    default_cfl_target(integrator)

The CFL setpoint of an [`AdaptiveTimeStepper`](@ref) built without `cfl_target`, for the
[`AbstractIntegrator`](@ref) `integrator`: `0.3` for [`LeapfrogIntegrator`](@ref), `0.4` for
[`FBRK3Integrator`](@ref).
"""
function default_cfl_target end

default_cfl_target(::LeapfrogIntegrator) = 0.3
default_cfl_target(::FBRK3Integrator) = 0.4

# The settings an integrator needs from the rest of `Params`, checked by `Params(; ...)`.
_check_integrator(::LeapfrogIntegrator, _) = nothing
function _check_integrator(::FBRK3Integrator, p)
    p.closure_level isa PresentClosureLevel || throw(ArgumentError(
        "FBRK3Integrator needs closure_level = PresentClosureLevel(): its stages take the " *
        "closures at the stage level"))
    p.tracer_form isa FluxTracerForm || throw(ArgumentError(
        "FBRK3Integrator needs tracer_form = FluxTracerForm()"))
    p.momentum_form isa FluxMomentumForm || throw(ArgumentError(
        "FBRK3Integrator needs momentum_form = FluxMomentumForm()"))
    return nothing
end
