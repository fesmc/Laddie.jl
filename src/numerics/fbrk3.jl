# ============================================================================
# Forward–backward RK3 (`FBRK3Integrator`)
# ============================================================================

# One FB-RK3 step (Lilly et al. 2023; Lambert et al. 2026, eqs. 16–24): three stages from
# qⁿ in `past`, each an explicit step with its tendencies at the previous stage, held in
# `present`,
#
#     q^(n+1/3) = qⁿ + Δt/3 F(qⁿ),   q^(n+1/2) = qⁿ + Δt/2 F(q^(n+1/3)),   qⁿ⁺¹ = qⁿ + Δt F(q^(n+1/2)),
#
# with the pressure and Coriolis terms on the thickness blend of `_momentum_thickness!`:
# (a, b) = (β₁, 0), (β₂, 1 − β₂), (β₃, β₃).  `present` and `future` swap between stages.
# Viscosity and diffusion act only in the last stage, from `past` (as v2).  The secondary
# fields are recomputed from each stage, so that a change of dt or of the forcing time
# needs no bootstrap.  Keywords as `_step_fields!`.
function _step_model!(m, s::FBRK3Integrator, dt; kwargs...)
    third = dt / 3
    _fbrk3_stage!(m, third, (s.beta1, zero(s.beta1)), Val(false); kwargs...)
    _swap_stage!(m)
    _fbrk3_stage!(m, dt / 2, (s.beta2, one(s.beta2) - s.beta2), Val(false); kwargs...)
    _swap_stage!(m)
    _fbrk3_stage!(m, dt, (s.beta3, s.beta3), Val(true); kwargs...)
    _end_fbrk3_step!(m, third)
    return
end

# One stage: secondary fields from `present`, then every prognostic from `past` over `span`.
# The preventive bounds on H (`ent2`, `detr_cap`) and dH/dt are taken over the stage span.
function _fbrk3_stage!(m, span, blend, lateral; kwargs...)
    step_secondary_fields!(m, span)
    _step_fields!(m, span, span, blend, lateral; kwargs...)
    return
end

function _swap_stage!(m)
    for var in _prognostics(m)
        var.present, var.future = var.future, var.present
    end
    return
end

# `past .= present` for every prognostic.
function _sync_levels!(m)
    for var in _prognostics(m)
        launch!(_copy_kernel!, var.past, var.present)
    end
    return
end

# After the last stage: qⁿ⁺¹ becomes `present` and `past`, and the secondary fields are
# refreshed on it, for the output and the diagnostics.  The refresh may adjust the state
# (`ResetToAmbientConvection` resets T and S), so `past` is synced again after it: the
# adjustment then holds for the next step, as with the leapfrog's filtered level.
function _end_fbrk3_step!(m, span)
    _swap_stage!(m)
    _sync_levels!(m)
    step_secondary_fields!(m, span)
    _sync_levels!(m)
    return
end

# The levels of a model that has not been stepped yet are identical: refresh the secondary
# fields for the first stage and keep `past` equal to `present`.
function _bootstrap!(m, ::FBRK3Integrator, dt; kwargs...)
    step_secondary_fields!(m, dt / 3)
    _sync_levels!(m)
    return
end

# Nothing to do after a change of dt: each step starts from one level.
_rebootstrap!(_, ::FBRK3Integrator, _; kwargs...) = nothing
