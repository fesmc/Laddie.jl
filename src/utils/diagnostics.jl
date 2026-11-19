# ============================================================================
# Diagnostics: melt and speed statistics, the log line
# ============================================================================

"""
$(TYPEDSIGNATURES)

Return melt-rate and velocity statistics for the current state of a model or
simulation, as a named tuple.  All reductions execute on the model's device (GPU-safe),
except for a simulation on the Reactant backend, which is reduced on a CPU copy;
results are returned as CPU scalars.

- `max_melt`   — maximum net basal melt rate over ice-covered cells, m yr⁻¹
- `mean_melt`  — area-mean net basal melt rate over ice-covered cells, m yr⁻¹
- `max_speed`  — maximum depth-averaged current speed |u|, m s⁻¹
- `total_melt` — area-integrated net basal melt (mass flux), Gt yr⁻¹
- `refreezing` — area-integrated refreezing (mass flux), Gt yr⁻¹

The net basal melt is the interface melt rate `melt` (negative where the base
refreezes) minus the frazil-ice formation `frazil`, the basal mass balance of the ice.
`refreezing` is the ice formed, `Σ (max(−melt, 0) + frazil)`, so `total_melt +
refreezing` is the melting alone.  Rates are freshwater equivalent, the unit the model
integrates; multiply by `rho_freshwater / rho_ice` for ice equivalent.  The mass fluxes
are `Σ rate · dx · dy · rho_freshwater`, the same in either unit.

The melt statistics average over `imask` (ice-covered cells), not `tmask`.  Under
[`ConnectedGapsBC`](@ref) the two differ: gap cells are dynamically active but ice-free
and melt exactly zero, so averaging over `tmask` would dilute the mean by the gap area.
This diverges from LADDIE v2, whose `domain_a` integration includes gaps; with no gaps
in the mask the two masks coincide and the statistics are unchanged.  `max_speed` stays
on `tmask` — the layer really does flow through gaps, and that flow is the point.
"""
# On the model the log line reads (`_diag_model`): under Reactant a CPU copy refreshed from
# the device, since eager reductions on device arrays compile one program per call.
meltstats(sim::Simulation) = meltstats(_diag_model(sim.exec, sim))

const _KG_PER_GT = 1e12     # kilogrammms per gigatonne

function meltstats(m::Model)
    imask = m.imask
    d = m.diag
    n = sum(imask)
    to_gigatonnes = m.dx * m.dy * m.rho_freshwater / _KG_PER_GT
    # The net melt, m yr⁻¹; the maximum over ice cells only, so that it is negative
    # where every ice cell refreezes.
    @. d = ifelse(imask > 0, (m.melt - m.frazil) * m.seconds_per_year, -Inf)
    max_melt = maximum(d)
    @. d = (m.melt - m.frazil) * m.seconds_per_year * imask
    net_sum = sum(d)                              # m yr⁻¹ summed over ice cells
    mean_melt = net_sum / n
    total_melt = net_sum * to_gigatonnes
    @. d = (max(-m.melt, 0) + m.frazil) * m.seconds_per_year * imask
    refreezing = sum(d) * to_gigatonnes
    _launch_tpoint_diag!(_speed_norm_kernel!, m)
    max_speed = maximum(d)
    return (; max_melt, mean_melt, max_speed, total_melt, refreezing)
end

# Per-cell diagnostics on T-points, written into `m.diag`. Velocities are
# averaged onto the T-point as `im_half(U)`, `jm_half(V)`. The kernel shares its
# argument list with `_cfl_rate_kernel!` (see `_launch_tpoint_diag!`), so it ignores
# some of it.
@kernel function _speed_norm_kernel!(
    out,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(H),
    @Const(tmask),
    g,
    dx,
    dy,
    Nx,
    Ny,
)
    g, dx, dy = _val(g), _val(dx), _val(dy)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u = (U[i, j] + U[_xm1(i, Nx), j]) / 2
        v = (V[i, j] + V[i, _ym1(j, Ny)]) / 2
        out[i, j] = _safe_sqrt(u^2 + v^2) * tmask[i, j]
    end
end

function _launch_tpoint_diag!(kernel!, m)
    nx, ny = size(m.diag)
    launch!(
        kernel!,
        m.diag,
        m.U.present,
        m.V.present,
        m.drho,
        m.H.present,
        m.tmask,
        m.g,
        m.dx,
        m.dy,
        nx,
        ny,
    )
    return nothing
end

"""
$(TYPEDSIGNATURES)

Write a one-line diagnostic to the log file at every `output.diagday`-day interval.
"""
function printdiags(sim)
    _event_due(sim, sim.io.nextdiag) || return
    m = _diag_model(sim.exec, sim)
    sim.io.nextdiag += sim.output.diagday * _primal(m.seconds_per_day)
    t_days = _t_days(sim)

    # Device reductions over the scratch field `diag`, so no field is copied to the host.
    d = m.diag
    active_sum(x) = (@. d = x * m.tmask; sum(d))
    dxdy = m.dx * m.dy
    area = sum(m.tmask) * dxdy

    d_Hav = active_sum(m.H.present) * dxdy / area
    @. d = ifelse(m.tmask > 0, m.H.present, Inf)
    d_Hmin = minimum(d)
    @. d = ifelse(m.tmask > 0, m.H.present, -Inf)
    d_Hmax = maximum(d)

    # Melt statistics over ice-covered cells, and the largest T-point speed, as
    # reported by `meltstats`.
    stats = meltstats(m)
    d_Mmax = stats.max_melt
    d_Mav = stats.mean_melt
    d_Mtot = stats.total_melt
    d_Vmax = stats.max_speed

    @. d = (m.melt + m.entr + m.ent2 - m.detr - m.detr_cap - m.frazil) * m.tmask
    total = sum(d) * dxdy
    d_MWF = total > 0 ? 100.0 * active_sum(m.melt) * dxdy / total : 0.0

    d_Etot = 1e-6 * active_sum(m.entr) * dxdy
    d_E2tot = 1e-6 * active_sum(m.ent2) * dxdy
    d_DEtot = 1e-6 * active_sum(m.detr) * dxdy
    d_PSI = -1e-6 * active_sum(m.convH) * dxdy

    @. d = ifelse(m.tmask > 0, m.drho, 100.0)
    d_drho = 1000.0 * minimum(d)
    @. d = m.convection * (m.tmask > 0)
    d_conv = sum(d)

    line = @sprintf(
        "%8.3f days || %5.1f [%4.2f %4.0f] m || %5.2f | %3.0f m/yr | %7.2f Gt/yr || %5.2f %% || %5.3f + %5.3f - %5.3f | %5.3f Sv || %3.2f m/s || %5.5f %3.0f []",
        t_days,
        d_Hav,
        d_Hmin,
        d_Hmax,
        d_Mav,
        d_Mmax,
        d_Mtot,
        d_MWF,
        d_Etot,
        d_E2tot,
        d_DEtot,
        d_PSI,
        d_Vmax,
        d_drho,
        d_conv
    )
    _print2log(sim, line)
    return nothing
end
