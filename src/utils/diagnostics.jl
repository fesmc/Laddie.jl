# ============================================================================
# Diagnostics: melt and speed statistics, the log line
# ============================================================================

"""
$(TYPEDSIGNATURES)

Return melt-rate and velocity statistics for the current state of a model or
simulation, as a named tuple.  All reductions execute on the device (GPU-safe);
results are returned as CPU scalars.

- `max_meltrate`  — maximum basal melt rate, m yr⁻¹
- `mean_meltrate` — area-mean basal melt rate over ice-covered cells, m yr⁻¹
- `max_speed`     — maximum depth-averaged current speed |u|, m s⁻¹
- `total_melt`    — area-integrated basal melt (mass flux), Gt yr⁻¹

Melt rates are freshwater equivalent, the unit the model integrates; multiply by
`rho_freshwater / rho_ice` for ice equivalent.  `total_melt` is the corresponding
mass flux, `Σ melt · dx · dy · rho_freshwater`, and is the same in either unit.

The fields keep their order, so `max_melt, mean_melt, max_speed = meltstats(sim)`
still destructures the first three.

The melt statistics average over `imask` (ice-covered cells), not `tmask`.  Under
[`ConnectedGapsBC`](@ref) the two differ: gap cells are dynamically active but ice-free
and melt exactly zero, so averaging over `tmask` would dilute the mean by the gap area.
This diverges from LADDIE v2, whose `domain_a` integration includes gaps; with no gaps
in the mask the two masks coincide and the statistics are unchanged.  `max_speed` stays
on `tmask` — the layer really does flow through gaps, and that flow is the point.
"""
meltstats(sim::Simulation) = meltstats(sim.model)

const _KG_PER_GT = 1e12     # kilogrammms per gigatonne

function meltstats(m::Model)
    imask = m.imask
    d = m.diag
    n = sum(imask)
    @. d = m.melt * m.seconds_per_year * imask
    max_meltrate = maximum(d)
    melt_sum = sum(d)                             # m yr⁻¹ summed over ice cells
    mean_meltrate = melt_sum / n
    total_melt = melt_sum * m.dx * m.dy * m.rho_freshwater / _KG_PER_GT
    _launch_tpoint_diag!(_speed_norm_kernel!, m)
    max_speed = maximum(d)
    return (; max_meltrate, mean_meltrate, max_speed, total_melt)
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
    d_Mmax = stats.max_meltrate
    d_Mav = stats.mean_meltrate
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
