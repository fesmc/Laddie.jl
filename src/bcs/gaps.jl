#############################
# Gaps BC
#############################

"""
Abstract supertype for the treatment of ice-shelf gaps — ice-free cells inside
the shelf domain, marked `4` in the mask.  Pass a concrete instance as
`BoundaryConditions(; gaps = ...)`: [`SinkGapsBC`](@ref) (the default, gaps act as sinks),
[`ConnectedGapsBC`](@ref) (gaps stay dynamically connected), or [`LeakyGapsBC`](@ref)
(connected and partly covered by mélange, with an open-water surface that loses heat
and lets the layer convect).
"""
abstract type AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps — ice-free cells inside the ice-shelf domain, marked `4` in
the domain mask — as sinks of the meltwater layer (the default, and the behaviour
of LADDIE v1.x).  Gap cells are demoted to open ocean (`0`) at model construction,
so a meltwater current reaching a gap loses its heat and momentum to the ambient
ocean exactly as it would at the calving front, and cannot drive melting further
downstream.

Select via `BoundaryConditions(; gaps = SinkGapsBC())` (the default).

See also [`ConnectedGapsBC`](@ref).
"""
struct SinkGapsBC <: AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps as dynamically connected: the meltwater layer keeps being
integrated across ice-free cells inside the ice-shelf domain, so heat and momentum
are advected downstream and can drive melting on the far side of the gap.  No
melting occurs within a gap itself — there is no glacial ice to melt — so no
meltwater volume or buoyancy is added there.

Gap cells are those marked `4` in the domain mask.  Mark them either directly (a
coupled ice-sheet driver knows its own reference geometry) or from a reference ice
footprint with [`MarkGapsPreprocess`](@ref) in the `preprocess` list of [`Grid`](@ref).  This
boundary condition only decides what happens in the cells so marked.

Motivated by [Jesse et al. (2026, https://doi.org/10.5194/egusphere-2026-4237)](@cite jesse_ice-sheet_2026),
who show that the choice between this treatment and [`SinkGapsBC`](@ref) can alter
coupled ice-volume loss by an amount comparable to 1 °C of ocean warming.  The two
are deliberate end-members; reality lies somewhere between them.

# Example

```julia
boundary = BoundaryConditions(; gaps = ConnectedGapsBC())
Model(Grid(mask, z_draft, dx, dy); forcing, boundary)   # gaps already marked as 4
grid = Grid(mask, z_draft, dx, dy; preprocess = [MarkGapsPreprocess(refgeo_H)])
Model(grid; forcing, boundary)                          # gaps from a footprint
```
"""
struct ConnectedGapsBC <: AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps as connected and partly covered by mélange and sea ice, with
cover fraction α ∈ [0, 1] (`melange_cover`, default `0.9`).  The covered part acts on
the layer as the ice shelf does; the open part exchanges heat with the atmosphere, and
the layer convects where it loses its buoyancy.

The gaps stay dynamically active as under [`ConnectedGapsBC`](@ref): the layer's
thickness is set by transport, entrainment, the sea ice the open water forms and
convection, never by a prescribed relaxation.  Melting is zero in gaps, as under
`ConnectedGapsBC`.

**Drag.**  The ice-ocean drag `Params.C_d` is scaled by α in gap cells (by the mean
cover of the two cells of a velocity face).  `ConnectedGapsBC` applies the full drag in
gaps, as if they were covered; open water (α = 0) applies none.

**Surface heat loss.**  The open fraction loses `(1 − α) Q` to the atmosphere, for
`Q = surface_heat_loss` (W m⁻², default `0`).  It first cools the layer, by
`(1 − α) Q dt/(ρ₀ c_p H)`.  Once the layer reaches the surface freezing point, the rest
forms sea ice of salinity `ice_salinity` (default 5 psu): the frozen water leaves the
layer, its salt beyond `ice_salinity` stays (brine rejection).

**Convection.**  Where the layer is denser than the water beneath it (δρ < 0), its T
and S relax towards that water's on `convection_time` (s, default `10⁴`), as under
[`RelaxToAmbientConvection`](@ref), and its thickness relaxes towards `H_min` at the same rate:
the convected water leaves the layer.  This acts in every gap cell, whatever the cover
and whatever the model's `convection`, which leaves gap cells alone.

The two ends:

- `α = 1`: full drag, no surface exchange.  This is `ConnectedGapsBC`, plus convection
  in gaps: bit-identical to it wherever no gap cell turns unstable.
- `α = 0`: open water, no drag, the full heat loss `Q`.  Whether this comes close to
  [`SinkGapsBC`](@ref) depends on `Q`: the sink is not built in.

Neither α nor `Q` is constrained by observations in this setting; `Q` can be compared
with estimates of the heat loss of coastal polynyas (a few hundred W m⁻² in winter).

# Example

```jldoctest
julia> BoundaryConditions(; gaps = LeakyGapsBC()).gaps       # α = 0.9, Q = 0
LeakyGapsBC{Float64}(0.9, 0.0, 5.0, 10000.0)

julia> LeakyGapsBC(; melange_cover = 0.3, surface_heat_loss = 300)
LeakyGapsBC{Float64}(0.3, 300.0, 5.0, 10000.0)
```

# Fields
$(TYPEDFIELDS)
"""
struct LeakyGapsBC{FT} <: AbstractGapsBC
    "mélange and sea-ice cover of the gaps, α ∈ [0, 1]"
    melange_cover::FT
    "heat loss of open water to the atmosphere (W m⁻², ≥ 0); the gap loses (1 − α) of it"
    surface_heat_loss::FT
    "salinity of the sea ice the heat loss forms (psu), which keeps this much of the salt"
    ice_salinity::FT
    "time scale of convection in gap cells (s, > 0)"
    convection_time::FT
    function LeakyGapsBC(
        melange_cover::Real, surface_heat_loss::Real = 0.0, ice_salinity::Real = 5.0,
        convection_time::Real = 1e4,
    )
        0 <= melange_cover <= 1 || throw(
            ArgumentError("LeakyGapsBC melange_cover must lie in [0, 1], got $melange_cover"),
        )
        surface_heat_loss >= 0 || throw(
            ArgumentError("LeakyGapsBC surface_heat_loss must be non-negative, got $surface_heat_loss"),
        )
        ice_salinity >= 0 || throw(
            ArgumentError("LeakyGapsBC ice_salinity must be non-negative, got $ice_salinity"),
        )
        convection_time > 0 || throw(
            ArgumentError("LeakyGapsBC convection_time must be positive, got $convection_time"),
        )
        f = promote(float(melange_cover), float(surface_heat_loss), float(ice_salinity), float(convection_time))
        return new{eltype(f)}(f...)
    end
end
LeakyGapsBC(; melange_cover = 0.9, surface_heat_loss = 0.0, ice_salinity = 5.0, convection_time = 1e4) =
    LeakyGapsBC(melange_cover, surface_heat_loss, ice_salinity, convection_time)

# Resolve mask value 4 (gap) into the classification the selected treatment implies.
# Runs at model construction, before the Geometry is derived: SinkGapsBC demotes gaps
# to open ocean, so every mask derivation sees exactly the v1.x geometry;
# ConnectedGapsBC keeps them.  Marking gaps in the first place is
# `MarkGapsPreprocess`'s job.
_apply_gaps_bc(mask, ::SinkGapsBC) = ifelse.(mask .== 4, 0, mask)
_apply_gaps_bc(mask, ::ConnectedGapsBC) = mask
_apply_gaps_bc(mask, ::LeakyGapsBC) = mask

# The per-cell fields of `LeakyGapsBC`, fixed at build: the drag factor (α in gaps, 1
# elsewhere) and the open-water heat loss ((1 − α) Q in gaps, 0 elsewhere).
_init_gaps!(cache, ::AbstractGapsBC, mask) = nothing
function _init_gaps!(cache, bc::LeakyGapsBC, mask)
    α, gap = bc.melange_cover, mask .== 4
    cache.gap_drag .= ifelse.(gap, α, 1)
    cache.gap_heat .= ifelse.(gap, (1 - α) * bc.surface_heat_loss, 0)
    return nothing
end

# `LeakyGapsBC`: the open-water surface of a gap, operator-split after the T and S
# steps.  `gap_heat` holds the heat loss of each gap cell, (1 − α) Q for mélange
# cover α.  It first cools the layer, by Q dt/(ρ₀ c_p H).  Once the layer reaches the
# surface freezing point, the rest forms sea ice, with the latent heat holding T at
# the freezing point.  The ice takes the water thickness h = c_p δT H / L out of the
# layer (δT the part of the cooling below freezing) and keeps `ice_salinity`; the rest
# of its salt stays in the thinner layer, as for frazil:
#
#     H' = H − h,   S' = (S H − S_ice h) / H'.
#
# Then convection: where the layer is denser than the water beneath it (δρ < 0), T
# and S relax towards that water (Ta, Sa) on `convection_time`, as under
# `RelaxToAmbientConvection`, and H relaxes towards H_min at the same rate: the convected water
# leaves the layer.  Every gap cell convects, whatever its cover.
# Without leaky gaps `gap_heat` is a scalar and this is skipped.
_gap_surface!(m, ::Number, dt) = nothing
function _gap_surface!(m, ::AbstractArray, dt)
    gaps = m.boundary.gaps
    launch!(
        _gap_surface_kernel!,
        m.T.future,
        m.S.future,
        m.H.future,
        m.Ta,
        m.Sa,
        m.z_draft,
        m.tmask,
        m.imask,
        m.gap_heat,
        m.H_min,
        m.rho0_seawater,
        m.c_p,
        m.L,
        m.l1,
        m.l2,
        m.l3,
        gaps.ice_salinity,
        gaps.convection_time,
        _eos_coefficients(m),
        dt,
    )
    return
end

@kernel function _gap_surface_kernel!(
    T, S, H, @Const(Ta), @Const(Sa), @Const(z_draft), @Const(tmask), @Const(imask), @Const(Q),
    H_min, rho0, c_p, L, l1, l2, l3, S_ice, tau, eos, dt,
)
    rho0, c_p, L, dt, H_min, tau = _val(rho0), _val(c_p), _val(L), _val(dt), _val(H_min), _val(tau)
    l1, l2, l3, S_ice, eos = _val(l1), _val(l2), _val(l3), _val(S_ice), _val(eos)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(T)
        Ti, Si, Hi, z = T[i, j], S[i, j], H[i, j], z_draft[i, j]
        gap = (tmask[i, j] > 0) & iszero(imask[i, j])
        # Cooling over the step (K), zero where Q is.  `H > 0` in active cells; the
        # max only guards inactive ones (H = 0, Q = 0).
        cool = Q[i, j] * dt / (rho0 * c_p * max(Hi, eps(FT)))
        Tf = l1 * Si + l2 + l3 * z
        # The part of the cooling that would take T below freezing forms ice instead;
        # all of it if the layer is already at or below freezing.
        freeze = min(cool, max(Tf - Ti + cool, zero(FT)))
        T1 = Ti - cool + freeze
        # Frozen water thickness, never more than takes the layer to H_min.
        h = min(c_p * freeze * Hi / L, max(Hi - H_min, zero(FT)))
        H1 = Hi - h
        # (S H − S_ice h)/H', written so that h = 0 leaves S bit-identical.
        S1 = Si + (Si - S_ice) * h / max(H1, eps(FT))
        # Convection, backward Euler on `tau`.
        unstable = gap & (_drho(Sa[i, j], S1, Ta[i, j], T1, _layer_depth(z, H1), eos) < 0)
        k = ifelse(unstable, dt / tau, zero(FT))
        T[i, j] = ifelse(unstable, (T1 + k * Ta[i, j]) / (1 + k), T1)
        S[i, j] = ifelse(unstable, (S1 + k * Sa[i, j]) / (1 + k), S1)
        H[i, j] = ifelse(unstable, (H1 + k * H_min) / (1 + k), H1)
    end
end
