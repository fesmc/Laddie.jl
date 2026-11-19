# Wall slip

Grounding lines (mask value `2`) and land (mask value `1`: exposed bedrock, islands and
the outer border ring) are zero-flux walls. The condition on the tangential momentum
at each wall is one **slip factor** ``s`` on the faces it bounds, with the convention of
NEMO (`rn_shlat`): the velocity behind the wall is taken as a ghost ``(1 - s)\,u``. The
two kinds of wall are set independently, with
`BoundaryConditions(; grounding_line = ..., land = ...)`:

| | free slip, ``s = 0`` | partial slip | no slip, ``s = 2`` |
|---|---|---|---|
| grounding line, [`AbstractSlipAtGL`](@ref) | [`FreeSlipAtGL`](@ref) | [`PartialSlipAtGL`](@ref)`(s)`, default `s = 1` | [`NoSlipAtGL`](@ref) |
| land, [`AbstractSlipAtLand`](@ref) | [`FreeSlipAtLand`](@ref) | [`PartialSlipAtLand`](@ref)`(s)`, default `s = 1` | [`NoSlipAtLand`](@ref) |

## What ``s`` means

Take a ``u``-point next to a wall, with the wall parallel to ``u``. On the C-grid the wall
is the face between two rows of T-cells, half a grid spacing ``\Delta`` from the
``u``-point. The ``u``-point mirrored across the wall, a full ``\Delta`` away, lies behind
the wall and carries no water. The stencils still read a velocity there, the **ghost**
``u_g``, and the wall condition is nothing but the choice of ``u_g``. With the velocity
varying linearly between the two points, the velocity at the wall is their mean,
``u_w = (u + u_g)/2``. Two physical limits fix ``u_g``:

- **No slip**: the water sticks to the wall, ``u_w = 0``. The ghost must mirror the
  velocity with the opposite sign, ``u_g = -u``.
- **Free slip**: the wall exerts no stress, so the velocity has no gradient across it,
  ``u_g = u`` and ``u_w = u``.

Writing the ghost as ``u_g = (1 - s)\,u`` puts free slip at ``s = 0`` and no slip at
``s = 2``, and gives

```math
u_w = \left(1 - \frac{s}{2}\right) u :
```

``s/2`` is the fraction of the velocity the wall takes away. Halfway between the limits,
``s = 1`` makes the ghost zero and keeps half the velocity at the wall: **partial slip**.
It is also what a wall becomes when it is treated as an ordinary cell holding zero
velocity, which is why it is LADDIE v2's wall and the default.

```@raw html
<svg viewBox="0 0 700 262" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Velocity profile next to a wall for slip factors 0, 1 and 2" style="width:100%;max-width:700px;height:auto;font-family:inherit;fill:currentColor">
<defs>
<pattern id="ws-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="currentColor" stroke-width="1" opacity="0.3"/></pattern>
<marker id="ws-blue" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4c8eda"/></marker>
<marker id="ws-orange" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e0803a"/></marker>
<marker id="ws-tick" viewBox="0 0 10 2" refX="5" refY="1" markerWidth="8" markerHeight="2" orient="auto"><rect width="10" height="2" fill="currentColor"/></marker>
</defs>
<g transform="translate(0,0)">
<text x="115" y="22" text-anchor="middle" font-weight="bold">free slip, s = 0</text>
<rect x="10" y="150" width="215" height="80" fill="url(#ws-hatch)"/>
<line x1="10" y1="150" x2="225" y2="150" stroke="currentColor" stroke-width="2"/>
<line x1="110" y1="40" x2="110" y2="230" stroke="currentColor" stroke-dasharray="2 3" opacity="0.5"/>
<line x1="190" y1="85" x2="190" y2="215" stroke="currentColor" stroke-dasharray="6 4"/>
<line x1="110" y1="100" x2="184" y2="100" stroke="#4c8eda" stroke-width="3" marker-end="url(#ws-blue)"/>
<line x1="110" y1="200" x2="184" y2="200" stroke="#e0803a" stroke-width="3" marker-end="url(#ws-orange)"/>
<circle cx="190.0" cy="150" r="4" fill="currentColor"/>
<text x="198.0" y="143" font-size="12">u<tspan baseline-shift="sub" font-size="9">w</tspan> = u</text>
<text x="115" y="250" text-anchor="middle" font-size="12">ghost u<tspan baseline-shift="sub" font-size="9">g</tspan> = (1 − s) u = u</text>
<line x1="66" y1="100" x2="66" y2="150" stroke="currentColor" opacity="0.6" marker-start="url(#ws-tick)" marker-end="url(#ws-tick)"/><text x="72" y="129" font-size="11">Δ/2</text><line x1="66" y1="150" x2="66" y2="200" stroke="currentColor" opacity="0.6" marker-start="url(#ws-tick)" marker-end="url(#ws-tick)"/><text x="72" y="179" font-size="11">Δ/2</text><text x="14" y="104" font-size="12">u</text>
<text x="14" y="204" font-size="12">u<tspan baseline-shift="sub" font-size="9">g</tspan></text>
<text x="14" y="168" font-size="11" font-style="italic">wall</text>
<text x="14" y="60" font-size="11" font-style="italic">fluid</text>
</g>
<g transform="translate(235,0)">
<text x="115" y="22" text-anchor="middle" font-weight="bold">partial slip, s = 1</text>
<rect x="10" y="150" width="215" height="80" fill="url(#ws-hatch)"/>
<line x1="10" y1="150" x2="225" y2="150" stroke="currentColor" stroke-width="2"/>
<line x1="110" y1="40" x2="110" y2="230" stroke="currentColor" stroke-dasharray="2 3" opacity="0.5"/>
<line x1="202" y1="85" x2="98" y2="215" stroke="currentColor" stroke-dasharray="6 4"/>
<line x1="110" y1="100" x2="184" y2="100" stroke="#4c8eda" stroke-width="3" marker-end="url(#ws-blue)"/>
<circle cx="110" cy="200" r="4.5" fill="#e0803a"/>
<circle cx="150.0" cy="150" r="4" fill="currentColor"/>
<text x="142.0" y="143" font-size="12" text-anchor="end">u<tspan baseline-shift="sub" font-size="9">w</tspan> = u/2</text>
<text x="115" y="250" text-anchor="middle" font-size="12">ghost u<tspan baseline-shift="sub" font-size="9">g</tspan> = (1 − s) u = 0</text>
</g>
<g transform="translate(470,0)">
<text x="115" y="22" text-anchor="middle" font-weight="bold">no slip, s = 2</text>
<rect x="10" y="150" width="215" height="80" fill="url(#ws-hatch)"/>
<line x1="10" y1="150" x2="225" y2="150" stroke="currentColor" stroke-width="2"/>
<line x1="110" y1="40" x2="110" y2="230" stroke="currentColor" stroke-dasharray="2 3" opacity="0.5"/>
<line x1="214" y1="85" x2="6" y2="215" stroke="currentColor" stroke-dasharray="6 4"/>
<line x1="110" y1="100" x2="184" y2="100" stroke="#4c8eda" stroke-width="3" marker-end="url(#ws-blue)"/>
<line x1="110" y1="200" x2="36" y2="200" stroke="#e0803a" stroke-width="3" marker-end="url(#ws-orange)"/>
<circle cx="110.0" cy="150" r="4" fill="currentColor"/>
<text x="102.0" y="143" font-size="12" text-anchor="end">u<tspan baseline-shift="sub" font-size="9">w</tspan> = 0</text>
<text x="115" y="250" text-anchor="middle" font-size="12">ghost u<tspan baseline-shift="sub" font-size="9">g</tspan> = (1 − s) u = −u</text>
</g>
</svg>
```

*The tangential velocity ``u`` next to the wall (blue), its ghost ``u_g`` behind the wall
(orange) and the linear profile through them (dashed). The dot marks the velocity ``u_w``
on the wall face.*

The viscous flux through the wall face follows from the same ghost:

```math
A\,H\,\frac{u_g - u}{\Delta^2} = -s\,A\,H\,\frac{u}{\Delta^2},
```

so the wall stress grows linearly with ``s``. Equivalently, the profile reaches zero at
``\Delta/s`` from the ``u``-point: never for free slip, at the ghost point for ``s = 1``
and on the wall face for no slip.

The kernels form it from the velocity stored behind the wall, which is zero, so the
plain stencil already holds ``s = 1``; they add the remaining ``s - 1``. Under
[`NonlinearLateralViscosity`](@ref) the ghost enters the shear-scaled coefficient as
well, so its wall stress grows as ``s^2``.

The same ghost forms the momentum carried across a wall face by the advection, unless
[wall momentum advection](wall_advection.md) says otherwise.

At a coastline corner whose stencil touches both grounded ice and exposed rock, the
grounding-line condition takes precedence, so each wall face gets exactly one slip
factor.

## LADDIE v1 and v2

LADDIE v2 treats grounded and ice-free land neighbours as ordinary cells with zero
velocity, one grid spacing away (`laddie_velocity.f90:259-267`), and carries no momentum
across them. This is the default, ``s = 1`` with [`NoWallAdvection`](@ref). v2 calls it
"no slip" ([Lambert et al., 2026](@cite lambert_one-layer_2026)); in this convention it
is partial slip.

LADDIE v1 describes its walls as NEMO-like partial slip with a factor between 0 (free
slip) and 2 (no slip), and its published runs use 1 (Lambert et al., 2023). Its code,
however, adds the factor to a viscous stencil that already holds the stored zero, so
its viscous wall is one step stronger than labelled: ``s = 1`` is no slip there. In the
advection the factor is used as labelled. Reproducing v1 therefore takes
[`V1WallAdvection`](@ref):

```julia
# LADDIE v1 (slip = 1): no slip in the viscosity, partial slip in the advection
BoundaryConditions(; grounding_line = NoSlipAtGL(), land = NoSlipAtLand(),
                   wall_advection = V1WallAdvection())

# LADDIE v2: the defaults
BoundaryConditions()
```

## API

```@docs
AbstractSlipAtGL
NoSlipAtGL
FreeSlipAtGL
PartialSlipAtGL
AbstractSlipAtLand
NoSlipAtLand
FreeSlipAtLand
PartialSlipAtLand
```
