
abstract type AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

Grounding-line momentum boundary condition of LADDIE v1.x (the default):
grounding-line walls use the same slip factor as land walls, i.e.
`Params.slip` (`1.0` = free slip).

Select via `Params(; glbc = FreeSlipGL())` (the default).
"""
struct FreeSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

No-slip momentum boundary condition at the grounding line: the tangential
velocity is forced to zero at walls bordering grounded ice (mask value `2`),
while land/border walls (mask value `1`) keep the global `Params.slip`
factor.  Implemented as a slip factor of `2` on grounding-line faces
(ghost velocity = −interior velocity).

Motivated by LADDIE v2.0 (Lambert et al., in review, 2026), where a no-slip
grounding-line condition improves melt patterns near the grounding line
compared to observations.  Not yet validated against LADDIE v2.0 output.

Select via `Params(; glbc = NoSlipGL())`.
"""
struct NoSlipGL <: AbstractGroundingLineBC end

# Slip factor applied at grounding-line faces (land faces always use `slip`).
# Ghost tangential velocity is (1 − factor)·u: 1 → free slip, 2 → no slip.
_gl_slip(::FreeSlipGL, slip) = slip
_gl_slip(::NoSlipGL, slip) = oftype(slip, 2)
