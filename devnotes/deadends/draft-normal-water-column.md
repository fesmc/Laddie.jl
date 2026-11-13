# Water column measured normal to the draft

- **Type:** dead end
- **Status:** not implemented
- **Commits:** none

## Idea

Measure the water column that sets the layer cap normal to the sloping ice draft instead of vertically, to give thin, sloping cells more room.

## Why it is a dead end

- The normal distance is *shorter* (factor cos θ), so it would lower the cap rather than raise it.
- At 1 km, with draft slopes of a few degrees, it changes the cap by < 0.2 %.
- The real problem was the *mean* water column of a 1 km cell, which hides channels.
  The subgrid water column fixes that (`improvements/flux-form-tracers.md`): at the Larsen C problem cells it is 13–29 m instead of 1.4–2.4 m.

## What would reopen it

Nothing foreseen. With no relative cap in reference runs (`relative-thickness-cap.md`), the water column no longer enters the cap at all.
