#!/bin/bash
# Full sweep of bench.jl: KA and Reactant, CPU and GPU, Float64 and Float32.
# Appends to results.csv next to this script (~1 h, mostly Reactant compiles).
# Keep the machine otherwise idle while it runs.
#
#   benchmark/reactant/run_sweep.sh            # all targets
#   TARGETS="r-gpu ka-cuda" benchmark/reactant/run_sweep.sh
#
# CPU runs use cores 0-7; adjust CORES for another machine (physical cores only:
# SMT siblings slow the stencils down, see the thread-scaling notes).
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$here/results.csv"
TARGETS="${TARGETS:-ka-cpu r-cpu ka-cuda r-gpu}"
CORES="${CORES:-0-7}"
NCORES=$(( $(echo "$CORES" | cut -d- -f2) - $(echo "$CORES" | cut -d- -f1) + 1 ))
[ -f "$out" ] || echo "target,FT,nx,ny,ms_per_step,ms_per_step_hostloop,compile_s,maxrelerr" > "$out"
cd "$root"
for FT in Float64 Float32; do
  for T in $TARGETS; do
    case $T in
      ka-cpu) cmd=(julia -t "$NCORES" --project=benchmark) ;;
      r-cpu)  cmd=(taskset -c "$CORES" julia --project=benchmark) ;;
      *)      cmd=(julia --project=benchmark) ;;
    esac
    echo "== $T $FT" >&2
    TARGET=$T FT=$FT "${cmd[@]}" benchmark/reactant/bench.jl 2>>"$here/bench.err" \
      | grep -v "^target" >> "$out"
  done
done
