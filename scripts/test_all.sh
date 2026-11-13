#!/usr/bin/env bash
# One command to test everything. Prints PASS / FAIL per check and a summary.
#
#   bash scripts/test_all.sh            # standard run, ~8 min on a laptop
#   bash scripts/test_all.sh --quick    # 3 of the 8 synthetic data situations, no real areas, ~4 min
#   bash scripts/test_all.sh --full     # also the slow automated tests + the 100k-unit register benchmark
#   bash scripts/test_all.sh --network  # also download a fresh real Indian scene (needs internet)
#
# Logs and outputs: test_runs/<date-time>/
set -uo pipefail
cd "$(dirname "$0")/.."
MODE_QUICK=0; MODE_FULL=0; MODE_NET=0
for a in "$@"; do case $a in --quick) MODE_QUICK=1;; --full) MODE_FULL=1;; --network) MODE_NET=1;; esac; done
RUN="test_runs/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$RUN"
PASS=0; FAIL=0; RESULTS=()
step() {  # step "name" command...
  local name="$1"; shift
  printf "\n\033[1m== %s\033[0m\n" "$name"
  local log="$RUN/$(echo "$name" | tr ' /' '__' | tr -cd '[:alnum:]_-').log"
  local t0=$(date +%s)
  if "$@" >"$log" 2>&1; then
    PASS=$((PASS+1)); RESULTS+=("PASS  $name ($(( $(date +%s)-t0 )) s)"); tail -3 "$log"; printf "\033[32mPASS\033[0m\n"
  else
    FAIL=$((FAIL+1)); RESULTS+=("FAIL  $name   -> see $log"); tail -15 "$log"; printf "\033[31mFAIL\033[0m (log: %s)\n" "$log"
  fi
}
PY=${PY:-python}

step "1 environment" $PY -c "
import sys, ulpin3d, lightgbm, torch, rasterio, geopandas, laspy, ezdxf, fitz, onnxruntime
import numpy as np, lightgbm as lgb
X=np.random.rand(500,5); y=(X[:,0]>.5).astype(int)
lgb.train({'objective':'binary','verbose':-1}, lgb.Dataset(X,y), 5)   # LightGBM + torch in one process (macOS OpenMP check)
print('python', sys.version.split()[0], '| torch', torch.__version__, '| lightgbm', lgb.__version__, '| rasterio', rasterio.__version__)"

step "2 automated tests (fast)" $PY -m pytest -q tests -m "not slow" -p no:warnings
if [ $MODE_FULL = 1 ]; then step "2b automated tests (slow)" $PY -m pytest -q tests -m slow -p no:warnings; fi

step "3 3D ULPIN verifier: valid id" $PY -m ulpin3d.cli verify-id 27PUNEW10101H4S024
step "3b 3D ULPIN verifier: typo is rejected" bash -c "! $PY -m ulpin3d.cli verify-id 27PUNEW10101H4S034"

step "4 floor-plan reader on the 4 ward plans" bash -c "
for f in tests/data/ward_w1/plans/*; do $PY -m ulpin3d.cli parse-plan --plan \"\$f\" | head -1; done
n=\$(for f in tests/data/ward_w1/plans/*; do $PY -m ulpin3d.cli parse-plan --plan \"\$f\" | grep -c '^  FLAT'; done | paste -sd+ - | bc)
echo \"flats per typical floor across 4 plans: \$n (expected 16)\"; [ \"\$n\" = 16 ]"

# every example area: 7 synthetic data situations (graded against what their inputs make possible) + real areas
EX="$RUN/examples"
if [ $MODE_QUICK = 1 ]; then ONLY="--only 1_ 3_ 6_"; else ONLY=""; fi
step "5 run + grade the 8 data situations (synthetic ward, known answer)" $PY -m ulpin3d.cli test-scenarios --out "$RUN/scenarios" $ONLY
W="$RUN/scenarios/1_drone_lidar_plans"
step "5c property cards: every seal matches the register, a forged seal is rejected" $PY -c "
import json, sys
from ulpin3d.card import verify_card
u = '$W/units.geojson'; seals = json.load(open('$W/cards/seals.json'))
bad = [i for i, s in seals.items() if not verify_card(u, i, s)[0]]
i = next(iter(seals)); forged = verify_card(u, i, '0' * 32)[0]
print(len(seals), 'cards,', len(bad), 'seal mismatches, forged seal accepted:', forged)
sys.exit(1 if bad or forged or not seals else 0)"
step "5d indexed register answers the API queries" $PY -c "
from ulpin3d.register_db import Register
r = Register('$W'); s = r.summary(); u = r.units(owner='sharma', limit=5)
print(r.n_units, 'units,', r.n_ids(), 'IDs, by status', s['by_status'], '| owner search hits', len(u))
assert r.n_units > 0 and r.n_ids() > 0 and u"

if [ $MODE_QUICK = 0 ]; then
  step "6 run the real areas (Delft, Sao Paulo, Delhi, BKC; every data combination each has)" $PY -m ulpin3d.cli run-examples --out "$EX"
  step "6b Sao Paulo outlines vs the city's building classes (IoU >= 0.75)" \
       $PY -m ulpin3d.cli check-outlines --buildings "$EX/saopaulo_paraisopolis/1_drone_lidar/buildings.geojson" --truth examples/saopaulo_paraisopolis/truth/buildings_from_city_classes.tif
  step "6c Delft outlines vs the official building register (BAG), LiDAR scenario (IoU >= 0.5; one block postdates the LiDAR)" \
       $PY -m ulpin3d.cli check-outlines --min-iou 0.5 --buildings "$EX/delft_apartments/2_lidar_only/buildings.geojson" --truth examples/delft_apartments/truth/buildings_from_bag.tif
fi
if [ $MODE_FULL = 1 ]; then
  step "10 register benchmark at ~100k units (old in-memory vs indexed SQLite)" $PY scripts/bench_register.py --units "$W/units.geojson" --copies 500 --md "$RUN/register_benchmark.md"
fi

if [ $MODE_NET = 1 ]; then
  step "8 fresh real Indian scene from the internet (IIT BHU, Varanasi)" bash -c "
$PY -m ulpin3d.cli fetch-area --title 'IIT BHU' --size 200 --out $RUN/iit_bhu && \
$PY -m ulpin3d.cli run-area --area $RUN/iit_bhu --out $RUN/iit_bhu_out"
fi

step "9 viewer files written" bash -c "ls -la $W/viewer $W/viewer/vendor && test -s $W/viewer/cadastre_data.js"

printf "\n\033[1m================ SUMMARY ================\033[0m\n"
for r in "${RESULTS[@]}"; do echo "$r"; done
printf "\n%d passed, %d failed. Outputs and logs: %s\n" $PASS $FAIL "$RUN"
echo "Results table: $EX/RESULTS.md   3D cadastre: open $W/viewer/index.html"
[ $FAIL = 0 ]
