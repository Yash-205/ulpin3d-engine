# Testing guide: ulpin3d-engine

How to check, step by step, that every part of the system works: from the 3D ULPIN arithmetic up to a full survey area rendered as a 3D cadastre. Each step gives the command, how long it takes, and the result you should see.

## Contents
1. [What gets tested](#1-what-gets-tested)
2. [One-time setup](#2-one-time-setup-10-minutes)
3. [Run everything with one command](#3-run-everything-with-one-command)
4. [Step by step](#4-step-by-step)
5. [The 3D viewer, click by click](#5-the-3d-viewer-click-by-click)
6. [Testing with the judges' data](#6-testing-with-the-judges-data)
7. [Models: which to use, and how to (re)train them](#7-models-which-to-use-and-how-to-retrain-them)
8. [Reference results](#8-reference-results)
9. [Troubleshooting](#9-troubleshooting)
10. [Finale rehearsal checklist](#10-finale-rehearsal-checklist)

---

## 1. What gets tested

| Level | What | Data | Time |
|---|---|---|---|
| Automated tests | 47 unit and integration tests (IDs, footprints, LiDAR, plans, plan-first logic, cadastre, basements, property cards, API, indexed register, CPU model, ONNX) | generated on the fly | ~2.5 min (+3 min slow ones) |
| ID and plan checks | 3D ULPIN verifier, floor-plan reader | `tests/data/ward_w1/plans` | ~2 min |
| 8 data situations | the whole pipeline against a known right answer, once per input combination (drone / LiDAR / plans / surveyed outlines), cut on the fly from `tests/data/ward_w1`, each graded against what its inputs make possible | `run-examples` | ~8 min |
| Real LiDAR area | the whole pipeline on real city LiDAR, scored against the city's own building classes | `examples/saopaulo_paraisopolis` (grade B) | ~1.5 min |
| Real Indian drone area | the pipeline with weak data: the image alone, outlines from our model | `examples/sangam_vihar_delhi` (grade C) | ~30 s |
| Viewer | the 3D cadastre in a browser | any `examples/results/*/viewer` | manual, 5 min |

**Data grades.** A = drone image + LiDAR + plans (flats get IDs). B = image + LiDAR (buildings and floors get IDs). C = image or outlines only (buildings get provisional IDs).

---

## 2. One-time setup (~10 minutes)

On macOS, use this conda-forge environment. It avoids the PyTorch/LightGBM crash you hit in the base environment.

```bash
conda create -n ulpin3d -c conda-forge python=3.11 lightgbm pytorch torchvision rasterio geopandas pyproj shapely \
      scikit-image scikit-learn laspy lazrs-python ezdxf pymupdf matplotlib onnx onnxruntime pytest -y
conda activate ulpin3d
cd /Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine
pip install -e ".[dev]"
```

Every new terminal: `conda activate ulpin3d` and `cd` into the folder.

---

## 3. Run everything with one command

```bash
bash scripts/test_all.sh            # ~7 min
bash scripts/test_all.sh --quick    # ~4 min, skips the two real-data areas
bash scripts/test_all.sh --full     # adds the slow automated tests
bash scripts/test_all.sh --network  # adds a fresh real Indian scene downloaded from the internet
```

Expected ending:

```
================ SUMMARY ================
PASS  1 environment
PASS  2 automated tests (fast)
PASS  3 3D ULPIN verifier: valid id
PASS  3b 3D ULPIN verifier: typo is rejected
PASS  4 floor-plan reader on the 4 ward plans
PASS  5 run-area: synthetic ward (grade A)
PASS  5b check against the ward's ground truth
PASS  6 run-area: real LiDAR, Sao Paulo favela (grade B, CPU model + height)
PASS  6b outlines vs the city's building classes (IoU >= 0.75)
PASS  7 run-area: real drone image, Sangam Vihar Delhi (grade C)
PASS  9 viewer files written

11 passed, 0 failed. Outputs and logs: test_runs/<date-time>
```

Every step writes a log to `test_runs/<date-time>/`. A failing step prints the last 15 lines of its log.

---

## 4. Step by step

Run these yourself when you want to understand or demo one part.

### 4.1 Automated tests
```bash
pytest -q tests -m "not slow"      # 32 tests, ~2 min
pytest -q tests                    # all 34, ~4 min
```
Expect `… passed`. Warnings about deprecated functions in ezdxf, rasterio or pymupdf are harmless.

### 4.2 3D ULPIN: generate and verify offline
```bash
python -m ulpin3d.cli verify-id 27PUNEW10101H4S024
#  VALID  27PUNEW10101H4-S-02-4  parcel 27PUNEW10101 · floor 4 · class S · unit 2
python -m ulpin3d.cli verify-id 27PUNEW10101H4S034     # one digit changed
#  INVALID: check character should be 2, not 4
python -m ulpin3d.cli verify-id 853ZQTS2B2B2H0S01X     # the example on slide 2
#  INVALID: check character should be G, not X       <- fix the slide
```

### 4.3 Floor-plan reader
```bash
python -m ulpin3d.cli parse-plan --plan tests/data/ward_w1/plans/BP-2025-3117_typical_floor.dxf
python -m ulpin3d.cli parse-plan --plan tests/data/ward_w1/plans/BP-2025-3134_typical_floor.png   # a scan
```
Expect `floors [1, …]  scale 1:100`, then 4 flats per plan with labels (`FLAT 101` …), BHK type, carpet area, confidence, and common areas (stair, lift, lobby, duct). On the scan some flat numbers are misread by OCR. They're repaired from the numbering sequence and shown with confidence 0.7, or 0.4 when it's ambiguous.

**Try a real plan.** Any sanctioned plan PDF or DXF from a municipal building-permission portal works. The parser expects 230 mm external and party walls and 115 mm partitions, `FLAT/UNIT/SHOP <number>` labels, and `SCALE 1:N` on the sheet. On a real plan, look for: is every flat found, are the labels right, are the carpet areas plausible?

### 4.4 Whole pipeline, every data situation (known answer)
```bash
python -m ulpin3d.cli test-scenarios                     # the 8 situations, ~10 min; table in test_runs/scenarios/RESULTS.md
python -m ulpin3d.cli run-area   --area tests/data/ward_w1 --out out/ward_w1      # or one area by hand
python -m ulpin3d.cli check-town --area tests/data/ward_w1 --out out/ward_w1
open out/ward_w1/viewer/index.html
```
`check-town` grades a run against the `expect:` block in the area's `area.yaml` (what those inputs make possible). Expected for the full package:
```
  buildings detected           8/8 (8 outlines)                   PASS
  outlines accurate (IoU >= 0.8) 8/8                              PASS
  mean footprint IoU >= 0.9    0.97                               PASS
  no 3D ID on an inaccurate outline 0 given an ID                 PASS
  floor count exact            8/8                                PASS
  plans on the right plot      4/4                                PASS
  unsanctioned floor found     1/1 (false alarms 0)               PASS
  encroachment found           1/1 (false alarms 0)               PASS
  flats cut from plans         88/88                              PASS
  owners linked to flats       88/88                              PASS
  3D ULPINs unique             198 issued, unique=True            PASS
PASS: result matches what these inputs make possible
```
In scenarios with fewer inputs some lines read `n/a` (impossible with those inputs, e.g. no extra floor can be seen without heights); that is not a failure.
Also read `out/ward_w1/area_report.md`. Each plan should be listed as matched to a building, with *orientation confirmed from the roof cabin seen in LiDAR*.

**Make a new ward** with a different layout to see that it isn't tuned to one example:
```bash
python -m ulpin3d.cli demo-town --out out/ward_seed7 --seed 7
python -m ulpin3d.cli run-area  --area out/ward_seed7 --out out/ward_seed7_out
python -m ulpin3d.cli check-town --area out/ward_seed7 --out out/ward_seed7_out
```

### 4.5 Real LiDAR: São Paulo favela (held out from all training)
```bash
python -m ulpin3d.cli run-area --area examples/saopaulo_paraisopolis --out out/sp \
       --footprint-model models/footprint_lite_height
python -m ulpin3d.cli check-outlines --buildings out/sp/buildings.geojson \
       --truth examples/saopaulo_paraisopolis/truth/buildings_from_city_classes.tif
```
Expect ~470 buildings, ~990 units, and `building outline IoU 0.834 precision 0.900 recall 0.919`. The cloud arrives **unclassified**, so the LiDAR model classifies it, the CPU footprint model draws outlines from the image + height, and floors come from LiDAR. About 40 % of units are flagged. That's correct behaviour for dense informal housing with no plans: low-confidence outlines don't get an ID.

**Another tile:** `python -m ulpin3d.cli fetch-area-lidar --key MDS_color_3325-411.laz --out out/heliopolis` (Heliópolis, also held out), then run it the same way.

### 4.6 Real Indian drone image: Sangam Vihar, Delhi
```bash
python -m ulpin3d.cli run-area --area examples/sangam_vihar_delhi --out out/sangam
open out/sangam/viewer/index.html
```
Expect 108 buildings at data grade C: single level, provisional parcel IDs, about 25 flagged. In the viewer you'll see the Google/Microsoft outlines sit 2–5 m off the real roofs. That's why the training recipe registers labels before using them.

**Fresh Indian scenes from the internet** (OpenAerialMap; titles that work: `Sangam Vihar`, `IIT BHU`, `singhnagar`, `Raichur`, `Lokhandwala`):
```bash
python -m ulpin3d.cli fetch-area --title "IIT BHU" --size 250 --out out/iit_bhu
python -m ulpin3d.cli run-area   --area out/iit_bhu --out out/iit_bhu_out
```

---

## 5. The 3D viewer, click by click

Open `out/ward_w1/viewer/index.html` (double-click works; no server needed; works offline). Check each row:

| # | Do this | You should see |
|---|---|---|
| 1 | Page loads | Drone image as the ground, 8 buildings, panel says 8 buildings · 198 units · 198 IDs, "Synthetic demo data" tag |
| 2 | Drag / right-drag / scroll | Orbit, pan, zoom |
| 3 | Click any green flat | Card with the 3D ULPIN split into parcel · level · class · unit · check, "Check character verified offline", storey height ~3.1 m, carpet area in m² and sq ft, owner name and sale deed |
| 4 | Click **Copy 3D ULPIN** | "Copied" |
| 5 | Under *Needs attention*, click "Unsanctioned floor" | Camera flies to the top floor of the 8-storey block on 27PUNEW10106, purple, status UNSANCTIONED, still has an ID |
| 6 | Click "Encroachment" | The house on 27PUNEW10109; the parcel line visibly cuts through it |
| 7 | Drag **Floor separation** to 4 m | Floors pull apart; flats, stair/lift (grey) and stilt level visible |
| 8 | Tick **Isolate level**, move the level slider | One floor at a time across the ward |
| 9 | Click the *Provisional* chip | Provisional units hide; click again to show |
| 10 | Search `FLAT 402`, then an owner's surname | Flies to the matching flat |
| 11 | Paste an ID into **Verify a 3D ULPIN**, change one character | Valid → Invalid with the expected check character |
| 12 | Click empty ground inside a plot | Parcel card: plot area, buildings, IDs issued |
| 13 | Switch the OS to dark mode | Page and scene recolour |
| 14 | Open on a phone (or a narrow window) | 3D view on top, panel below |

---

## 6. Testing with the judges' data

1. Copy `examples/_template_area` and rename it, e.g. `areas/judges_ward`.
2. Put the files in the sub-folders (see `examples/_template_area/README.md`): `ortho/` (GeoTIFF), `lidar/` (LAS/LAZ), `parcels/` (with a ULPIN column), `plans/` (DXF/PDF/scans), `ownership/` (CSV).
3. Edit `area.yaml`: name, `parcel_chars` (12 or 14), height limit, parcel ID column name.
4. Run, picking the footprint model by what you were given:

| You have | Command |
|---|---|
| Image + LiDAR (most likely) | `run-area --area areas/judges_ward --out out/judges --footprint-model models/footprint_lite_height` |
| Image + LiDAR, GPU model trained | `... --footprint-model models/kaggle/footprint_india.onnx` (compare both; keep the better one) |
| LiDAR only | `run-area --area areas/judges_ward --out out/judges` (outlines from LiDAR) |
| Image only | `... --footprint-model models/kaggle/footprint_india.onnx`, or `models/footprint_lite_rgb` as a weaker CPU fallback |
| Surveyed outlines given | put them in `footprints/`; they override any model |

5. Read `out/judges/area_report.md` first:
   - **data grade and missing**: did it find every file?
   - **Plans**: every plan should say `matched`. `no matching building` means the plan's outline doesn't fit any footprint: wrong scale, wrong site, or a footprint error.
   - **orientation**: `confirmed …` is good; `ambiguous` means the flats may be rotated 180° (check the north arrow).
   - **Ownership**: `linked` should equal `records`. Unlinked rows usually mean the `unit_label` spelling differs from the plan (`FLAT 402` vs `402`).
   - **Unsanctioned floors / encroachments**: verify each one in the viewer before presenting it.
6. Open `out/judges/viewer/index.html` and walk through section 5.

**Formats seen in Indian survey deliveries that work:** UTM 43N/44N/45N, WGS84 or Web Mercator GeoTIFFs; LAS 1.2–1.4 with or without RGB/intensity; Shapefile/GeoJSON parcels; AutoCAD DXF (export DWG to DXF first, e.g. with the free ODA File Converter); vector or scanned PDF plans. **Not yet supported:** DWG directly, multi-page PDFs (use page 0, or split the file), point clouds in E57 (convert with `pdal translate in.e57 out.laz`).

---

## 7. Models: which to use, and how to (re)train them

| Model | File | Trained on | Needs | Good for |
|---|---|---|---|---|
| LiDAR classifier (LightGBM) | `models/pointcloud/model.txt` | São Paulo LiDAR, 28 tiles | CPU | always, when there's a point cloud |
| Footprints, CPU, image + height (LightGBM) | `models/footprint_lite_height/` | São Paulo image + LiDAR height | CPU + LiDAR | **default when LiDAR is given**: IoU 0.78 on unseen tiles, trains in 78 s |
| Footprints, CPU, image only (LightGBM) | `models/footprint_lite_rgb/` | African + S. Asian + Indian drone chips | CPU | fallback with no LiDAR and no GPU model (weak: IoU 0.31 on Indian test) |
| Footprints, U-Net (deep) | from Kaggle: `footprint_india.pt` / `.onnx` | full recipe, GPU | GPU to train, CPU to run (ONNX) | image-only areas; separating attached houses |

### Retrain the CPU models yourself (no GPU)
```bash
# image + height model from LiDAR chips (rebuild chips first if you add tiles)
python -m ulpin3d.cli train-footprint-lite --chips data/footprint/chips_sp --out models/footprint_lite_height --height yes
python -m ulpin3d.cli eval-footprint-lite  --model models/footprint_lite_height --chips data/footprint/chips_sp --split test
# LiDAR classifier
python -m ulpin3d.cli pc-prepare --config configs/pointcloud.yaml && python -m ulpin3d.cli pc-train --config configs/pointcloud.yaml \
  && python -m ulpin3d.cli pc-eval --config configs/pointcloud.yaml
```

### Train the U-Net on a free Kaggle GPU (one command)
1. kaggle.com → *Settings → API → Create New Token* (downloads `kaggle.json`). Verify your phone number on Kaggle (needed for GPU and internet).
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. Start it:
   ```bash
   QUICK=1 bash scripts/kaggle_train.sh push    # ~1 h smoke run first
   bash scripts/kaggle_train.sh status          # RUNNING / COMPLETE / ERROR
   bash scripts/kaggle_train.sh fetch           # -> models/kaggle/
   bash scripts/kaggle_train.sh push            # then the full run (~6-8 h)
   ```
4. Check `models/kaggle/reports/stage2_india_test/report.json`. Pass bar for the demo: pixel IoU ≥ 0.70 and object F1 ≥ 0.50 on the Indian test set, with "validated" precision ≥ 0.95 in the calibration table. Compare stage 1 with stage 2 to show what fine-tuning on India bought you.

The run saves `footprint_india.onnx`, an ONNX export that is **2.7× faster on CPU with identical masks** (measured: 2.9 s → 1.1 s per 512×512 tile). Use it at the venue. `export-onnx --int8` makes the file 4× smaller but isn't faster on most laptop CPUs and changes about 3 % of the mask.

---

## 8. Reference results

| Test | Data | Result |
|---|---|---|
| Synthetic ward | 8 buildings, 4 plans, 88 flats | all checks pass (section 4.4) |
| Floor plans | 7 synthetic Indian plans, 52 flats | DXF/PDF: unit F1 1.00, IoU 0.99, carpet error 0.13 %; scans: F1 0.92 |
| LiDAR classifier | 4 unseen São Paulo tiles | building IoU 0.81, overall 0.86; 0.78 without RGB and intensity |
| CPU footprints, image + height | 4 unseen São Paulo tiles | pixel IoU 0.775, F1 0.873, object F1 0.39 |
| CPU footprints, image + height, whole area | Paraisópolis 250 m, unclassified cloud | outline IoU 0.834, precision 0.90, recall 0.92 |
| CPU footprints, image only | 5 held-out Indian scenes | pixel IoU 0.31 (weak, needs the GPU model) |
| U-Net CPU pilot (12 short epochs) | 5 held-out Indian scenes, registered labels | pixel IoU 0.47, object F1 0.17 (under-trained) |
| ONNX export | 512×512 tile, laptop-class CPU | 2.7× faster than PyTorch, identical masks |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Segmentation fault` during tests (macOS) | You're in the base environment. Use the conda-forge `ulpin3d` environment (section 2). |
| `ModuleNotFoundError: ulpin3d` | `conda activate ulpin3d` and `pip install -e .` inside the project folder. |
| Viewer shows "The 3D library could not load" | The `viewer/vendor/` folder is missing next to `index.html`; copy the whole `viewer/` folder, not just the HTML. |
| Viewer is blank or grey | Use Chrome, Edge or Safari with hardware acceleration on. |
| `needs LiDAR height` error | You passed `footprint_lite_height` for an area without LiDAR; use the ONNX/U-Net model or `footprint_lite_rgb`. |
| Plan says `no matching building` | Check the plan's scale text (`SCALE 1:100`), that it belongs to this site, and that a footprint exists for that building in `buildings.geojson`. |
| Many units `flagged` | Expected with weak data (grade C, image-only CPU model). Add LiDAR, plans or a trained U-Net, or review in the viewer's *Needs attention* list. |
| `fetch-area`: no scene found | Use a title from section 4.6; titles are matched as substrings. |
| Kaggle push fails with 403 | Phone not verified on Kaggle, or `kaggle.json` not in `~/.kaggle/` with `chmod 600`. |

---

## 10. Finale rehearsal checklist

- [ ] `bash scripts/test_all.sh` passes on the demo laptop, **with Wi-Fi off** (the viewer and all models work offline).
- [ ] Models are in `models/`: `pointcloud/`, `footprint_lite_height/`, `footprint_lite_rgb/`, and `kaggle/footprint_india.onnx` once trained.
- [ ] A copy of `examples/_template_area` is ready to receive the judges' files.
- [ ] Timed dry run on a full example: survey folder → `run-area` → viewer, under 3 minutes for a 250 m area.
- [ ] Slide 2's example ID corrected (check character **G**, not X), and the ID format decision (12 or 14-character parcel) made.
- [ ] Each flagged, unsanctioned and encroachment item in your demo area opened once in the viewer, so you can explain it.
