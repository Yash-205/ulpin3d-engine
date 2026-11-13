# ULPIN-3D Engine — Project Prep Guide

> **SIH26011 · 3D ULPIN Generation & Vertical Property Mapping · Team Dimension Zero**

This document explains the overall architecture, every module's role, and the end-to-end data flow of the project — written to give a newcomer a complete mental model before touching any code.

---

## 1. What the Project Does

The engine converts a **survey package** (drone orthophoto + LiDAR point cloud + sanctioned floor plans + parcel records + ownership CSVs) into a **3D digital cadastre**:

- One **building polygon per structure** with confidence, height, floor count and status
- One **volumetric unit record per flat / common area / unsanctioned floor / air-rights slab** — each unit carrying a globally-unique, offline-verifiable **3D ULPIN**
- An **interactive Three.js viewer** (`viewer/index.html`) — no server needed, opens from disk

The single entry point for a whole area is:

```bash
python -m ulpin3d.cli run-area --area <survey_folder> --out <output_folder>
```

---

## 2. High-Level Pipeline

```
Drone Orthophoto ──► Model 1 (U-Net / LightGBM-lite)
                     Building Footprints + Confidence
                                   │
                                   ▼
LiDAR Point Cloud ──► Model 2 (LightGBM)          ──► Fusion ──► 3D Buildings
                       Ground / Building /                        (footprint +
                       Vegetation / Other                         height + floors)
                       + Ground/HAG filter                              │
                                                                        │
Sanctioned Plans ──► Model 3 (DXF / PDF / Scan)                        │
   (DXF / PDF /       Floor Units + Labels +                            │
    Scan)             Common Areas                                       │
                                   │                                    │
                                   ▼                                    ▼
                           Georeference ──────────────────────► Cadastre Builder
                           (fit plan to                           (stack floors,
                            footprint)                            air rights,
                                                                  unsanctioned,
                                                                  confidence)
                                                                        │
                                                                        ▼
2D Parcels + Ownership ─────────────────────────────────────► 3D ULPIN Assignment
                                                              + Anomaly Checks
                                                                        │
                                                                        ▼
                                                              Outputs: GeoJSON
                                                              CityJSON 2.0
                                                              Viewer HTML
                                                              Area Report
```

---

## 3. Module Map

```
ulpin3d/
├── cli.py                   ← entry point for ALL commands
├── area.py                  ← run-area orchestrator (the main pipeline)
├── fuse.py                  ← merge Model 1 + Model 2 into 3D buildings
├── ulpin.py                 ← 3D ULPIN encode / parse / verify / assign
├── cadastre.py              ← build volumetric units per building; GeoJSON + CityJSON export
├── viewer_export.py         ← write viewer/ folder (index.html + cadastre_data.js)
├── card.py                  ← sealed 3D Property Cards (HTML + QR) + verify-card
├── api.py                   ← read-only FastAPI over a run-area result (`serve`)
│
├── common/
│   └── raster.py            ← COG/remote reading on a metric UTM grid at fixed GSD
│
├── data/
│   ├── sources.py           ← Open Cities + OpenAerialMap catalog; Open Buildings labels; splits
│   ├── chips.py             ← tile orthophotos into 512 px chips with building/boundary masks
│   ├── register.py          ← FFT cross-correlation to correct label-to-image offset per scene
│   └── synthetic.py         ← generate synthetic Indian settlement orthophotos for tests
│
├── footprint/
│   ├── model.py             ← U-Net (EfficientNet-B4 encoder, 2-head: building + boundary)
│   ├── dataset.py           ← PyTorch Dataset + augmentations for chip training
│   ├── train.py             ← training loop, LR schedule, mixed precision, checkpointing
│   ├── infer.py             ← sliding-window inference on any-size orthophoto; ONNX export
│   ├── postprocess.py       ← watershed instance segmentation; confidence scoring
│   ├── metrics.py           ← pixel IoU, object F1, calibration table
│   ├── evaluate.py          ← evaluate a checkpoint on a chip split
│   └── lite.py              ← LightGBM footprint model (CPU, no GPU needed)
│
├── pointcloud/
│   ├── io.py                ← read LAS/LAZ; write classified LAZ
│   ├── ground.py            ← cloth-simulation ground filter; height-above-ground (HAG)
│   ├── features.py          ← geometric features per point (neighbourhood stats, HAG, RGB, intensity)
│   ├── classify.py          ← LightGBM train / evaluate / infer; feature engineering pipeline
│   ├── buildings.py         ← cluster building points into footprints; attach roof heights + floors
│   ├── rasterize.py         ← point cloud → nDSM / intensity raster (for lite model)
│   ├── data.py              ← class-name ↔ ASPRS code tables; label remapping
│   └── synthetic.py         ← synthetic Indian LiDAR scenes for tests
│
├── floorplan/
│   ├── parse.py             ← DXF / vector PDF / scanned image → units + labels + floor range
│   ├── georef.py            ← fit plan outline to building footprint (similarity + ICP); resolve 180° orientation ambiguity via stair cabin
│   ├── evaluate.py          ← evaluate plan parsing + georef against ground truth
│   └── synthetic.py         ← generate synthetic Indian sanctioned floor plans (DXF / PDF / scan)
│
├── demo/
│   ├── town.py              ← generate a complete synthetic ward (8 buildings, plans, LiDAR, ownership)
│   ├── check.py             ← compare a run-area result to the synthetic ward's ground truth
│   └── real_area.py         ← fetch a real Indian OAM drone scene or São Paulo LiDAR tile for testing
│
└── viewer/
    ├── index.html           ← self-contained Three.js 3D cadastre viewer
    └── vendor/              ← Three.js + OrbitControls (offline copy)
```

---

## 4. Module-by-Module Detail

### 4.1 `cli.py` — Command Dispatcher

All user-facing operations are sub-commands of `python -m ulpin3d.cli`. Key commands:

| Command | What it does |
|---|---|
| `run-area` | Full pipeline from survey folder to viewer |
| `demo-town` | Generate synthetic ward for testing |
| `check-town` | Verify run-area output against truth |
| `train-footprint` | Train U-Net Model 1 |
| `infer-footprint` | Run Model 1 on a single orthophoto |
| `export-onnx` | Export U-Net to ONNX (~2.7× faster on CPU) |
| `train-footprint-lite` | Train LightGBM CPU footprint model |
| `pc-prepare / pc-train / pc-eval` | LiDAR Model 2 pipeline |
| `pc-infer` | Classify a new point cloud |
| `fuse` | Merge footprints + classified LiDAR |
| `parse-plan` | Parse a single floor plan |
| `cadastre` | Build units for a single building |
| `catalog / labels / chips / register` | Training data pipeline |
| `verify-id` | Offline check a 3D ULPIN |
| `fetch-area` | Download a real Indian drone scene |

---

### 4.2 `area.py` — Pipeline Orchestrator

The `run_area()` function is the master pipeline. It runs these steps in order:

1. **Ingest** (`ingest()`) — discovers all files in the survey folder and assigns a data grade:
   - Grade **A**: orthophoto + LiDAR + plans
   - Grade **B**: two of three
   - Grade **C**: minimal data
2. **LiDAR Classification** (`_classified_cloud()`) — applies Model 2 to the cloud (or uses pre-classified labels)
3. **Building Outlines** (`_buildings()`) — sourced from (in priority order):
   - Surveyed `footprints/` file
   - Model 1 (orthophoto) ± LiDAR height
   - Model 2 building points directly
4. **Parcel Assignment** (`_parcels()`, `_assign_parcels()`) — matches each building to its 2D ULPIN; flags encroachments
5. **Plan Matching** (`_match_plans()`) — parses every plan, scores IoU against candidate buildings, greedily assigns one plan per building, resolves 180° orientation from stair-cabin seen in LiDAR
6. **Cadastre Building** (`build_cadastre()` per building) — stacks units per floor, marks unsanctioned floors, adds air-rights slab
7. **Ownership Linking** (`_ownership()`) — merges ownership CSV records into each unit by parcel + flat label
8. **Anomaly Checks** (`_checks()`) — detects overlapping units, storey heights outside 2.4–6.0 m, and statistical outliers via Isolation Forest
9. **3D ULPIN Assignment** (`assign_ids()`) — deterministic numbering across each parcel
10. **Output** — writes `buildings.geojson`, `units.geojson`, `parcels.geojson`, `cadastre.city.json`, `area_report.json/.md`, and the viewer

---

### 4.3 `footprint/` — Model 1: Building Extraction from Orthophotos

**Architecture:** U-Net with EfficientNet-B4 encoder and two output heads:
- Head 0: building probability map
- Head 1: boundary probability (walls between adjacent buildings)

**Training data flow:**
```
Open Cities AI + OpenAerialMap → catalog.py → labels.py → chips.py → register.py
           Stage 1 pretrain (Africa + South Asia) → Stage 2 fine-tune (India)
```

**Key files:**
- [`model.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/footprint/model.py) — U-Net definition; `FootprintLoss` = BCE + Dice (building) + focal BCE (boundary)
- [`train.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/footprint/train.py) — training loop with AdamW, OneCycleLR, mixed precision
- [`infer.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/footprint/infer.py) — sliding 4096 px blocks, any CRS/GSD (warped to UTM at 0.15 m/px), optional TTA, ONNX export
- [`postprocess.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/footprint/postprocess.py) — watershed segmentation on `(building - boundary)` probability
- [`lite.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/footprint/lite.py) — LightGBM on colour + texture patches (+ optional LiDAR height) — no GPU, trains in ~78 s

**Confidence & status thresholds:**
- ≥ 0.85 → **validated**
- 0.60–0.85 → **provisional**
- < 0.60 → **flagged** (goes to surveyor review, no ID issued)

---

### 4.4 `pointcloud/` — Model 2: LiDAR Classification + Heights

**Algorithm:** LightGBM gradient-boosted classifier on per-point geometric features.

**Classes:** ground · building · vegetation · other (IGNORE for ambiguous São Paulo class 19 at ground level)

**Feature engineering (`features.py`):**
- Height above ground (HAG)
- Neighbourhood statistics at k = 10, 25, 50 (linearity, planarity, scatter, verticality)
- RGB colour (dropped if sensor lacks it)
- Intensity, return-number attributes (dropped randomly during training to handle unknown judges' sensor)

**`buildings.py`** — given classified points:
- Clusters `building` class into footprint polygons (DBSCAN on 2D projection)
- Computes `ground_z`, `roof_h_main`, `roof_levels`, `floors_est` with low/high band
- Detects roof features (stair cabins, lift machine rooms) above main roof + 1.8 m — used to resolve plan orientation ambiguity

---

### 4.5 `fuse.py` — Merge Models 1 + 2

Combines orthophoto footprints with classified LiDAR to produce **3D building records**:

- Attaches `ground_z`, `roof_h_main`, `floors_est`, `height_conf` from LiDAR to each photo footprint
- `confidence_3d = footprint_conf × height_factor`
- Adds **LiDAR-only buildings** (hidden under trees / missed by photo model) with `status = flagged`
- Outputs `buildings_3d.geojson` (EPSG:4326) and `.gpkg` (UTM)

---

### 4.6 `floorplan/` — Model 3: Floor Plan Reader

Parses three formats to a common output: **units + common areas + floor range + scale**.

**`parse.py` pipeline (all three formats follow the same steps after loading):**
1. Load wall geometry (DXF lines, PDF vector paths, or rasterized scan at 20 mm/px)
2. Morphological separation: thick walls (≥ 160 mm — exterior/party) vs. thin partitions
3. Close door gaps (up to ~1.2 m) in thick walls
4. Flood-fill enclosed regions → one flat / common area polygon
5. Assign labels via regex: `FLAT 401`, `2BHK`, `CORRIDOR`, `LIFT`, `STAIRCASE` …
6. OCR correction for scan confusions (`F1AT → FLAT`, `L1FT → LIFT`, etc.)
7. Extract floor range: `"TYPICAL FLOOR PLAN (1ST TO 6TH FLOOR)"` → `floors=[1,2,3,4,5,6]`
8. Extract scale: DXF `$INSUNITS`, PDF/scan `SCALE 1:N`

**`georef.py`** — fits parsed plan outline to building footprint:
- Similarity transform (rotation, scale, translation) via ICP
- Quality gates: RMS ≤ 0.5 m → OK; RMS ≤ 1.5 m → flagged; > 1.5 m → halt
- Resolves 180° orientation ambiguity using stair-cabin centroids from LiDAR

---

### 4.7 `cadastre.py` — Volumetric Unit Builder

Per building:

1. **Georef** each matched plan onto the footprint
2. **Floor heights** — derived from `(roof_h - 0.4) / n_floors` (clipped to 2.7–4.5 m), ground floor +0.4 m
3. **Stack units** — for each level in the plan, transform unit polygons and assign `z_min / z_max`
4. **Reality vs sanction** — LiDAR levels without a plan → one whole-level unit, `status = unsanctioned`
5. **Air rights** — from roof up to permissible height → `zclass = "A"`
6. **Assign 3D ULPINs** (via `ulpin.py`)
7. **Export** — GeoJSON (2D footprints + z attributes) + CityJSON 2.0 (LOD1 extruded solids)

**Confidence propagation:** `min(unit_confidence, georef_quality, building_3d_confidence)`

---

### 4.8 `ulpin.py` — 3D ULPIN Format

**Format (18 characters, no separators):**

```
853ZQTS2B2B2  H4  S  02  G
└─ parcel ──┘ │   │  │   └ Luhn mod-36 check character
              │   │  └ unit number, base-32, 2 chars (00 = whole level / common)
              │   └ class: S surface · A air-rights · I infrastructure
              └ level, base-32, 2 chars: H0=ground, H1=floor 1, GV=basement 1
```

Key properties:
- **Deterministic** — same space always gets the same ID; AI decides which spaces exist, this module names them
- **Offline-verifiable** — single character typo always caught; > 90% of adjacent swaps caught
- Plan flat numbers (e.g. `FLAT 402`) preserved as `preferred_unit=02` where clean and unique
- `parcel_chars` = 12 (slide demo) or 14 (full national ULPIN, 20-character total ID)

---

### 4.9 `viewer_export.py` + `viewer/index.html` — 3D Cadastre Viewer

Writes `viewer/` next to the run-area output:

- **`cadastre_data.js`** — `window.CADASTRE = {meta, ortho, parcels, buildings, units}` (JSON)
- **`cadastre_view.json`** — same data, loadable via "Load area" button
- **`index.html`** — Three.js viewer (offline vendor copy included)

Viewer features:
- Orthophoto as ground texture (embedded as base64 JPEG — works from `file://`)
- Every unit as a 3D extruded box coloured by status (validated / provisional / unsanctioned / flagged; common areas grey; air rights translucent blue)
- Click a flat → panel with full 3D ULPIN breakdown, storey height, carpet area, volume, owner, document, confidence, issues
- Click a parcel → its buildings
- Filter by status, show floors up to level N, isolate one floor, explode floors apart
- Search by 3D ULPIN, flat label, or owner name
- Offline ID verifier (catches single-character typos)
- Load another `cadastre_view.json`
- Light / dark theme; mobile-friendly

---

### 4.10 `data/` — Training Data Pipeline

| File | Role |
|---|---|
| [`sources.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/data/sources.py) | Catalog Open Cities AI + OpenAerialMap scenes; fetch Google/Microsoft Open Buildings labels; geographic train/val/test splits |
| [`chips.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/data/chips.py) | Tile orthophotos into 512 px × 512 px chips at 0.15 m/px; create building + boundary + instance masks |
| [`register.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/data/register.py) | Estimate per-scene label-to-image shift (satellite labels sit 1–4 m off on 5 cm drone imagery); rebuild chips with corrected labels |
| [`synthetic.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/data/synthetic.py) | Generate synthetic Indian settlement orthophotos (RCC roofs, water tanks, attached blocks) for unit tests |

---

### 4.11 `demo/` — Test Harness

| File | Role |
|---|---|
| [`town.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/demo/town.py) | Generate a complete synthetic ward: 8 buildings, 4 plans (PDF/DXF/scan), 88 flats, 1 unsanctioned floor, 1 encroachment, synthetic LiDAR, ownership CSV |
| [`check.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/demo/check.py) | Assert the run-area result matches the ward's `truth.json` (buildings found, floor counts, plans matched, unsanctioned + encroachment detected, all ULPINs unique) |
| [`real_area.py`](file:///Users/yashagarwal/Downloads/WORK/Projects_Main/Hackathons/SIH/ulpin3d-engine/ulpin3d/demo/real_area.py) | Download a real Indian OAM drone scene (`fetch-area`) or a São Paulo LiDAR tile (`fetch-area-lidar`) for end-to-end testing |

---

## 5. Input Survey Folder Layout

```
area.yaml               name, parcel_chars (12), permissible_height_m, parcel_id_field
ortho/*.tif             drone orthophoto (any CRS / GSD)
lidar/*.la[sz]          LiDAR point cloud (classified or unclassified)
parcels/*.geojson       2D cadastral parcels with a ULPIN column
plans/**                sanctioned plans: DXF / PDF / scan
                        │  sub-folder named after a parcel ULPIN pins its plans to that parcel
ownership/*.csv         parcel_ulpin, unit_label, owner_name, share_pct, right_type, document
footprints/*.geojson    (optional) surveyed building outlines — override the AI
```

Anything missing is reported; the pipeline uses what is there.

---

## 6. Output Files

| File | Description |
|---|---|
| `buildings.geojson` | One polygon per building (EPSG:4326) with confidence, floor count, status, issues |
| `units.geojson` | One polygon per unit (EPSG:4326) with 3D ULPIN, z range, area, owner |
| `parcels.geojson` | 2D parcels with building + ID counts |
| `cadastre.city.json` | CityJSON 2.0 — LOD1 extruded solids for every unit and building |
| `area_report.json` | Full run report: data grade, building source, plan matching, checks, totals |
| `area_report.md` | Human-readable markdown summary |
| `cards/<id>.html` | 3D Property Card per issued ID: QR + SHA-256/HMAC seal; `cards/seals.json` is the seal register |
| `viewer/index.html` | Interactive 3D viewer (open from disk, no server) |
| `viewer/cadastre_data.js` | Data bundle loaded by the viewer |
| `viewer/cadastre_view.json` | Same data as JSON (loadable at runtime) |

---

## 7. Configs

```
configs/
├── data_footprint.yaml           catalog, label fetch, chip, register settings
├── footprint_stage1_pretrain.yaml   Stage 1: Open Cities + OAM pretrain
├── footprint_stage2_india.yaml   Stage 2: Indian OAM fine-tune
├── footprint_pilot_cpu.yaml      CPU pilot (ResNet-34, fast test)
└── pointcloud.yaml               LiDAR Model 2: tile list, voxel size, feature config
```

---

## 8. Training Data Strategy

India has no open drone+LiDAR+label package, so the pipeline uses a **two-stage regional transfer**:

| Stage | Footprint Model | LiDAR Model |
|---|---|---|
| **Pretrain** | Open Cities AI (Africa, dense/informal) + OpenAerialMap Bangladesh/Nepal | São Paulo 2017 city LiDAR (favelas + high-rises — closest open match to Indian metro morphology) |
| **Fine-tune** | 25 Indian OAM scenes + Open Buildings labels (with offset correction) | Synthetic Indian buildings (water tanks, stair cabins, RCC) |
| **Test (held out)** | Sangam Vihar Delhi, Vijayawada, Raichur, IIT BHU, Lokhandwala | 4 São Paulo tiles with 1.5 km spatial buffer |

**Three data problems addressed:**
1. **Label shift** (satellite labels 1–4 m off on 5 cm drone imagery) — `register.py` corrects per scene
2. **Imperfect Indian test labels** — `export-review` exports for hand-correction in QGIS; reviewed labels override raw ones automatically
3. **Unknown judges' sensor** — RGB/intensity/return-number attributes randomly blanked during training

---

## 9. Quick-Start Commands

```bash
# Install
pip install -e .
pytest -q tests   # 41 tests, ~5 min on CPU

# Generate + run synthetic ward
python -m ulpin3d.cli demo-town --out examples/ward_w1
python -m ulpin3d.cli run-area  --area examples/ward_w1 --out examples/ward_w1_out
open examples/ward_w1_out/viewer/index.html

# Check synthetic ward result against ground truth
python -m ulpin3d.cli check-town --area examples/ward_w1 --out examples/ward_w1_out

# Run on real survey folder
python -m ulpin3d.cli run-area --area data/judges_area --out out/site \
    --footprint-model models/footprint_india.pt \
    --pc-model models/pointcloud/model.txt

# Verify a 3D ULPIN offline
python -m ulpin3d.cli verify-id 853ZQTS2B2B2H4S02G
```

---

## 10. Synthetic Ward Ground-Truth Checklist

| Check | Expected |
|---|---|
| Buildings found / footprint IoU | 8/8 · 0.956 |
| Floor count from LiDAR | 8/8 exact |
| Plans matched to right parcel | 4/4, orientation confirmed 4/4 |
| Unsanctioned floor / encroachment | 1/1 · 1/1, 0 false alarms |
| Flats / owners linked | 88/88 · 88/88 |
| 3D ULPINs issued | 198 (177 validated, 20 provisional, 1 unsanctioned), all unique |

