# ulpin3d-engine — AI models for Bhu-Aadhaar 3D

**SIH26011 · 3D ULPIN Generation & Vertical Property Mapping · Team Dimension Zero**

This repo holds the trained-model side of the system. The models turn a survey package (drone orthophoto, LiDAR point cloud, later floor plans) into **buildings with footprints, heights, floor counts and a confidence score**. That output feeds 3D ULPIN assignment and the 3D cadastre viewer.

```
drone orthophoto ──► Model 1  building footprints ─┐
                     (U-Net, 2 heads)              ├─► fused buildings ─► floors, units ─► validation ─► 3D ULPIN ─► cadastre
LiDAR point cloud ─► Model 2  point classes ───────┘    + heights           (plans)        (thresholds)
                     (LightGBM)  + ground/HAG
```

| Model | Input → output | Status |
|---|---|---|
| **1. Building extraction** | RGB orthophoto (any CRS/GSD) → one polygon per building, with confidence and validated/provisional/flagged status | ✅ code + tests; train on GPU |
| **2. LiDAR classification** | LAS/LAZ → ground / building / vegetation / other per point, with confidence | ✅ code + tests; trained on São Paulo |
| **2b. Heights & floors** | classified LiDAR + footprints → roof height, roof levels, floor count with a low/high band | ✅ rule-based; ML version next |
| **3. Floor-plan reader** | sanctioned DXF / vector PDF / scan → flats + common areas with labels, BHK, carpet area, floor range | ✅ geometry + OCR (PP-OCR via RapidOCR) |
| **Georeferencing** | plan → building footprint (similarity + ICP), with quality gates (0.5 m RMS → halt) | ✅ |
| **3D ULPIN + cadastre** | building + plans → volumetric units, unsanctioned floors, air rights, IDs, GeoJSON + CityJSON 2.0 | ✅ |
| **Anomaly validator** | 3D units → overlaps, storey heights, Isolation-Forest outliers | ✅ in run-area |
| **Model 1-lite (CPU)** | LightGBM on colour/texture (+ LiDAR height): no GPU to train or run | ✅ image+height IoU 0.775 on unseen tiles, trains in 78 s |
| **Underground levels** | declared basement depth → one volumetric unit per level with its own 3D ULPIN (class S, negative level), capped at provisional because LiDAR cannot see below ground | ✅ `basements:` in area.yaml |
| **REST API** | FastAPI over a run-area result: ID lookup, filters, **which unit contains this 3D point**, cards, seal check | ✅ `serve` |
| **3D Property Card** | unit record → printable card with QR and a SHA-256 / HMAC seal, checked by `verify-card` | ✅ in run-area |
| **ONNX export** | U-Net → ONNX for laptop CPUs | ✅ 2.7× faster, identical masks |

## Training data: regional transfer to India

India has no open package of drone imagery + LiDAR + labels. So each model learns first on regions that **look like Indian settlements**, then adapts to Indian data, and is **scored only on Indian scenes it has never seen**.

| | Model 1 (footprints) | Model 2 (LiDAR) |
|---|---|---|
| **Pretrain (look-alike regions)** | [Open Cities AI](https://source.coop/open-cities) — World Bank drone imagery, 2–20 cm, 7 African cities, dense/informal (tier 1, ~790k buildings, labels traced on the drone image). **OpenAerialMap drone scenes from Bangladesh (461), Nepal (86), Sri Lanka, Pakistan**, labelled with Google + Microsoft Open Buildings ([VIDA merge](https://source.coop/vida/google-microsoft-open-buildings)) | [São Paulo city LiDAR 2017](https://registry.opendata.aws/pmsp-lidar/) — ~22 pts/m², RGB, classified. **Favelas (Paraisópolis, Heliópolis, Brasilândia) next to high-rises**, the closest open match to Indian metro morphology |
| **Fine-tune (India)** | 25 Indian OAM drone scenes (Manipur, Karnataka, Gujarat, Kerala, Rajasthan …) + Open Buildings labels | synthetic Indian buildings (water tanks, stair cabins, parapets, attached RCC) |
| **Test (held out, 1 km buffer)** | Sangam Vihar (Delhi), Singh Nagar (Vijayawada), Raichur village (Karnataka), IIT BHU (Varanasi), Lokhandwala (Mumbai) | 4 São Paulo tiles held out with a 1.5 km buffer: 3 favelas + the Paulista high-rise core |

### Three data problems the pipeline deals with
1. **Labels shifted on drone imagery.** Google/Microsoft footprints were traced on satellite images and sit 1–4 m off on a 5 cm drone mosaic (clear on Sangam Vihar). → `register` estimates each scene's offset with the stage-1 model (FFT cross-correlation) and rebuilds the chips with corrected labels.
2. **Indian test labels are imperfect** (merged row-houses, missing buildings). → `export-review` exports the test scenes for hand-correction in QGIS. Reviewed labels in `labels_reviewed/` override the raw ones automatically. **Do this before quoting accuracy numbers.**
3. **The judges' sensor is unknown** (RGB? intensity? return numbers?). → Training randomly blanks those attributes, and `pc-eval` reports accuracy with and without them.

## Results so far

**Model 2 — LiDAR classification**: trained on 28 São Paulo tiles, tested on 4 unseen tiles (1.5 km buffer). Model in `models/pointcloud/`.

| Test tiles (unseen) | Overall acc. | Building IoU | Ground IoU | Vegetation IoU |
|---|---|---|---|---|
| all sensors | 0.857 | **0.811** | 0.759 | 0.822 |
| no RGB | 0.855 | 0.804 | 0.771 | 0.799 |
| no RGB, no intensity | 0.839 | 0.776 | 0.766 | 0.754 |

Per tile, building IoU: Paraisópolis 0.85 · Brasilândia 0.81 · Paulista high-rise 0.78 · Heliópolis 0.73. São Paulo's class 19 is used for ground in some districts and for objects in others. "Other" points lying at ground level (< 0.3 m) are therefore ignored in training and scoring; before that fix, Heliópolis scored 0.48 overall accuracy purely because of the label inconsistency.

**Model 1 — footprints, current model** (`models/footprint_india.pt` / `.onnx`; U-Net + EfficientNet-B4, trained on a Kaggle T4 GPU, 25 + 14 epochs). Trained on **human-labelled** buildings (ramp: Karnataka, Bangladesh, Myanmar; Open Cities Africa drone) plus down-weighted drone scenes with machine labels. Scored on **619 Karnataka tiles locked away from training** (4,943 buildings drawn by people, reviewed by a second person):

| | object F1 (IoU ≥ 0.5) | pixel IoU | precision | recall |
|---|---|---|---|---|
| validation (Karnataka, seen during model selection) | 0.725 | 0.786 | | |
| **test (Karnataka, never seen)** | **0.729** | **0.789** | 0.78 | 0.69 |

Val ≈ test, so no overfitting. Confidence means what it says: buildings marked *validated* were right 96% of the time, *provisional* 83%, *flagged* 41%. Weak spot: on very sharp drone images it merges attached houses into one outline; with a sanctioned plan the plan's outline is used instead, and outlines over several plots are split at the plot line. On real Indian drone imagery (Sangam Vihar) its outlines sit on the roofs while the open Google/Microsoft outlines are 2–5 m off (`reports/sangam_model_vs_open_outlines.jpg`).

**Model 1, earlier runs (kept for the record).** On synthetic held-out scenes (sanity check only), pixel IoU 0.94 and object F1 0.77 after a short CPU run. Confidence calibration: 100% of "validated", 93% of "provisional" and 53% of "flagged" buildings were real, so the thresholds mean what they say. Real-data numbers come from the GPU run in the notebook, plus the CPU pilot (`configs/footprint_pilot_cpu.yaml`, `reports/pilot_india_test`).

**Model 1 on real Indian drone imagery: CPU pilot** (ResNet-34, 480 steps on 1,744 real chips; *not* the competition model). Scored on 159 chips from the 5 held-out Indian scenes:

| labels used for scoring | pixel IoU | pixel F1 | object F1 |
|---|---|---|---|
| raw Google/Microsoft | 0.36 | 0.53 | 0.04 |
| registered (offset corrected, 3–5 m per scene) | 0.47 | 0.64 | 0.17 |

This shows two things. Label registration is essential (the estimated offsets were Sangam Vihar +3.2/+4.7 m, Vijayawada −4.7/−0.7 m, Raichur −3.2/−1.7 m). And the model is badly under-trained on CPU: it misses grey RCC roofs and hasn't learned shared walls yet. The GPU recipe (≈25k steps, EfficientNet-B4, full data) is what produces the real numbers. The safety logic already holds: **0 buildings were marked "validated"**, so nothing would get an ID on weak evidence.

**Model 3: floor plans** (7 synthetic Indian sanctioned plans, 52 flats):

| input | unit F1 | unit IoU after georef | carpet-area error | floor range read |
|---|---|---|---|---|
| DXF | 1.00 | 0.99 | 0.13 % | 7/7 |
| vector PDF | 1.00 | 0.99 | 0.13 % | 7/7 |
| scan (150 dpi, skewed, noisy) | 0.92 | 0.91 | 1.6 % | 7/7 |

On scans, outlines are right; the errors are OCR-misread flat numbers. Those are repaired from the numbering series (floor × flat count) and flagged. Where two flats read the same way, both drop to *flagged* rather than risk a wrong ID.

**Heights / floors** (synthetic Indian LiDAR, photo footprints given): floor count exact for 100% of buildings, roof height error 0.12 m.

> **Testing:** see [TESTING.md](TESTING.md). One command, `bash scripts/test_all.sh`, runs the automated tests and every example area (~15 minutes). Test areas, one per data situation plus real Indian drone imagery and real LiDAR, are in `examples/`.

## Run a whole area in one command

```bash
conda activate ulpin3d
python -m ulpin3d.cli run-area --area <survey folder> --out out/<area> \
        [--footprint-model models/footprint_india.onnx]    # Model 1 (fast CPU); without it outlines come from LiDAR
open out/<area>/viewer/index.html                         # the 3D cadastre (opens from disk, no server)
```

The survey folder holds whatever the judges hand over; anything missing is reported and the pipeline uses what is there:

```
area.yaml          name, parcel_chars (12), permissible_height_m, parcel_id_field,
                   basements (levels below ground: a number, or {building id / parcel ULPIN: n}), basement_height_m (3.0)
ortho/*.tif        drone orthophoto           lidar/*.laz     point cloud (classified or not)
parcels/*.geojson  2D parcels with ULPIN      plans/**        sanctioned plans: DXF / PDF / scans
plans/plans_index.csv  optional: file, parcel_ulpin (the building-permission register; or put plans in a
                   sub-folder named after the parcel ULPIN). Pinned plans are followed even when the AI outline is poor.
ownership/*.csv    parcel_ulpin, unit_label, owner_name, share_pct, right_type, document
footprints/*.geojson  optional outlines: trusted as surveyed, or set footprints_source: ai in area.yaml
```

**With or without a sanctioned plan.** The image model is never the last word on a footprint:

| Situation | Footprint used | What gets flagged |
|---|---|---|
| Plan available (pinned to its plot, or matched by shape) | the plan's true-to-scale outline, placed on the plot's LiDAR roof | built area beyond the plan ("possible unauthorised extension"), floors above the plan ("unsanctioned") |
| Plan available, no LiDAR, photo model caught only part of the roof | the plan slid over the plot to where the model's roof probability is high inside it, its outline sits on the model's roof-edge map, and the confidently seen part lies inside it | ID stays *provisional* (height not measured) |
| No plan, LiDAR available | LiDAR roof outline when the photo outline disagrees with it (IoU < 0.85), cut at the plot line | "LiDAR roof outline used"; overhangs past the plot as encroachment |
| No plan, no LiDAR | photo outline, split along plot lines when it covers several plots | lower confidence; whole levels instead of flats |

Tested on the synthetic ward with deliberately bad AI outlines (roofs half lost in shadow, blobs running into the neighbour's plot): with plans, plans on the right plot went from 0/4 to 4/4 and flats from 52/88 to 88/88; without plans, footprint IoU went from 0.68 to 0.96. Encroachment was still caught (1/1, no false alarms) in both.

It runs: ingestion report and data grade → building outlines → LiDAR classification, heights and floors → parcel assignment (outlines over several plots split at the plot line), LiDAR roof check of AI outlines and encroachment check → **plans matched to buildings** (pinned plots first, then by shape) (shape fit, scale and floor-count evidence, one plan per building; orientation settled from the stair cabin seen in LiDAR) → flats, common areas, unsanctioned floors and air rights per building → **3D ULPINs per parcel** → ownership linking → overlap, storey-height and Isolation-Forest checks → outputs:

`buildings.geojson` · `units.geojson` · `parcels.geojson` · `cadastre.city.json` (CityJSON 2.0) · `register.sqlite` (indexed, for the API) · `area_report.json` / `.md` · `viewer/` · `cards/`

**Tested on real places** (`examples/README.md`; `python -m ulpin3d.cli run-examples`), each run once per combination of the data it really has, scored against the official building map where one exists:

| Area | Data | Best result vs the official map |
|---|---|---|
| Delft, Netherlands | 8 cm aerial photo, AHN4 LiDAR, Kadaster parcels, BAG building + apartment register | official outlines + LiDAR: IoU 1.00; LiDAR only: IoU 0.54 (one block postdates the LiDAR) |
| São Paulo, Brazil | city LiDAR (held-out tile) | IoU 0.82 (precision 0.92, recall 0.88) |
| Sangam Vihar, Delhi | real drone image only | outlines on the roofs; all flagged (no parcels, no heights) |
| Bandra Kurla Complex, Mumbai | OpenStreetMap outlines, 7 with floor counts | towers with floor counts get stacked floors; rest flagged |

**Every data situation is tested** on the synthetic ward in `tests/data/ward_w1` (`python -m ulpin3d.cli test-scenarios`), the only data with a known right answer for plans, flats and owners (no real area anywhere publishes sanctioned plans and owners). It is cut into 8 packages on the fly and each is graded against what its inputs make possible:

| inputs | outlines accurate | floors | plans | extra floor | encroachment | flats · owners |
|---|---|---|---|---|---|---|
| drone + LiDAR + plans | 8/8 | 8/8 | 4/4 | 1/1 | 1/1 | 88/88 · 88/88 |
| drone + LiDAR | 8/8 | 8/8 | — | — | 1/1 | whole levels |
| drone + plans (no LiDAR) | 8/8 | — | 4/4 | — | 1/1 | 88/88 · 88/88 |
| drone only | 6/8, the other 2 flagged (no ID) | — | — | — | 1/1 | whole levels |
| LiDAR + plans | 8/8 | 8/8 | 4/4 | 1/1 | 1/1 | 88/88 · 88/88 |
| LiDAR only | 8/8 | 8/8 | — | — | 1/1 | whole levels |
| municipal outlines + plans | 8/8 | — | 4/4 | — | 1/1 | 88/88 · 88/88 |
| drone + LiDAR + plans, no permit register (plans matched by shape) | 8/8 | 8/8 | 4/4 | 1/1 | 1/1 | 88/88 · 88/88 |

All PASS, 0 false alarms, and in every case **no 3D ULPIN was issued on an inaccurate outline**. "—" = impossible with those inputs.

### The 3D cadastre viewer
`viewer/index.html` shows the orthophoto as the ground, parcel boundaries, and every unit as a 3D volume coloured by status (validated / provisional / unsanctioned / flagged; common areas grey, air rights translucent blue). Click a flat for its 3D ULPIN broken into parcel · level · class · unit · check (verified offline), storey height and elevation, carpet area in m² and sq ft, volume, owner and document, confidence and findings. Click a plot for its buildings. You can filter by status, show floors up to a level or isolate one, pull the floors apart, search by ID, flat or owner, verify any typed ID, and load another `cadastre_view.json`. Works in light and dark, and on phones.

Every unit of a kind (flats, whole levels, common areas, air rights) shares one GPU mesh and one edge set, so a ward is at most 8 draw calls however many flats it has; filters and the level slider rewrite the shared buffers in place, and a click is mapped from the hit triangle back to its unit. On a 9,000-unit ward: 60 FPS and ready in 1.5 s, against 10 FPS and 2.0 s when every unit was its own mesh (`reports/viewer_benchmark.md`).

### Underground levels
LiDAR cannot see below ground, so basements come from a declared depth (a permit or a surveyed depth marker), not from the sensors. Add `basements: 2` to `area.yaml` (or `basements: {B0003: 2, 27PUNEW10106: 1}`) and every such building gets `BASEMENT 1`, `BASEMENT 2` … as volumes under its footprint, with elevations stacked down from the ground (`basement_height_m`, default 3.0 m) and IDs such as `27PUNEW10107GVS01B` (`GV` = basement 1). They are marked *provisional* with the finding "not observed by LiDAR". The viewer makes the ground see-through when basements exist and its level slider reaches below ground.

### REST API
```bash
pip install -e ".[api]"
python -m ulpin3d.cli serve --out out/<area>          # docs at http://127.0.0.1:8000/docs
curl "http://127.0.0.1:8000/point?lon=73.7977&lat=18.5199&z=101.2"   # the volume that owns this 3D point
curl "http://127.0.0.1:8000/ulpin/27PUNEW10108H1S03Y"                # verify + record
curl "http://127.0.0.1:8000/units?status=provisional&level=-1"       # filter
```
Also `/summary`, `/parcel/{ulpin}/units`, `/card/{id}` (the property card) and `/verify-card`. Read-only; it never changes the register.

**Fast at district scale.** Queries come from `register.sqlite`, an indexed copy of `units.geojson` that `run-area` writes next to it (and the API rebuilds when `units.geojson` changes; `build-db --out <area>` by hand). B-tree indexes on ID, parcel, building, status + level and kind, an FTS5 trigram index for owner-name search, and SQLite's built-in **3D R-tree** (longitude, latitude, elevation), so "which unit contains this 3D point" only touches the units stacked at that spot. Stdlib only, one file. On 99,000 units with owners (the synthetic ward tiled 500x; `scripts/bench_register.py`, `reports/register_benchmark.md`):

| | before (GeoJSON in RAM) | indexed SQLite |
|---|---|---|
| API start-up | 2.7 s | 1.3 ms |
| units on a parcel / status + level filter / owner-name search | 25 ms each | 0.9 / 1.9 / 3.4 ms |
| verify a card's seal | 1.8 s (re-read the whole file) | 0.3 ms |
| 3D ULPIN lookup, 3D point query | < 1 ms | < 1 ms |

### 3D Property Cards
`run-area` writes one card per issued 3D ULPIN to `cards/<id>.html` (plus `cards/index.html` and `cards/seals.json`). A card shows the ID broken into its parts, the unit's level, elevations, areas and volume, the rights on record, a QR code and a seal. It prints to A4 or PDF from the browser, and the viewer's unit panel links to it.

The seal is SHA-256 over the record and the unit outline, so any later change to the owner, the area, the heights or the shape is detected:

```bash
python -m ulpin3d.cli verify-card --units out/<area>/units.geojson --id 27PUNEW10108H1S03Y --seal <seal on the card, or the 32 characters in its QR>
python -m ulpin3d.cli property-card --units out/<area>/units.geojson --id 27PUNEW10108H1S03Y   # re-issue one card
```

A plain hash shows that a card matches the register, not who issued it. Set `ULPIN3D_SEAL_KEY` (the issuing authority's secret) before `run-area` and `verify-card` and the seal becomes HMAC-SHA256, which nobody can recompute without the key. Cards for units that are not `validated` say so and state that they are not proof of title.

## Install
```bash
pip install -e .            # Python 3.10+, CPU or CUDA; Apple MPS works too
pytest -q tests             # 41 tests, ~5 min on CPU
```

## Train (GPU: Kaggle / Colab)
Use **`notebooks/train_kaggle.ipynb`**, which runs every step below in order. Or by hand:
```bash
# Model 1 — footprints
python -m ulpin3d.cli catalog  --config configs/data_footprint.yaml        # ~620 scenes, with splits
python -m ulpin3d.cli labels   --config configs/data_footprint.yaml        # Open Cities + Open Buildings
python -m ulpin3d.cli chips    --config configs/data_footprint.yaml        # 512 px @ 0.15 m, ~8 GB
python -m ulpin3d.cli train-footprint --config configs/footprint_stage1_pretrain.yaml
python -m ulpin3d.cli register --config configs/data_footprint.yaml --checkpoint runs/footprint_stage1/best.pt
python -m ulpin3d.cli chips    --config configs/data_footprint.yaml --sources oam --out data/footprint/chips_reg
python -m ulpin3d.cli train-footprint --config configs/footprint_stage2_india.yaml --set data.chips_root=data/footprint/chips_reg
python -m ulpin3d.cli eval-footprint  --checkpoint runs/footprint_stage2_india/best.pt --chips data/footprint/chips_reg --split test --out reports/india

# Model 2 — LiDAR (CPU)
python -m ulpin3d.cli pc-prepare --config configs/pointcloud.yaml
python -m ulpin3d.cli pc-train   --config configs/pointcloud.yaml
python -m ulpin3d.cli pc-eval    --config configs/pointcloud.yaml
```

## Run on the judges' data
```bash
python -m ulpin3d.cli infer-footprint --image their_ortho.tif --checkpoint models/footprint_india.pt --out out/site
python -m ulpin3d.cli pc-infer --laz their_cloud.laz --model models/pointcloud/model.txt --out out/site/classified.laz
```
Outputs: `buildings.geojson` (EPSG:4326, for the web viewer), `buildings.gpkg` (UTM), `probs.tif`, and a classified LAZ with a per-point `cls_conf`. Any image size works (processed in 4096 px blocks), in any CRS (warped to UTM on the fly), at any GSD (resampled to the model's 0.15 m).

## 3D ULPIN format
`853ZQTS2B2B2` `H4` `S` `02` `G` = parcel (12) · level (base-32, 544 = "H0" = ground, so H4 = 4th floor, GV = basement 1) · class (S surface / A air-rights / I infrastructure) · unit (base-32; the plan's flat number is kept where clean, so FLAT 402 → 02; 00 = whole level / common area) · Luhn mod-36 check. It is bit-compatible with the web demo's `luhn36()`. Every single-character typo is caught, and >90 % of adjacent swaps.
**Note for the deck:** slide 2's example `853ZQTS2B2B2H0-S-01-X` has check character **G**, not X, so the demo's verifier would reject it. **Open decision:** 12-char parcel (slide) vs the full 14-char national ULPIN (`parcel_chars=14`, giving 20-character IDs).

Run the last mile:
```bash
python -m ulpin3d.cli fuse      --buildings out/site/buildings.gpkg --laz out/site/classified.laz --out out/site/buildings_3d.geojson
python -m ulpin3d.cli parse-plan --plan typical_floor.pdf
python -m ulpin3d.cli cadastre  --buildings out/site/buildings_3d.geojson --building-id SITE-B00042 \
        --plans ground.pdf typical_1_to_6.pdf --parcel 27MH0PUN0A01 --permissible-height 36 --out out/site/B00042
```
→ `units.geojson` (web viewer) and `units.city.json` (CityJSON 2.0, LOD1 solids: Building + BuildingUnits). Each unit carries its 3D ULPIN, level, z-range, carpet area, volume, status and issues. Levels LiDAR sees beyond the sanctioned plans become `unsanctioned` whole-level units (flag, don't fail). The space above the roof up to the permissible height becomes an air-rights volume (class A).

## Confidence and thresholds
Each building gets `confidence ∈ [0, 1]`: mean building probability, discounted for boundary ambiguity and very small size. Statuses use the cut-offs in the config:
`≥ 0.85 validated` · `0.60–0.85 provisional` · `< 0.60 flagged` (goes to surveyor review).
`eval-footprint` prints a **calibration table**: the share of "validated" buildings that really match a true building. If it's below ~0.95, raise `conf_validated`.

## Layout
```
ulpin3d/common/raster.py      COG/remote reading on a metric UTM grid at fixed GSD
ulpin3d/data/sources.py       Open Cities + OpenAerialMap catalog, Open Buildings labels, geographic splits
ulpin3d/data/chips.py         chips + building / boundary (party-wall) / instance masks
ulpin3d/data/register.py      label-to-image offset correction
ulpin3d/data/synthetic.py     synthetic Indian settlement orthophotos (tests)
ulpin3d/footprint/            dataset+augs, U-Net, loss, train, metrics, post-processing, inference, evaluation
ulpin3d/pointcloud/           LAS IO, ground filter/HAG, features, LightGBM classifier, buildings+floors, synthetic LiDAR
ulpin3d/fuse.py               footprints + classified LiDAR -> 3D building records
ulpin3d/floorplan/            plan generator (Indian sanctioned plans), DXF/PDF/scan parser, georef, evaluation
ulpin3d/ulpin.py              3D ULPIN encode / parse / verify / deterministic assignment
ulpin3d/cadastre.py           building + plans -> volumetric units with IDs; GeoJSON + CityJSON export
ulpin3d/card.py               sealed 3D Property Cards (HTML + QR) and seal verification
configs/                      data + training configs
notebooks/train_kaggle.ipynb  end-to-end GPU recipe
```

## Licences of training data
Open Cities: ODbL / per-scene. OpenAerialMap: CC-BY 4.0. Google Open Buildings: CC-BY 4.0; Microsoft footprints: ODbL. São Paulo LiDAR: as published by PMSP/GeoSampa. All fine for a hackathon; a government deployment would need a licence review, and retraining on SVAMITVA/NAKSHA data.
