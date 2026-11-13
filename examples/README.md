# Test areas: real places only

```
examples/
  delft_apartments/        Delft, Netherlands: the most complete open package anywhere
  saopaulo_paraisopolis/   São Paulo, Brazil: real city LiDAR of a dense favela + high-rises
  sangam_vihar_delhi/      Sangam Vihar, Delhi: real drone image
  bkc_mumbai/              Bandra Kurla Complex, Mumbai: real OpenStreetMap building outlines
  _template_area/          empty folder in the expected layout, for the judges' data: copy, fill, run
  results/                 what run-examples produced, grouped by area, then by scenario
    RESULTS.md             the table of all results
```

```bash
python -m ulpin3d.cli run-examples                  # runs every area and scenario (~10 min) -> examples/results/
python -m ulpin3d.cli run-examples --only delft     # just one area
open examples/results/delft_apartments/1_drone_lidar/viewer/index.html            # any result in 3D (double-click works too)
python -m ulpin3d.cli serve --out examples/results/delft_apartments/1_drone_lidar  # its REST API
```

Each area is run once per combination of the data it **really** has (the `scenarios:` block in its
`area.yaml`), so you can see what each extra source adds. Where an official building map exists, our
outlines are scored against it. Each result folder holds `viewer/index.html` (3D cadastre),
`cards/index.html` (property cards), `area_report.md`, the GeoJSON / CityJSON outputs, `register.sqlite`
(indexed, for the API) and `run.log`.

## What each area has

| Area | Photo | LiDAR | Parcels | Official outlines | Units register | Plans | Owners |
|---|---|---|---|---|---|---|---|
| `delft_apartments` | 8 cm aerial | AHN4, 3.3 M points | Kadaster, 162 | BAG, 191 buildings | BAG, 595 homes with address + floor area | not open | not open |
| `saopaulo_paraisopolis` | rendered from LiDAR | city LiDAR | — | city building classes | — | — | — |
| `sangam_vihar_delhi` | drone, 8 cm | — | — | Google/Microsoft (not official, offset 2–5 m) | — | — | — |
| `bkc_mumbai` | none open | none open | none open | OpenStreetMap, 204 (7 with floor counts) | — | — | — |

No open dataset anywhere has sanctioned floor plans and owners for real buildings, so flats cut from
plans and owner linking are tested on the synthetic ward in `tests/data/ward_w1`
(`python -m ulpin3d.cli test-scenarios`, all 8 situations PASS), where the right answer is known.

## Results

| Area | Scenario | Inputs | Buildings | 3D units | IDs issued | Outlines vs official map (IoU · precision · recall) |
|---|---|---|---|---|---|---|
| **bkc_mumbai** | [—](results/bkc_mumbai/viewer/index.html) | official building outlines (OpenStreetMap) | 204 | 266 | 69 | no official map open |
| **delft_apartments** | [1_drone_lidar](results/delft_apartments/1_drone_lidar/viewer/index.html) | drone/aerial photo, LiDAR, parcels | 158 | 497 | 402 | 0.54 · 0.83 · 0.60 |
|  | [2_lidar_only](results/delft_apartments/2_lidar_only/viewer/index.html) | LiDAR, parcels | 221 | 484 | 484 | 0.54 · 0.84 · 0.60 |
|  | [3_drone_only](results/delft_apartments/3_drone_only/viewer/index.html) | drone/aerial photo, parcels | 158 | 158 | 0 | 0.47 · 0.74 · 0.57 |
|  | [4_city_outlines_lidar](results/delft_apartments/4_city_outlines_lidar/viewer/index.html) | drone/aerial photo, LiDAR, parcels, official building outlines | 207 | 391 | 367 | 1.00 · 1.00 · 1.00 |
| **sangam_vihar_delhi** | [1_drone_only](results/sangam_vihar_delhi/1_drone_only/viewer/index.html) | drone/aerial photo | 80 | 80 | 0 | no official map open |
|  | [2_open_outlines](results/sangam_vihar_delhi/2_open_outlines/viewer/index.html) | drone/aerial photo, official building outlines | 108 | 108 | 83 | no official map open |
| **saopaulo_paraisopolis** | [1_drone_lidar](results/saopaulo_paraisopolis/1_drone_lidar/viewer/index.html) | drone/aerial photo, LiDAR | 472 | 992 | 709 | 0.81 · 0.91 · 0.89 |
|  | [2_lidar_only](results/saopaulo_paraisopolis/2_lidar_only/viewer/index.html) | LiDAR | 221 | 440 | 424 | 0.82 · 0.92 · 0.88 |
|  | [3_drone_only](results/saopaulo_paraisopolis/3_drone_only/viewer/index.html) | drone/aerial photo | 106 | 106 | 0 | 0.69 · 0.86 · 0.77 |

IoU 1.0 = our outlines cover exactly the official ones. Fewer IDs on poorer data is intended: IDs are
withheld (flagged) wherever the evidence is too weak.

**What the numbers say**

- **Delft:** with the official outlines (scenario 4) the result matches the official map exactly and LiDAR
  adds the heights: the realistic municipal-update case. From LiDAR or photo alone we reach 0.57–0.60
  recall. The ceiling is about 0.64: one large block in the south-west was built after the LiDAR survey
  (AHN4, ~2020–22) and is invisible to it, while the aerial photo and the register are newer. Delft also
  exposed a real bug, now fixed: a row of terraced houses, each on its own narrow plot, lost its middle
  houses when split along plot lines. That fix applies directly to Indian row-house streets.
- **São Paulo:** LiDAR is what makes outlines reliable (0.81–0.82 IoU with or without the photo, vs 0.69
  from the photo alone), and only LiDAR gives floor counts.
- **Delhi:** from the image alone the model's outlines sit on the roofs (Google's are 2–5 m off,
  `reports/sangam_model_vs_open_outlines.jpg`), but with no parcels or heights every building stays flagged.
- **BKC:** there is no open drone, LiDAR, parcel or plan data for BKC. The 7 towers with recorded floor
  counts get stacked floors and provisional IDs; the rest stay flagged. The BKC page in the parent folder
  (`bkc-vertical-cadastre-v2.html`) is a hand-modelled illustration, not survey data.
