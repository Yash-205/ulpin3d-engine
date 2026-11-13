# Survey folder layout (what run-area reads)

| Folder | Put here | Formats | Required? |
|---|---|---|---|
| `ortho/` | drone orthophoto (one mosaic) | GeoTIFF, any CRS, any resolution | yes, unless LiDAR or footprints are given |
| `lidar/` | point cloud | LAS / LAZ (1.0–1.4), classified or not | strongly recommended: heights and floor counts |
| `parcels/` | 2D cadastral parcels with their ULPIN | GeoJSON, GeoPackage, Shapefile (with .shx/.dbf/.prj) | recommended: without it parcel IDs are provisional |
| `plans/` | sanctioned building plans | DXF, PDF (vector or scanned), PNG/JPG scans | for flat-level units; matched to buildings automatically |
| `plans/<ULPIN>/` | optional: plans you know belong to one parcel | as above | optional |
| `ownership/` | record of rights | CSV: `parcel_ulpin, unit_label, owner_name, share_pct, right_type, document` | optional |
| `footprints/` | surveyed building outlines, if you have them | GeoJSON / GeoPackage / Shapefile | optional (overrides the AI outlines) |

Missing pieces are listed in `area_report.md`; the pipeline uses what is there (data grade A/B/C).

`unit_label` must match the flat number on the plan, e.g. `FLAT 402`. Plans for a typical floor
should carry the floor range in the title, e.g. "TYPICAL FLOOR PLAN (1ST TO 6TH FLOOR)".
