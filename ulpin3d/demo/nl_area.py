"""A real, nearly complete survey package from Dutch open data (no synthetic data).

    python -m ulpin3d.cli fetch-area-nl --bbox 84250 446500 84500 446750 --laz data/nl/37EN1_15.LAZ \
        --out examples/delft_apartments

The Netherlands is one of the few places where the public registers a 3D cadastre needs are open
(CC-BY / CC0), for the same block:

    ortho/ortho.tif               PDOK aerial photo, 8 cm (Beeldmateriaal Nederland, CC-BY 4.0)
    lidar/cloud.laz               AHN4 airborne LiDAR, ~20 pts/m2, classes as delivered (CC0)
    parcels/parcels.geojson       Kadaster cadastral parcels; the 14-digit parcel id is the 2D parcel ID (CC-BY 4.0)
    reference/bag_buildings.geojson   BAG: the official, surveyed building outlines (CC0)
    reference/bag_units.csv       BAG: every registered apartment / unit with address, floor area, use, year (CC0)
    truth/buildings_from_bag.tif  the BAG outlines rasterised, to score our outlines against

Not open: sanctioned floor plans and owners (the Kadaster sells ownership extracts). So flats are not
cut from plans here; the BAG unit register is the check on how many apartments each building holds.
All coordinates EPSG:28992 (RD New).
"""
from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

EPSG = 28992
WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
BAG = "https://service.pdok.nl/lv/bag/wfs/v2_0"
KAD = "https://service.pdok.nl/kadaster/kadastralekaart/wfs/v5_0"


def _wfs(url, typename, bbox, page=1000):
    feats, start = [], 0
    while True:
        p = dict(SERVICE="WFS", VERSION="2.0.0", REQUEST="GetFeature", TYPENAMES=typename, COUNT=page, STARTINDEX=start,
                 BBOX=f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},urn:ogc:def:crs:EPSG::{EPSG}", OUTPUTFORMAT="application/json")
        f = requests.get(url, params=p, timeout=120).json()["features"]
        feats += f
        if len(f) < page:
            break
        start += page
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=EPSG)
    return gpd.GeoDataFrame.from_features(feats, crs=EPSG)


def _ortho(bbox, out: Path, gsd=0.08, tile_px=2000):
    """Aerial photo through the WMS in tiles (servers cap image size), stitched into one GeoTIFF."""
    from PIL import Image

    L, B, R, T = bbox
    W, H = int(round((R - L) / gsd)), int(round((T - B) / gsd))
    img = np.zeros((H, W, 3), np.uint8)
    for r0 in range(0, H, tile_px):
        for c0 in range(0, W, tile_px):
            w, h = min(tile_px, W - c0), min(tile_px, H - r0)
            x0, y1 = L + c0 * gsd, T - r0 * gsd
            p = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS="Actueel_orthoHR", STYLES="", CRS=f"EPSG:{EPSG}",
                     BBOX=f"{x0},{y1 - h * gsd},{x0 + w * gsd},{y1}", WIDTH=w, HEIGHT=h, FORMAT="image/png")
            r = requests.get(WMS, params=p, timeout=180)
            r.raise_for_status()
            img[r0:r0 + h, c0:c0 + w] = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=f"EPSG:{EPSG}",
                       transform=from_origin(L, T, gsd, gsd), compress="jpeg", photometric="ycbcr", tiled=True) as d:
        d.write(img.transpose(2, 0, 1))
    return W, H


def _lidar(laz: Path, bbox, out: Path):
    import laspy

    L, B, R, T = bbox
    keep = []
    with laspy.open(str(laz)) as f:
        hdr = f.header
        for chunk in f.chunk_iterator(5_000_000):
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            m = (x >= L) & (x <= R) & (y >= B) & (y <= T)
            if m.any():
                keep.append(chunk[m])
    las = laspy.LasData(hdr)
    las.points = laspy.ScaleAwarePointRecord(np.concatenate([k.array for k in keep]), hdr.point_format, hdr.scales, hdr.offsets)
    from pyproj import CRS

    try:
        las.header.add_crs(CRS.from_epsg(EPSG))
    except Exception:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out))
    return len(las.points)


def fetch_area_nl(bbox, laz, out, name="Delft (real Dutch open data)", pad=10.0):
    out = Path(out)
    L, B, R, T = bbox
    win = (L - pad, B - pad, R + pad, T + pad)
    print(f"[fetch-area-nl] aerial photo {R - L:.0f} x {T - B:.0f} m at 8 cm ...", flush=True)
    W, H = _ortho(win, out / "ortho" / "ortho.tif")
    print(f"[fetch-area-nl] LiDAR crop from {Path(laz).name} ...", flush=True)
    n = _lidar(Path(laz), win, out / "lidar" / "cloud.laz")
    print("[fetch-area-nl] parcels, buildings, units from the national registers ...", flush=True)
    par = _wfs(KAD, "kadastralekaart:Perceel", bbox)
    par = par[par.geometry.intersects(box(*bbox))].copy()
    par["ulpin"] = par["identificatieLokaalID"].astype(str).str.zfill(14)
    keep = [c for c in ("ulpin", "kadastraleGemeenteWaarde", "sectie", "perceelnummer", "kadastraleGrootteWaarde") if c in par]
    (out / "parcels").mkdir(parents=True, exist_ok=True)
    par[keep + ["geometry"]].to_file(out / "parcels" / "parcels.geojson", driver="GeoJSON")
    pand = _wfs(BAG, "bag:pand", bbox)
    pand = pand[pand.geometry.intersects(box(*bbox)) & pand.status.astype(str).str.contains("in gebruik", case=False)].copy()
    vbo = _wfs(BAG, "bag:verblijfsobject", bbox)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("identificatie", "bouwjaar", "status", "aantal_verblijfsobjecten") if c in pand]
    pand[cols + ["geometry"]].to_file(out / "reference" / "bag_buildings.geojson", driver="GeoJSON")
    ucols = [c for c in ("identificatie", "pandidentificatie", "openbare_ruimte", "huisnummer", "huisletter", "toevoeging",
                         "postcode", "woonplaats", "oppervlakte", "gebruiksdoel", "bouwjaar", "status") if c in vbo]
    pd.DataFrame(vbo[ucols]).to_csv(out / "reference" / "bag_units.csv", index=False)
    gsd = 0.25
    tw, th = int((R - L + 2 * pad) / gsd), int((T - B + 2 * pad) / gsd)
    tr = from_origin(win[0], win[3], gsd, gsd)
    lab = rasterize([(g, 1) for g in pand.geometry], out_shape=(th, tw), transform=tr).astype(np.uint8)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    with rasterio.open(out / "truth" / "buildings_from_bag.tif", "w", driver="GTiff", width=tw, height=th, count=1,
                       dtype="uint8", crs=f"EPSG:{EPSG}", transform=tr) as d:
        d.write(lab, 1)
    homes = vbo[vbo.gebruiksdoel.astype(str).str.contains("woon")] if len(vbo) else vbo
    cfg = {"name": name, "synthetic": False, "parcel_chars": 14, "parcel_id_field": "ulpin",
           "source": {"ortho": "PDOK Luchtfoto RGB 8 cm (Beeldmateriaal Nederland), CC-BY 4.0",
                      "lidar": "AHN4 (Rijkswaterstaat / Het Waterschapshuis) via TU Delft GeoTiles, CC0",
                      "parcels": "Kadaster, Kadastrale Kaart (BRK), CC-BY 4.0",
                      "reference": "BAG buildings + units (Kadaster), CC0"}}
    (out / "area.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[fetch-area-nl] {W}x{H} px photo, {n:,} LiDAR points, {len(par)} parcels, {len(pand)} buildings, "
          f"{len(vbo)} registered units ({len(homes)} homes) -> {out}")
    return out
