"""Raster helpers: open local or remote (COG over HTTPS) imagery and read it on a
metric UTM grid at a fixed ground sampling distance (GSD).

Every dataset we touch (Open Cities, OpenAerialMap, the judges' drone orthophoto)
comes in a different CRS and resolution. The models are trained at a fixed GSD,
so all reads go through `MetricReader`, which warps on the fly into the scene's
UTM zone at the requested GSD.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin

# Sensible defaults for streaming Cloud-Optimised GeoTIFFs over HTTPS.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_CACHEMAX": "512",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "100000000",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
for _k, _v in _GDAL_ENV.items():
    os.environ.setdefault(_k, _v)


def to_gdal_path(path: str) -> str:
    if path.startswith(("http://", "https://")):
        return "/vsicurl/" + path
    if path.startswith("s3://"):
        return "/vsis3/" + path[5:]
    return path


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


@dataclass
class MetricGrid:
    crs: CRS
    bounds: tuple  # (left, bottom, right, top) in metres
    gsd: float

    @property
    def width(self) -> int:
        return int(round((self.bounds[2] - self.bounds[0]) / self.gsd))

    @property
    def height(self) -> int:
        return int(round((self.bounds[3] - self.bounds[1]) / self.gsd))

    @property
    def transform(self):
        return from_origin(self.bounds[0], self.bounds[3], self.gsd, self.gsd)


class MetricReader:
    """Read RGB windows of an image on a UTM grid at a target GSD.

    Returns (rgb uint8 HxWx3, valid bool HxW). `valid` excludes nodata, alpha=0
    and pure-black collar pixels common in drone mosaics.
    """

    def __init__(self, path: str, gsd: float, crs: CRS | None = None):
        self.path = path
        self.src = rasterio.open(to_gdal_path(path))
        b = transform_bounds(self.src.crs, "EPSG:4326", *self.src.bounds)
        self.lonlat_bounds = b
        self.crs = crs or utm_crs_for_lonlat((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        self.gsd = gsd
        self.native_gsd = self._native_gsd()
        mb = transform_bounds(self.src.crs, self.crs, *self.src.bounds)
        # snap to the GSD grid
        self.bounds = (
            math.floor(mb[0] / gsd) * gsd,
            math.floor(mb[1] / gsd) * gsd,
            math.ceil(mb[2] / gsd) * gsd,
            math.ceil(mb[3] / gsd) * gsd,
        )
        self.grid = MetricGrid(self.crs, self.bounds, gsd)
        self.vrt = WarpedVRT(
            self.src,
            crs=self.crs,
            transform=self.grid.transform,
            width=self.grid.width,
            height=self.grid.height,
            resampling=Resampling.bilinear,
            src_nodata=self.src.nodata,
            nodata=self.src.nodata,
            add_alpha=self.src.count == 3 and self.src.nodata is not None,
        )
        self.alpha_band = self._find_alpha()

    def _native_gsd(self) -> float:
        """True ground resolution in metres (fixes EPSG:3857 scale distortion)."""
        rx = abs(self.src.res[0])
        if self.src.crs.is_geographic:
            lat = (self.lonlat_bounds[1] + self.lonlat_bounds[3]) / 2
            return rx * 111320 * math.cos(math.radians(lat))
        if self.src.crs.to_epsg() == 3857:
            lat = (self.lonlat_bounds[1] + self.lonlat_bounds[3]) / 2
            return rx * math.cos(math.radians(lat))
        return rx

    def _find_alpha(self):
        ci = [c.name for c in self.vrt.colorinterp]
        if "alpha" in ci:
            return ci.index("alpha") + 1
        if self.vrt.count >= 4:
            return 4
        return None

    def window(self, left: float, top: float, size_px: int):
        from rasterio.windows import Window

        col = int(round((left - self.bounds[0]) / self.gsd))
        row = int(round((self.bounds[3] - top) / self.gsd))
        return Window(col, row, size_px, size_px)

    def _read_padded(self, bands, win, width_px, height_px=None):
        """WarpedVRT forbids boundless reads, so clip the window and zero-pad."""
        from rasterio.windows import Window

        height_px = height_px or width_px
        c0, r0 = int(win.col_off), int(win.row_off)
        c1, r1 = c0 + width_px, r0 + height_px
        cc0, rr0 = max(c0, 0), max(r0, 0)
        cc1, rr1 = min(c1, self.vrt.width), min(r1, self.vrt.height)
        n = len(bands) if isinstance(bands, (list, tuple)) else 1
        out = np.zeros((n, height_px, width_px), np.uint8)
        if cc1 > cc0 and rr1 > rr0:
            data = self.vrt.read(bands, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0),
                                 resampling=Resampling.bilinear)
            if data.ndim == 2:
                data = data[None]
            out[:, rr0 - r0: rr1 - r0, cc0 - c0: cc1 - c0] = data
        return out

    def read(self, left: float, top: float, size_px: int, height_px: int | None = None):
        """RGB (H, W, 3) uint8 and valid (H, W) bool for a window whose top-left
        corner is (left, top) in the UTM grid."""
        h = height_px or size_px
        win = self.window(left, top, size_px)
        rgb = self._read_padded([1, 2, 3], win, size_px, h)
        valid = np.ones((h, size_px), bool)
        if self.alpha_band:
            a = self._read_padded([self.alpha_band], win, size_px, h)[0]
            valid &= a > 0
        if self.vrt.nodata is not None:
            valid &= ~np.all(rgb == self.vrt.nodata, axis=0)
        valid &= ~np.all(rgb <= 1, axis=0)  # black collars
        valid &= ~np.all(rgb >= 254, axis=0)  # white collars
        rgb = np.ascontiguousarray(np.moveaxis(rgb, 0, -1).astype(np.uint8))
        return rgb, valid

    def read_overview(self, max_px: int = 1024):
        """Low-res whole-scene read for coverage masks (uses COG overviews)."""
        scale = max(self.grid.width, self.grid.height) / max_px
        w, h = max(1, int(self.grid.width / scale)), max(1, int(self.grid.height / scale))
        rgb = self.vrt.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        valid = ~np.all(rgb <= 1, axis=0)
        if self.alpha_band:
            a = self.vrt.read(self.alpha_band, out_shape=(h, w), resampling=Resampling.nearest)
            valid &= a > 0
        return valid, scale * self.gsd

    def close(self):
        self.vrt.close()
        self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
