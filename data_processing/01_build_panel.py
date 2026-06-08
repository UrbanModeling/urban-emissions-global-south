"""
02_build_panel_v3.py  --  panel v3

Changes from v2: Urban built-up area now uses a precomputed mapping table,
bypassing the full-grid reproject entirely.
  - build_urban_mapping(): one-time computation of which MODIS 500m source pixels
    map into each 0.1 deg target cell; result cached to urban_mapping.pkl
  - urban_extract_fast(): per year, only a windowed read + np.bincount (no reproject)
  - Coordinate transform uses closed-form NumPy formula (no pyproj call)

All other variables (EDGAR, height, NTL) are identical to v2.
Output panel saved as panel_v3.pkl; cache written to cache_v3/.
"""

import os, time, pickle
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window

import netCDF4 as nc

from pathlib import Path
ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# -- Paths --------------------------------------------------------------------
BASE_DB    = 'F:/Database'  # TODO: set to your local raw-data directory
# OUT_DIR set above via pathlib
V_DIR      = f'{OUT_DIR}/cache_v3'
os.makedirs(V_DIR, exist_ok=True)

EDGAR_DIR  = f'{BASE_DB}/Emission/EDGAR_total2025/TOTALS_emi_nc'
NTL_DIR    = f'{BASE_DB}/NTT_2026'
HEIGHT_DIR = f'{BASE_DB}/height/height_2026/27650880'
URBAN_DIR  = f'{BASE_DB}/Landcover/GEE_exports2/GEE_exports'

YEARS = list(range(2001, 2019))   # 2001-2018

# -- Target grid (0.1 deg WGS84) ----------------------------------------------
TARGET_CRS       = CRS.from_epsg(4326)
TARGET_WIDTH     = 3600
TARGET_HEIGHT    = 1800
TARGET_TRANSFORM = from_bounds(-180, -90, 180, 90, TARGET_WIDTH, TARGET_HEIGHT)

# MODIS Sinusoidal sphere radius (matches file metadata)
SINU_R = 6371007.181

# -- Load pixel list ----------------------------------------------------------
pixels = pd.read_pickle(f'{OUT_DIR}/pixels_v3.pkl')
rows   = pixels['row'].values
cols   = pixels['col'].values
n_pix  = len(pixels)
print(f'Pixels: {n_pix}  Years: {YEARS[0]}-{YEARS[-1]}  Total rows: {n_pix*len(YEARS)}')

# =============================================================================
# Helper functions
# =============================================================================

def edgar_extract(year, pixels):
    """Compute row indices from lat difference to avoid north-south flip bug."""
    fpath = f'{EDGAR_DIR}/EDGAR_2025_GHG_CO2_{year}_TOTALS_emi.nc'
    ds   = nc.Dataset(fpath)
    emi  = np.array(ds.variables['emissions'][:])
    lats = np.array(ds.variables['lat'][:])
    lons = np.array(ds.variables['lon'][:])
    ds.close()
    lat_step = lats[1] - lats[0]
    lon_step = lons[1] - lons[0]
    if year == YEARS[0]:
        print(f'    EDGAR lat {"asc" if lat_step>0 else "desc"}  step={lat_step:.4f}  lats[0]={lats[0]:.4f}')
    e_rows = np.round((pixels['pixel_lat'].values + 0.05 - lats[0]) / lat_step).astype(int).clip(0, len(lats)-1)
    e_cols = np.round((pixels['pixel_lon'].values + 0.05 - lons[0]) / lon_step).astype(int).clip(0, len(lons)-1)
    return emi[e_rows, e_cols] / 1e3


def _sinu_to_wgs84(x, y):
    """MODIS Sinusoidal (meters) -> WGS84 (degrees); pure NumPy closed-form formula."""
    lat_rad = y / SINU_R
    cos_lat = np.cos(lat_rad)
    lon_rad = np.where(np.abs(cos_lat) > 1e-10, x / (SINU_R * cos_lat), 0.0)
    return np.degrees(lon_rad), np.degrees(lat_rad)


def _wgs84_to_sinu(lon_deg, lat_deg):
    """WGS84 (degrees) -> MODIS Sinusoidal (meters); used to determine source window."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = lon_rad * SINU_R * np.cos(lat_rad)
    y = lat_rad * SINU_R
    return x, y


def build_urban_mapping(pixels, cache_path):
    """
    One-time precomputation: map each 0.1 deg target cell to the flat indices of
    MODIS 500m source pixels that fall within it.
    Result cached to cache_path; subsequent calls just load the cache.

    Returns dict:
      'tiles': list of (None | dict{'window', 'src_idx', 'tgt_idx'}) for tile1, tile2
      'n_pix': int
    """
    if os.path.exists(cache_path):
        print('  Loading cached mapping...')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    t_map = time.time()
    print('  Building mapping (first run only; cached afterwards)...')

    # Fast lookup: 2D array, value = pixel index in pixels df (-1 = not a target)
    lookup = np.full((TARGET_HEIGHT, TARGET_WIDTH), -1, dtype=np.int32)
    for i, (r, c) in enumerate(zip(pixels['row'].values, pixels['col'].values)):
        lookup[r, c] = i

    # Target bounding box (WGS84, with 0.2 deg buffer to avoid edge gaps)
    lat_max = 90.0 - pixels['row'].min() * 0.1 + 0.2
    lat_min = 90.0 - (pixels['row'].max() + 1) * 0.1 - 0.2
    lon_min = pixels['col'].min() * 0.1 - 180.0 - 0.2
    lon_max = (pixels['col'].max() + 1) * 0.1 - 180.0 + 0.2

    # Sample bbox boundary points -> Sinusoidal to determine source window
    # (bbox is non-rectangular under the oblique projection)
    n_edge = 200
    lons_b = np.concatenate([np.linspace(lon_min, lon_max, n_edge),
                              np.full(n_edge, lon_max),
                              np.linspace(lon_max, lon_min, n_edge),
                              np.full(n_edge, lon_min)])
    lats_b = np.concatenate([np.full(n_edge, lat_max),
                              np.linspace(lat_max, lat_min, n_edge),
                              np.full(n_edge, lat_min),
                              np.linspace(lat_min, lat_max, n_edge)])
    xs_b, ys_b = _wgs84_to_sinu(lons_b, lats_b)

    tile_paths = [
        f'{URBAN_DIR}/Global_Urban_2001-0000000000-0000000000.tif',
        f'{URBAN_DIR}/Global_Urban_2001-0000000000-0000065536.tif',
    ]
    tile_data_list = []

    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            tf     = src.transform   # a>0, e<0
            bounds = src.bounds

            # Source window: intersection of Sinusoidal bbox with tile extent
            x0 = max(xs_b.min() - 500, bounds.left)
            x1 = min(xs_b.max() + 500, bounds.right)
            y0 = max(ys_b.min() - 500, bounds.bottom)
            y1 = min(ys_b.max() + 500, bounds.top)

            if x0 >= x1 or y0 >= y1:
                tile_data_list.append(None)
                print(f'  {os.path.basename(tile_path)}: no overlap, skipping')
                continue

            col_off = max(0, int((x0 - bounds.left) / tf.a))
            col_end = min(src.width,  int((x1 - bounds.left) / tf.a) + 2)
            row_off = max(0, int((bounds.top - y1) / (-tf.e)))
            row_end = min(src.height, int((bounds.top - y0) / (-tf.e)) + 2)
            win_w = col_end - col_off
            win_h = row_end - row_off

            print(f'  {os.path.basename(tile_path)}: window {win_w}x{win_h} = {win_w*win_h/1e6:.0f}M pixels')

            # Column x-coordinates (fixed, computed once)
            src_c_arr = np.arange(col_off, col_end)
            sx = tf.c + (src_c_arr + 0.5) * tf.a   # (win_w,)

            CHUNK = 200   # rows per chunk; ~160 MB/chunk memory budget
            all_src, all_tgt = [], []

            for r0 in range(0, win_h, CHUNK):
                r1 = min(r0 + CHUNK, win_h)
                chunk_h = r1 - r0

                src_r_arr = np.arange(row_off + r0, row_off + r1)
                sy = tf.f + (src_r_arr + 0.5) * tf.e   # (chunk_h,), tf.e<0

                # Broadcast to chunk_h x win_w coordinate matrix, then flatten
                XX = np.tile(sx, chunk_h)        # (chunk_h * win_w,)
                YY = np.repeat(sy, win_w)        # (chunk_h * win_w,)

                lons, lats = _sinu_to_wgs84(XX, YY)

                tgt_r = np.floor((90.0 - lats) / 0.1).astype(np.int32)
                tgt_c = np.floor((lons + 180.0) / 0.1).astype(np.int32)

                in_bounds = ((tgt_r >= 0) & (tgt_r < TARGET_HEIGHT) &
                             (tgt_c >= 0) & (tgt_c < TARGET_WIDTH))
                if not in_bounds.any():
                    continue

                tgt_idx = np.full(len(tgt_r), -1, dtype=np.int32)
                tgt_idx[in_bounds] = lookup[tgt_r[in_bounds], tgt_c[in_bounds]]

                valid = tgt_idx >= 0
                if not valid.any():
                    continue

                # Flat index within the window
                local_row = np.repeat(np.arange(r0, r1), win_w)
                flat_idx  = local_row * win_w + np.tile(np.arange(win_w), chunk_h)

                all_src.append(flat_idx[valid].astype(np.int32))
                all_tgt.append(tgt_idx[valid])

            src_arr = np.concatenate(all_src) if all_src else np.array([], dtype=np.int32)
            tgt_arr = np.concatenate(all_tgt) if all_tgt else np.array([], dtype=np.int32)

            tile_data_list.append({
                'window' : Window(col_off, row_off, win_w, win_h),
                'src_idx': src_arr,
                'tgt_idx': tgt_arr,
            })
            print(f'    -> {len(src_arr):,} valid mappings')

    mapping = {'tiles': tile_data_list, 'n_pix': n_pix}
    with open(cache_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f'  Mapping cached ({time.time()-t_map:.0f}s) -> {cache_path}')
    return mapping


def urban_extract_fast(year, mapping):
    """
    Urban extraction using precomputed mapping: windowed read + bincount; no reproject.
    Per-year cost = disk read time (LZW decompress) + O(n_nonzero) NumPy ops.
    """
    n_pix  = mapping['n_pix']
    result = np.zeros(n_pix, dtype=np.float64)

    tile_paths = [
        f'{URBAN_DIR}/Global_Urban_{year}-0000000000-0000000000.tif',
        f'{URBAN_DIR}/Global_Urban_{year}-0000000000-0000065536.tif',
    ]

    for tile_path, tile_info in zip(tile_paths, mapping['tiles']):
        if tile_info is None:
            continue
        with rasterio.open(tile_path) as src:
            data = src.read(1, window=tile_info['window']).ravel()

        src_idx = tile_info['src_idx']
        tgt_idx = tile_info['tgt_idx']
        if len(src_idx):
            result += np.bincount(tgt_idx,
                                  weights=data[src_idx].astype(np.float64),
                                  minlength=n_pix)

    return result * 250000   # pixel count -> m2


def height_extract(year, rows, cols):
    """0.05 deg WGS84 -> 0.1 deg average; NaN-safe."""
    fpath = f'{HEIGHT_DIR}/{year}_esti_mean_height.tif'
    with rasterio.open(fpath) as src:
        data  = src.read(1).astype(np.float64)
        valid = (~np.isnan(data)).astype(np.float64)
        data  = np.where(np.isnan(data), 0.0, data)
        dst_val = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float64)
        dst_cnt = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float64)
        reproject(source=data,  destination=dst_val,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=TARGET_TRANSFORM, dst_crs=TARGET_CRS,
                  dst_nodata=0.0, resampling=Resampling.average)
        reproject(source=valid, destination=dst_cnt,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=TARGET_TRANSFORM, dst_crs=TARGET_CRS,
                  dst_nodata=0.0, resampling=Resampling.sum)
    vals = dst_val[rows, cols]
    vals[dst_cnt[rows, cols] == 0] = np.nan
    return vals


def ntl_extract(year, rows, cols):
    """NTL is already 0.1 deg WGS84; average resampling."""
    fpath = f'{NTL_DIR}/nppviirs_like_V2_{year}.tif'
    with rasterio.open(fpath) as src:
        dst = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst,
                  dst_transform=TARGET_TRANSFORM, dst_crs=TARGET_CRS,
                  dst_nodata=0.0, resampling=Resampling.average)
        dst = np.where(dst < 0, 0.0, dst)
    return dst[rows, cols]


# =============================================================================
# Extract each variable by year; skip if cached
# =============================================================================

# Step 1: EDGAR CO2
print('\nStep 1: EDGAR CO2...')
for yr in YEARS:
    out = f'{V_DIR}/edgar_{yr}.pkl'
    if os.path.exists(out):
        print(f'  {yr}: skip'); continue
    t = time.time()
    vals = edgar_extract(yr, pixels)
    pd.Series(vals).to_pickle(out)
    print(f'  {yr}: {time.time()-t:.1f}s  median={np.median(vals[vals>0]):.3g} kt  max={vals.max():.1f} kt')

# Step 2: Urban built-up area (precomputed mapping -> windowed read + bincount per year)
print('\nStep 2: Urban...')
urban_mapping = build_urban_mapping(pixels, f'{V_DIR}/urban_mapping.pkl')

for yr in YEARS:
    out = f'{V_DIR}/urban_{yr}.pkl'
    if os.path.exists(out):
        print(f'  {yr}: skip'); continue
    t = time.time()
    vals = urban_extract_fast(yr, urban_mapping)
    pd.Series(vals).to_pickle(out)
    print(f'  {yr}: {time.time()-t:.1f}s  max={vals.max():.3g} m2')

# Step 3: Building height
print('\nStep 3: Height...')
for yr in YEARS:
    out = f'{V_DIR}/height_{yr}.pkl'
    if os.path.exists(out):
        print(f'  {yr}: skip'); continue
    t = time.time()
    vals = height_extract(yr, rows, cols)
    pd.Series(vals).to_pickle(out)
    print(f'  {yr}: {time.time()-t:.1f}s  NaN={np.isnan(vals).mean()*100:.1f}%  max={np.nanmax(vals):.3f}m')

# Step 4: NTL
print('\nStep 4: NTL...')
for yr in YEARS:
    out = f'{V_DIR}/ntl_{yr}.pkl'
    if os.path.exists(out):
        print(f'  {yr}: skip'); continue
    t = time.time()
    vals = ntl_extract(yr, rows, cols)
    pd.Series(vals).to_pickle(out)
    print(f'  {yr}: {time.time()-t:.1f}s  max={vals.max():.2f}')

# =============================================================================
# Step 5: Assemble panel
# =============================================================================
print('\nStep 5: Assembling panel...')
pix_area_m2 = (0.1 * 111320) ** 2

records = []
for yr in YEARS:
    r = pixels[['city_id','city_name','country','region','p20','pixel_lat','pixel_lon','row','col']].copy()
    r['year']      = yr
    r['co2_kt']    = pd.read_pickle(f'{V_DIR}/edgar_{yr}.pkl').values
    r['urban_m2']  = pd.read_pickle(f'{V_DIR}/urban_{yr}.pkl').values
    r['height_m']  = pd.read_pickle(f'{V_DIR}/height_{yr}.pkl').values
    r['ntl']       = pd.read_pickle(f'{V_DIR}/ntl_{yr}.pkl').values
    records.append(r)
    print(f'  {yr}: loaded')

panel = pd.concat(records, ignore_index=True)
panel = panel.sort_values(['pixel_lat','pixel_lon','year']).reset_index(drop=True)
panel['urban_frac'] = panel['urban_m2'] / pix_area_m2   # built-up fraction

out_panel = f'{OUT_DIR}/panel_v3.pkl'
panel.to_pickle(out_panel)

print(f'\nPanel shape: {panel.shape}')
print(f'Cities: {panel["city_id"].nunique()}  Pixels: {n_pix}  Years: {len(YEARS)}')
p2018 = panel[panel.year == 2018]
print('\nVariable stats (2018):')
for c in ['co2_kt','urban_m2','height_m','ntl']:
    s = p2018[c]
    print(f'  {c:12s}  zero={( s==0).mean()*100:4.1f}%  NaN={s.isna().mean()*100:4.1f}%  median={s.median():.4g}  max={s.max():.4g}')

# Spot-check Shanghai 2018
sh = panel[(panel['city_name'].str.contains('Shanghai', na=False)) & (panel['year']==2018)]
if len(sh):
    print('\nShanghai 2018 check:')
    print(sh[['pixel_lat','pixel_lon','co2_kt','urban_m2','height_m','ntl']].to_string(index=False))

print(f'\nSaved: {out_panel}  ({os.path.getsize(out_panel)/1e6:.1f} MB)')
print(f'Total: {(time.time()-t0)/60:.1f} min')
