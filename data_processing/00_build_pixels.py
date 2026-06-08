"""
01_preprocess_v2.py
修复版：用 rasterio.transform.rowcol() 从各数据集自身 transform 直接反算行列号，
不再手写南起/北起公式，彻底消除坐标翻转 bug。

输出文件（均在 data/processed/ 下）：
  pixels.pkl              城市像素基础信息（行列号已修正）
  ghs_{var}_{epoch}.pkl   GHS单文件提取结果（11014行）
  edgar_2000_2020.pkl     EDGAR逐年CO2
  gdp_2000_2020.pkl       GDP逐年
  panel_v1.pkl            最终面板
"""

import os, sys, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import rowcol
import netCDF4 as nc
import warnings

from pathlib import Path
ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

t0 = time.time()

# ── 路径 ─────────────────────────────────────────────────────
BASE_DB   = 'F:/Database'  # TODO: set to your local raw-data directory
# OUT_DIR set above via pathlib
os.makedirs(OUT_DIR, exist_ok=True)

UCDB_CSV  = f'{BASE_DB}/GHSL/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv'
EDGAR_DIR = f'{BASE_DB}/Emission/EDGAR_total2025/TOTALS_emi_nc'
GDP_TIF   = f'{BASE_DB}/GDP/rast_gdpTot_1990_2022_5arcmin.tif'

GHS_FILES = {
    'built_v': {yr: f'{BASE_DB}/GHSL/GHS_BUILT_V_E{yr}_GLOBE_R2023A_54009_1000_V1_0/GHS_BUILT_V_E{yr}_GLOBE_R2023A_54009_1000_V1_0.tif'
                for yr in [2000, 2005, 2010, 2015, 2020]},
    'built_s': {yr: f'{BASE_DB}/GHSL/GHS_BUILT_S_E{yr}_GLOBE_R2023A_54009_1000_V1_0/GHS_BUILT_S_E{yr}_GLOBE_R2023A_54009_1000_V1_0.tif'
                for yr in [2000, 2005, 2010, 2015, 2020]},
    'pop':     {yr: f'{BASE_DB}/Pop/ghs-pop/GHS_POP_E{yr}_GLOBE_R2023A_54009_1000_V1_0.tif'
                for yr in [2000, 2005, 2010, 2015, 2020]},
}

YEARS     = list(range(2000, 2021))
GDP_YEARS = list(range(1990, 2023))

TARGET_CRS       = CRS.from_epsg(4326)
TARGET_WIDTH     = 3600
TARGET_HEIGHT    = 1800
TARGET_TRANSFORM = rasterio.transform.from_bounds(-180, -90, 180, 90, TARGET_WIDTH, TARGET_HEIGHT)

GS_REGIONS = [
    'Western Africa','Eastern Africa','Middle Africa','Southern Africa','Northern Africa',
    'South America','Central America','Caribbean',
    'South-Central Asia','South-Eastern Asia',
    'Western Asia','Eastern Asia'
]
EXCL_ISO = ['JPN','KOR','ISR','SAU','ARE','KWT','QAT','BHR','OMN','SGP','BRN']

# ══════════════════════════════════════════════════════════════
# Step 1: 城市像素
# ══════════════════════════════════════════════════════════════
out_pixels = f'{OUT_DIR}/pixels.pkl'
if os.path.exists(out_pixels):
    print('Step 1: pixels.pkl exists, loading...')
    pixels = pd.read_pickle(out_pixels)
else:
    print('Step 1: Building city pixel list...')
    df = pd.read_csv(UCDB_CSV, encoding='latin1', low_memory=False)
    gs = df[df['GRGN_L2'].isin(GS_REGIONS) & ~df['CTR_MN_ISO'].isin(EXCL_ISO)].copy()
    gs = gs.dropna(subset=['GCPNT_LAT','GCPNT_LON'])
    gs['pixel_lat'] = np.floor(gs['GCPNT_LAT'] / 0.1) * 0.1
    gs['pixel_lon'] = np.floor(gs['GCPNT_LON'] / 0.1) * 0.1
    pixels = (gs.groupby(['pixel_lat','pixel_lon'])
                .agg(n_cities=('UC_NM_MN','count'),
                     region=('GRGN_L2', lambda x: x.value_counts().index[0]))
                .reset_index())

    # 用 TARGET_TRANSFORM 直接反算行列号，无需关心北起/南起
    pix_lat_c = pixels['pixel_lat'].values + 0.05   # 像素中心纬度
    pix_lon_c = pixels['pixel_lon'].values + 0.05   # 像素中心经度
    r, c = rowcol(TARGET_TRANSFORM, xs=pix_lon_c, ys=pix_lat_c)
    pixels['row'] = np.array(r).clip(0, TARGET_HEIGHT - 1)
    pixels['col'] = np.array(c).clip(0, TARGET_WIDTH  - 1)

    pixels.to_pickle(out_pixels)
    print(f'  Saved pixels.pkl  ({len(pixels)} pixels)')

rows = pixels['row'].values
cols = pixels['col'].values

# ══════════════════════════════════════════════════════════════
# Step 2: GHS — 每个文件独立处理输出
# ══════════════════════════════════════════════════════════════
print('\nStep 2: Processing GHS files...')

def warp_and_extract(src_path, rows, cols):
    with rasterio.open(src_path) as src:
        data = src.read(1, masked=True).astype(np.float64)
        data = data.filled(0.0)   # nodata -> 0，不参与求和
        dst_arr = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float64)
        reproject(
            source=data,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            dst_nodata=0.0,
            resampling=Resampling.sum,
        )
    return dst_arr[rows, cols]

for var, epoch_files in GHS_FILES.items():
    for epoch, fpath in epoch_files.items():
        out_path = f'{OUT_DIR}/ghs_{var}_{epoch}.pkl'
        if os.path.exists(out_path):
            print(f'  {var} {epoch}: exists, skip')
            continue
        print(f'  {var} {epoch}...', end=' ', flush=True)
        t = time.time()
        values = warp_and_extract(fpath, rows, cols)
        df_out = pixels[['pixel_lat','pixel_lon']].copy()
        df_out['value'] = values
        df_out.to_pickle(out_path)
        print(f'{time.time()-t:.1f}s  -> {out_path}')

# ══════════════════════════════════════════════════════════════
# Step 3: EDGAR
# ══════════════════════════════════════════════════════════════
out_edgar = f'{OUT_DIR}/edgar_2000_2020.pkl'
if os.path.exists(out_edgar):
    print('\nStep 3: edgar_2000_2020.pkl exists, loading...')
    edgar_df = pd.read_pickle(out_edgar)
else:
    print('\nStep 3: Extracting EDGAR CO2...')
    # 读取 EDGAR 自身的经纬度坐标，用 rowcol 反算行列号，不手动翻转
    fpath0 = f'{EDGAR_DIR}/EDGAR_2025_GHG_CO2_2000_TOTALS_emi.nc'
    ds0 = nc.Dataset(fpath0)
    edgar_lats = np.array(ds0.variables['lat'][:])
    edgar_lons = np.array(ds0.variables['lon'][:])
    ds0.close()
    edgar_transform = rasterio.transform.from_bounds(
        edgar_lons[0] - (edgar_lons[1]-edgar_lons[0])/2,
        edgar_lats[0] - (edgar_lats[1]-edgar_lats[0])/2,
        edgar_lons[-1] + (edgar_lons[-1]-edgar_lons[-2])/2,
        edgar_lats[-1] + (edgar_lats[-1]-edgar_lats[-2])/2,
        len(edgar_lons), len(edgar_lats)
    )
    er, ec = rowcol(edgar_transform,
                    xs=pixels['pixel_lon'].values + 0.05,
                    ys=pixels['pixel_lat'].values + 0.05)
    edgar_rows = np.array(er).clip(0, len(edgar_lats) - 1)
    edgar_cols = np.array(ec).clip(0, len(edgar_lons) - 1)

    records = []
    for yr in YEARS:
        fpath = f'{EDGAR_DIR}/EDGAR_2025_GHG_CO2_{yr}_TOTALS_emi.nc'
        ds = nc.Dataset(fpath)
        emi = np.array(ds.variables['emissions'][:])
        ds.close()
        records.append(pd.Series(emi[edgar_rows, edgar_cols] / 1e3, name=yr))
    edgar_df = pd.concat(records, axis=1)
    edgar_df.index = pixels.index
    edgar_df.to_pickle(out_edgar)
    print(f'  Saved edgar_2000_2020.pkl  shape={edgar_df.shape}')

# ══════════════════════════════════════════════════════════════
# Step 4: GDP
# ══════════════════════════════════════════════════════════════
out_gdp = f'{OUT_DIR}/gdp_2000_2020.pkl'
if os.path.exists(out_gdp):
    print('\nStep 4: gdp_2000_2020.pkl exists, loading...')
    gdp_df = pd.read_pickle(out_gdp)
else:
    print('\nStep 4: Extracting GDP...')
    with rasterio.open(GDP_TIF) as src:
        gdp_shape  = src.shape
        gdp_nodata = src.nodata
        gdp_transform = src.transform
        gdp_all = src.read().astype(np.float32)
    if gdp_nodata is not None:
        gdp_all[gdp_all == gdp_nodata] = np.nan

    # 用文件自身 transform 反算，不手写公式
    gr, gc = rowcol(gdp_transform,
                    xs=pixels['pixel_lon'].values + 0.05,
                    ys=pixels['pixel_lat'].values + 0.05)
    gdp_rows = np.array(gr).clip(0, gdp_shape[0] - 1)
    gdp_cols = np.array(gc).clip(0, gdp_shape[1] - 1)

    records = {yr: gdp_all[GDP_YEARS.index(yr), gdp_rows, gdp_cols].astype(np.float64)
               for yr in YEARS}
    gdp_df = pd.DataFrame(records, index=pixels.index)
    gdp_df.to_pickle(out_gdp)
    print(f'  Saved gdp_2000_2020.pkl  shape={gdp_df.shape}')

# ══════════════════════════════════════════════════════════════
# Step 5: 合并面板
# ══════════════════════════════════════════════════════════════
print('\nStep 5: Merging into panel...')

GHS_EPOCHS = [2000, 2005, 2010, 2015, 2020]

def interp_annual(epoch_vals, years, epochs=GHS_EPOCHS):
    """epoch_vals: dict {epoch: Series(n_pixels)}，返回 dict {year: Series}"""
    result = {}
    for yr in years:
        if yr <= epochs[0]:
            result[yr] = epoch_vals[epochs[0]]
        elif yr >= epochs[-1]:
            result[yr] = epoch_vals[epochs[-1]]
        else:
            e0 = max(e for e in epochs if e <= yr)
            e1 = min(e for e in epochs if e > yr)
            w = (yr - e0) / (e1 - e0)
            result[yr] = (1-w) * epoch_vals[e0] + w * epoch_vals[e1]
    return result

# 加载GHS
ghs_epochs = {}
for var in ['built_v','built_s','pop']:
    ghs_epochs[var] = {ep: pd.read_pickle(f'{OUT_DIR}/ghs_{var}_{ep}.pkl')['value']
                       for ep in GHS_EPOCHS}

ghs_annual = {var: interp_annual(ghs_epochs[var], YEARS) for var in ghs_epochs}

# 组装
pix_area_m2 = (0.1 * 111320) ** 2

records = []
for yr in YEARS:
    row = pixels[['pixel_lat','pixel_lon','n_cities','region']].copy()
    row['year']       = yr
    row['co2_kt']     = edgar_df[yr].values
    row['built_s_m2'] = ghs_annual['built_s'][yr].values
    row['built_v_m3'] = ghs_annual['built_v'][yr].values
    row['pop']        = ghs_annual['pop'][yr].values
    row['gdp_usd']    = gdp_df[yr].values
    records.append(row)

panel = pd.concat(records, ignore_index=True)
panel = panel.sort_values(['pixel_lat','pixel_lon','year']).reset_index(drop=True)

panel['bua_frac']    = panel['built_s_m2'] / pix_area_m2
panel['far_proxy']   = panel['built_v_m3'] / pix_area_m2
panel['pop_density'] = panel['pop'] / (panel['built_s_m2'] / 1e6).replace(0, np.nan)

out_panel = f'{OUT_DIR}/panel_v1.pkl'
panel.to_pickle(out_panel)

print(f'\nPanel shape: {panel.shape}')
print(panel.dtypes)
print(f'\nSaved: {out_panel}  ({os.path.getsize(out_panel)/1e6:.1f} MB)')
print(f'Total time: {(time.time()-t0)/60:.1f} min')
