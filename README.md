# Divergent emission trajectories reveal contrasting roles of outward and upward urban growth in Global South cities

This repository contains the analysis code accompanying the paper. The compiled pixel-level panel dataset is publicly available at https://doi.org/10.6084/m9.figshare.32605185.

## Repository structure

```
├── data_processing/
│   ├── 00_build_pixels.py      # Extract city pixels from GHS-UCDB; reproject EDGAR, MODIS, and building height to 0.1° grid
│   └── 01_build_panel.py       # Assemble annual pixel-year panel dataset
│
├── analysis/
│   ├── kmeans_trajectory.py    # K-means clustering of emission trajectories (k = 4)
│   ├── reg_by_cluster.py       # Two-way FE panel regressions by trajectory cluster
│   ├── reg_by_region.py        # Two-way FE panel regressions by UN sub-region
│   └── reg_by_agglom.py        # Two-way FE panel regressions by urban agglomeration
│
└── figures/
    ├── utils.py                # Shared plotting utilities
    ├── fig1_clusters.py        # Fig. 1 — Cluster feature box plots and geographic distribution
    ├── fig_a1_clusters.py      # Fig. A1 — Pairwise feature scatter plots and k-selection elbow curve
    ├── fig2_composite.py       # Fig. 2 — Cluster-level emission trajectories and regression coefficients
    ├── fig3_regional.py        # Fig. 3 — Regional shifts in morphological predictors
    ├── fig3_arrow_schematic.py # Fig. 3 — Cardinal direction schematic panel
    ├── fig4_attrib_map.py      # Fig. 4 — Pixel-level attributed emission contribution maps
    └── fig4_attrib_legend.py   # Fig. 4 — Bivariate legend
```

## Reproducing the analysis

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare raw data

Download the following datasets and place them under a local database directory (paths are configured at the top of each script):

| Dataset | Source |
|---------|--------|
| EDGAR v2025 gridded CO₂ | https://edgar.jrc.ec.europa.eu/ |
| MODIS MCD12Q1 (Land Cover) | https://lpdaac.usgs.gov/ |
| Building height (Yu et al.) | [REPO URL] |
| GHS-UCDB R2019A | https://ghsl.jrc.ec.europa.eu/ |
| GHS-POP R2023A | https://ghsl.jrc.ec.europa.eu/ |

Alternatively, the compiled panel dataset can be downloaded directly from Figshare (https://doi.org/10.6084/m9.figshare.32605185), in which case steps 3–4 can be skipped.

### 3. Build pixel index and panel

```bash
python data_processing/00_build_pixels.py
python data_processing/01_build_panel.py
```

### 4. Run analysis

```bash
python analysis/kmeans_trajectory.py
python analysis/reg_by_cluster.py
python analysis/reg_by_region.py
python analysis/reg_by_agglom.py
```

### 5. Generate figures

```bash
python figures/fig1_clusters.py
python figures/fig_a1_clusters.py
python figures/fig2_composite.py
python figures/fig3_regional.py
python figures/fig3_arrow_schematic.py
python figures/fig4_attrib_map.py
python figures/fig4_attrib_legend.py
```

## License

Code is released under the MIT License.
