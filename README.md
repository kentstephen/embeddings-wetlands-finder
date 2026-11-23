# embeddings-wetlands-finder

Tests of how well yearly 10 m satellite embeddings find wetlands, alone and with lidar.

Three embeddings are compared on the same pixels, all read from Source Cooperative: AlphaEarth Foundations
(64 numbers a pixel), TESSERA v1.1 (128) and OlmoEarth v1.3 (128). Labels are National Wetlands Inventory (NWI)
polygons from the USFWS map service. Lidar is USGS 3DEP: the 1 m elevation models and the raw point clouds.

Three 10 km squares in different landscapes:

| Square | Landscape | NWI photo year |
|---|---|---|
| Delmarva (Maryland and Delaware) | coastal plain, forested depressions among crop fields | 2019 |
| Stutsman County, North Dakota | prairie potholes | 2019 |
| Lee, New Hampshire | glaciated forest, swamps under tree canopy | 2015 |

Every score is on ground the model was not taught on: each square is cut into 1 km blocks and the blocks are
dealt into folds. Open water is left out of both sides, so the question is vegetated wetland against upland.

## Scripts

Each script is self-contained (`uv run scripts/<name>.py`), reads everything over the network at run time and
caches nothing.

- `scripts/embedding_signal.py`: a linear probe on each embedding. Wetland against upland, the same within 30 m
  of the NWI line, forested wetland against upland forest (tree cover only, ESA WorldCover 2021), emergent
  wetland against upland in low cover, and taught on two squares and scored on the third. About 2 minutes,
  7 GB peak.
- `scripts/terrain_adds.py`: eleven terrain numbers per pixel from the 3DEP 1 m elevation model, and a small
  head (one hidden layer of 64) on terrain alone, each embedding alone, and each embedding with terrain.
  About 15 minutes, 8 GB peak.
- `scripts/lpc_adds.py`: every point of the 3DEP point cloud over each square (about 6 billion points, 42 GB
  over the network) folded into per-pixel numbers (density, returns, intensity, height above ground), then the
  same head with the points added.
- `scripts/ept.py`: a small reader for the USGS public Entwine Point Tile clouds.

## Results so far

Linear probe on the embedding alone, AUC:

| Test | Square | AlphaEarth | TESSERA | OlmoEarth |
|---|---|---|---|---|
| Vegetated wetland vs upland | Delmarva | .950 | .952 | .952 |
| | Stutsman | .970 | .979 | .981 |
| | Lee | .953 | .944 | .966 |
| Within 30 m of the NWI line | Delmarva | .841 | .859 | .864 |
| | Stutsman | .888 | .922 | .922 |
| | Lee | .833 | .861 | .881 |
| Forested wetland vs upland forest | Delmarva | .868 | .878 | .873 |
| | Lee | .943 | .933 | .956 |
| Taught on the other two squares | Delmarva | .915 | .561 | .942 |
| | Stutsman | .845 | .852 | .854 |
| | Lee | .897 | .667 | .934 |

Small head, with 1 m lidar terrain added, AUC:

| Test | Square | Terrain alone | Best embedding alone | Best embedding + terrain |
|---|---|---|---|---|
| Vegetated wetland vs upland | Delmarva | .932 | .954 | .961 |
| | Stutsman | .976 | .981 | .986 |
| | Lee | .954 | .965 | .971 |
| Within 30 m of the NWI line | Delmarva | .823 | .862 | .879 |
| | Stutsman | .912 | .926 | .942 |
| | Lee | .859 | .882 | .896 |
| Forested wetland vs upland forest | Delmarva | .873 | .877 | .895 |
| | Lee | .946 | .956 | .962 |

![NWI beside the held-out probability from the AlphaEarth probe](figures/embedding_signal.png)

## What these numbers do not show

- The labels are NWI, so every score is agreement with NWI. A wetland NWI missed counts as upland here.
  Nothing has been scored against a field delineation.
- Three squares. The scores differ by landscape and a model taught in one does not carry cleanly to another.
- NWI's line is least exact at the boundary, which is also where the scores are lowest.
