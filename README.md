# `SWATPollution` Class

The `SWATPollution` class is designed to manage the setup and optional execution of a SWAT+ simulation for modeling contaminant transport in a specified watershed. It facilitates input preparation, parameter configuration, and integration of observational data for analysis and calibration.

> **This code was developed for the paper:**  
> **[Integrated modeling of the generation, attenuation, and transport of point-source pollutants at the watershed-scale using SWAT+](https://www.sciencedirect.com/science/article/abs/pii/S1364815225003159)**
> **Library used:**  
> [`pyswatplus`](https://github.com/swat-model/pySWATPlus) – Python interface for SWAT+ modeling and file handling.

---

## Features

- Supports multiple predefined watersheds.
- Integrates observational data for calibration and validation.
- Handles temporary file management and execution setup.
- Allows simulation of various contaminants.
- Customizes SWAT+ input files via user-defined modifications.

---

## Parameters

| Parameter | Type | Description |
|----------|------|-------------|
| `conca` | `str` | Name of the watershed. Valid options: `'muga'`, `'fluvia'`, `'ter'`, `'llobregat'`, `'besos'`, `'tordera'`, `'sud'`. |
| `contaminant` | `str` | Name of the pollutant to simulate (e.g., `'Venlafaxina'`, `'Ciprofloxacina'`). |
| `txtinout_folder` | `str` or `Path` | Path to directory containing SWAT `txtinout` files. |
| `channels_geom_path` | `str` or `Path`, optional | Path to the shapefile representing channel geometry. If `None`, a default is used. |
| `tmp_path` | `str` or `Path`, optional | Temporary directory for simulation input/output files. Used only if `run=True`. |
| `run` | `bool`, optional | If `True`, prepares and runs the model. Default: `False`. |
| `compound_features` | `dict`, optional | Dictionary defining SWAT file modifications. Format: `{ 'filename': (id_col, [(id, col, value)]) }`. |
| `show_output` | `bool`, optional | If `True`, prints SWAT run output logs. Default: `True`. |
| `copy_txtinout` | `bool`, optional | If `True`, copies `txtinout` files to `tmp_path` before execution. Used only if `run=True`. |
| `overwrite_txtinout` | `bool`, optional | If `True`, allows overwriting existing files in `tmp_path`. |
| `observacions` | `pd.DataFrame`, optional | Observational data for model calibration. If `None`, defaults are loaded. |
| `lod_path` | `str` or `Path`, optional | Path to file with limits of detection (LOD). If `None`, a default is used. |
| `year_start` | `int`, optional | Start year for the simulation. Default: `2000`. Used only if `run=True`. |
| `year_end` | `int`, optional | End year for the simulation. Default: `2022`. Used only if `run=True`. |
| `warmup` | `int`, optional | Number of warm-up years. Default: `1`. Used only if `run=True`. |

---

## Example Usage

```python
from pathlib import Path
import pandas as pd

contaminant = 'Venlafaxina'
conca = 'besos'

cwd = Path('C:/Users/joans/OneDrive/Escriptori/icra/traca_contaminacio/traca_contaminacio')
txtinout_folder = cwd / 'data' / 'txtinouts' / 'tmp' / contaminant / conca
channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'

# Load observational data
recall_points_path = cwd.parent / 'traca' / 'traca' / 'inputs compound generator' / 'inputs' / 'recall_points.xlsx'
recall_points_df = pd.read_excel(recall_points_path)
edars = recall_points_df[recall_points_df['conca'] == conca][['lat', 'lon', 'edar_code']].dropna()

observacions = generate_pollution_observations(contaminant)
df = observacions_from_conca(channels_geom_path, observacions, conca)

first_observation = df.year.min()
year_end = 2022
year_start = max(first_observation - 3, 2000)
warmup = max(1, first_observation - year_start)

swatpy = SWATPollution(
    conca=conca,
    contaminant=contaminant,
    txtinout_folder=txtinout_folder,
    channels_geom_path=channels_geom_path,
    run=False,
    year_start=year_start,
    year_end=year_end,
    warmup=warmup,
    observacions=observacions
)
