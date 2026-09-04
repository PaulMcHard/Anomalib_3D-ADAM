# 3D-ADAM Benchmarks

Benchmarking suite for running [anomalib](https://github.com/open-edge-platform/anomalib) anomaly detection models against **3D-ADAM**, a multimodal (RGB + depth) industrial anomaly detection dataset, with MVTec AD / MVTec 3D-AD scripts included for baseline comparison.

Dataset: [huggingface.co/datasets/pmchard/3D-ADAM](https://huggingface.co/datasets/pmchard/3D-ADAM) — released under CC BY-NC-SA 4.0.

## Repository structure

```
adam3d/                    Anomalib DataModule/Dataset classes for 3D-ADAM
  adam_3d_datamodule.py    ADAM3D LightningDataModule (mirrors anomalib's MVTec3D)
  adam_3d_dataset.py       ADAM3DDataset + sample-list parsing for the on-disk layout

*_adam.py                  3D-ADAM benchmark entry points (PaDiM, CFA, PatchCore)

scripts/
  mvtec/                   Equivalent benchmark scripts against MVTec AD / MVTec 3D-AD
  adam/2d/                 2D-only benchmarks against 3D-ADAM (PatchCore, UniNet, Dinomaly)

utils/                     Dataset preparation, reorganization, and cleanup scripts
```

### Expected dataset layout

`ADAM3DDataset` expects each category to follow the MVTec 3D-AD convention:

```
<root>/<category>/<split>/<label>/rgb/<file>.png
<root>/<category>/<split>/<label>/xyz/<file>.tiff
<root>/<category>/test/<label>/ground_truth/<file>.png   # anomalous test samples only
```

where `split` is `train` or `test`, and `label` is `good` for normal samples or the defect type for anomalous ones. The 3D-ADAM categories are:

`1m1, 1m2, 1m3, 2m1, 2m2h, 2m2m, 3m1, 3m2, 3m2c, 4m1, 4m2, 4m2c, gripper_closed, gripper_open, helicalgear1, helicalgear2, rackgear, spiralgear, spurgear, tapa2m1, tapa3m1, tapa4m1, tapatbb`

## Setup

Requires Python with CUDA-enabled PyTorch (see `requirements.txt`, pinned to `torch==2.8.0+cu126`).

```bash
pip install -r requirements.txt
```

## Running a benchmark

Each benchmark script iterates over every category directory under a dataset root, trains and tests a model per category via anomalib's `Engine`, and writes aggregated results to a JSON log.

```bash
python PaDiM3D_adam.py
python cfa3D_adam.py
python patchcore3D_adam.py
```

Edit the `dataset_base_dir` and `log_file_path` constants at the bottom of each script to point at your local copy of the dataset and the desired results file (written under `scores/` by convention, which is gitignored).

`scripts/mvtec/` contains the same benchmarks run against `MVTec3D`/`MVTecAD` for comparison, and `scripts/adam/2d/` contains 2D-only (RGB-only) benchmarks against 3D-ADAM using anomalib's generic `Folder` datamodule.

## Dataset utilities (`utils/`)

Scripts used to prepare and maintain the dataset on disk:

- **`create_depth_img.py` / `ply_to_tiff_all.py`** — convert `.ply` point clouds to 3-channel `.tiff` depth images for anomalib compatibility.
- **`rgba_converter.py`** — strip the alpha channel from RGBA images (anomalib requires 3-channel input).
- **`consolidate_masks.py` / `new_consolidate_masks.py` / `consolidate_all_masks.py`** — merge per-defect ground-truth masks into the labeller/anomalib-expected format.
- **`shorten_filenames.py` / `new_shorten_filenames.py`** — shorten long capture filenames down to image numbers, disambiguating clashes with instance IDs.
- **`reorg_adam2mvtec.py`** — reorganize a raw 3D-ADAM export into the MVTec 3D-AD directory layout.
- **`reorg_adam_blob.py` / `reorg_dataset_2d.py` / `reorg_dataset_3d.py`** — additional dataset layout transforms (flattening splits, 2D/3D-specific restructuring).
- **`dataset_cleanup.py`** — removes stale `tiff`/`gt` directories, renames `gt_consolidated` to `ground_truth`, and drops redundant `.ply` files once a `.tiff` exists. Supports `--dry-run` (default).
- **`new_copy_nano.py`** — extract a reduced "Nano" subset of the dataset while preserving directory structure.
- **`get_dir_struct.py` / `get_dirs_claude.py` / `get_dirfiles_claude.py`** — dump a directory tree to JSON for inspection/diffing.
- **`compress_json.py`** — gzip+base64 compress/decompress JSON result files.
- **`download.py`** — fetches the MVTec AD dataset via `kagglehub` for baseline comparisons.

Most scripts accept `--dry-run` where destructive; check `--help` on each before running against a full dataset copy.
