# Rare Disease Classification in Video Capsule Endoscopy Videos

Supplementary code release. This repo reproduces the training, decoding, and table/figure
pipeline reported in the paper: a training-free, video-conditioned decoding method (HCE) that
composes video-level disease-presence conditioning with a macro-F1-optimal per-frame correction
(EGA) to improve rare-class recall on VCE triage, evaluated on Kvasir-Capsule and GALAR.

This is a curated subset of a larger research repo, only the code that produces the reported
numbers is included here (many exploratory scripts and configs from the research process are
not part of this release).

## Data access

- **Kvasir-Capsule** is publicly available. `datasets/kvasir.py`'s `official_split` will
  auto-download the official 2-fold split CSVs; the underlying video frames/images must still be
  obtained from the official Kvasir-Capsule release and placed under `--data-root`.
- **GALAR** is not bundled and not auto-downloadable by this code. It is a separate published
  dataset; obtaining it requires going through its own data access process. Everything here
  (`datasets/galar.py`, `analysis/build_galar_pathology.py`, the GALAR k8s jobs) assumes you
  already have the raw GALAR shards and are pointing `--galar-root`/`--shards-dir` at them.
  **Full end-to-end reproduction is achievable for Kvasir-Capsule; for GALAR, reproduction is
  blocked on obtaining that dataset separately, not on anything in this code.**
- The exact GALAR pathology split we used (2-fold, video-disjoint, fold $k$ tests on set $k$)
  is included under `tables/report/galar_pathology_splits/` (`split_0/{train,val,test}.csv`,
  `split_1/{train,val,test}.csv`, `classes.txt`), each row is `label,path` (a relative
  `video/frame_NNNNNN.PNG` path), no image content. `build_galar_pathology.py` is fully
  deterministic given the same raw GALAR CSVs (every set is sorted before use, the only
  randomness is a seeded RNG), so re-running it should reproduce this exactly, but we ship
  the actual split files too so reproduction doesn't depend on your copy of raw GALAR being
  byte-identical to ours.

## Environment

```
pip install -r requirements.txt
```

Requires a CUDA GPU for training/logit collection at a practical speed; decoding, tables, and
figures are CPU-only.

## Path convention (important)

The decoding script (`analysis/hcg_decode.py`) and several helper scripts use **hardcoded
absolute paths** matching the original cluster's PVC layout (`/pvc/results/...`,
`/pvc/kvasir-capsule`, `/pvc/results/galar_pathology_splits`, etc.), they are not CLI flags.
To run outside that cluster, either:
- create the same directory structure locally (e.g. `mkdir -p /pvc/results/logits
  /pvc/results/logits_kvtemporal /pvc/results/experimental`), or
- symlink `/pvc` to wherever your data/results actually live.

The k8s YAMLs under `k8s/` are the exact job configs used to produce the reported results,
included for reference/provenance. `<NAMESPACE>`, `<COMPUTE_PVC>`, and `<GALAR_DATA_PVC>` are
placeholders where the original cluster identity was scrubbed, fill in your own before
submitting them to a cluster.

## Pipeline

Four stages, in order. Kvasir-Capsule example paths shown; GALAR follows the same shape with
`--dataset galar` and the GALAR-specific flags (see each script's `--help`).

### 1. Train the classifier
```
python training/train.py --model densenet201_swint --dataset kvasir \
    --data-root /pvc/kvasir-capsule --output-dir /pvc/results \
    --split-mode official --split-id 0 --seed 0 \
    --loss focal --sampler random --experiment densenet201_swint_focal_official_f0_seed0
```
Repeat over `--split-id {0,1}` x `--seed {0,1,42}`. Architecture/loss/sampler sweeps for Tables
1/2 use the same script with different `--model`/`--loss`/`--sampler`, see `k8s/job-cnn-all.yaml`,
`k8s/job-baselines-vit.yaml`, and `k8s/job-recipe-screen.yaml` for the exact configs run.

### 2. Collect per-frame logits
```
python analysis/collect_logits.py --model densenet201_swint \
    --checkpoint /pvc/results/checkpoints/densenet201_swint_focal_official_f0_seed0_best.pt \
    --data-root /pvc/kvasir-capsule --output-dir /pvc/results \
    --split-id 0 --seed 0 --exp densenet201_swint_focal_otdecode_official
```
Writes `{output_dir}/logits/{exp}_f{fold}_seed{seed}.npz` (keys `vL,vY,tL,tY`). See
`k8s/job-collect-recipes.yaml` (Kvasir) / `k8s/job-galar-patho-collect.yaml` (GALAR).

### 3. Tag logits with video/frame ids
Needed for the per-video cluster bootstrap and temporal smoothing, both require knowing which
frames belong to which video.
```
python analysis/tag_logits_kvasir.py --root /pvc/kvasir-capsule \
    --logits-dir /pvc/results/logits --exp densenet201_swint_focal_otdecode_official \
    --split-mode official --folds 0,1 --seeds 0,1,42
```
GALAR equivalent: `analysis/tag_logits.py`. See `k8s/job-kvasir-temporal.yaml` /
`k8s/job-galar-patho-temporal.yaml`.

### 4. Decode and evaluate
```
python analysis/hcg_decode.py --dataset kvasir --device cpu
python analysis/hcg_decode.py --dataset galar_pathology --device cpu
```
Add `--ensemble` for the 3-seed ensemble scope. Writes `results_hcg.csv`,
`results_hcg_significance.csv`, `per_class_{dataset}_{scope}_hcg.csv`, and
`confmat_{dataset}_{scope}.npz` to `/pvc/results/experimental/`. See `k8s/job-hcg.yaml`,
`k8s/job-hcg-compare.yaml`, `k8s/job-hcg-cbsampler.yaml`.

### 5. Tables and figures
Copy the CSVs/npz from step 4 into `tables/report/` (already populated here with the CSVs/npz
used for the paper, so these scripts run standalone without repeating steps 1-4):
```
python tables/make_main_results.py        # Tables 3/4
python tables/make_perclass_rare.py
python tables/make_ablation_mechanism.py
python tables/make_ablation_granularity.py
python tables/make_ablation_cbsampler.py
python tables/make_arch_baseline.py       # Table 1
python tables/make_baseline_kvasir.py     # Table 2
python tables/make_loss_sampler.py
python analysis/plot_confmat.py
python analysis/plot_pareto.py
python analysis/plot_dataset_dist.py
```

## Repository layout

```
analysis/    decoding pipeline (hcg_decode.py + its 3 dependencies), logit collection/tagging,
             GALAR split builder, table/figure plotting scripts
training/    classifier training entrypoint + imbalance-handling (loss/sampler) utilities
models/      architecture registry (densenet201_swint and other baselines, via timm)
datasets/    Kvasir-Capsule and GALAR dataset loaders + split logic
utils/       metrics and results-logging helpers
tables/      table-generation scripts and tables/report/ (the CSV/npz artifacts they read)
k8s/         reference job configs for the runs that produced the reported results
```
