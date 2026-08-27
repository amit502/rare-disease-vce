"""
Saves per-epoch metrics and a final summary row to CSV on the PVC.

Layout under output_dir:
  {exp_name}_seed{seed}/
    epochs.csv       — one row per epoch (train + val metrics)
    test_metrics.csv — final test-set metrics (one row)
  summary.csv        — one row per completed run (appended)
"""
import csv
import os
from pathlib import Path
from datetime import datetime


class ResultsLogger:
    def __init__(self, output_dir, exp_name, seed, config, summary_name="summary.csv"):
        self.run_dir  = Path(output_dir) / f"{exp_name}_seed{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_csv   = self.run_dir / "epochs.csv"
        self.test_csv    = self.run_dir / "test_metrics.csv"
        # separate summary per dataset when its per-class columns differ (e.g. GALAR
        # section classes vs Kvasir findings) so DictWriter never misaligns columns
        self.summary_csv = Path(output_dir) / summary_name

        self.exp_name = exp_name
        self.seed     = seed
        self.config   = config          # full argparse namespace → dict
        self._epoch_writer  = None
        self._epoch_file    = None

    # ── per-epoch ────────────────────────────────────────────────────────────

    def log_epoch(self, epoch, train_loss, val_loss, val_metrics):
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            **{k: round(v, 6) for k, v in val_metrics.items()},
        }
        if self._epoch_writer is None:
            self._epoch_file   = open(self.epoch_csv, "w", newline="")
            self._epoch_writer = csv.DictWriter(
                self._epoch_file, fieldnames=list(row.keys())
            )
            self._epoch_writer.writeheader()
        self._epoch_writer.writerow(row)
        self._epoch_file.flush()

    def close(self):
        if self._epoch_file:
            self._epoch_file.close()

    # ── final test ───────────────────────────────────────────────────────────

    def log_test(self, test_metrics, best_epoch):
        row = {"best_epoch": best_epoch, **{k: round(v, 6) for k, v in test_metrics.items()}}
        with open(self.test_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        self._append_summary(test_metrics, best_epoch)

    def _append_summary(self, test_metrics, best_epoch):
        cfg = vars(self.config) if hasattr(self.config, "__dict__") else self.config
        row = {
            "timestamp":        datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exp_name":         self.exp_name,
            "model":            cfg.get("model"),
            "dataset":          cfg.get("dataset"),
            "split_mode":       cfg.get("split_mode"),
            "split_id":         cfg.get("split_id"),
            "train_size":       cfg.get("train_size"),
            "val_size":         cfg.get("val_size"),
            "test_size":        cfg.get("test_size"),
            "seed":             self.seed,
            "epochs_max":       cfg.get("epochs"),
            "epochs_run":       best_epoch,
            "optimizer":        cfg.get("optimizer"),
            "lr":               cfg.get("lr"),
            "momentum":         cfg.get("momentum"),
            "weight_decay":     cfg.get("weight_decay"),
            "batch_size":       cfg.get("batch_size"),
            "loss":             cfg.get("loss"),
            "scheduler":        cfg.get("scheduler"),
            "plateau_patience": cfg.get("plateau_patience"),
            "plateau_factor":   cfg.get("plateau_factor"),
            "early_stop_lr":    cfg.get("early_stop_lr"),
            "image_size":       224,
            "pretrained_backbone": cfg.get("pretrained_backbone"),
            **{k: round(v, 6) for k, v in test_metrics.items()},
        }
        write_header = not self.summary_csv.exists()
        with open(self.summary_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
