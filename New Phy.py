#!/usr/bin/env python3

from __future__ import annotations

import csv
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from phys2cvr import io as ng  # provides load_nifti_get_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_ROOT = Path("/Neurodata/M3PI/derivatives")
VESSEL_ROOT = DATA_ROOT / "vessels"
MELODIC_ROOT = DATA_ROOT / "Mayra_derivatives" / "vessels" / "vessels_melodic"
OUTPUT_DIR = Path.home() / "vessel_analysis_results"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 7)]
SESSIONS = [f"ses-{i:02d}" for i in range(1, 9)]
TASKS = ("rest", "simon", "breathhold")
NUM_COMPONENTS = 30  # = numbercomponents in the Bash script

# Vessel segmentation (EDIT: file name is a placeholder)
SEG_TEMPLATE = "{root}/{sub}/ses-7T/reg/{sub}_vesselseg.nii.gz"
# Optional whole-brain mask in vessel space, or "" for none
WHOLEBRAIN_TEMPLATE = ""

# Breathhold averaging
TARGET_TR = 0.5       # s, interpolation grid
DISCARD_POINTS = 66   # 15 s of nothing + 18 s of first paced breathing
WINDOW_POINTS = 116   # 58 s per repetition (6 repetitions expected)
MAX_LAG_S = 20.0      # cross-correlation search range

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("vessel_analysis")


def melodic_dir(sub: str, task: str) -> Path:
    return MELODIC_ROOT / sub / task / f"{sub}_{task}{NUM_COMPONENTS}_melodic"


def comp_number(path: Path) -> int:
    """1-based component number from 0-based fslsplit names."""
    return int(re.search(r"_(\d+)$", path.name.split(".")[0]).group(1)) + 1


def write_csv(rows: list[dict], name: str) -> None:
    if not rows:
        log.error("No rows for %s", name)
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d rows: %s", len(rows), OUTPUT_DIR / name)


# ---------------------------------------------------------
# Step 6: density
# ---------------------------------------------------------

def density(brain_mask_path, vessel_mask_path, wholebrain_mask=None):
    _, brain_mask, _ = ng.load_nifti_get_mask(brain_mask_path, is_mask=True, ndim=3)

    if wholebrain_mask is not None:
        _, wholebrain_mask, _ = ng.load_nifti_get_mask(wholebrain_mask, is_mask=True, ndim=3)
        brain_mask = brain_mask * wholebrain_mask

    brain_mask_voxels = np.sum(brain_mask)

    _, vessel_mask, _ = ng.load_nifti_get_mask(vessel_mask_path, is_mask=True, ndim=3)
    vessel_inside_brain = vessel_mask * brain_mask
    vessel_mask_voxels = np.sum(vessel_inside_brain)

    vascular_density = vessel_mask_voxels / brain_mask_voxels * 100

    return brain_mask_voxels, vessel_mask_voxels, vascular_density


def density_table() -> list[dict]:
    rows = []
    for sub in SUBJECTS:
        seg = Path(SEG_TEMPLATE.format(root=VESSEL_ROOT, sub=sub))
        if not seg.is_file():
            log.warning("Segmentation not found: %s", seg)
            continue
        whole = (
            Path(WHOLEBRAIN_TEMPLATE.format(root=VESSEL_ROOT, sub=sub))
            if WHOLEBRAIN_TEMPLATE else None
        )
        for task in TASKS:
            files = sorted(
                (melodic_dir(sub, task) / "IC_masks_vessel_split").glob("IC_mask_vessel_*.nii.gz")
            )
            if not files:
                log.warning("%s %s: no vessel-space masks", sub, task)
            for f in files:
                n_mask, n_vessel, dens = density(f, seg, whole)
                rows.append({
                    "subject": sub, "task": task, "component": comp_number(f),
                    "mask_voxels": int(n_mask),
                    "vessel_voxels_in_mask": int(n_vessel),
                    "density_percent": float(dens),
                })
    return rows


# ---------------------------------------------------------
# Step 9: average breathhold response
# ---------------------------------------------------------

def average_response(series: np.ndarray, tr: float) -> np.ndarray | None:
    """Linear interpolation to TARGET_TR (as in phys2cvr), drop the first
    DISCARD_POINTS, chop into WINDOW_POINTS windows, drop the remainder,
    average the windows."""
    t_in = np.arange(len(series)) * tr
    t_out = np.arange(0.0, t_in[-1] + 1e-9, TARGET_TR)
    y = np.interp(t_out, t_in, series)[DISCARD_POINTS:]
    n_win = len(y) // WINDOW_POINTS
    if n_win < 1:
        return None
    if n_win != 6:
        log.warning("%d windows (expected 6)", n_win)
    return y[: n_win * WINDOW_POINTS].reshape(n_win, WINDOW_POINTS).mean(axis=0)


def all_responses() -> dict:
    """{(subject, task, session): {component: response}}"""
    out = {}
    for sub in SUBJECTS:
        for ses in SESSIONS:
            spc = MELODIC_ROOT / sub / "breathhold" / f"{sub}_{ses}_breathhold_spc.nii.gz"
            if not spc.is_file():
                continue
            tr = float(nib.load(str(spc)).header.get_zooms()[3])
            for task in TASKS:
                ts_dir = melodic_dir(sub, task) / "breathhold_timeseries" / ses
                comps = {}
                for f in sorted(ts_dir.glob("IC_mask_*.txt")):
                    resp = average_response(np.loadtxt(f, ndmin=1), tr)
                    if resp is not None:
                        comps[comp_number(f)] = resp
                if comps:
                    out[(sub, task, ses)] = comps
    return out


# ---------------------------------------------------------
# Step 10: cross-correlation between IC time series
# Every component pair (a < b) within a task/session, on the average
# breathhold responses. r = signed r at the lag with the largest |r|;
# shift_s > 0 means component a is delayed relative to b.
# ---------------------------------------------------------

def xcorr_peak(a: np.ndarray, b: np.ndarray):
    if a.std() == 0 or b.std() == 0:
        return None
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = len(a)
    c = np.correlate(a, b, mode="full") / n
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) * TARGET_TR <= MAX_LAG_S
    i = int(np.argmax(np.abs(c[keep])))
    return float(c[keep][i]), float(lags[keep][i] * TARGET_TR)


def xcorr_table(responses: dict) -> list[dict]:
    rows = []
    for (sub, task, ses), comps in responses.items():
        ids = sorted(comps)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                peak = xcorr_peak(comps[a], comps[b])
                if peak:
                    rows.append({
                        "subject": sub, "task": task, "session": ses,
                        "component_a": a, "component_b": b,
                        "r": peak[0], "shift_s": peak[1],
                    })
    return rows


# ---------------------------------------------------------
# Images
# ---------------------------------------------------------

def plot_ica_maps(sub: str, task: str, out_dir: Path) -> None:
    mdir = melodic_dir(sub, task)
    masks_file = mdir / "IC_masks.nii.gz"
    if not masks_file.is_file():
        return
    masks = np.asarray(nib.load(str(masks_file)).dataobj) > 0
    underlay = np.asarray(nib.load(str(mdir / "mean.nii.gz")).dataobj)

    n, ncols = masks.shape[3], 6
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= n:
            continue
        z = int(np.argmax(masks[..., k].sum(axis=(0, 1))))
        ax.imshow(np.rot90(underlay[..., z]), cmap="gray")
        overlay = np.ma.masked_where(~masks[..., k][..., z], masks[..., k][..., z])
        ax.imshow(np.rot90(overlay), cmap="autumn", alpha=0.8)
        ax.set_title(f"IC {k + 1}", fontsize=8)
    fig.suptitle(f"{sub} | {task} | ICA maps")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sub}_{task}_ICA_maps.png", dpi=150)
    plt.close(fig)


def plot_breathhold(responses: dict, sub: str, task: str, out_dir: Path) -> None:
    per_ses = [c for (s, t, _), c in responses.items() if s == sub and t == task]
    if not per_ses:
        return
    ids = sorted({k for c in per_ses for k in c})
    ncols = 6
    nrows = -(-len(ids) // ncols)
    t = np.arange(WINDOW_POINTS) * TARGET_TR
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows),
                             squeeze=False, sharex=True)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(ids):
            ax.axis("off")
            continue
        stack = [c[ids[k]] for c in per_ses if ids[k] in c]
        for s in stack:
            ax.plot(t, s, color="0.7", lw=0.6)
        ax.plot(t, np.mean(stack, axis=0), color="C3", lw=1.4)
        ax.set_title(f"IC {ids[k]}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{sub} | {task} components | avg breathhold SPC (grey: sessions, red: mean)")
    fig.supxlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sub}_{task}_breathhold_avg.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    write_csv(density_table(), "density.csv")

    responses = all_responses()
    write_csv(xcorr_table(responses), "cross_correlation.csv")

    for sub in SUBJECTS:
        for task in TASKS:
            plot_ica_maps(sub, task, fig_dir)
            plot_breathhold(responses, sub, task, fig_dir)


if __name__ == "__main__":
    main()
