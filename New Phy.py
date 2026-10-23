
#!/usr/bin/env python3

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy.interpolate import interp1d


# ============================================================
# SETTINGS
# ============================================================

TR = 1.5
TARGET_TR = 0.5
N_COMPONENTS = 30

TASKS = ("simon", "rest", "breathhold")

# Breath-hold protocol
DISCARD_POINTS = 66
WINDOW_LENGTH = 116
N_WINDOWS = 6

# Set to None for all positive segmentation voxels.
# For labelled segmentations, specify the vessel label(s).
VESSEL_LABELS = None

BASE_DIR = Path(
    "/Neurodata/M3PI/derivatives/Mayra_derivatives/"
    "vessels/vessels_melodic"
)

SEG_DIR = Path(
    "/Neurodata/M3PI/derivatives/Mayra_derivatives/"
    "vessels/segmentations"
)

SEG_NAME = "segmentation_mask.nii.gz"


# ============================================================
# TIME SERIES
# ============================================================

def load_ica(melodic_dir):
    """Load MELODIC mixing matrix: time points x components."""

    path = melodic_dir / "melodic_mix"

    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.loadtxt(path, ndmin=2)

    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f"Invalid ICA time series: {path}")

    return data


def interpolate_ica(data):
    """Linearly interpolate ICA time series onto TARGET_TR grid."""

    n_time = data.shape[0]

    if n_time < 2:
        raise ValueError("At least two samples are required.")

    original_time = np.arange(n_time) * TR
    target_time = np.arange(
        0,
        original_time[-1] + 1e-10,
        TARGET_TR
    )

    interpolator = interp1d(
        original_time,
        data,
        axis=0,
        kind="linear",
        bounds_error=True
    )

    return interpolator(target_time), target_time


def average_breathhold(data):
    """
    Discard initial samples, extract six complete windows,
    discard trailing samples, and average corresponding
    time points across windows.

    Input shape: (time, components)
    """

    trimmed = data[DISCARD_POINTS:]

    required = WINDOW_LENGTH * N_WINDOWS

    if len(trimmed) < required:
        raise ValueError(
            f"Insufficient data: {len(trimmed)} samples remain; "
            f"{required} required."
        )

    remainder = len(trimmed) - required

    windows = trimmed[:required].reshape(
        N_WINDOWS, WINDOW_LENGTH, data.shape[1]
    )

    average = windows.mean(axis=0)

    return average, windows, remainder


def save_timeseries(data, times, path):
    """Save component time courses in wide format."""

    columns = [
        f"component_{i + 1}" for i in range(data.shape[1])
    ]

    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "time_seconds", times)
    df.to_csv(path, index=False)


def save_breathhold_results(
    melodic_dir, subject, average, windows, remainder
):
    """Save averaged response, individual windows, and metadata."""

    component_names = [
        f"component_{i + 1}" for i in range(average.shape[1])
    ]

    # Averaged response
    average_df = pd.DataFrame(
        average, columns=component_names
    )
    average_df.insert(
        0, "time_seconds", np.arange(WINDOW_LENGTH) * TARGET_TR
    )

    average_df.to_csv(
        melodic_dir / f"{subject}_breathhold_average.csv",
        index=False
    )

    # Individual windows: long format
    rows = []

    for w in range(N_WINDOWS):
        for t in range(WINDOW_LENGTH):
            rows.append({
                "window": w + 1,
                "time_seconds": t * TARGET_TR,
                **{
                    name: windows[w, t, c]
                    for c, name in enumerate(component_names)
                }
            })

    pd.DataFrame(rows).to_csv(
        melodic_dir / f"{subject}_breathhold_windows.csv",
        index=False
    )

    pd.DataFrame([{
        "initial_samples_discarded": DISCARD_POINTS,
        "window_length": WINDOW_LENGTH,
        "window_duration_seconds": WINDOW_LENGTH * TARGET_TR,
        "number_of_windows": N_WINDOWS,
        "trailing_samples_discarded": remainder,
        "average_samples": WINDOW_LENGTH
    }]).to_csv(
        melodic_dir / f"{subject}_breathhold_metadata.csv",
        index=False
    )


# ============================================================
# SEGMENTATION AND ICA MASK OVERLAP
# ============================================================

def binary_mask(image, labels=None):
    """Create a binary mask from finite image values."""

    data = image.get_fdata()
    valid = np.isfinite(data)

    if labels is None:
        return valid & (data > 0)

    return valid & np.isin(data, labels)


def calculate_mask_density(mask_path, seg_path):
    """
    Calculate:
      coverage = overlap/segmentation voxels
      precision = overlap / ICA voxels

    ICA mask is resampled to the segmentation grid if needed.
    """

    ica_img = nib.load(str(mask_path))
    seg_img = nib.load(str(seg_path))

    if len(ica_img.shape) != 3 or len(seg_img.shape) != 3:
        raise ValueError("Both masks must be 3D.")

    ica_mask = binary_mask(ica_img)
    seg_mask = binary_mask(seg_img, VESSEL_LABELS)

    if not seg_mask.any():
        raise ValueError("Segmentation mask is empty.")

    same_grid = (
        ica_img.shape == seg_img.shape
        and np.allclose(
            ica_img.affine, seg_img.affine,
            rtol=0, atol=1e-5
        )
    )

    if not same_grid:
        # Grid compatibility does not guarantee anatomical alignment.
        ica_img = nib.Nifti1Image(
            ica_mask.astype(np.uint8), ica_img.affine
        )

        ica_mask = resample_from_to(
            ica_img,
            (seg_img.shape, seg_img.affine),
            order=0
        ).get_fdata() > 0

    n_ica = int(ica_mask.sum())
    n_seg = int(seg_mask.sum())
    n_overlap = int((ica_mask & seg_mask).sum())

    voxel_volume = abs(np.linalg.det(seg_img.affine[:3, :3]))

    return {
        "ica_voxels": n_ica,
        "segmentation_voxels": n_seg,
        "overlap_voxels": n_overlap,
        "coverage": n_overlap / n_seg,
        "precision": n_overlap / n_ica if n_ica else np.nan,
        "overlap_volume_mm3": n_overlap * voxel_volume,
        "segmentation_volume_mm3": n_seg * voxel_volume,
        "ica_volume_on_segmentation_grid_mm3": n_ica * voxel_volume
    }


def process_masks(subject, task, melodic_dir, seg_path):
    """Calculate overlap for all component masks."""

    mask_dir = melodic_dir / "IC_masks_split"

    if not mask_dir.is_dir() or not seg_path.is_file():
        print(f"Skipping mask analysis: {subject}, {task}")
        return

    results = []

    for mask_path in sorted(mask_dir.glob("IC_mask_*.nii.gz")):
        try:
            metrics = calculate_mask_density(mask_path, seg_path)

            results.append({
                "subject": subject,
                "task": task,
                "mask": mask_path.name,
                **metrics
            })

        except Exception as exc:
            print(f"Mask error {mask_path.name}: {exc}")

    if results:
        pd.DataFrame(results).to_csv(
            melodic_dir / "component_mask_summary.csv",
            index=False
        )


# ============================================================
# SUBJECT / TASK PROCESSING
# ============================================================

def process_task(subject, task):
    melodic_dir = (
        BASE_DIR / subject / task
        / f"{subject}_{task}{N_COMPONENTS}_melodic"
    )

    if not melodic_dir.is_dir():
        print(f"Missing MELODIC directory: {melodic_dir}")
        return

    seg_path = SEG_DIR / subject / SEG_NAME

    # Spatial mask analysis
    process_masks(subject, task, melodic_dir, seg_path)

    # Temporal analysis
    ica = load_ica(melodic_dir)
    resampled, times = interpolate_ica(ica)

    save_timeseries(
        resampled,
        times,
        melodic_dir / f"{subject}_{task}_ICA_timeseries_0.5s.csv"
    )

    # Protocol-specific averaging
    if task == "breathhold":
        average, windows, remainder = average_breathhold(resampled)

        save_breathhold_results(
            melodic_dir, subject, average, windows, remainder
        )

        print(
            f"{subject} {task}: "
            f"{N_WINDOWS} windows averaged; "
            f"{remainder} trailing samples discarded."
        )


def main():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(BASE_DIR)

    subjects = sorted(
        p.name for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )

    for subject in subjects:
        for task in TASKS:
            try:
                process_task(subject, task)
            except Exception as exc:
                print(f"ERROR {subject} {task}: {exc}")

    print("Processing completed.")


if __name__ == "__main__":
    main()
