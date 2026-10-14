
#!/usr/bin/env python3

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ============================================================
# 1. SETTINGS
# ============================================================

# MELODIC functional data: TR = 1.5 seconds
ICA_FREQ = 1 / 1.5

# Target time interval = 0.5 seconds
TARGET_FREQ = 1 / 0.5

NUMBER_COMPONENTS = 30


# ============================================================
# 2. DIRECTORIES
# ============================================================

# Host:
# /Neurodata/M3PI/derivatives/Mayra_derivatives/vessels_melodic
#
# Container:
# /data/Mayra_derivatives/vessels_melodic

BASE_DIRECTORY = Path(
    "/data/Mayra_derivatives/vessels/vessels_melodic"
)

if not BASE_DIRECTORY.is_dir():
    raise FileNotFoundError(
        f"MELODIC base directory not found:\n{BASE_DIRECTORY}\n"
        "Check the Apptainer bind mount and directory."
    )


# Host:
# /Neurodata/M3PI/derivatives/func_net_comp
#
# Container:
# /data/func_net_comp

NETWORK_DIRECTORY = Path(
    "/data/func_net_comp"
)


# ============================================================
# 3. LOAD NETWORK MASKS
# ============================================================

def load_network_masks(subject):
    """
    Find the subject's available network masks.

    Prefer vessel-reference masks if they already exist.
    No transformations are performed by this script.
    """

    masks = sorted(
        NETWORK_DIRECTORY.glob(
            f"map_*_mask_{subject}_vesselres.nii.gz"
        )
    )

    if masks:
        print("Using existing vessel-reference masks.")
        return masks

    # Fall back to original masks if no vessel-reference
    # masks are available.
    masks = sorted(
        NETWORK_DIRECTORY.glob(
            f"map_*_mask_{subject}.nii.gz"
        )
    )

    # Do not include vessel-reference masks twice.
    masks = [
        mask for mask in masks
        if not mask.name.endswith("_vesselres.nii.gz")
    ]

    return masks


# ============================================================
# 4. LOAD ICA TIME SERIES
# ============================================================

def load_ica_time_series(melodic_directory):
    """
    Load the MELODIC time series from melodic_mix.

    Expected dimensions:
        rows    = time points
        columns = ICA components
    """

    mix_file = melodic_directory / "melodic_mix"

    if not mix_file.is_file():
        raise FileNotFoundError(
            f"ICA time-series file not found: {mix_file}"
        )

    time_series = np.loadtxt(mix_file)

    if time_series.ndim == 1:
        time_series = time_series[:, np.newaxis]

    if time_series.ndim != 2:
        raise ValueError(
            f"Unexpected melodic_mix shape: {time_series.shape}"
        )

    if not np.all(np.isfinite(time_series)):
        raise ValueError(
            f"Non-finite values found in {mix_file}"
        )

    print("Original ICA time-series shape:", time_series.shape)

    return time_series


# ============================================================
# 5. RESAMPLE ICA TIME SERIES TO 0.5 SECONDS
# ============================================================

def resample_signal_freqs(
    ts,
    freq1=ICA_FREQ,
    freq2=TARGET_FREQ,
    axis=0
):
    """
    Linearly interpolate a signal from freq1 to freq2.

    Original sampling interval: 1.5 seconds
    Target sampling interval:   0.5 seconds

    No CO2 data are used.
    """

    ts = np.asarray(ts, dtype=float)

    if ts.ndim == 0:
        raise ValueError("Input must be a time series.")

    if freq1 <= 0 or freq2 <= 0:
        raise ValueError(
            "Sampling frequencies must be positive."
        )

    n_original = ts.shape[axis]

    if n_original < 2:
        raise ValueError(
            "At least two time points are required."
        )

    original_times = np.arange(n_original) / freq1

    # Include the original start and end time points.
    n_target = int(
        round((n_original - 1) * freq2 / freq1)
    ) + 1

    target_times = np.arange(n_target) / freq2

    interpolator = interp1d(
        original_times,
        ts,
        axis=axis,
        kind="linear",
        bounds_error=True
    )

    resampled = interpolator(target_times)

    print("Original frequency:", freq1, "Hz")
    print("Target frequency:", freq2, "Hz")
    print("Original time points:", n_original)
    print("Resampled time points:", resampled.shape[axis])
    print("Original duration:", original_times[-1], "seconds")

    return resampled, target_times


# ============================================================
# 6. INSPECT NETWORK MASKS
# ============================================================

def inspect_network_masks(subject, output_directory):
    """
    Summarise available network masks.

    Does not load melodic_IC.nii.gz.
    Does not transform masks.
    """

    masks = load_network_masks(subject)

    if not masks:
        print(
            f"No network masks found for {subject} "
            f"in {NETWORK_DIRECTORY}"
        )
        return

    mask_results = []

    for mask_file in masks:

        print(f"\nLoading mask: {mask_file.name}")

        mask_image = nib.load(str(mask_file))
        mask_data = mask_image.get_fdata()

        binary_mask = mask_data > 0

        voxel_count = int(np.count_nonzero(binary_mask))

        voxel_volume = float(
            abs(np.linalg.det(mask_image.affine[:3, :3]))
        )

        mask_results.append({
            "subject": subject,
            "mask": mask_file.name,
            "shape": str(mask_image.shape),
            "voxel_count": voxel_count,
            "voxel_volume_mm3": voxel_volume,
            "mask_volume_mm3": voxel_count * voxel_volume
        })

        print("Shape:", mask_image.shape)
        print("Nonzero voxels:", voxel_count)
        print("Voxel volume (mm3):", voxel_volume)

    output_file = (
        output_directory / "network_mask_summary.csv"
    )

    pd.DataFrame(mask_results).to_csv(
        output_file,
        index=False
    )

    print("\nMask summary saved to:", output_file)


# ============================================================
# 7. PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):

    print("\n" + "=" * 55)
    print(f"Processing {subject}")
    print("=" * 55)

    melodic_directory = (
        BASE_DIRECTORY
        / subject
        / f"{subject}{NUMBER_COMPONENTS}_melodic"
    )

    if not melodic_directory.is_dir():
        print(
            f"WARNING: MELODIC directory not found:\n"
            f"{melodic_directory}"
        )
        return

    # A. Inspect available network masks.
    inspect_network_masks(
        subject,
        melodic_directory
    )

    # B. Load ICA time series.
    try:
        ica_time_series = load_ica_time_series(
            melodic_directory
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"WARNING: Skipping {subject}: {error}")
        return

    # C. Interpolate from 1.5-second to 0.5-second intervals.
    resampled_ica, target_times = resample_signal_freqs(
        ica_time_series,
        freq1=1 / 1.5,
        freq2=1 / 0.5,
        axis=0
    )

    # D. Save resampled ICA time series.
    component_names = [
        f"component_{i + 1}"
        for i in range(resampled_ica.shape[1])
    ]

    results = pd.DataFrame(
        resampled_ica,
        columns=component_names
    )

    results.insert(0, "time_seconds", target_times)

    output_file = (
        melodic_directory
        / f"{subject}_ICA_timeseries_0.5s.csv"
    )

    results.to_csv(output_file, index=False)

    print("\nResampled ICA shape:", resampled_ica.shape)
    print("Saved:", output_file)


# ============================================================
# 8. MAIN: PROCESS ALL SUBJECTS
# ============================================================

def main():

    if not BASE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"MELODIC base directory not found:\n"
            f"{BASE_DIRECTORY}\n"
            "Check the Apptainer bind mount and directory."
        )

    if not NETWORK_DIRECTORY.is_dir():
        print(
            "WARNING: Network mask directory not found:\n"
            f"{NETWORK_DIRECTORY}\n"
            "ICA time-series processing can still continue."
        )

    subjects = sorted(
        folder.name
        for folder in BASE_DIRECTORY.iterdir()
        if folder.is_dir()
        and folder.name.startswith("sub-")
    )

    if not subjects:
        print(
            f"No subject folders found in {BASE_DIRECTORY}"
        )
        return

    print("Subjects found:", subjects)
    print("ICA frequency:", ICA_FREQ, "Hz")
    print("Target frequency:", TARGET_FREQ, "Hz")
    print("Target sampling interval:", 1 / TARGET_FREQ, "seconds")

    for subject in subjects:
        try:
            process_subject(subject)
        except Exception as error:
            print(f"ERROR processing {subject}: {error}")
            continue

    print("\n" + "=" * 55)
    print("ALL SUBJECTS PROCESSED")
    print("=" * 55)


if __name__ == "__main__":
    main()

