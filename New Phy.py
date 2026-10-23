
#!/usr/bin/env python3

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from nibabel.processing import resample_from_to
from scipy.interpolate import interp1d


# ============================================================
# 1. SETTINGS
# ============================================================

ICA_TR = 1.5
TARGET_TR = 0.5

ICA_FREQ = 1.0 / ICA_TR
TARGET_FREQ = 1.0 / TARGET_TR

NUMBER_COMPONENTS = 30

TASKS = ("simon", "rest", "breathhold")

# Breath-hold response averaging settings
DISCARD_INITIAL_POINTS = 66
WINDOW_LENGTH = 116
NUMBER_OF_WINDOWS = 6

# Apply the six-window averaging protocol only to breathhold.
AVERAGE_TASKS = ("breathhold",)

# If your segmentation is a binary mask, all positive voxels
# are included. For a labelled segmentation, see the note below.
SEGMENTATION_THRESHOLD = 0


# ============================================================
# 2. DIRECTORIES — APPTAINER CONTAINER PATHS
# ============================================================

# Host bind mount:
# /Neurodata -> /Neurodata
#
# Confirmed directory from your terminal screenshot:
# /Neurodata/M3PI/derivatives/Mayra_derivatives/vessels/vessels_melodic

BASE_DIRECTORY = Path(
    "/Neurodata/M3PI/derivatives/Mayra_derivatives/"
    "vessels/vessels_melodic"
)

SEGMENTATION_ROOT = Path(
    "/Neurodata/M3PI/derivatives/Mayra_derivatives/"
    "vessels/segmentations"
)

SEGMENTATION_FILENAME = "segmentation_mask.nii.gz"


# ============================================================
# 3. LOAD ICA TIME SERIES
# ============================================================

def load_ica_time_series(melodic_directory):
    """
    Load MELODIC melodic_mix.

    Expected format:
        rows = time points
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
# 4. LINEAR INTERPOLATION TO 0.5-SECOND SAMPLING
# ============================================================

def resample_signal_freqs(
    ts,
    freq1=ICA_FREQ,
    freq2=TARGET_FREQ,
    axis=0
):
    """
    Linearly interpolate a time series from the original
    sampling frequency to the target sampling frequency.

    For ICA_TR = 1.5 seconds and TARGET_TR = 0.5 seconds,
    the target interval is one-third of the original TR.

    The output does not extrapolate beyond the original
    time-series duration.
    """

    ts = np.asarray(ts, dtype=float)

    if ts.ndim == 0:
        raise ValueError("Input must be a time series.")

    if freq1 <= 0 or freq2 <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    n_original = ts.shape[axis]

    if n_original < 2:
        raise ValueError("At least two time points are required.")

    original_times = np.arange(n_original) / freq1

    duration = original_times[-1]

    # Include target samples only within the original duration.
    n_target = int(np.floor(duration * freq2 + 1e-10)) + 1
    target_times = np.arange(n_target) / freq2

    interpolator = interp1d(
        original_times,
        ts,
        axis=axis,
        kind="linear",
        bounds_error=True
    )

    resampled = interpolator(target_times)

    print(f"Original TR: {1 / freq1:.3f} seconds")
    print(f"Target interval: {1 / freq2:.3f} seconds")
    print(f"Original time points: {n_original}")
    print(f"Resampled time points: {resampled.shape[axis]}")
    print(f"Original duration: {duration:.3f} seconds")

    return resampled, target_times


# ============================================================
# 5. AVERAGE SIX COMPLETE BREATH-HOLD WINDOWS
# ============================================================

def calculate_average_spc(
    interpolated_timeseries,
    discard_points=DISCARD_INITIAL_POINTS,
    window_length=WINDOW_LENGTH,
    number_of_windows=NUMBER_OF_WINDOWS
):
    """
    Process an interpolated time series.

    Step 1: Input is the already-read ICA time series.
    Step 2: Input has already been linearly interpolated to 0.5 s.
    Step 3: Discard the first 66 interpolated samples.
    Step 4: Extract six windows, each containing 116 samples.
    Step 5: Discard any incomplete trailing samples.
    Step 6: Average corresponding time points across the windows.

    Input:
        (time_points, components)

    Returns:
        average_response: (116, components)
        windows: (6, 116, components)
        discarded_remainder: number of trailing samples not used
    """

    timeseries = np.asarray(
        interpolated_timeseries,
        dtype=float
    )

    if timeseries.ndim == 1:
        timeseries = timeseries[:, np.newaxis]

    if timeseries.ndim != 2:
        raise ValueError(
            "Expected time series with shape "
            "(time_points, components)."
        )

    if not np.all(np.isfinite(timeseries)):
        raise ValueError(
            "Time series contains NaN or infinite values."
        )

    if discard_points < 0 or window_length <= 0:
        raise ValueError("Invalid discard or window length.")

    if number_of_windows <= 0:
        raise ValueError("number_of_windows must be positive.")

    # Step 3: discard the initial 66 samples.
    trimmed = timeseries[discard_points:]

    required_samples = window_length * number_of_windows

    if trimmed.shape[0] < required_samples:
        raise ValueError(
            f"Insufficient data after discarding the first "
            f"{discard_points} samples: {trimmed.shape[0]} remain, "
            f"but {required_samples} are needed for "
            f"{number_of_windows} complete windows."
        )

    # Steps 4 and 5: retain exactly six complete windows.
    usable_samples = trimmed[:required_samples]

    discarded_remainder = (
        trimmed.shape[0] - required_samples
    )

    # Reshape into:
    # (number of windows, samples per window, components)
    windows = usable_samples.reshape(
        number_of_windows,
        window_length,
        timeseries.shape[1]
    )

    # Step 6: average the six windows at each corresponding time point.
    average_response = np.mean(windows, axis=0)

    return average_response, windows, discarded_remainder


# ============================================================
# 6. SAVE AVERAGE RESPONSE AND INDIVIDUAL WINDOWS
# ============================================================

def save_average_response(
    melodic_directory,
    subject,
    task,
    average_response,
    windows,
    discarded_remainder
):
    """Save the averaged response and six windows as CSV files."""

    component_names = [
        f"component_{i + 1}"
        for i in range(average_response.shape[1])
    ]

    # Save the 116-point averaged response.
    average_df = pd.DataFrame(
        average_response,
        columns=component_names
    )

    average_df.insert(
        0,
        "time_seconds",
        np.arange(WINDOW_LENGTH) * TARGET_TR
    )

    average_file = (
        melodic_directory
        / f"{subject}_{task}_average_response_0.5s.csv"
    )

    average_df.to_csv(average_file, index=False)

    print("Saved average response:", average_file)

    # Save each window in long format for quality control.
    records = []

    for window_index in range(windows.shape[0]):
        for time_index in range(windows.shape[1]):

            record = {
                "window": window_index + 1,
                "time_seconds": time_index * TARGET_TR
            }

            for component_index, component_name in enumerate(
                component_names
            ):
                record[component_name] = windows[
                    window_index,
                    time_index,
                    component_index
                ]

            records.append(record)

    windows_df = pd.DataFrame(records)

    windows_file = (
        melodic_directory
        / f"{subject}_{task}_six_windows_0.5s.csv"
    )

    windows_df.to_csv(windows_file, index=False)

    print("Saved six individual windows:", windows_file)

    # Save processing metadata.
    metadata = pd.DataFrame([{
        "subject": subject,
        "task": task,
        "target_TR_seconds": TARGET_TR,
        "initial_samples_discarded": DISCARD_INITIAL_POINTS,
        "window_length_samples": WINDOW_LENGTH,
        "window_duration_seconds": WINDOW_LENGTH * TARGET_TR,
        "number_of_windows": windows.shape[0],
        "samples_in_average": average_response.shape[0],
        "trailing_samples_discarded": discarded_remainder
    }])

    metadata_file = (
        melodic_directory
        / f"{subject}_{task}_averaging_metadata.csv"
    )

    metadata.to_csv(metadata_file, index=False)

    print("Saved averaging metadata:", metadata_file)


# ============================================================
# 7. CALCULATE ICA COMPONENT MASK DENSITY
# ============================================================

def calculate_mask_density(
    component_mask_file,
    segmentation_file
):
    """
    Calculate spatial overlap between an ICA component mask
    and a segmentation mask.

    Binary masks include finite values > SEGMENTATION_THRESHOLD.

    Density / segmentation coverage:
        overlap voxels / segmentation voxels

    ICA precision:
        overlap voxels / ICA voxels

    The ICA mask is resampled to the segmentation grid if
    the image dimensions or affine differ.

    NOTE:
    Resampling aligns voxel grids; it does not correct a
    mismatch in anatomical registration or coordinate space.
    """

    component_img = nib.load(str(component_mask_file))
    segmentation_img = nib.load(str(segmentation_file))

    if len(component_img.shape) != 3:
        raise ValueError(
            f"ICA mask must be 3D: {component_img.shape}"
        )

    if len(segmentation_img.shape) != 3:
        raise ValueError(
            f"Segmentation must be 3D: {segmentation_img.shape}"
        )

    component_data = component_img.get_fdata()
    segmentation_data = segmentation_img.get_fdata()

    component_mask = (
        np.isfinite(component_data)
        & (component_data > 0)
    )

    segmentation_mask = (
        np.isfinite(segmentation_data)
        & (segmentation_data > SEGMENTATION_THRESHOLD)
    )

    segmentation_voxels = int(
        np.count_nonzero(segmentation_mask)
    )

    if segmentation_voxels == 0:
        raise ValueError(
            "Segmentation contains no voxels above the threshold."
        )

    same_grid = (
        component_img.shape == segmentation_img.shape
        and np.allclose(
            component_img.affine,
            segmentation_img.affine,
            rtol=0,
            atol=1e-5
        )
    )

    if not same_grid:
        print(
            "Resampling ICA mask to segmentation grid:",
            component_mask_file.name
        )

        component_binary_img = nib.Nifti1Image(
            component_mask.astype(np.uint8),
            component_img.affine
        )

        resampled_img = resample_from_to(
            component_binary_img,
            (segmentation_img.shape, segmentation_img.affine),
            order=0
        )

        component_mask = resampled_img.get_fdata() > 0

    component_voxels = int(
        np.count_nonzero(component_mask)
    )

    overlap_voxels = int(
        np.count_nonzero(component_mask & segmentation_mask)
    )

    density = overlap_voxels / segmentation_voxels

    ica_precision = (
        overlap_voxels / component_voxels
        if component_voxels > 0
        else np.nan
    )

    voxel_volume_mm3 = float(
        abs(np.linalg.det(segmentation_img.affine[:3, :3]))
    )

    return {
        "ica_mask_voxels": component_voxels,
        "segmentation_voxels": segmentation_voxels,
        "overlap_voxels": overlap_voxels,
        "density": density,
        "density_percent": density * 100,
        "ica_precision": ica_precision,
        "ica_precision_percent": (
            ica_precision * 100
            if np.isfinite(ica_precision)
            else np.nan
        ),
        "voxel_volume_mm3": voxel_volume_mm3,
        "segmentation_volume_mm3": (
            segmentation_voxels * voxel_volume_mm3
        ),
        "ica_volume_on_segmentation_grid_mm3": (
            component_voxels * voxel_volume_mm3
        ),
        "overlap_volume_mm3": (
            overlap_voxels * voxel_volume_mm3
        )
    }


# ============================================================
# 8. INSPECT COMPONENT MASKS AND SAVE DENSITY SUMMARY
# ============================================================

def inspect_component_masks(
    subject,
    task,
    melodic_directory,
    segmentation_file
):
    """Inspect component masks and save a CSV density summary."""

    split_directory = melodic_directory / "IC_masks_split"

    if not split_directory.is_dir():
        print("WARNING: Component mask directory not found:", split_directory)
        return

    if not segmentation_file.is_file():
        print("WARNING: Segmentation file not found:", segmentation_file)
        return

    masks = sorted(split_directory.glob("IC_mask_*.nii.gz"))

    if not masks:
        print("WARNING: No component masks found:", split_directory)
        return

    mask_results = []

    for mask_file in masks:
        print(f"\nProcessing component mask: {mask_file.name}")

        try:
            mask_image = nib.load(str(mask_file))

            if len(mask_image.shape) != 3:
                print("WARNING: Expected 3D mask; skipping:", mask_file.name)
                continue

            mask_data = mask_image.get_fdata()

            binary_mask = (
                np.isfinite(mask_data)
                & (mask_data > 0)
            )

            original_voxels = int(np.count_nonzero(binary_mask))

            original_voxel_volume = float(
                abs(np.linalg.det(mask_image.affine[:3, :3]))
            )

            original_volume = original_voxels * original_voxel_volume

            density_results = calculate_mask_density(
                mask_file,
                segmentation_file
            )

            result = {
                "subject": subject,
                "task": task,
                "mask": mask_file.name,
                "shape": str(mask_image.shape),
                "original_ica_mask_voxels": original_voxels,
                "original_voxel_volume_mm3": original_voxel_volume,
                "original_mask_volume_mm3": original_volume,
                "segmentation_file": segmentation_file.name,
                **density_results
            }

            mask_results.append(result)

            print("Original ICA voxels:", original_voxels)
            print("Overlap voxels:", density_results["overlap_voxels"])
            print(
                "Segmentation coverage:",
                f'{density_results["density_percent"]:.2f}%'
            )
            print(
                "ICA precision:",
                f'{density_results["ica_precision_percent"]:.2f}%'
            )

        except Exception as error:
            print(f"ERROR processing {mask_file.name}: {error}")

    if mask_results:
        output_file = melodic_directory / "component_mask_summary.csv"

        pd.DataFrame(mask_results).to_csv(output_file, index=False)

        print("Saved component mask density summary:", output_file)
    else:
        print("WARNING: No component mask results to save.")


# ============================================================
# 9. PROCESS ONE SUBJECT AND TASK
# ============================================================

def process_subject_task(subject, task):
    """Process ICA time series and component masks for one task."""

    print("\n" + "=" * 65)
    print(f"Subject: {subject} | Task: {task}")
    print("=" * 65)

    melodic_directory = (
        BASE_DIRECTORY
        / subject
        / task
        / f"{subject}_{task}{NUMBER_COMPONENTS}_melodic"
    )

    if not melodic_directory.is_dir():
        print("WARNING: MELODIC directory not found:", melodic_directory)
        return

    print("MELODIC directory:", melodic_directory)

    # Expected subject-level segmentation.
    segmentation_file = (
        SEGMENTATION_ROOT
        / subject
        / SEGMENTATION_FILENAME
    )

    if segmentation_file.is_file():
        inspect_component_masks(
            subject,
            task,
            melodic_directory,
            segmentation_file
        )
    else:
        print("WARNING: Segmentation not found:", segmentation_file)
        print("Component density calculation skipped.")

    # Step 1: Read original ICA time series.
    try:
        ica_time_series = load_ica_time_series(melodic_directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"WARNING: Cannot load ICA time series: {error}")
        return

    # Step 2: Linearly interpolate to a 0.5-second interval.
    resampled_ica, target_times = resample_signal_freqs(
        ica_time_series,
        freq1=ICA_FREQ,
        freq2=TARGET_FREQ,
        axis=0
    )

    # Save full interpolated time series for all tasks.
    component_names = [
        f"component_{i + 1}"
        for i in range(resampled_ica.shape[1])
    ]

    results = pd.DataFrame(
        resampled_ica,
        columns=component_names
    )

    results.insert(0, "time_seconds", target_times)

    resampled_file = (
        melodic_directory
        / f"{subject}_{task}_ICA_timeseries_0.5s.csv"
    )

    results.to_csv(resampled_file, index=False)

    print("Saved interpolated time series:", resampled_file)

    # Steps 3-6: Apply the averaging protocol only to breathhold.
    if task in AVERAGE_TASKS:
        try:
            (
                average_response,
                windows,
                discarded_remainder
            ) = calculate_average_spc(resampled_ica)

        except ValueError as error:
            print(
                f"WARNING: Average response could not be calculated "
                f"for {subject}, {task}: {error}"
            )
            return

        print("Initial samples discarded:", DISCARD_INITIAL_POINTS)
        print("Complete windows:", windows.shape[0])
        print("Samples per window:", windows.shape[1])
        print("Trailing samples discarded:", discarded_remainder)
        print("Average response shape:", average_response.shape)

        save_average_response(
            melodic_directory,
            subject,
            task,
            average_response,
            windows,
            discarded_remainder
        )


# ============================================================
# 10. MAIN — PROCESS ALL SUBJECTS AND TASKS
# ============================================================

def main():

    if not BASE_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"MELODIC base directory not found:\n{BASE_DIRECTORY}\n\n"
            "Check the bind mount and the BASE_DIRECTORY setting."
        )

    if not SEGMENTATION_ROOT.is_dir():
        print(
            "WARNING: Segmentation root does not exist:",
            SEGMENTATION_ROOT
        )
        print("Density calculations will be skipped until it is available.")

    subjects = sorted(
        folder.name
        for folder in BASE_DIRECTORY.iterdir()
        if folder.is_dir() and folder.name.startswith("sub-")
    )

    if not subjects:
        print("No subject folders found:", BASE_DIRECTORY)
        return

    print("MELODIC root:", BASE_DIRECTORY)
    print("Segmentation root:", SEGMENTATION_ROOT)
    print("Subjects found:", subjects)
    print("Tasks:", TASKS)
    print("TR:", ICA_TR, "seconds")
    print("Target interval:", TARGET_TR, "seconds")
    print("Averaging tasks:", AVERAGE_TASKS)

    for subject in subjects:
        for task in TASKS:
            try:
                process_subject_task(subject, task)
            except Exception as error:
                print(f"ERROR processing {subject}, {task}: {error}")
                continue

    print("\n" + "=" * 65)
    print("PYTHON PROCESSING COMPLETED")
    print("Output root:", BASE_DIRECTORY)
    print("=" * 65)


if __name__ == "__main__":
    main()
