
#!/usr/bin/env python3

from pathlib import Path
import subprocess
import itertools

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags


# ============================================================
# 1. SETTINGS AND PATHS
# ============================================================

BASE_DIR = Path("/data/Mayra_derivatives/vessels/vessels_melodic")
DATA_DIR = Path("/data/vessels")

# Update if your segmentation files are stored elsewhere.
SEG_DIR = Path("/data/Mayra_derivatives/vessels/segmentations")
SEG_NAME = "segmentation_mask.nii.gz"

OUTPUT_DIR = BASE_DIR / "python_analysis"

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 7)]
SESSIONS = [f"ses-{i:02d}" for i in range(1, 9)]
TASKS = ("simon", "rest", "breathhold")

N_COMPONENTS = 30
ICA_TR = 1.5
TARGET_TR = 0.5

DISCARD_POINTS = 66
WINDOW_LENGTH = 116
N_WINDOWS = 6


# ============================================================
# 2. FILE PATHS
# ============================================================

def melodic_dir(subject, task):
    return (
        BASE_DIR / subject / task
        / f"{subject}_{task}{N_COMPONENTS}_melodic"
    )


def vessel_reference(subject):
    return (
        DATA_DIR / subject / "ses-7T" / "reg"
        / f"{subject}_vesselref.nii.gz"
    )


def segmentation_file(subject):
    return SEG_DIR / subject / SEG_NAME


def breathhold_spc(subject, session):
    return (
        BASE_DIR / subject / "breathhold"
        / f"{subject}_{session}_breathhold_spc.nii.gz"
    )


def component_number(path):
    import re
    match = re.search(r"IC_mask_(\d+)", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot identify component number: {path}")
    return int(match.group(1))


def transform_paths(subject):
    """ANTs transforms used by the existing registration workflow."""
    return [
        DATA_DIR / subject / "ses-7T" / "reg"
        / f"{subject}_ses-02_UNIT12vesselref0GenericAffine.mat",

        DATA_DIR / subject / "ses-02" / "reg"
        / f"{subject}_ses-02_T2w2UNIT10GenericAffine.mat",

        DATA_DIR / subject / "ses-01" / "reg"
        / f"{subject}_ses-02_T2w2sbref0GenericAffine.mat",
    ]


# ============================================================
# 3. TRANSFORM COMPONENT MASKS TO VESSEL-REFERENCE SPACE
# ============================================================

def transform_masks(subject, task):
    directory = melodic_dir(subject, task)
    source_dir = directory / "IC_masks_split"
    output_dir = directory / "IC_masks_vesselref"

    reference = vessel_reference(subject)
    transforms = transform_paths(subject)

    for path in [reference, *transforms]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    masks = sorted(source_dir.glob("IC_mask_*.nii.gz"))
    if not masks:
        raise FileNotFoundError(f"No component masks found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for mask in masks:
        number = component_number(mask)
        output_mask = (
            output_dir / f"IC_mask_{number:04d}_vesselres.nii.gz"
        )

        if output_mask.exists():
            continue

        command = [
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(mask),
            "-r", str(reference),
            "-o", str(output_mask),
            "-n", "NearestNeighbor",
            "-t", str(transforms[0]),
            "-t", str(transforms[1]),
            "-t", f"[{transforms[2]},1]",
        ]

        subprocess.run(command, check=True)

    return output_dir


# ============================================================
# 4. CALCULATE COMPONENT DENSITY
# ============================================================

def calculate_density(subject, task):
    reference_path = vessel_reference(subject)
    segmentation_path = segmentation_file(subject)
    mask_dir = melodic_dir(subject, task) / "IC_masks_vesselref"

    reference_img = nib.load(str(reference_path))
    segmentation_img = nib.load(str(segmentation_path))

    reference_mask = reference_img.get_fdata() != 0
    segmentation = segmentation_img.get_fdata() != 0

    if (
        segmentation.shape != reference_mask.shape
        or not np.allclose(
            segmentation_img.affine, reference_img.affine, atol=1e-4
        )
    ):
        raise ValueError(
            f"{subject}: segmentation must be on the vessel-reference grid."
        )

    # Segmented vessel voxels inside the analysis mask.
    denominator_mask = segmentation & reference_mask
    denominator = np.count_nonzero(denominator_mask)

    if denominator == 0:
        raise ValueError(f"{subject}: segmentation contains no analysis voxels.")

    rows = []

    for path in sorted(mask_dir.glob("IC_mask_*_vesselres.nii.gz")):
        image = nib.load(str(path))
        component = image.get_fdata() != 0

        if (
            component.shape != reference_mask.shape
            or not np.allclose(image.affine, reference_img.affine, atol=1e-4)
        ):
            raise ValueError(f"Component mask is not in vessel space: {path}")

        overlap = np.count_nonzero(component & denominator_mask)

        rows.append({
            "subject": subject,
            "task": task,
            "component": component_number(path),
            "overlap_voxels": int(overlap),
            "segmentation_voxels": int(denominator),
            "density": overlap / denominator,
        })

    return rows


# ============================================================
# 5. AVERAGE BREATH-HOLD SPC RESPONSES
# ============================================================

def average_response(timeseries):
    """Interpolate, discard the initial samples, and average six windows."""
    timeseries = np.asarray(timeseries, dtype=float).squeeze()

    if timeseries.ndim != 1 or len(timeseries) < 2:
        raise ValueError("Timeseries must contain at least two samples.")

    original_time = np.arange(len(timeseries)) * ICA_TR
    interpolated_time = np.arange(
        0, original_time[-1] + 1e-9, TARGET_TR
    )

    interpolated = interp1d(
        original_time, timeseries, kind="linear"
    )(interpolated_time)

    required = DISCARD_POINTS + WINDOW_LENGTH * N_WINDOWS
    if len(interpolated) < required:
        raise ValueError(
            f"Only {len(interpolated)} interpolated samples; "
            f"{required} are required."
        )

    start = DISCARD_POINTS
    stop = start + WINDOW_LENGTH * N_WINDOWS
    windows = interpolated[start:stop].reshape(
        N_WINDOWS, WINDOW_LENGTH
    )

    return np.mean(windows, axis=0)


# ============================================================
# 6. CROSS-CORRELATION BETWEEN COMPONENT RESPONSES
# ============================================================

def cross_correlation(first, second):
    """Return maximum normalized correlation and its lag in seconds."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    first = first - np.mean(first)
    second = second - np.mean(second)

    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return np.nan, np.nan

    correlation = correlate(second, first, mode="full") / denominator
    lags = correlation_lags(len(second), len(first), mode="full")

    best = np.argmax(correlation)

    return (
        float(correlation[best]),
        float(lags[best] * TARGET_TR),
    )


# ============================================================
# 7. RUN ANALYSIS AND SAVE RESULTS
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    density_results = []
    response_results = []
    correlation_results = []

    for subject in SUBJECTS:
        for task in TASKS:
            print(f"Processing masks: {subject}, {task}")

            transform_masks(subject, task)
            density_results.extend(calculate_density(subject, task))

            # The Bash pipeline extracts each task's component masks
            # from each session's breath-hold SPC image.
            timeseries_root = (
                melodic_dir(subject, task) / "breathhold_timeseries"
            )

            for session in SESSIONS:
                require_spc = breathhold_spc(subject, session)
                if not require_spc.is_file():
                    print(f"Skipping missing SPC image: {require_spc}")
                    continue

                session_dir = timeseries_root / session
                files = sorted(session_dir.glob("*.txt"))

                if not files:
                    print(f"Skipping missing timeseries: {session_dir}")
                    continue

                responses = {}

                for path in files:
                    component = component_number(path)
                    values = np.loadtxt(path, dtype=float).squeeze()
                    response = average_response(values)
                    responses[component] = response

                    for point, value in enumerate(response):
                        response_results.append({
                            "subject": subject,
                            "task": task,
                            "session": session,
                            "component": component,
                            "time_seconds": point * TARGET_TR,
                            "average_spc": float(value),
                        })

                for first, second in itertools.combinations(
                    sorted(responses), 2
                ):
                    r, shift = cross_correlation(
                        responses[first], responses[second]
                    )

                    correlation_results.append({
                        "subject": subject,
                        "task": task,
                        "session": session,
                        "component_1": first,
                        "component_2": second,
                        "correlation_r": r,
                        "shift_seconds": shift,
                    })

    pd.DataFrame(density_results).to_csv(
        OUTPUT_DIR / "component_density.csv", index=False
    )
    pd.DataFrame(response_results).to_csv(
        OUTPUT_DIR / "breathhold_average.csv", index=False
    )
    pd.DataFrame(correlation_results).to_csv(
        OUTPUT_DIR / "cross_correlation.csv", index=False
    )

    print(f"Analysis complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
