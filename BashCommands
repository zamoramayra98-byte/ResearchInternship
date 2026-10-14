
#!/bin/bash

# ============================================================
# Multi-task, multi-session SPC + MELODIC ICA Pipeline
# Includes --Ostats, component masks, average responses,
# and pairwise cross-correlation.
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# 1. Define subjects, sessions, and tasks
# ------------------------------------------------------------

subjects=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

sessions=("ses-01" "ses-02" "ses-03" "ses-04" \
          "ses-05" "ses-06" "ses-07" "ses-08")

tasks=("simon" "rest" "breathhold")

numbercomponents=30

# ------------------------------------------------------------
# 2. Define paths inside the Apptainer container
#
# Bind mounts:
# -B /Neurodata/M3PI/derivatives:/data
# -B /home/mayraz/ResearchInternship:/scripts
# ------------------------------------------------------------

DATA_DIR="/data/vessels"
OUTPUT_ROOT="/data/Mayra_derivatives/vessels/vessels_melodic"

# Acquisition sampling interval (assumes TR = 1.5 seconds)
ICA_TR=1.5

# Interpolated sampling interval
TARGET_TR=0.5

# Breathhold response averaging parameters
DISCARD_POINTS=66
WINDOW_POINTS=116
REPETITIONS=6

# ------------------------------------------------------------
# 3. Validate paths and output permissions
# ------------------------------------------------------------

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Input directory not found: $DATA_DIR"
    echo "Check your Apptainer bind mount."
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

if [[ ! -w "$OUTPUT_ROOT" ]]; then
    echo "ERROR: Output directory is not writable: $OUTPUT_ROOT"
    echo "Check permissions on the host directory mounted as /data."
    exit 1
fi

echo "Input directory:  $DATA_DIR"
echo "Output directory: $OUTPUT_ROOT"

# ------------------------------------------------------------
# 4. Process each subject
# ------------------------------------------------------------

for subject in "${subjects[@]}"; do

    echo "================================================"
    echo "Processing subject: $subject"
    echo "================================================"

    subject_out="${OUTPUT_ROOT}/${subject}"
    mkdir -p "$subject_out"

    # --------------------------------------------------------
    # 5. Process each task: calculate SPC, concatenate, run ICA
    # --------------------------------------------------------

    for task in "${tasks[@]}"; do

        echo "----------------------------------------------"
        echo "Subject: $subject | Task: $task"
        echo "----------------------------------------------"

        task_out="${subject_out}/${task}"
        mkdir -p "$task_out"

        func_list=()
        mask=""

        # ----------------------------------------------------
        # 6. Loop through sessions
        # ----------------------------------------------------

        for session in "${sessions[@]}"; do

            echo "Checking ${subject} ${session} ${task}"

            func="${DATA_DIR}/${subject}/${session}/func/00.${subject}_${session}_task-${task}_optcom_native_preprocessed.nii.gz"

            session_mask="${DATA_DIR}/${subject}/${session}/reg/sbref_brain_mask.nii.gz"

            if [[ ! -f "$func" ]]; then
                echo "WARNING: Functional image not found: $func"
                continue
            fi

            if [[ ! -f "$session_mask" ]]; then
                echo "WARNING: Brain mask not found: $session_mask"
                continue
            fi

            # Select the first available session mask.
            if [[ -z "$mask" ]]; then
                mask="$session_mask"
            fi

            # ------------------------------------------------
            # 7. Calculate SPC
            # SPC = (functional - temporal mean) / temporal mean
            # ------------------------------------------------

            spc="${task_out}/${subject}_${session}_${task}_spc.nii.gz"
            mean="${task_out}/${subject}_${session}_${task}_mean.nii.gz"

            echo "Calculating SPC: ${subject} ${session} ${task}"

            fslmaths "$func" -Tmean "$mean"
            fslmaths "$func" -sub "$mean" -div "$mean" "$spc"

            if [[ ! -f "$spc" ]]; then
                echo "ERROR: SPC image was not created: $spc"
                exit 1
            fi

            func_list+=("$spc")

        done

        # ----------------------------------------------------
        # 8. Check that enough runs were found
        # ----------------------------------------------------

        if [[ "${#func_list[@]}" -lt 2 ]]; then
            echo "WARNING: Fewer than two usable runs for ${subject} ${task}."
            echo "Skipping concatenation and MELODIC."
            continue
        fi

        if [[ -z "$mask" || ! -f "$mask" ]]; then
            echo "ERROR: No valid brain mask for ${subject} ${task}."
            continue
        fi

        # ----------------------------------------------------
        # 9. Concatenate SPC runs
        # Keep the existing fslmerge implementation
        # ----------------------------------------------------

        concatenated="${task_out}/concatenated_func_${task}.nii.gz"

        TR=$(fslval "${func_list[0]}" pixdim4)

        echo "Concatenating SPC runs for ${subject} ${task}"

        fslmerge -tr \
            "$concatenated" \
            "${func_list[@]}" \
            "$TR"

        # ----------------------------------------------------
        # 10. Run MELODIC with --Ostats
        # ----------------------------------------------------

        melodic_output="${task_out}/${subject}_${task}${numbercomponents}_melodic"

        echo "Running MELODIC with ${numbercomponents} components"
        echo "Output: $melodic_output"

        melodic \
            -i "$concatenated" \
            -o "$melodic_output" \
            -m "$mask" \
            --report \
            --nobet \
            --Ostats \
            --dim="$numbercomponents"

        echo "Finished MELODIC: ${subject} ${task}"

    done

    # --------------------------------------------------------
    # 11. Create component masks and extract Breathhold signals
    # --------------------------------------------------------

    for task in "${tasks[@]}"; do

        task_out="${subject_out}/${task}"
        melodic_output="${task_out}/${subject}_${task}${numbercomponents}_melodic"

        ic_maps="${melodic_output}/melodic_IC.nii.gz"
        ic_masks="${melodic_output}/IC_masks.nii.gz"
        split_dir="${melodic_output}/IC_masks_split"

        if [[ ! -f "$ic_maps" ]]; then
            echo "WARNING: MELODIC maps not found: $ic_maps"
            echo "Skipping component extraction for ${task}."
            continue
        fi

        # ----------------------------------------------------
        # 12. Threshold ICA maps at 2.3
        # ----------------------------------------------------

        echo "Thresholding ICA maps for ${subject} ${task}"

        fslmaths "$ic_maps" -thr 2.3 -bin "$ic_masks"

      
        # ----------------------------------------------------
        # 13. Split 4D component masks into separate 3D masks
        # ----------------------------------------------------

        mkdir -p "$split_dir"

        fslsplit "$ic_masks" "${split_dir}/IC_mask_" -t

        # ----------------------------------------------------
        # 13a. Transform component masks into vessel-reference
        #      space using ANTs
        # ----------------------------------------------------

        subject_number="${subject#sub-}"

        anat_reference="${DATA_DIR}/sub-${subject_number}/ses-7T/reg/sub-${subject_number}_vesselref.nii.gz"

        t1w_to_t2starw="${DATA_DIR}/sub-${subject_number}/ses-7T/reg/sub-${subject_number}_ses-02_UNIT12vesselref0GenericAffine.mat"

        t2w_to_t1w="${DATA_DIR}/sub-${subject_number}/ses-02/reg/sub-${subject_number}_ses-02_T2w2UNIT10GenericAffine.mat"

        t2w_to_func="${DATA_DIR}/sub-${subject_number}/ses-01/reg/sub-${subject_number}_ses-02_T2w2sbref0GenericAffine.mat"

        # Store transformed masks separately from the original
        # functional-space masks.
        vessel_mask_dir="${melodic_output}/IC_masks_vesselref"
        mkdir -p "$vessel_mask_dir"

        # Check whether ANTs is available.
        if ! command -v antsApplyTransforms >/dev/null 2>&1; then

            echo "WARNING: antsApplyTransforms is unavailable."
            echo "Skipping vessel-reference transformations for ${subject}."

        # Check that all registration inputs exist.
        elif [[ ! -f "$anat_reference" ||
                ! -f "$t1w_to_t2starw" ||
                ! -f "$t2w_to_t1w" ||
                ! -f "$t2w_to_func" ]]; then

            echo "WARNING: Registration files missing for ${subject}."
            echo "Required files:"
            echo "  $anat_reference"
            echo "  $t1w_to_t2starw"
            echo "  $t2w_to_t1w"
            echo "  $t2w_to_func"
            echo "Skipping vessel-reference transformations."

        else

            for component_mask in "${split_dir}"/IC_mask_*.nii.gz; do

                [[ -f "$component_mask" ]] || continue

                component_name=$(basename "$component_mask" .nii.gz)

                output_mask="${vessel_mask_dir}/${component_name}_vesselres.nii.gz"

                # Avoid repeating completed transformations.
                if [[ -f "$output_mask" ]]; then
                    echo "Already transformed: $output_mask"
                    continue
                fi

                echo "Transforming ${component_name} to vessel-reference space"

                antsApplyTransforms \
                    -d 3 \
                    -i "$component_mask" \
                    -r "$anat_reference" \
                    -o "$output_mask" \
                    -n NearestNeighbor \
                    -t "$t1w_to_t2starw" \
                    -t "$t2w_to_t1w" \
                    -t "[${t2w_to_func},1]"

                if [[ -f "$output_mask" ]]; then
                    echo "Created transformed mask: $output_mask"
                else
                    echo "ERROR: Transformed mask was not created: $output_mask"
                    exit 1
                fi

            done

        fi


        # ----------------------------------------------------
        # 14. Extract component time series from each session's
        #     Breathhold SPC data
        # ----------------------------------------------------

        for session in "${sessions[@]}"; do

            # Created during the SPC step for task=breathhold.
            breathhold_spc="${subject_out}/breathhold/${subject}_${session}_breathhold_spc.nii.gz"

            if [[ ! -f "$breathhold_spc" ]]; then
                echo "WARNING: Breathhold SPC not found: $breathhold_spc"
                continue
            fi

            timeseries_dir="${melodic_output}/breathhold_timeseries/${session}"
            analysis_dir="${melodic_output}/breathhold_analysis/${session}"

            mkdir -p "$timeseries_dir" "$analysis_dir"

            for component_mask in "${split_dir}"/IC_mask_*.nii.gz; do

                [[ -f "$component_mask" ]] || continue

                component_name=$(basename "$component_mask" .nii.gz)
                timeseries_file="${timeseries_dir}/${component_name}.txt"

                echo "Extracting ${component_name}: ${subject} ${session} ${task}"

                fslmeants \
                    -i "$breathhold_spc" \
                    -m "$component_mask" \
                    -o "$timeseries_file"

            done

            # ------------------------------------------------
            # 15. Interpolate, average repetitions, and calculate
            #     pairwise cross-correlations using Python
            # ------------------------------------------------

            echo "Averaging responses and computing cross-correlations"

            python3 - \
                "$timeseries_dir" \
                "$analysis_dir" \
                "$ICA_TR" \
                "$TARGET_TR" \
                "$DISCARD_POINTS" \
                "$WINDOW_POINTS" \
                "$REPETITIONS" <<'PYTHON'

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

timeseries_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

source_tr = float(sys.argv[3])
target_tr = float(sys.argv[4])
discard_points = int(sys.argv[5])
window_points = int(sys.argv[6])
repetitions = int(sys.argv[7])

output_dir.mkdir(parents=True, exist_ok=True)

required_points = discard_points + window_points * repetitions

average_responses = {}

# ------------------------------------------------------------
# 1. Read and interpolate each component time series
# ------------------------------------------------------------

for file in sorted(timeseries_dir.glob("IC_mask_*.txt")):

    signal = np.asarray(np.loadtxt(file), dtype=float).ravel()

    if signal.size < 2:
        print(f"WARNING: Too few samples in {file.name}")
        continue

    if not np.all(np.isfinite(signal)):
        print(f"WARNING: Non-finite samples in {file.name}")
        continue

    # Original sampling interval is assumed to be 1.5 seconds.
    original_time = np.arange(signal.size) * source_tr

    # Sample the signal every 0.5 seconds.
    target_time = np.arange(
        0,
        original_time[-1] + 1e-9,
        target_tr
    )

    interpolated = np.interp(
        target_time,
        original_time,
        signal
    )

    if interpolated.size < required_points:
        print(
            f"WARNING: {file.name} has {interpolated.size} "
            f"interpolated points; {required_points} required."
        )
        continue

    # --------------------------------------------------------
    # 2. Discard the first 66 interpolated samples
    # --------------------------------------------------------

    remaining = interpolated[discard_points:]

    # --------------------------------------------------------
    # 3. Retain six complete windows of 116 samples each
    #    Ignore any remaining samples after those six windows
    # --------------------------------------------------------

    usable = remaining[:repetitions * window_points]

    windows = usable.reshape(repetitions, window_points)

    # --------------------------------------------------------
    # 4. Average the six repetitions point by point
    # --------------------------------------------------------

    average_response = windows.mean(axis=0)
    component_name = file.stem

    average_responses[component_name] = average_response

    np.savetxt(
        output_dir / f"{component_name}_average.txt",
        average_response,
        fmt="%.8f"
    )

    np.savetxt(
        output_dir / f"{component_name}_six_windows.txt",
        windows,
        fmt="%.8f"
    )

    print(f"Saved averaged response: {component_name}")

# ------------------------------------------------------------
# 5. Pairwise cross-correlation between averaged components
# ------------------------------------------------------------

correlation_rows = []

for component_a, component_b in combinations(
    sorted(average_responses), 2
):

    a = average_responses[component_a]
    b = average_responses[component_b]

    # Mean-center each response.
    a = a - a.mean()
    b = b - b.mean()

    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        print(
            f"WARNING: Cannot correlate {component_a} and "
            f"{component_b}: a response is constant."
        )
        continue

    # Full cross-correlation across all lags.
    correlation = np.correlate(a, b, mode="full") / norm

    lags = np.arange(-(len(b) - 1), len(a))
    lag_seconds = lags * target_tr

    for lag, value in zip(lag_seconds, correlation):
        correlation_rows.append({
            "component_a": component_a,
            "component_b": component_b,
            "lag_seconds": lag,
            "cross_correlation": value
        })

if correlation_rows:

    df = pd.DataFrame(correlation_rows)

    df.to_csv(
        output_dir / "pairwise_cross_correlations.csv",
        index=False
    )

    # Save the lag with the largest absolute correlation
    # for each pair of components.
    peak_rows = []

    for (component_a, component_b), pair_df in df.groupby(
        ["component_a", "component_b"]
    ):

        peak_idx = pair_df["cross_correlation"].abs().idxmax()
        peak = pair_df.loc[peak_idx]

        peak_rows.append({
            "component_a": component_a,
            "component_b": component_b,
            "peak_lag_seconds": peak["lag_seconds"],
            "peak_cross_correlation": peak["cross_correlation"]
        })

    pd.DataFrame(peak_rows).to_csv(
        output_dir / "peak_cross_correlations.csv",
        index=False
    )

    print("Saved pairwise cross-correlation results.")

else:
    print("WARNING: No valid component pairs to correlate.")

PYTHON

        done

    done

done

echo "================================================"
echo "Pipeline completed."
echo "Outputs saved under: $OUTPUT_ROOT"
echo "================================================"

