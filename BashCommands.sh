#!/bin/bash

# ============================================================
# Multi-subject, multi-session SPC + MELODIC ICA pipeline
# Creates component masks and extracts Breathhold SPC signals.
# ANTs transformations have been removed.
# ============================================================

set -Eeuo pipefail
shopt -s nullglob

# ------------------------------------------------------------
# 1. Subjects, sessions, and tasks
# ------------------------------------------------------------

subjects=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")

sessions=("ses-01" "ses-02" "ses-03" "ses-04"
          "ses-05" "ses-06" "ses-07" "ses-08")

tasks=("simon" "rest" "breathhold")

numbercomponents=30

# ------------------------------------------------------------
# 2. Paths inside Apptainer
# ------------------------------------------------------------

DATA_DIR="/data/vessels"
OUTPUT_ROOT="/data/Mayra_derivatives/vessels/vessels_melodic"

# Acquisition and post-processing parameters
ICA_TR=1.5
TARGET_TR=0.5

DISCARD_POINTS=66
WINDOW_POINTS=116
REPETITIONS=6

# Threshold for binarizing each MELODIC component map
IC_THRESHOLD=2.3

# ------------------------------------------------------------
# 3. Validate directories and required commands
# ------------------------------------------------------------

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Input directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

if [[ ! -w "$OUTPUT_ROOT" ]]; then
    echo "ERROR: Output directory is not writable: $OUTPUT_ROOT"
    exit 1
fi

for command_name in fslmaths fslmerge fslval melodic fslsplit fslmeants 3dTproject; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $command_name"
        exit 1
    fi
done

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
    # 5. Calculate SPC and run MELODIC for each task
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
        # 6. Process sessions
        # ----------------------------------------------------

        for session in "${sessions[@]}"; do

            echo "Checking ${subject} ${session} ${task}"

            func_dir="${DATA_DIR}/${subject}/${session}/func"

            func="${func_dir}/00.${subject}_${session}_task-${task}_optcom_native_preprocessed.nii.gz"

            session_mask="${DATA_DIR}/${subject}/${session}/reg/sbref_brain_mask.nii.gz"

            # Check the mask BEFORE selecting it.
            if [[ ! -f "$session_mask" ]]; then
                echo "WARNING: Brain mask not found: $session_mask"
                continue
            fi

            # Select the first available valid session mask.
            if [[ -z "$mask" ]]; then
                mask="$session_mask"
            fi

            if [[ ! -f "$func" ]]; then
                echo "WARNING: Functional image not found: $func"
                continue
            fi

            nuisance_file="${func_dir}/${subject}_${session}_task-${task}_optcom_nuisreg_mat.1D"

            if [[ ! -f "$nuisance_file" ]]; then
                echo "WARNING: Nuisance regression file not found:"
                echo "  $nuisance_file"
                echo "Skipping this session."
                continue
            fi

            # ------------------------------------------------
            # 6a. Nuisance regression
            # ------------------------------------------------

            echo "Nuisance regression: ${subject} ${session} ${task}"

            nuisance_mean="${task_out}/${subject}_${session}_${task}_avgfornuisance.nii.gz"

            denoised="${task_out}/${subject}_${session}_task-${task}_denoised_bold.nii.gz"

            fslmaths "$func" -Tmean "$nuisance_mean"

            3dTproject \
                -polort -1 \
                -input "$func" \
                -mask "$mask" \
                -ort "$nuisance_file" \
                -prefix "$denoised" \
                -overwrite

            if [[ ! -f "$denoised" ]]; then
                echo "ERROR: Denoised image not created: $denoised"
                exit 1
            fi

            # Restore the temporal mean after nuisance regression.
            fslmaths "$denoised" -add "$nuisance_mean" "$denoised"

            func="$denoised"

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
        # ----------------------------------------------------

        concatenated="${task_out}/concatenated_func_${task}.nii.gz"

        TR=$(fslval "${func_list[0]}" pixdim4)

        echo "Concatenating SPC runs for ${subject} ${task}"

        fslmerge -tr \
            "$concatenated" \
            "${func_list[@]}" \
            "$TR"

        # ----------------------------------------------------
        # 10. Run MELODIC
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
        # 12. Threshold and split the 4D component maps
        # ----------------------------------------------------

        mkdir -p "$split_dir"

        echo "Thresholding component maps at ${IC_THRESHOLD}"

        fslmaths "$ic_maps" \
            -thr "$IC_THRESHOLD" \
            -bin "$ic_masks"

        # Remove old split masks so stale components are not used.
        rm -f "${split_dir}"/IC_mask_*.nii.gz

        fslsplit "$ic_masks" "${split_dir}/IC_mask_" -t

        component_masks=("${split_dir}"/IC_mask_*.nii.gz)

        if [[ "${#component_masks[@]}" -eq 0 ]]; then
            echo "WARNING: No component masks created for ${subject} ${task}."
            continue
        fi

        echo "Created ${#component_masks[@]} component masks."

        # ----------------------------------------------------
        # 13. Extract Breathhold SPC time series
        # ----------------------------------------------------
        # Use this task's component masks to sample each session's
        # Breathhold SPC image.
        #
        # No ANTs transformations are performed.
        # Component masks and SPC images must be in the same
        # spatial reference for fslmeants results to be valid.
        # ----------------------------------------------------

        for session in "${sessions[@]}"; do

            breathhold_spc="${subject_out}/breathhold/${subject}_${session}_breathhold_spc.nii.gz"

            if [[ ! -f "$breathhold_spc" ]]; then
                echo "WARNING: Breathhold SPC not found: $breathhold_spc"
                continue
            fi

            timeseries_dir="${melodic_output}/breathhold_timeseries/${session}"

            analysis_dir="${melodic_output}/breathhold_analysis/${session}"

            mkdir -p "$timeseries_dir" "$analysis_dir"

            for component_mask in "${component_masks[@]}"; do

                component_name=$(basename "$component_mask" .nii.gz)

                timeseries_file="${timeseries_dir}/${component_name}.txt"

                echo "Extracting ${component_name}: ${subject} ${session} ${task}"

                fslmeants \
                    -i "$breathhold_spc" \
                    -m "$component_mask" \
                    -o "$timeseries_file"

                if [[ ! -s "$timeseries_file" ]]; then
                    echo "WARNING: Time series is empty: $timeseries_file"
                fi

            done

        done

    done

done

echo "================================================"
echo "SPC and MELODIC pipeline finished."
echo "================================================"
