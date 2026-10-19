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

            func_dir="${DATA_DIR}/${subject}/${session}/func"

            func="${func_dir}/00.${subject}_${session}_task-${task}_optcom_native_preprocessed.nii.gz"

            session_mask="${DATA_DIR}/${subject}/${session}/reg/sbref_brain_mask.nii.gz"

            # Select the first available session mask.
            if [[ -z "$mask" ]]; then
                mask="$session_mask"
            fi

            if [[ ! -f "$session_mask" ]]; then
                echo "WARNING: Brain mask not found: $session_mask"
                continue
            fi


            if [[ ! -f "$func" ]]; then
                echo "WARNING: Functional image not found: $func"
                continue
            else
                echo "Nuisance regression for ${subject}, ${session}"
                fslmaths ${func} -Tmean ${task_out}/${subject}_${session}_${task}_avgfornuisance
                3dTproject -polort -1 -input ${func} -mask ${mask} \
                    -ort ${func_dir}/${subject}_${session}_task-${task}_optcom_nuisreg_mat.1D \
		-prefix ${task_out}/${subject}_${session}_task-${task}_denoised_bold.nii.gz \
		-overwrite
                func=${task_out}/${subject}_${session}_task-${task}_denoised_bold.nii.gz
                fslmaths ${func} -add ${task_out}/${subject}_${session}_${task}_avgfornuisance ${func}
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
        melodic_stats="${task_out}/${subject}_${task}${numbercomponents}_melodic/stats"

        ic_maps="${melodic_output}/melodic_IC.nii.gz"
        ic_masks="${melodic_output}/IC_masks.nii.gz"
        split_dir="${melodic_output}/IC_masks_split"

        if [[ ! -f "$ic_maps" ]]; then
            echo "WARNING: MELODIC maps not found: $ic_maps"
            echo "Skipping component extraction for ${task}."
            continue
        fi

		mkdir -p ${split_dir}

		for component_mask in "${melodic_stats}"/thresh_zstat_*.nii.gz; do

		    compnum=$( printf "%02d" $(basename ${component_mask#*_zstat} .nii.gz) )

		    fslmaths ${component_mask} -bin "${split_dir}"/IC_mask_${compnum}
		done
		
		    

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

         
