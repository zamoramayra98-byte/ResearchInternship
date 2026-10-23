
#!/usr/bin/env python3
"""Input validation and SPC summaries for the M3PI vessel pipeline."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_ROOT = Path("/Neurodata/M3PI/derivatives")
VESSEL_ROOT = DATA_ROOT / "vessels"
MELODIC_ROOT = (
    DATA_ROOT / "Mayra_derivatives" / "vessels" / "vessels_melodic"
)

OUTPUT_DIR = Path.home() / "vessel_analysis_results"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 7)]
SESSIONS = range(1, 9)

MASK_PATTERNS = ("*mask*.nii.gz", "*vessel*.nii.gz", "*seg*.nii.gz")

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = np = None

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            OUTPUT_DIR / "vessel_analysis.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("vessel_analysis")


# ---------------------------------------------------------
# PATHS AND VALIDATION
# ---------------------------------------------------------

def reference_path(subject: str) -> Path:
    return (
        VESSEL_ROOT / subject / "ses-7T" / "reg"
        / f"{subject}_vesselref.nii.gz"
    )


def spc_path(subject: str, session: int) -> Path:
    return (
        MELODIC_ROOT / subject / "breathhold"
        / f"{subject}_ses-{session:02d}_breathhold_spc.nii.gz"
    )


def registration_files(subject: str) -> list[Path]:
    directory = reference_path(subject).parent
    if not directory.is_dir():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and (
            p.name.endswith((".mat", ".txt", ".h5", ".nii"))
            or p.name.endswith(".nii.gz")
        )
    )


def candidate_masks(subject: str) -> list[Path]:
    directory = VESSEL_ROOT / subject
    if not directory.is_dir():
        return []

    files = {
        p
        for pattern in MASK_PATTERNS
        for p in directory.rglob(pattern)
        if p.is_file()
        and not p.name.endswith("_vesselref.nii.gz")
    }
    return sorted(files)


def validate_environment() -> None:
    for path in (VESSEL_ROOT, MELODIC_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Required dataset directory is unavailable: {path}"
            )

    test_file = OUTPUT_DIR / ".write_test"
    try:
        test_file.write_text("ok\n", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        raise PermissionError(
            f"Output directory is not writable: {OUTPUT_DIR}"
        ) from exc

    logger.info("Dataset: %s", DATA_ROOT)
    logger.info("Output: %s", OUTPUT_DIR)


# ---------------------------------------------------------
# NIFTI SUMMARY
# ---------------------------------------------------------

def summarize_spc(path: Path) -> dict:
    result = {
        "file": str(path),
        "status": "error",
        "shape": "",
        "voxel_count": "",
        "finite_voxels": "",
        "mean": "",
        "std": "",
        "minimum": "",
        "maximum": "",
        "nonzero_voxels": "",
        "nonzero_fraction": "",
        "error": "",
    }

    if nib is None or np is None:
        result.update(
            status="dependency_missing",
            error="Install nibabel and numpy to summarize NIfTI images.",
        )
        return result

    try:
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
        finite = np.isfinite(data)
        values = data[finite]

        result["shape"] = "x".join(map(str, data.shape))
        result["voxel_count"] = int(data.size)
        result["finite_voxels"] = int(values.size)

        if not values.size:
            result["error"] = "No finite voxel values."
            return result

        nonzero = int(np.count_nonzero(values))
        result.update(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            nonzero_voxels=nonzero,
            nonzero_fraction=float(nonzero / values.size),
            status="ok",
        )
    except Exception as exc:
        result["error"] = str(exc)
        logger.exception("Failed to read %s", path)

    return result


# ---------------------------------------------------------
# AUDIT
# ---------------------------------------------------------

def audit_subject(subject: str) -> tuple[dict, list[dict]]:
    ref = reference_path(subject)
    masks = candidate_masks(subject)
    transforms = registration_files(subject)

    subject_row = {
        "subject": subject,
        "reference_path": str(ref),
        "reference_exists": ref.is_file(),
        "candidate_mask_count": len(masks),
        "candidate_mask_paths": ";".join(map(str, masks)),
        "registration_file_count": len(transforms),
        "registration_file_paths": ";".join(map(str, transforms)),
        "nifti_support": nib is not None and np is not None,
        "status": (
            "inputs_discovered"
            if ref.is_file() and masks
            else "incomplete_inputs"
        ),
    }

    if not ref.is_file():
        logger.warning("Missing reference: %s", ref)
    if not masks:
        logger.warning("%s: no candidate masks found", subject)
    if not transforms:
        logger.warning(
            "%s: no registration files found in %s",
            subject, ref.parent,
        )

    spc_rows = []

    for session in SESSIONS:
        path = spc_path(subject, session)

        if not path.is_file():
            logger.warning("Missing SPC: %s", path)
            spc_rows.append({
                "subject": subject,
                "session": f"ses-{session:02d}",
                "file": str(path),
                "status": "missing",
                "shape": "",
                "voxel_count": "",
                "finite_voxels": "",
                "mean": "",
                "std": "",
                "minimum": "",
                "maximum": "",
                "nonzero_voxels": "",
                "nonzero_fraction": "",
                "error": "File not found",
            })
            continue

        spc_rows.append({
            "subject": subject,
            "session": f"ses-{session:02d}",
            **summarize_spc(path),
        })

    return subject_row, spc_rows


def write_csv(rows: list[dict], filename: str) -> bool:
    if not rows:
        logger.error("No records available for %s", filename)
        return False

    path = OUTPUT_DIR / filename
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Saved %d records: %s", len(rows), path)
        return True
    except OSError:
        logger.exception("Could not write %s", path)
        return False


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> int:
    validate_environment()

    subject_rows = []
    spc_rows = []

    for subject in SUBJECTS:
        try:
            subject_row, image_rows = audit_subject(subject)
            subject_rows.append(subject_row)
            spc_rows.extend(image_rows)
        except Exception:
            logger.exception("Audit failed for %s", subject)
            subject_rows.append({
                "subject": subject,
                "reference_path": str(reference_path(subject)),
                "reference_exists": False,
                "candidate_mask_count": 0,
                "candidate_mask_paths": "",
                "registration_file_count": 0,
                "registration_file_paths": "",
                "nifti_support": nib is not None and np is not None,
                "status": "audit_failed",
            })

    audit_ok = write_csv(subject_rows, "input_audit.csv")
    spc_ok = write_csv(spc_rows, "spc_image_summary.csv")

    logger.info(
        "Audit finished: %d subjects, %d SPC records",
        len(subject_rows), len(spc_rows),
    )
    logger.warning(
        "This script audits inputs and summarizes SPC images only. "
        "It does not calculate vessel density or cross-correlations."
    )

    return 0 if audit_ok and spc_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
