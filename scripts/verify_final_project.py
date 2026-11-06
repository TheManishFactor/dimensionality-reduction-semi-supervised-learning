"""Verify the Phase 7-8 submission package without training or rewriting results."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "results" / "final_report"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"
COMPARISON_CSV = REPORT_DIR / "final_results_comparison.csv"
COMPARISON_JSON = REPORT_DIR / "final_results_comparison.json"
REPORT_MD = REPORT_DIR / "final_research_report.md"
REPORT_PDF = REPORT_DIR / "final_research_report.pdf"
EXEC_SUMMARY = REPORT_DIR / "executive_summary.txt"
LIMITATIONS = REPORT_DIR / "limitations_and_future_work.md"
PHASE7_REPORT = REPORT_DIR / "phase7_verification_report.txt"
MANIFEST = REPORT_DIR / "report_manifest.json"
SNAPSHOT = REPORT_DIR / "provenance" / "phase1_phase6_snapshot.json"
OUTPUT = REPORT_DIR / "final_verification_report.txt"

PHASE5_CSV = PROJECT_ROOT / "results" / "classifier_evaluation" / "tables" / "classifier_results.csv"
PHASE6_CSV = PROJECT_ROOT / "results" / "self_training" / "tables" / "self_training_results.csv"
PHASE5_REPORT = PROJECT_ROOT / "results" / "classifier_evaluation" / "reports" / "phase5_verification_report.txt"
PHASE6_REPORT = PROJECT_ROOT / "results" / "self_training" / "reports" / "phase6_verification_report.txt"

PPTX = PROJECT_ROOT / "final_project_presentation.pptx"
VIVA = PROJECT_ROOT / "viva_questions_and_answers.md"
DEMO = PROJECT_ROOT / "project_demonstration_guide.md"
CHECKLIST = PROJECT_ROOT / "final_submission_checklist.md"
README = PROJECT_ROOT / "README.md"

KEY_COLUMNS = ("dataset", "labelled_percentage", "classifier", "feature_representation")
METRICS = ("accuracy", "macro_f1")
EXPECTED_EXPERIMENTS = 81
TOLERANCE = 1e-9

FIGURES = (
    "fig01_accuracy_supervised_vs_self_training.png",
    "fig02_macro_f1_supervised_vs_self_training.png",
    "fig03_paired_improvement_distributions.png",
    "fig04_representation_comparison.png",
    "fig05_labelled_proportion_trend.png",
    "fig06_classifier_comparison.png",
    "fig07_training_time.png",
    "fig08_prediction_time.png",
    "fig09_pseudo_label_acceptance.png",
    "fig10_iteration_statistics.png",
    "fig11_improvement_heatmaps.png",
)
ESSENTIAL_FIGURES = (
    "fig01_accuracy_supervised_vs_self_training.png",
    "fig02_macro_f1_supervised_vs_self_training.png",
    "fig04_representation_comparison.png",
    "fig05_labelled_proportion_trend.png",
    "fig06_classifier_comparison.png",
    "fig07_training_time.png",
    "fig09_pseudo_label_acceptance.png",
)
TABLE_ROWS = {
    "summary_overall.csv": 4,
    "summary_by_factor.csv": 114,
    "summary_best_configurations.csv": 12,
    "summary_warnings.csv": 36,
    "summary_degenerate_fits.csv": 3,
    "summary_runtime.csv": 12,
    "summary_pseudo_labels.csv": 18,
}
REQUIRED_REPORT_SECTIONS = (
    "abstract",
    "introduction",
    "objectives",
    "dataset",
    "methodology",
    "preprocessing",
    "pca",
    "lda",
    "supervised",
    "self-training",
    "results",
    "discussion",
    "limitations",
    "future",
    "conclusion",
    "references",
)
AUTHORIZED_PHASE8_CHANGES = {"README.md", "scripts/verify_final_project.py"}


class Verification:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str, str]] = []

    def check(self, section: str, name: str, condition: bool, detail: str = "") -> bool:
        passed = bool(condition)
        self.checks.append((section, name, passed, detail))
        return passed

    def close(
        self,
        section: str,
        name: str,
        reported: float,
        expected: float,
        tolerance: float = 1e-6,
    ) -> bool:
        delta = abs(float(reported) - float(expected))
        return self.check(
            section,
            name,
            delta <= tolerance,
            f"reported={reported:.8f}, recomputed={expected:.8f}, delta={delta:.2e}",
        )

    @property
    def failures(self) -> list[tuple[str, bool, str, str]]:
        return [item for item in self.checks if not item[2]]

    def summary(self) -> tuple[int, int, int]:
        failed = len(self.failures)
        return len(self.checks), len(self.checks) - failed, failed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_sources(v: Verification) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section = "1 Phase 5/6 coverage"
    v.check(section, "Phase 5 results file exists", PHASE5_CSV.exists(), str(PHASE5_CSV))
    v.check(section, "Phase 6 results file exists", PHASE6_CSV.exists(), str(PHASE6_CSV))
    if not (PHASE5_CSV.exists() and PHASE6_CSV.exists()):
        empty = pd.DataFrame()
        return empty, empty, empty
    phase5 = pd.read_csv(PHASE5_CSV)
    phase6 = pd.read_csv(PHASE6_CSV)
    v.check(section, "Phase 5 has 81 rows", len(phase5) == EXPECTED_EXPERIMENTS, str(len(phase5)))
    v.check(section, "Phase 6 has 81 rows", len(phase6) == EXPECTED_EXPERIMENTS, str(len(phase6)))
    for label, frame in (("Phase 5", phase5), ("Phase 6", phase6)):
        v.check(
            section,
            f"{label} all completed",
            set(frame["status"].unique()) == {"completed"},
            str(frame["status"].value_counts().to_dict()),
        )
        v.check(
            section,
            f"{label} has no error messages",
            int(frame["error_message"].notna().sum()) == 0,
            f"{int(frame['error_message'].notna().sum())} error(s)",
        )
        duplicated = int(frame.duplicated(subset=list(KEY_COLUMNS)).sum())
        v.check(section, f"{label} condition keys unique", duplicated == 0, f"{duplicated} duplicate(s)")
        for metric in METRICS:
            v.check(
                section,
                f"{label} {metric} in [0,1]",
                bool(frame[metric].between(0.0, 1.0).all()),
                f"{int((~frame[metric].between(0.0, 1.0)).sum())} invalid",
            )
    keys5 = set(map(tuple, phase5[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    keys6 = set(map(tuple, phase6[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    v.check(section, "All 81 configurations are shared by both phases", keys5 == keys6)
    v.check(
        section,
        "162 experiment rows are accounted for",
        len(phase5) + len(phase6) == 2 * EXPECTED_EXPERIMENTS,
        f"{len(phase5) + len(phase6)} rows",
    )
    v.check(
        section,
        "81 experiment IDs per phase",
        len(set(phase5["experiment_id"])) == EXPECTED_EXPERIMENTS
        and len(set(phase6["experiment_id"])) == EXPECTED_EXPERIMENTS,
        f"{len(set(phase5['experiment_id']))} / {len(set(phase6['experiment_id']))}",
    )
    return phase5, phase6, pd.DataFrame()


def verify_comparison(v: Verification, phase5: pd.DataFrame, phase6: pd.DataFrame) -> pd.DataFrame:
    section = "2 Comparison integrity"
    v.check(section, "Comparison CSV exists", COMPARISON_CSV.exists())
    v.check(section, "Comparison JSON exists", COMPARISON_JSON.exists())
    v.check(section, "Comparison manifest exists", MANIFEST.exists())
    if not (COMPARISON_CSV.exists() and COMPARISON_JSON.exists()):
        return pd.DataFrame()
    comparison = pd.read_csv(COMPARISON_CSV)
    payload = read_json(COMPARISON_JSON)
    v.check(section, "Comparison CSV has 81 rows", len(comparison) == EXPECTED_EXPERIMENTS, str(len(comparison)))
    v.check(
        section,
        "Comparison JSON has 81 records",
        len(payload.get("comparison", [])) == EXPECTED_EXPERIMENTS,
        str(len(payload.get("comparison", []))),
    )
    v.check(
        section,
        "Comparison keys are unique",
        int(comparison.duplicated(subset=list(KEY_COLUMNS)).sum()) == 0,
    )
    metric_columns = [
        f"{side}_{metric}"
        for metric in METRICS
        for side in ("supervised", "self_training")
    ]
    v.check(
        section,
        "No missing metric values",
        not comparison[metric_columns].isna().any().any(),
    )

    p5 = phase5[list(KEY_COLUMNS) + list(METRICS)].rename(
        columns={metric: f"phase5_{metric}" for metric in METRICS}
    )
    p6 = phase6[list(KEY_COLUMNS) + list(METRICS) + ["numerically_degenerate_fit"]].rename(
        columns={metric: f"phase6_{metric}" for metric in METRICS}
    )
    merged = (
        comparison.merge(p5, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        .merge(p6, on=list(KEY_COLUMNS), how="left", suffixes=("", "_p6"), validate="one_to_one")
    )
    v.check(
        section,
        "Every comparison row links to both source phases",
        bool(merged[[f"phase5_{m}" for m in METRICS] + [f"phase6_{m}" for m in METRICS]].notna().all().all()),
    )
    for metric in METRICS:
        v.check(
            section,
            f"{metric}: comparison matches Phase 5",
            bool(
                ((merged[f"supervised_{metric}"] - merged[f"phase5_{metric}"]).abs() <= TOLERANCE).all()
            ),
        )
        v.check(
            section,
            f"{metric}: comparison matches Phase 6",
            bool(
                ((merged[f"self_training_{metric}"] - merged[f"phase6_{metric}"]).abs() <= TOLERANCE).all()
            ),
        )
    for metric in METRICS:
        recomputed = merged[f"self_training_{metric}"] - merged[f"supervised_{metric}"]
        delta = float((comparison[f"{metric}_improvement"] - recomputed).abs().max())
        v.check(
            section,
            f"{metric} improvement is recomputable",
            delta <= TOLERANCE,
            f"max delta={delta:.2e}",
        )

    manifest = read_json(MANIFEST) if MANIFEST.exists() else {}
    v.check(
        section,
        "Manifest records the comparison hash",
        manifest.get("source_comparison_sha256") == sha256_file(COMPARISON_CSV),
        manifest.get("source_comparison_sha256", ""),
    )
    json_rows = pd.DataFrame(payload.get("comparison", []))
    if not json_rows.empty:
        merged_json = comparison.merge(
            json_rows[list(KEY_COLUMNS) + [f"self_training_{m}" for m in METRICS]],
            on=list(KEY_COLUMNS),
            suffixes=("_csv", "_json"),
            validate="one_to_one",
        )
        max_delta = max(
            float((merged_json[f"self_training_{m}_csv"] - merged_json[f"self_training_{m}_json"]).abs().max())
            for m in METRICS
        )
        v.check(section, "JSON and CSV metrics agree", max_delta <= TOLERANCE, f"max delta={max_delta:.2e}")
    else:
        v.check(section, "JSON and CSV metrics agree", False, "no JSON records")

    overall = pd.read_csv(TABLES_DIR / "summary_overall.csv")
    for metric in METRICS:
        for scope, subset in (
            ("all_conditions", comparison),
            ("excluding_degenerate_fits", comparison[~comparison["numerically_degenerate_fit"]]),
        ):
            row = overall[(overall["metric"] == metric) & (overall["scope"] == scope)]
            if len(row) != 1:
                v.check(section, f"overall row {metric}/{scope}", False, f"{len(row)} rows")
                continue
            row = row.iloc[0]
            v.close(section, f"{metric}/{scope} n", row["n"], len(subset), 0.5)
            v.close(
                section,
                f"{metric}/{scope} supervised mean",
                row["mean_supervised"],
                subset[f"supervised_{metric}"].mean(),
                2e-6,
            )
            v.close(
                section,
                f"{metric}/{scope} self-training mean",
                row["mean_self_training"],
                subset[f"self_training_{metric}"].mean(),
                2e-6,
            )
            v.close(
                section,
                f"{metric}/{scope} mean difference",
                row["mean_difference"],
                subset[f"{metric}_improvement"].mean(),
                2e-6,
            )

    best = pd.read_csv(TABLES_DIR / "summary_best_configurations.csv")
    for metric in METRICS:
        for dataset in ("breast_cancer", "mnist", "har"):
            subset = comparison[comparison["dataset"] == dataset]
            for method, column in (
                ("supervised", f"supervised_{metric}"),
                ("self_training", f"self_training_{metric}"),
            ):
                row = best[
                    (best["metric"] == metric)
                    & (best["dataset"] == dataset)
                    & (best["method"] == method)
                ]
                expected = float(subset[column].max())
                v.check(
                    section,
                    f"best {metric} {dataset} {method}",
                    len(row) == 1 and abs(float(row.iloc[0]["score"]) - expected) <= 2e-6,
                    f"{len(row)} row(s)",
                )
    return comparison


def verify_figures(v: Verification, comparison: pd.DataFrame) -> None:
    section = "3 Figures"
    source_hash = sha256_file(COMPARISON_CSV) if COMPARISON_CSV.exists() else ""
    for name in FIGURES:
        path = FIGURES_DIR / name
        exists = path.exists()
        v.check(section, f"{name} exists", exists)
        if not exists:
            continue
        v.check(section, f"{name} is non-trivial", path.stat().st_size > 20_000, f"{path.stat().st_size} bytes")
        try:
            with Image.open(path) as image:
                width, height = image.size
                metadata = dict(image.info)
        except Exception as error:  # noqa: BLE001
            v.check(section, f"{name} opens", False, str(error))
            continue
        v.check(section, f"{name} opens", True, f"{width}x{height}")
        v.check(section, f"{name} has sufficient resolution", width >= 2000 and height >= 800, f"{width}x{height}")
        v.check(section, f"{name} records the source hash", metadata.get("SourceSHA256") == source_hash)
        v.check(section, f"{name} has a title", bool(metadata.get("Title")))
    for name in ESSENTIAL_FIGURES:
        v.check(section, f"Essential figure {name}", (FIGURES_DIR / name).exists())


def verify_tables(v: Verification) -> None:
    section = "4 Summary tables"
    files = sorted(path.name for path in TABLES_DIR.glob("*.csv"))
    v.check(section, "Seven summary tables exist", files == sorted(TABLE_ROWS), str(files))
    for name, rows in TABLE_ROWS.items():
        path = TABLES_DIR / name
        if not path.exists():
            v.check(section, f"{name} row count", False, "missing")
            continue
        frame = pd.read_csv(path)
        v.check(section, f"{name} row count", len(frame) == rows, f"{len(frame)} rows (expected {rows})")


def verify_documents(v: Verification) -> None:
    section = "5 Documents"
    for path in (REPORT_MD, REPORT_PDF, EXEC_SUMMARY, LIMITATIONS):
        v.check(section, f"{path.name} exists", path.exists(), str(path.stat().st_size) if path.exists() else "missing")

    report = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else ""
    headings = re.findall(r"^## (.+)$", report, re.MULTILINE)
    v.check(section, "Report has at least 15 sections", len(headings) >= 15, str(len(headings)))
    lowered = [heading.lower() for heading in headings]
    for keyword in REQUIRED_REPORT_SECTIONS:
        v.check(
            section,
            f"Report section includes {keyword!r}",
            any(keyword in heading for heading in lowered),
        )
    for phrase in (
        "0.9063",
        "0.8826",
        "0.0038",
        "0.2670",
        "7.59",
        "521,861",
        "16 Liblinear ConvergenceWarnings",
        "17 LinearSVC convergence warnings",
        "3 NumericalStabilityWarnings",
        "68 conditions stopped at the three-iteration cap",
        "13 stopped because no remaining sample",
        "Phase 5 did not record prediction time",
        "No inferential test is reported",
    ):
        v.check(section, f"Report states {phrase!r}", phrase in report)
    for name in FIGURES:
        v.check(section, f"Report references {name}", f"figures/{name}" in report)
    v.check(section, "Report references Table 23", "Table 23" in report)

    if EXEC_SUMMARY.exists():
        summary = EXEC_SUMMARY.read_text(encoding="utf-8")
        for phrase in ("81", "0.8826", "7.59", "68", "13"):
            v.check(section, f"Executive summary includes {phrase!r}", phrase in summary)
    if LIMITATIONS.exists():
        limitations = LIMITATIONS.read_text(encoding="utf-8")
        for phrase in (
            "warnings, limitations, and failed experiments",
            "Phase 5 did not record prediction time",
            "68 of 81",
            "three numerically degenerate",
            "no p-values",
            "single split",
            "Preservation statement",
        ):
            v.check(section, f"Limitations includes {phrase!r}", phrase in limitations)

    if REPORT_PDF.exists():
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(REPORT_PDF))
            pages = len(reader.pages)
            first = reader.pages[0].extract_text() or ""
            last = reader.pages[-1].extract_text() or ""
            v.check(section, "PDF opens successfully", pages > 0, f"{pages} pages")
            v.check(section, "PDF has at least 10 pages", pages >= 10, str(pages))
            v.check(section, "PDF title is readable", "Dimensionality Reduction" in first)
            v.check(section, "PDF closes with references", "References" in last or "Pedregosa" in last)
        except Exception as error:  # noqa: BLE001
            v.check(section, "PDF opens successfully", False, str(error))


def _pptx_text(path: Path) -> tuple[int, str, int]:
    namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        media = [name for name in archive.namelist() if name.startswith("ppt/media/")]
        text = []
        for name in slides:
            root = ET.fromstring(archive.read(name))
            text.extend(node.text or "" for node in root.iter(namespace))
    return len(slides), " ".join(text), len(media)


def verify_presentation(v: Verification) -> None:
    section = "6 Presentation"
    v.check(section, "PPTX exists", PPTX.exists(), str(PPTX.stat().st_size) if PPTX.exists() else "missing")
    if not PPTX.exists():
        return
    v.check(section, "PPTX is a valid zip package", zipfile.is_zipfile(PPTX))
    try:
        slides, text, media = _pptx_text(PPTX)
    except Exception as error:  # noqa: BLE001
        v.check(section, "PPTX opens successfully", False, str(error))
        return
    v.check(section, "PPTX opens successfully", True, f"{slides} slides, {media} media files")
    v.check(section, "PPTX has 15 slides", slides == 15, str(slides))
    v.check(section, "PPTX embeds existing figures", media >= 4, f"{media} media files")
    lowered = text.lower()
    for phrase in (
        "problem",
        "research objectives",
        "datasets",
        "methodology",
        "preprocessing",
        "pca",
        "lda",
        "supervised",
        "self-training",
        "results",
        "limitations",
        "conclusion",
        "future scope",
    ):
        v.check(section, f"Slide content covers {phrase!r}", phrase in lowered)
    for phrase in ("0.0237", "0.0038", "7.59", "162"):
        v.check(section, f"Slide content states {phrase!r}", phrase in text)


def verify_preparation(v: Verification) -> None:
    section = "7 Preparation files"
    for path in (VIVA, DEMO, CHECKLIST):
        v.check(section, f"{path.name} exists", path.exists())
    if VIVA.exists():
        viva = VIVA.read_text(encoding="utf-8")
        questions = len(re.findall(r"^\*\*\d+\.", viva, re.MULTILINE))
        v.check(section, "At least 30 viva questions", questions >= 30, str(questions))
        for phrase in ("0.90", "3-iteration", "7.59", "1,769", "Cohen's d_z", "LDA"):
            v.check(section, f"Viva content includes {phrase!r}", phrase in viva)
    if DEMO.exists():
        demo = DEMO.read_text(encoding="utf-8")
        for phrase in ("Do not retrain", "verify_phase7.py", "fig01", "final_project_presentation.pptx"):
            v.check(section, f"Demonstration guide includes {phrase!r}", phrase in demo)
    if CHECKLIST.exists():
        checklist = CHECKLIST.read_text(encoding="utf-8")
        for phrase in ("Phase 7 verification", "162", "Submission statement", "final_verification_report.txt"):
            v.check(section, f"Checklist includes {phrase!r}", phrase in checklist)


def verify_readme(v: Verification) -> None:
    section = "8 README"
    v.check(section, "README exists", README.exists())
    if not README.exists():
        return
    readme = README.read_text(encoding="utf-8")
    for heading in (
        "Research objective",
        "Technologies used",
        "Datasets",
        "Installation",
        "Usage",
        "Project workflow",
        "Phase 7 results",
        "Phase 8 submission package",
        "Repository structure",
    ):
        v.check(section, f"README section {heading!r}", heading in readme)
    for phrase in (
        "81 supervised",
        "162",
        "final_research_report.pdf",
        "final_project_presentation.pptx",
        "results/final_report",
    ):
        v.check(section, f"README mentions {phrase!r}", phrase in readme)


def verify_existing_verification(v: Verification) -> None:
    section = "9 Prior verification"
    v.check(section, "Phase 5 verification report exists", PHASE5_REPORT.exists())
    v.check(section, "Phase 6 verification report exists", PHASE6_REPORT.exists())
    v.check(section, "Phase 7 verification report exists", PHASE7_REPORT.exists())
    if PHASE5_REPORT.exists():
        v.check(section, "Phase 5 verification passed", "66 / 66" in PHASE5_REPORT.read_text(encoding="utf-8"))
    if PHASE6_REPORT.exists():
        v.check(section, "Phase 6 verification passed", "86 / 86" in PHASE6_REPORT.read_text(encoding="utf-8"))
    if PHASE7_REPORT.exists():
        text = PHASE7_REPORT.read_text(encoding="utf-8")
        v.check(section, "Phase 7 verification passed", "STATUS: PASS" in text)
        v.check(section, "Phase 7 ran 1,769 checks", "1769" in text.replace(",", ""))


def verify_immutability(v: Verification) -> None:
    section = "10 Phase 1-6 immutability"
    v.check(section, "Immutability snapshot exists", SNAPSHOT.exists())
    if not SNAPSHOT.exists():
        return
    snapshot = read_json(SNAPSHOT)
    files = snapshot.get("files", {})
    v.check(section, "Snapshot covers 692 files", len(files) == 692, str(len(files)))
    v.check(
        section,
        "Snapshot excludes Phase 7 outputs",
        not any(path.startswith("results/final_report") for path in files),
    )
    missing, changed = [], []
    checked = 0
    for relative, record in files.items():
        if relative.replace("\\", "/") in AUTHORIZED_PHASE8_CHANGES:
            continue
        path = PROJECT_ROOT / relative
        checked += 1
        if not path.exists():
            missing.append(relative)
            continue
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            changed.append(relative)
    v.check(section, "Experimental artefacts are all present", not missing, ", ".join(missing[:5]))
    v.check(section, "Experimental artefacts are unchanged", not changed, ", ".join(changed[:5]))
    v.check(
        section,
        "Only the README was authorised to change",
        all(relative.replace("\\", "/") in AUTHORIZED_PHASE8_CHANGES for relative in changed)
        or not changed,
        f"{len(changed)} changed",
    )
    v.check(
        section,
        "Snapshot hashes were checked",
        checked >= 690,
        f"{checked} files",
    )


def write_report(v: Verification, elapsed: float) -> None:
    total, passed, failed = v.summary()
    lines = [
        "FINAL PROJECT VERIFICATION REPORT",
        "=" * 78,
        "Project       : Impact of Dimensionality Reduction on the Performance of",
        "                Subsequent Supervised Classifiers in a Semi-Supervised Setting",
        f"Generated UTC : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Elapsed       : {elapsed:.1f} s",
        "",
        "RESULT",
        "-" * 78,
        f"Checks run    : {total}",
        f"Checks passed : {passed}",
        f"Checks failed : {failed}",
        "",
        f"STATUS: {'PASS' if failed == 0 else 'FAIL'}",
        "",
    ]
    current = None
    for section, name, condition, detail in v.checks:
        if section != current:
            current = section
            lines.extend(["", section, "-" * 78])
        status = "PASS" if condition else "FAIL"
        suffix = f" ({detail})" if detail else ""
        lines.append(f"[{status}] {name}{suffix}")
    lines.extend(
        [
            "",
            "NOTES",
            "-" * 78,
            "* No model was trained and no Phase 1-6 result was modified.",
            "* README.md was updated in Phase 8 as required; it is documentation,",
            "  not an experimental artefact, and is excluded from the hash check.",
            "* scripts/verify_final_project.py is a Phase 8 addition and is also",
            "  excluded from the Phase 1-6 snapshot comparison.",
            "* PDF and PPTX checks confirm that the files open and contain the",
            "  expected pages/slides; they are not a visual design review.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import time

    started = time.perf_counter()
    v = Verification()
    phase5, phase6, _ = verify_sources(v)
    if not phase5.empty and not phase6.empty:
        comparison = verify_comparison(v, phase5, phase6)
        if not comparison.empty:
            verify_figures(v, comparison)
    verify_tables(v)
    verify_documents(v)
    verify_presentation(v)
    verify_preparation(v)
    verify_readme(v)
    verify_existing_verification(v)
    verify_immutability(v)
    elapsed = time.perf_counter() - started
    write_report(v, elapsed)

    total, passed, failed = v.summary()
    print(f"Checks run    : {total}")
    print(f"Checks passed : {passed}")
    print(f"Checks failed : {failed}")
    print(f"Status        : {'PASS' if failed == 0 else 'FAIL'}")
    print(f"Report        : {OUTPUT}")
    for section, name, _, detail in v.failures[:25]:
        print(f"FAILED: {section} :: {name} :: {detail}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
