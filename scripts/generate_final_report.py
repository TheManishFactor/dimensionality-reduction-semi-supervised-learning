"""Generate the Phase 7 documents, figures, and summary tables from saved results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_runner import load_config  # noqa: E402
from src.final_analysis import (  # noqa: E402
    AnalysisError,
    build_analysis,
    build_final_comparison,
    load_sources,
    sha256_file,
    validate_sources,
    write_comparison_outputs,
)
from src.final_reporting import (  # noqa: E402
    build_executive_summary,
    build_final_report,
    build_limitations_document,
    create_figures,
    write_pdf_from_html,
    write_summary_tables,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 7 final report, executive summary, "
        "limitations document, figures, tables, and PDF."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "final_report",
        help="Directory for final-report outputs.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write Markdown and HTML but skip PDF conversion.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    try:
        sources = load_sources(load_config(args.config))
        validation = validate_sources(sources)
        payload = build_analysis(sources)
    except AnalysisError as error:
        print(f"REPORT GENERATION FAILED: {error}", file=sys.stderr)
        return 1

    comparison = build_final_comparison(sources["phase5"], sources["phase6"])
    paths = write_comparison_outputs(comparison, payload, output_dir)
    # Match the saved JSON shape so the document builder sees the same records.
    payload["comparison"] = comparison.to_dict(orient="records")
    source_hash = sha256_file(paths["csv"])

    figures = create_figures(comparison, payload, figures_dir, source_hash)
    table_paths = write_summary_tables(payload, comparison, tables_dir)

    report, markdown = build_final_report(payload, figures, figures_dir)
    report_md = output_dir / "final_research_report.md"
    report_md.write_text(markdown, encoding="utf-8")
    report_html = output_dir / "final_research_report.html"
    report_html.write_text(report.render_html(), encoding="utf-8")

    summary_path = output_dir / "executive_summary.txt"
    summary_path.write_text(build_executive_summary(payload), encoding="utf-8")

    limitations_path = output_dir / "limitations_and_future_work.md"
    limitations_path.write_text(
        build_limitations_document(payload), encoding="utf-8"
    )

    pdf_info = None
    pdf_path = output_dir / "final_research_report.pdf"
    if not args.skip_pdf:
        try:
            pdf_info = write_pdf_from_html(report_html, pdf_path)
        except Exception as error:  # noqa: BLE001 - report the real cause
            print(f"PDF generation failed: {error}", file=sys.stderr)
            return 1

    manifest = {
        "phase": 7,
        "generated_utc": payload["generated_utc"],
        "validation_checks_passed": validation["passed"],
        "matched_conditions": validation["matched_experiments"],
        "source_comparison_sha256": source_hash,
        "figures": [
            {
                "number": item["number"],
                "filename": item["filename"],
                "sha256": sha256_file(item["path"]),
                "size_bytes": int(Path(item["path"]).stat().st_size),
            }
            for item in figures
        ],
        "tables": {
            name: {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for name, path in table_paths.items()
        },
        "documents": {
            "report_markdown": {
                "path": report_md.name,
                "sha256": sha256_file(report_md),
            },
            "report_html": {
                "path": report_html.name,
                "sha256": sha256_file(report_html),
            },
            "executive_summary": {
                "path": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
            "limitations": {
                "path": limitations_path.name,
                "sha256": sha256_file(limitations_path),
            },
        },
    }
    if pdf_info is not None:
        manifest["documents"]["report_pdf"] = {
            "path": pdf_path.name,
            "sha256": sha256_file(pdf_path),
            "size_bytes": pdf_info["size_bytes"],
            "renderer": pdf_info["browser"],
        }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Validation checks passed : {validation['passed']}")
    print(f"Matched conditions       : {validation['matched_experiments']}")
    print(f"Figures generated        : {len(figures)}")
    print(f"Summary tables           : {len(table_paths)}")
    print(f"Report Markdown          : {report_md}")
    print(f"Report HTML              : {report_html}")
    if pdf_info is not None:
        print(
            f"Report PDF               : {pdf_path} "
            f"({pdf_info['size_bytes'] / (1024 ** 2):.2f} MiB, "
            f"{Path(pdf_info['browser']).name})"
        )
    print(f"Executive summary        : {summary_path}")
    print(f"Limitations document     : {limitations_path}")
    print(f"Manifest                 : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
