"""One call to create XLSX then DOCX from the XLSX."""
from pathlib import Path
from core.qa_excel_export import export_run
from core.qa_word_report import build_from_workbook

def export_run_reports(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    xlsx = export_run(run_dir)
    docx = build_from_workbook(xlsx)
    return {"xlsx": str(xlsx), "docx": str(docx)}
