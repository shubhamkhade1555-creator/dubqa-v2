import argparse
import json
from pathlib import Path
from core.qa_report_bundle import export_run_reports
from core.qa_public_summary import write_public_summary

def main():
    p = argparse.ArgumentParser(description="Export DubQA run to Excel and Word.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--public-summary", action="store_true",
                   help="Create sanitized summary only; never publishes it.")
    p.add_argument("--approve-public-summary", action="store_true",
                   help="Explicitly approve writing sanitized public summary JSON.")
    args = p.parse_args()
    outputs = export_run_reports(args.run_dir)
    if args.public_summary:
        if not args.approve_public_summary:
            p.error("--public-summary also requires --approve-public-summary")
        results = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
        public_path = write_public_summary(results, args.run_dir, approved=True)
        outputs["sanitized_public_summary"] = str(public_path)
    print(json.dumps(outputs, indent=2))

if __name__ == "__main__":
    main()
