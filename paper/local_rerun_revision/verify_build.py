"""Compile the two manuscript inputs and record real release-gate results."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import fitz

root = Path(__file__).resolve().parent
out = root / "generated"
out.mkdir(exist_ok=True)
report = {"status": "not_verified", "visual_review": "pending", "documents": {}}
docs = {}
for name in ("source", "arxiv_source"):
    directory = root / name
    ok = True
    command_log = []
    for command in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ):
        result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        command_log.append("$ " + " ".join(command) + "\n" + result.stdout)
        if result.returncode:
            ok = False
            break
    (out / (name + "_commands.log")).write_text("\n".join(command_log))
    log = (directory / "main.log").read_text(errors="replace") if (directory / "main.log").exists() else ""
    (out / (name + "_latex.log")).write_text(log)
    warnings = [line for line in log.splitlines() if re.search(r"Overfull|undefined|multiply defined|Rerun to get", line, re.I)]
    row = {"compiled": ok, "warnings": warnings}
    if ok:
        shutil.copyfile(directory / "main.pdf", out / (name + "_preview.pdf"))
        doc = fitz.open(directory / "main.pdf")
        docs[name] = doc
        row["pages"] = len(doc)
        row["references_on_page_five"] = len(doc) == 5 and "REFERENCES" in doc[4].get_text().upper()
        row["page_text_lengths"] = [len(page.get_text()) for page in doc]
        row["page_sizes"] = [[page.rect.width, page.rect.height] for page in doc]
        row["render_hashes"] = []
        row["out_of_page_text"] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            row["render_hashes"].append(hashlib.sha256(pix.samples).hexdigest())
            if name == "source":
                pix.save(out / ("page_%02d.png" % (i + 1)))
            for block in page.get_text("blocks"):
                if block[0] < -1 or block[1] < -1 or block[2] > page.rect.width+1 or block[3] > page.rect.height+1:
                    row["out_of_page_text"].append({"page": i+1, "bbox": list(block[:4])})
        (out / (name + "_text.txt")).write_text("\n\n".join(page.get_text() for page in doc))
    report["documents"][name] = row
report["source_files_identical"] = (root / "source/main.tex").read_bytes() == (root / "arxiv_source/main.tex").read_bytes()
report["rendered_pages_identical"] = len(docs) == 2 and report["documents"]["source"]["render_hashes"] == report["documents"]["arxiv_source"]["render_hashes"]
passed = report["source_files_identical"] and report["rendered_pages_identical"] and all(
    r["compiled"] and r.get("pages") == 5 and r.get("references_on_page_five") and not r["warnings"] and not r.get("out_of_page_text")
    for r in report["documents"].values()
)
report["status"] = "automated_checks_passed_visual_review_pending" if passed else "needs_revision"
(out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
