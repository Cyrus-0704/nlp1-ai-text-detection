"""Check the frozen candidate report for identity leaks and write a small JSON result.

Standard library only. This is a mechanical check, not a substitute for reading
the report: it searches the Markdown source, DOCX package XML, and PDF bytes for
known identity strings and inspects basic metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_MD = ROOT / "reports" / "technical_report_draft.md"
REPORT_DOCX = ROOT / "reports" / "technical_report_v1_review.docx"
REPORT_PDF = ROOT / "reports" / "technical_report_v1_review.pdf"
OUT = ROOT / "reports" / "release_anonymity_check.json"

BANNED = [
    "Cyrus",
    "Cyrus-0704",
    "cat-dog-good",
    "qll",
    "zyj",
    "Lenovo",
    "F:\\",
    "C:\\",
    "github.com/Cyrus-0704",
    # Internal workflow locators that must not appear in the external PDF.
    "PR #",
    "PR#",
    "issue",
    "commit",
    "H04",
    "H05",
    "B08",
    "handoffs/",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(text: str) -> list[str]:
    found = []
    for token in BANNED:
        # The two path tokens are regexes; the rest are literal.
        if token.startswith("F:") or token.startswith("C:"):
            if re.search(re.escape(token), text):
                found.append(token)
        elif token in text:
            found.append(token)
    return found


def docx_check(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        xml_parts = [n for n in names if n.endswith(".xml")]
        blob = "\n".join(zf.read(n).decode("utf-8", errors="replace")
                         for n in xml_parts)
        core = zf.read("docProps/core.xml").decode("utf-8", errors="replace")
    creator = re.search(r"<dc:creator>(.*?)</dc:creator>", core)
    modified_by = re.search(r"<cp:lastModifiedBy>(.*?)</cp:lastModifiedBy>", core)
    return {
        "parts": len(names),
        "xml_parts": len(xml_parts),
        "hits": hits(blob),
        "creator": creator.group(1) if creator else None,
        "last_modified_by": modified_by.group(1) if modified_by else None,
    }


def pdf_check(path: Path) -> dict:
    data = path.read_bytes()
    text = data.decode("latin-1", errors="replace")

    # LibreOffice stores document metadata in XMP, not only in a PDF /Info
    # dictionary. Parse the XMP packet when present; fall back to /Info.
    meta = {
        "Author": None,
        "Title": None,
        "CreatorTool": None,
        "Producer": None,
        "Subject": None,
        "source": None,
    }
    xmp_match = re.search(rb"<x:xmpmeta\b.*?</x:xmpmeta>", data, re.S)
    if xmp_match:
        try:
            root = ET.fromstring(xmp_match.group(0))
        except ET.ParseError:
            root = None
        if root is not None:
            def local_text(name: str):
                for elem in root.iter():
                    if elem.tag.split("}")[-1] == name and elem.text and elem.text.strip():
                        return elem.text.strip()
                return None

            def local_list(name: str):
                values = []
                for elem in root.iter():
                    if elem.tag.split("}")[-1] != name:
                        continue
                    for child in elem.iter():
                        if child.tag.split("}")[-1] == "li" and child.text and child.text.strip():
                            values.append(child.text.strip())
                return values

            meta["CreatorTool"] = local_text("CreatorTool")
            meta["Producer"] = local_text("Producer")
            authors = local_list("creator")
            titles = local_list("title")
            meta["Author"] = "; ".join(authors) if authors else local_text("creator")
            meta["Title"] = "; ".join(titles) if titles else local_text("title")
            meta["source"] = "XMP"

    if meta["source"] is None:
        for key in ("Author", "Title", "Creator", "Producer", "Subject"):
            m = re.search(r"/" + key + r"\s*\(([^)]*)\)", text)
            if m:
                mapped = "CreatorTool" if key == "Creator" else key
                meta[mapped] = m.group(1)
        meta["source"] = "PDF Info"

    return {
        "bytes": len(data),
        "header": text[:5],
        "hits": hits(text),
        "metadata": meta,
    }


def main() -> None:
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            str(REPORT_MD.relative_to(ROOT)): sha256(REPORT_MD),
            str(REPORT_DOCX.relative_to(ROOT)): sha256(REPORT_DOCX),
            str(REPORT_PDF.relative_to(ROOT)): sha256(REPORT_PDF),
        },
        "banned_tokens": BANNED,
        "markdown": {
            "hits": hits(REPORT_MD.read_text(encoding="utf-8")),
        },
        "docx": docx_check(REPORT_DOCX),
        "pdf": pdf_check(REPORT_PDF),
    }
    md_ok = not result["markdown"]["hits"]
    docx_ok = (not result["docx"]["hits"]
               and not result["docx"]["creator"]
               and not result["docx"]["last_modified_by"])
    pdf_ok = (not result["pdf"]["hits"]
              and result["pdf"]["header"] == "%PDF-"
              and not result["pdf"]["metadata"].get("Author"))
    result["status"] = (
        "PASS_FROZEN_REPORT_PENDING_FINAL_RELEASE"
        if md_ok and docx_ok and pdf_ok
        else "FAIL_NEEDS_FIX"
    )
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS_FROZEN_REPORT_PENDING_FINAL_RELEASE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
