"""Turn any supported file into text. An unreadable file raises ParseError; it is never indexed as empty."""
import csv
import io
import logging
import mimetypes
from pathlib import Path

from . import llm

log = logging.getLogger("nexus.parsers")

IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MEDIA = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".mp4", ".mov", ".mkv", ".avi"}
TEXT = {".txt", ".md", ".markdown", ".rst", ".log", ".json", ".yaml", ".yml", ".py", ".js", ".ts", ".tsx", ".html", ".htm"}
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".tsv"} | IMAGE | MEDIA | TEXT


class ParseError(ValueError):
    pass


def _pdf(data: bytes) -> str:
    import pypdfium2

    pdf = pypdfium2.PdfDocument(data)
    pages = []
    for i, page in enumerate(pdf):
        t = page.get_textpage().get_text_range().strip()
        if t:
            pages.append(f"[Page {i + 1}]\n{t}")
    if not pages:
        raise ParseError("PDF has no extractable text (scanned PDFs are not supported yet)")
    return "\n\n".join(pages)


def _docx(data: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = [("## " + p.text if p.style.name.startswith("Heading") else p.text) for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        parts += [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
    return "\n\n".join(parts)


def _pptx(data: bytes) -> str:
    from pptx import Presentation

    slides = []
    for i, s in enumerate(Presentation(io.BytesIO(data)).slides, 1):
        lines = [sh.text_frame.text.strip() for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip():
            lines.append("Notes: " + s.notes_slide.notes_text_frame.text.strip())
        if lines:
            slides.append(f"## Slide {i}\n" + "\n".join(lines))
    return "\n\n".join(slides)


def _rows_to_text(name: str, rows: list[list[str]], limit: int = 2000) -> str:
    """Each row keeps its column names so one chunk still makes sense on its own."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    head = rows[0]
    lines = [f"## {name}", "Columns: " + ", ".join(head), f"Rows: {len(rows) - 1}"]
    lines += ["; ".join(f"{h}: {c}" for h, c in zip(head, r) if c.strip()) for r in rows[1:limit + 1]]
    return "\n".join(lines)


def _xlsx(data: bytes) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    return "\n\n".join(_rows_to_text(ws.title, [["" if c is None else str(c) for c in r] for r in ws.iter_rows(values_only=True)])
                       for ws in wb.worksheets)


def _csv(data: bytes, delimiter: str) -> str:
    text = data.decode("utf-8-sig", errors="replace")
    return _rows_to_text("Data", list(csv.reader(io.StringIO(text), delimiter=delimiter)))


def _html(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "svg"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def parse(filename: str, data: bytes, workspace_id: str | None = None) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ParseError(f"Unsupported file type: {ext or 'unknown'}")
    try:
        if ext == ".pdf":
            text = _pdf(data)
        elif ext == ".docx":
            text = _docx(data)
        elif ext == ".pptx":
            text = _pptx(data)
        elif ext == ".xlsx":
            text = _xlsx(data)
        elif ext in (".csv", ".tsv"):
            text = _csv(data, "," if ext == ".csv" else "\t")
        elif ext in (".html", ".htm"):
            text = _html(data)
        elif ext in IMAGE:
            mime = mimetypes.guess_type(filename)[0] or "image/png"
            text = llm.vision("Describe this image for a search index. Transcribe all visible text exactly.", data, mime, workspace_id)
        elif ext in MEDIA:
            from . import voice

            text = voice.transcribe(data, filename, None, workspace_id)["text"]
        else:
            text = data.decode("utf-8", errors="replace")
    except (ParseError, llm.LLMUnavailable):
        raise
    except Exception as e:
        raise ParseError(f"Could not read {filename}: {e}") from e
    text = text.replace("\x00", "").strip()
    if not text:
        raise ParseError(f"No text found in {filename}")
    return text
