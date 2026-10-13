"""Text extraction from uploaded resumes (PDF and DOCX). Files are read in memory and never stored."""

from __future__ import annotations

import io
import re
import zipfile

from app.errors import ValidationFailed

MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 20_000  # the resume field's limit (MatchIn.resume_text)
MAX_PDF_PAGES = 30


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValidationFailed("This PDF is password protected. Remove the password and upload it again.")
        return "\n\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES])
    except ValidationFailed:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise ValidationFailed("This PDF could not be read. It may be damaged.") from exc


def _docx_text(data: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ValidationFailed("This DOCX could not be read. It may be damaged.") from exc
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:  # resumes often lay out sections or skills in tables
        for row in table.rows:
            parts.append(" | ".join(dict.fromkeys(c.text.strip() for c in row.cells if c.text.strip())))
    return "\n".join(parts)


def extract_resume_text(filename: str, data: bytes) -> str:
    """The text of a PDF or DOCX file. The type is decided by the file's content, with the extension as a hint for the message."""
    if not data:
        raise ValidationFailed("The file is empty.")
    if len(data) > MAX_BYTES:
        raise ValidationFailed(f"The file is too large: the limit is {MAX_BYTES // (1024 * 1024)} MB.")
    name = filename.lower()
    if data.startswith(b"%PDF"):
        text = _pdf_text(data)
    elif data.startswith(b"PK") and (name.endswith(".docx") or not name.endswith(".doc")):
        text = _docx_text(data)
    else:
        raise ValidationFailed("Only PDF and DOCX files are supported (an old .doc file must be saved as .docx first).")
    text = _clean(text)
    if not text:
        raise ValidationFailed("No text was found in this file. A scanned PDF (an image) cannot be read; paste the text instead.")
    return text
