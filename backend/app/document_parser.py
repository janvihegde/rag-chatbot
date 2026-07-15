"""
Document text extraction (SRS Section: Functional Requirements ->
11. Document Ingestion (Admin)).

Supports the three source formats named in the SRS scope: PDF, HTML, DOCX.
Each parser takes raw bytes and returns extracted plain text; parse_file()
dispatches to the right one based on file extension.
"""
import io

from pypdf import PdfReader
from bs4 import BeautifulSoup
from docx import Document as DocxDocument


def parse_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def parse_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    # Drop non-content tags so scripts/styles don't pollute the extracted text.
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


def parse_docx(content: bytes) -> str:
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


_PARSERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".docx": parse_docx,
}


def parse_file(filename: str, content: bytes) -> str:
    """Dispatch to the right parser based on file extension."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Supported: {', '.join(_PARSERS)}"
        )
    return parser(content)