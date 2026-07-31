import re
import io
import requests
from pypdf import PdfReader
from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from PIL import Image


@dataclass(frozen=True)
class PageArtifact:
    page_number: int
    text: str
    image_jpeg: bytes


@dataclass(frozen=True)
class DocumentArtifact:
    document_id: str
    name: str
    kind: str
    pages: tuple[PageArtifact, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _jpeg_bytes(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((1800, 2400))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()


def _render_with_pdfplumber(page) -> bytes:
    # 144 DPI is usually sufficient for MLS tables and form checkboxes.
    rendered = page.to_image(resolution=144).original
    return _jpeg_bytes(rendered)


def ingest_pdf(path: str | Path, kind: str) -> DocumentArtifact:
    path = Path(path)
    pages: list[PageArtifact] = []

    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            pages.append(
                PageArtifact(
                    page_number=index,
                    text=text,
                    image_jpeg=_render_with_pdfplumber(page),
                )
            )

    return DocumentArtifact(
        document_id=_sha256(path),
        name=path.name,
        kind=kind,
        pages=tuple(pages),
    )


def anthropic_content(document: DocumentArtifact) -> list[dict]:
    """Create text/image content blocks without depending on the SDK types."""
    blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                f"DOCUMENT START\n"
                f"document_id: {document.document_id}\n"
                f"document_name: {document.name}\n"
                f"document_kind: {document.kind}"
            ),
        }
    ]

    for page in document.pages:
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"PAGE {page.page_number} NATIVE TEXT\n"
                    f"{page.text[:16000]}"
                ),
            }
        )
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(page.image_jpeg).decode("ascii"),
                },
            }
        )

    blocks.append({"type": "text", "text": "DOCUMENT END"})
    return blocks

def get_gdrive_direct_url(share_url: str) -> str:
    """Extracts the file ID from any standard Google Drive share link and builds a direct download URL."""
    pattern = r'(?:file/d/|id=)([\w-]+)'
    match = re.search(pattern, share_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return share_url

def extract_text_from_url(url: str) -> str:
    """Downloads a public Google Drive PDF directly into memory and extracts all text."""
    direct_url = get_gdrive_direct_url(url)
    
    # Download file stream into memory
    response = requests.get(direct_url, allow_redirects=True)
    response.raise_for_status()
    
    # Parse PDF directly from bytes stream
    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
            
    return extracted_text

def process_document_urls(urls: list[str]) -> str:
    """Iterates over a list of document URLs (e.g. MLS link + Disclosure link) 
    and combines their extracted text into a single prompt-ready string."""
    combined_text = ""
    for url in urls:
        doc_text = extract_text_from_url(url)
        combined_text += f"\n--- DOCUMENT SOURCE: {url} ---\n" + doc_text
    return combined_text