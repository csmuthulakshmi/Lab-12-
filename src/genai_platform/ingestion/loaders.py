"""Document loaders for PDF, HTML, Markdown, and TXT files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from bs4 import BeautifulSoup
from pypdf import PdfReader

from genai_platform.ingestion.schemas import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    # Strip NUL characters
    text = text.replace("\x00", "")
    # De-hyphenate line wraps (e.g. "connec-\ntion" -> "connection")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Normalize space characters
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path | str) -> list[Document]:
    """Extract pages from a PDF file as individual Documents."""
    path = Path(path)
    docs: list[Document] = []
    try:
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        title = ""
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            extracted = page.extract_text() or ""
            cleaned = clean_text(extracted)
            if not cleaned:
                logger.warning(f"No extractable text on page {page_num} of {path.name}")
            docs.append(
                Document(
                    text=cleaned,
                    source=str(path.resolve()),
                    doc_type="pdf",
                    page=page_num,
                    title=title,
                    metadata={"total_pages": num_pages},
                )
            )
    except Exception as e:
        logger.error(f"Failed to read PDF file {path}: {e}")
        raise
    return docs


def load_html(path: Path | str) -> list[Document]:
    """Extract structured text from an HTML file."""
    path = Path(path)
    try:
        html_content = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "lxml")

        # Drop unwanted tags
        unwanted_tags = [
            "script", "style", "nav", "footer", "header",
            "aside", "form", "noscript", "svg", "iframe",
        ]
        for tag in soup.find_all(unwanted_tags):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Convert headings to markdown-style
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            h_text = h.get_text(strip=True)
            h.replace_with(f"\n\n## {h_text}\n\n")

        # Convert list items
        for li in soup.find_all("li"):
            li_text = li.get_text(strip=True)
            li.replace_with(f"\n- {li_text}\n")

        # Convert paragraphs and divs to separate lines
        for p in soup.find_all(["p", "div", "br"]):
            p.insert_before("\n")
            p.insert_after("\n")

        text = clean_text(soup.get_text())
        return [
            Document(
                text=text,
                source=str(path.resolve()),
                doc_type="html",
                page=1,
                title=title,
                metadata={},
            )
        ]
    except Exception as e:
        logger.error(f"Failed to read HTML file {path}: {e}")
        raise


def load_text(path: Path | str) -> list[Document]:
    """Load plain text or markdown file."""
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    doc_type = "md" if ext in ("md", "markdown") else "txt"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(content)
        title = ""
        first_line = cleaned.splitlines()[0] if cleaned.splitlines() else ""
        if first_line.startswith("# "):
            title = first_line.lstrip("# ").strip()
        return [
            Document(
                text=cleaned,
                source=str(path.resolve()),
                doc_type=doc_type,
                page=1,
                title=title,
                metadata={},
            )
        ]
    except Exception as e:
        logger.error(f"Failed to read text file {path}: {e}")
        raise


def load_file(path: Path | str) -> list[Document]:
    """Dispatch file loading based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".html", ".htm"):
        return load_html(path)
    elif ext in (".txt", ".md", ".markdown", ".log", ".json"):
        return load_text(path)
    else:
        logger.warning(f"Unsupported file extension {ext} for {path}")
        return []


def load_path(path: Path | str) -> list[Document]:
    """
    Recursively load files from a file or directory.
    Corrupt files are logged with warnings and never crash the process.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    docs: list[Document] = []
    if target.is_file():
        try:
            docs.extend(load_file(target))
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable file {target}: {e}")
        return docs

    # Directory walk
    for p in sorted(target.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            try:
                docs.extend(load_file(p))
            except Exception as e:
                logger.warning(f"Skipping corrupt or unreadable file {p}: {e}")

    return docs
