from pathlib import Path
import pytest

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.chunking import (
    chunk_documents,
    chunk_stats,
    recursive_split,
    semantic_split,
)
from genai_platform.ingestion.loaders import (
    clean_text,
    load_html,
    load_path,
    load_text,
)
from genai_platform.ingestion.pipeline import build_chunks, load_chunks, save_chunks
from genai_platform.ingestion.schemas import Chunk, Document


def test_clean_text():
    raw = "Hello\x00 world! This is a hyph-\nenated word.   Extra   spaces.\n\n\n\nTriple line."
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "hyphenated" in cleaned
    assert "Extra spaces." in cleaned
    assert "\n\n\n" not in cleaned


def test_html_loader(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text(
        "<html><head><title>Test Page</title><script>var x = 1;</script></head>"
        "<body><nav>Menu</nav><h1>Header One</h1><p>First paragraph.</p>"
        "<ul><li>Item A</li><li>Item B</li></ul></body></html>",
        encoding="utf-8",
    )
    docs = load_html(html_file)
    assert len(docs) == 1
    assert docs[0].title == "Test Page"
    assert "var x" not in docs[0].text
    assert "## Header One" in docs[0].text
    assert "- Item A" in docs[0].text


def test_recursive_split_and_overlap():
    text = (
        "Paragraph one is here with some detailed explanations.\n\n"
        "Paragraph two continues the topic with further insights and examples.\n\n"
        "Paragraph three concludes the discussion on chunking strategies."
    )
    chunks = recursive_split(text, chunk_size=80, overlap=15)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120  # bounded near chunk_size with overlap


def test_semantic_split():
    embedder = HashEmbedder(dim=256)
    text = (
        "Quantum mechanics describes nature at the atomic scale. "
        "Wave functions represent quantum states. "
        "Superposition is a fundamental principle. "
        "In contrast, culinary arts involve cooking and gastronomy. "
        "Baking bread requires yeast and flour fermentation."
    )
    chunks = semantic_split(text, embedder=embedder, chunk_size=100)
    assert len(chunks) >= 2


def test_chunk_schemas_and_persistence(tmp_path):
    c1 = Chunk(
        text="Sample text for testing persistence.",
        source="test.txt",
        doc_type="txt",
        page=1,
        index=0,
        strategy="recursive",
    )
    assert len(c1.id) == 16
    assert c1.cite == "test.txt, p.1"

    dest = tmp_path / "test_chunks.jsonl"
    save_chunks([c1], path=dest)
    loaded = load_chunks(path=dest)
    assert len(loaded) == 1
    assert loaded[0].id == c1.id
    assert loaded[0].text == c1.text


def test_load_path_recursive(tmp_path):
    f1 = tmp_path / "doc1.txt"
    f1.write_text("Hello from doc1", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    f2 = sub / "doc2.md"
    f2.write_text("# Doc 2\nContent", encoding="utf-8")

    docs = load_path(tmp_path)
    assert len(docs) == 2
    sources = {Path(d.source).name for d in docs}
    assert "doc1.txt" in sources
    assert "doc2.md" in sources
