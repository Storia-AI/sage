"""Unit tests for the classes under chunker.py.

These are minimal happy-path tests to ensure that the chunkers don't crash.

Dependencies:
pip install pytest
pip install pytest-mock
"""

import os

import repo2vec.chunker


def test_text_chunker_happy_path():
    """Tests the happy path for the TextFileChunker."""
    chunker = repo2vec.chunker.TextFileChunker(max_tokens=100)

    file_path = os.path.join(os.path.dirname(__file__), "../README.md")
    with open(file_path, "r") as file:
        content = file.read()
    metadata = {"file_path": file_path}
    chunks = chunker.chunk(content, metadata)

    assert len(chunks) >= 1


def test_code_chunker_happy_path():
    """Tests the happy path for the CodeFileChunker."""
    chunker = repo2vec.chunker.CodeFileChunker(max_tokens=100)

    file_path = os.path.join(os.path.dirname(__file__), "../repo2vec/chunker.py")
    with open(file_path, "r") as file:
        content = file.read()
    metadata = {"file_path": file_path}
    chunks = chunker.chunk(content, metadata)

    assert len(chunks) >= 1


def test_ipynb_chunker_happy_path():
    """Tests the happy path for the IPynbChunker."""
    code_chunker = repo2vec.chunker.CodeFileChunker(max_tokens=100)
    chunker = repo2vec.chunker.IpynbFileChunker(code_chunker)

    file_path = os.path.join(os.path.dirname(__file__), "assets/sample-notebook.ipynb")
    with open(file_path, "r") as file:
        content = file.read()
    metadata = {"file_path": file_path}
    chunks = chunker.chunk(content, metadata)

    assert len(chunks) >= 1
