"""Chunker abstraction and implementations."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

import nbformat
import pygments
import tiktoken
from semchunk import chunk as chunk_via_semchunk
from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from sage.constants import TEXT_FIELD

logger = logging.getLogger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")


class Chunk:
    @abstractmethod
    def content(self) -> str:
        """The content of the chunk to be indexed."""

    @abstractmethod
    def metadata(self) -> Dict:
        """Metadata for the chunk to be indexed."""


@dataclass
class FileChunk(Chunk):
    """A chunk of code or text extracted from a file in the repository."""

    file_content: str  # The content of the entire file, not just this chunk.
    file_metadata: Dict  # Metadata of the entire file, not just this chunk.
    start_byte: int
    end_byte: int

    @cached_property
    def filename(self):
        if not "file_path" in self.file_metadata:
            raise ValueError("file_metadata must contain a 'file_path' key.")
        return self.file_metadata["file_path"]

    @cached_property
    def content(self) -> Optional[str]:
        """The text content to be embedded. Might contain information beyond just the text snippet from the file."""
        return self.filename + "\n\n" + self.file_content[self.start_byte : self.end_byte]

    @cached_property
    def metadata(self):
        """Converts the chunk to a dictionary that can be passed to a vector store."""
        # Some vector stores require the IDs to be ASCII.
        filename_ascii = self.filename.encode("ascii", "ignore").decode("ascii")
        chunk_metadata = {
            # Some vector stores require the IDs to be ASCII.
            "id": f"{filename_ascii}_{self.start_byte}_{self.end_byte}",
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "length": self.end_byte - self.start_byte,
            # Note to developer: When choosing a large chunk size, you might exceed the vector store's metadata
            # size limit. In that case, you can simply store the start/end bytes above, and fetch the content
            # directly from the repository when needed.
            TEXT_FIELD: self.content,
        }
        chunk_metadata.update(self.file_metadata)
        return chunk_metadata

    @cached_property
    def num_tokens(self):
        """Number of tokens in this chunk."""
        return len(tokenizer.encode(self.content, disallowed_special=()))

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return (
                self.filename == other.filename
                and self.start_byte == other.start_byte
                and self.end_byte == other.end_byte
            )
        return False

    def __hash__(self):
        return hash((self.filename, self.start_byte, self.end_byte))


class Chunker(ABC):
    """Abstract class for chunking a datum into smaller pieces."""

    @abstractmethod
    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        """Chunks a datum into smaller pieces."""


class CodeFileChunker(Chunker):
    """Splits a code file into chunks of at most `max_tokens` tokens each."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.text_chunker = TextFileChunker(max_tokens)

    @staticmethod
    def _get_language_from_filename(filename: str):
        """Returns a canonical name for the language of the file, based on its extension.
        Returns None if the language is unknown to the pygments lexer.
        """
        # pygments doesn't recognize .tsx files and returns None. So we need to special-case them.
        extension = os.path.splitext(filename)[1]
        if extension == ".tsx":
            return "tsx"

        try:
            lexer = pygments.lexers.get_lexer_for_filename(filename)
            return lexer.name.lower()
        except pygments.util.ClassNotFound:
            return None

    def _chunk_node(self, node: Node, file_content: str, file_metadata: Dict) -> List[FileChunk]:
        """Splits a node in the parse tree into a flat list of chunks."""
        node_chunk = FileChunk(file_content, file_metadata, node.start_byte, node.end_byte)

        if node_chunk.num_tokens <= self.max_tokens:
            return [node_chunk]

        if not node.children:
            # This is a leaf node, but it's too long. We'll have to split it with a text tokenizer.
            return self.text_chunker.chunk(file_content[node.start_byte : node.end_byte], file_metadata)

        chunks = []
        for child in node.children:
            chunks.extend(self._chunk_node(child, file_content, file_metadata))

        for chunk in chunks:
            # This should always be true. Otherwise there must be a bug in the code.
            assert chunk.num_tokens <= self.max_tokens

        # Merge neighboring chunks if their combined size doesn't exceed max_tokens. The goal is to avoid pathologically
        # small chunks that end up being undeservedly preferred by the retriever.
        merged_chunks = []
        for chunk in chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
            elif merged_chunks[-1].num_tokens + chunk.num_tokens < self.max_tokens - 50:
                # There's a good chance that merging these two chunks will be under the token limit. We're not 100% sure
                # at this point, because tokenization is not necessarily additive.
                merged = FileChunk(
                    file_content,
                    file_metadata,
                    merged_chunks[-1].start_byte,
                    chunk.end_byte,
                )
                if merged.num_tokens <= self.max_tokens:
                    merged_chunks[-1] = merged
                else:
                    merged_chunks.append(chunk)
            else:
                merged_chunks.append(chunk)
        chunks = merged_chunks

        for chunk in merged_chunks:
            # This should always be true. Otherwise there's a bug worth investigating.
            assert chunk.num_tokens <= self.max_tokens

        return merged_chunks

    @staticmethod
    def is_code_file(filename: str) -> bool:
        """Checks whether pygment & tree_sitter can parse the file as code."""
        language = CodeFileChunker._get_language_from_filename(filename)
        # tree-sitter-language-pack crashes on TypeScript files. We'll wait for a bit to see if the issue gets
        # resolved, otherwise we'll have to clone and fix the library.
        # See https://github.com/Goldziher/tree-sitter-language-pack/issues/5
        return language and language not in ["text only", "None", "typescript", "tsx"]

    @staticmethod
    def parse_tree(filename: str, content: str) -> List[str]:
        """Parses the code in a file and returns the parse tree."""
        language = CodeFileChunker._get_language_from_filename(filename)

        if not language or language in ["text only", "None"]:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None

        if language in ["typescript", "tsx"]:
            # tree-sitter-language-pack crashes on TypeScript files. We'll wait for a bit to see if the issue gets
            # resolved, otherwise we'll have to clone and fix the library.
            # See https://github.com/Goldziher/tree-sitter-language-pack/issues/5
            return None

        try:
            parser = get_parser(language)
        except LookupError:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None
        # This should never happen unless there's a bug in the code, but we'd rather not crash.
        except Exception as e:
            logging.warn("Failed to get parser for %s: %s", filename, e)
            return None

        tree = parser.parse(bytes(content, "utf8"))

        if not tree.root_node.children or tree.root_node.children[0].type == "ERROR":
            logging.warning("Failed to parse code in %s.", filename)
            return None
        return tree

    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        """Chunks a code file into smaller pieces."""
        file_content = content
        file_metadata = metadata
        file_path = metadata["file_path"]

        if not file_content.strip():
            return []

        tree = self.parse_tree(file_path, file_content)
        if tree is None:
            return []

        file_chunks = self._chunk_node(tree.root_node, file_content, file_metadata)
        for chunk in file_chunks:
            # Make sure that the chunk has content and doesn't exceed the max_tokens limit. Otherwise there must be
            # a bug in the code.
            assert (
                chunk.num_tokens <= self.max_tokens
            ), f"Chunk size {chunk.num_tokens} exceeds max_tokens {self.max_tokens}."

        return file_chunks


class TextFileChunker(Chunker):
    """Wrapper around semchunk: https://github.com/umarbutler/semchunk."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.count_tokens = lambda text: len(tokenizer.encode(text, disallowed_special=()))

    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        """Chunks a text file into smaller pieces."""
        file_content = content
        file_metadata = metadata
        file_path = file_metadata["file_path"]

        # We need to allocate some tokens for the filename, which is part of the chunk content.
        extra_tokens = self.count_tokens(file_path + "\n\n")
        text_chunks = chunk_via_semchunk(file_content, self.max_tokens - extra_tokens, self.count_tokens)

        file_chunks = []
        start = 0
        for text_chunk in text_chunks:
            # This assertion should always be true. Otherwise there's a bug worth finding.
            assert self.count_tokens(text_chunk) <= self.max_tokens - extra_tokens

            # Find the start/end positions of the chunks.
            start = file_content.index(text_chunk, start)
            if start == -1:
                logging.warning("Couldn't find semchunk in content: %s", text_chunk)
            else:
                end = start + len(text_chunk)
                file_chunks.append(FileChunk(file_content, file_metadata, start, end))

            start = end

        return file_chunks


class IpynbFileChunker(Chunker):
    """Extracts the python code from a Jupyter notebook, removing all the boilerplate.

    Based on https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_retrieval_augmented_generation.ipynb
    """

    def __init__(self, code_chunker: CodeFileChunker):
        self.code_chunker = code_chunker

    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        filename = metadata["file_path"]

        if not filename.lower().endswith(".ipynb"):
            logging.warn("IPYNBChunker is only for .ipynb files.")
            return []

        notebook = nbformat.reads(content, as_version=nbformat.NO_CONVERT)
        python_code = "\n".join([cell.source for cell in notebook.cells if cell.cell_type == "code"])

        tmp_metadata = {"file_path": filename.replace(".ipynb", ".py")}
        chunks = self.code_chunker.chunk(python_code, tmp_metadata)

        for chunk in chunks:
            # Update filenames back to .ipynb
            chunk.metadata = metadata
        return chunks


class UniversalFileChunker(Chunker):
    """Chunks a file into smaller pieces, regardless of whether it's code or text."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.code_chunker = CodeFileChunker(max_tokens)
        self.ipynb_chunker = IpynbFileChunker(self.code_chunker)
        self.text_chunker = TextFileChunker(max_tokens)

    def chunk(self, content: Any, metadata: Dict) -> List[Chunk]:
        if not "file_path" in metadata:
            raise ValueError("metadata must contain a 'file_path' key.")
        file_path = metadata["file_path"]

        # Figure out the appropriate chunker to use.
        if file_path.lower().endswith(".ipynb"):
            chunker = self.ipynb_chunker
        elif CodeFileChunker.is_code_file(file_path):
            chunker = self.code_chunker
        else:
            chunker = self.text_chunker

        return chunker.chunk(content, metadata)
