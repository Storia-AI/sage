"""Chunker abstraction and implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import pygments
import tiktoken
from semchunk import chunk as chunk_via_semchunk
from tree_sitter import Node
from tree_sitter_language_pack import get_parser

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of code or text extracted from a file in the repository."""

    filename: str
    start_byte: int
    end_byte: int
    _content: Optional[str] = None

    @property
    def content(self) -> Optional[str]:
        """The text content to be embedded. Might contain information beyond just the text snippet from the file."""
        return self._content

    def populate_content(self, file_content: str):
        """Populates the content of the chunk with the file path and file content."""
        self._content = (
            self.filename + "\n\n" + file_content[self.start_byte : self.end_byte]
        )

    def num_tokens(self, tokenizer):
        """Counts the number of tokens in the chunk."""
        if not self.content:
            raise ValueError("Content not populated.")
        return Chunk._cached_num_tokens(self.content, tokenizer)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_num_tokens(content: str, tokenizer):
        """Static method to cache token counts."""
        return len(tokenizer.encode(content, disallowed_special=()))

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
    """Abstract class for chunking a file into smaller pieces."""

    @abstractmethod
    def chunk(self, file_path: str, file_content: str) -> List[Chunk]:
        """Chunks a file into smaller pieces."""


class CodeChunker(Chunker):
    """Splits a code file into chunks of at most `max_tokens` tokens each."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_chunker = TextChunker(max_tokens)

    @staticmethod
    def _get_language_from_filename(filename: str):
        """Returns a canonical name for the language of the file, based on its extension.
        Returns None if the language is unknown to the pygments lexer.
        """
        try:
            lexer = pygments.lexers.get_lexer_for_filename(filename)
            return lexer.name.lower()
        except pygments.util.ClassNotFound:
            return None

    def _chunk_node(self, node: Node, filename: str, file_content: str) -> List[Chunk]:
        """Splits a node in the parse tree into a flat list of chunks."""
        node_chunk = Chunk(filename, node.start_byte, node.end_byte)
        node_chunk.populate_content(file_content)

        if node_chunk.num_tokens(self.tokenizer) <= self.max_tokens:
            return [node_chunk]

        if not node.children:
            # This is a leaf node, but it's too long. We'll have to split it with a text tokenizer.
            return self.text_chunker.chunk(
                filename, file_content[node.start_byte : node.end_byte]
            )

        chunks = []
        for child in node.children:
            chunks.extend(self._chunk_node(child, filename, file_content))

        for chunk in chunks:
            # This should always be true. Otherwise there must be a bug in the code.
            assert chunk.content and chunk.num_tokens(self.tokenizer) <= self.max_tokens

        # Merge neighboring chunks if their combined size doesn't exceed max_tokens. The goal is to avoid pathologically
        # small chunks that end up being undeservedly preferred by the retriever.
        merged_chunks = []
        for chunk in chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
            elif (
                merged_chunks[-1].num_tokens(self.tokenizer)
                + chunk.num_tokens(self.tokenizer)
                < self.max_tokens - 50
            ):
                # There's a good chance that merging these two chunks will be under the token limit. We're not 100% sure
                # at this point, because tokenization is not necessarily additive.
                merged = Chunk(
                    merged_chunks[-1].filename,
                    merged_chunks[-1].start_byte,
                    chunk.end_byte,
                )
                merged.populate_content(file_content)
                if merged.num_tokens(self.tokenizer) <= self.max_tokens:
                    merged_chunks[-1] = merged
                else:
                    merged_chunks.append(chunk)
        chunks = merged_chunks

        for chunk in merged_chunks:
            # This should always be true. Otherwise there's a bug worth investigating.
            assert chunk.content and chunk.num_tokens(self.tokenizer) <= self.max_tokens

        return merged_chunks

    @staticmethod
    def is_code_file(filename: str) -> bool:
        """Checks whether pygment & tree_sitter can parse the file as code."""
        language = CodeChunker._get_language_from_filename(filename)
        return language and language not in ["text only", "None"]

    @staticmethod
    def parse_tree(filename: str, content: str) -> List[str]:
        """Parses the code in a file and returns the parse tree."""
        language = CodeChunker._get_language_from_filename(filename)

        if not language or language in ["text only", "None"]:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None

        try:
            parser = get_parser(language)
        except LookupError:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None

        tree = parser.parse(bytes(content, "utf8"))

        if not tree.root_node.children or tree.root_node.children[0].type == "ERROR":
            logging.warning("Failed to parse code in %s.", filename)
            return None
        return tree

    def chunk(self, file_path: str, file_content: str) -> List[Chunk]:
        """Chunks a code file into smaller pieces."""
        tree = self.parse_tree(file_path, file_content)
        if tree is None:
            return []

        chunks = self._chunk_node(tree.root_node, file_path, file_content)
        for chunk in chunks:
            # Make sure that the chunk has content and doesn't exceed the max_tokens limit. Otherwise there must be
            # a bug in the code.
            assert chunk.content
            size = chunk.num_tokens(self.tokenizer)
            assert (
                size <= self.max_tokens
            ), f"Chunk size {size} exceeds max_tokens {self.max_tokens}."

        return chunks


class TextChunker(Chunker):
    """Wrapper around semchunk: https://github.com/umarbutler/semchunk."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

        tokenizer = tiktoken.get_encoding("cl100k_base")
        self.count_tokens = lambda text: len(
            tokenizer.encode(text, disallowed_special=())
        )

    def chunk(self, file_path: str, file_content: str) -> List[Chunk]:
        """Chunks a text file into smaller pieces."""
        # We need to allocate some tokens for the filename, which is part of the chunk content.
        extra_tokens = self.count_tokens(file_path + "\n\n")
        text_chunks = chunk_via_semchunk(
            file_content, self.max_tokens - extra_tokens, self.count_tokens
        )

        chunks = []
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
                chunks.append(Chunk(file_path, start, end, text_chunk))

            start = end
        return chunks


class UniversalChunker(Chunker):
    """Chunks a file into smaller pieces, regardless of whether it's code or text."""

    def __init__(self, max_tokens: int):
        self.code_chunker = CodeChunker(max_tokens)
        self.text_chunker = TextChunker(max_tokens)

    def chunk(self, file_path: str, file_content: str) -> List[Chunk]:
        if CodeChunker.is_code_file(file_path):
            return self.code_chunker.chunk(file_path, file_content)
        return self.text_chunker.chunk(file_path, file_content)
