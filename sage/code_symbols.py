"""Utilities to extract code symbols (class and method names) from code files."""

import logging
from typing import List, Tuple

from tree_sitter import Node

from sage.chunker import CodeFileChunker


def _extract_classes_and_methods(node: Node, acc: List[Tuple[str, str]], parent_class: str = None):
    """Extracts classes and methods from a tree-sitter node and places them in the `acc` accumulator."""
    if node.type in ["class_definition", "class_declaration"]:
        class_name_node = node.child_by_field_name("name")
        if class_name_node:
            class_name = class_name_node.text.decode("utf-8")
            acc.append((class_name, None))
            for child in node.children:
                _extract_classes_and_methods(child, acc, class_name)
    elif node.type in ["function_definition", "method_definition"]:
        function_name_node = node.child_by_field_name("name")
        if function_name_node:
            acc.append((parent_class, function_name_node.text.decode("utf-8")))
            # We're not going deeper into a method. This means we're missing nested functions.
    else:
        for child in node.children:
            _extract_classes_and_methods(child, acc, parent_class)


def get_code_symbols(file_path: str, content: str) -> List[Tuple[str, str]]:
    """Extracts code symbols from a file.

    Code symbols are tuples of the form (class_name, method_name). For classes, method_name is None. For methods
    that do not belong to a class, class_name is None.
    """
    if not CodeFileChunker.is_code_file(file_path):
        return []

    if not content:
        return []

    logging.info(f"Extracting code symbols from {file_path}")
    tree = CodeFileChunker.parse_tree(file_path, content)
    if not tree:
        return []

    classes_and_methods = []
    _extract_classes_and_methods(tree.root_node, classes_and_methods)
    return classes_and_methods
