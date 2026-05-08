import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOC_FILES = [
    "docs/warmup.md",
    "docs/linear_regression.md",
    "tests/README.md",
    "docs/setup_tool/README.md",
]


def test_docs_files_exist():
    for rel_path in DOC_FILES:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(abs_path), f"{rel_path} does not exist"


def test_docs_files_not_empty():
    for rel_path in DOC_FILES:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        assert content != "", f"{rel_path} is empty"