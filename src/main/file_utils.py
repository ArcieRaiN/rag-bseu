from pathlib import Path
from typing import List


def is_pdf_file(path: Path) -> bool:
    """Check if the given path points to a PDF file."""
    return path.suffix.lower() == ".pdf"


def ensure_pdf_exists(path: Path) -> Path:
    """
    Validate that a PDF file exists.
    Raises FileNotFoundError if the file is missing.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not is_pdf_file(path):
        raise ValueError(f"Expected a PDF file, got: {path}")
    return path


def list_pdfs(directory: Path) -> List[Path]:
    """Return all PDF files inside the given directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [p for p in directory.glob("*.pdf") if p.is_file()]


def read_pdf_bytes(path: Path) -> bytes:
    """
    Read a PDF file into memory.
    This is intentionally simple; parsing is handled elsewhere if needed.
    """
    file_path = ensure_pdf_exists(path)
    return file_path.read_bytes()

