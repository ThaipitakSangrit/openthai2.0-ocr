"""Read Thai and English text from pictures and PDFs with OpenThai 2.0 OCR Text.

Usage: python ocr_text.py document.pdf [more files ...]
Prints the text and saves it next to each file as <file>.txt.
"""
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "OpenThai2.0-OCR-Text"
sys.path.insert(0, str(MODEL_DIR))
from runtime.line_engine import LineOCR  # noqa: E402


def pages(path):
    """Each page of a PDF as a picture (drawn at 200 dpi), or the picture itself."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        for i in range(len(pdf)):
            yield pdf[i].render(scale=200 / 72).to_pil()
    else:
        yield str(path)


ocr = LineOCR(str(MODEL_DIR), threads=2)
for name in sys.argv[1:]:
    path = Path(name)
    texts = []
    for number, page in enumerate(pages(path), start=1):
        start = time.perf_counter()
        result = ocr.read(page)
        seconds = time.perf_counter() - start
        print(f"{path.name} page {number}: {len(result['lines'])} lines in {seconds:.2f} s", file=sys.stderr)
        texts.append(result["text"])
    text = "\n\n".join(texts)
    path.with_name(path.name + ".txt").write_text(text, encoding="utf-8")
    print(text)