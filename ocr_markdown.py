"""Read pictures and PDFs into Markdown with OpenThai 2.0 OCR Markdown.

  python ocr_markdown.py document.pdf
      runs the model on this computer: an NVIDIA GPU, a Mac's GPU, or the CPU
  python ocr_markdown.py document.pdf --server http://localhost:8000/v1
      sends the pages to a vLLM server instead (the fastest way)

Saves the Markdown next to each file as <file>.md. Tables come out as HTML.
"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from nemotron_parse.inference import DocumentParser, GenerationError, GenerationSettings, TransformersBackend

MODEL_DIR = Path(__file__).resolve().parent / "OpenThai2.0-OCR-Markdown"
sys.path.insert(0, str(MODEL_DIR))

def pages(path):
    """Each page of a PDF as a picture (drawn at 200 dpi), or the picture itself."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        for i in range(len(pdf)):
            yield pdf[i].render(scale=200 / 72).to_pil()
    else:
        yield Image.open(path)


def local_parser():
    import torch
    from transformers import AutoModel, AutoProcessor

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    class Backend(TransformersBackend):
        # a vLLM server reads the prompt after a new line; send the same tokens here
        def __call__(self, image, *, prompt, **kwargs):
            return super().__call__(image, prompt="\n" + prompt, **kwargs)

    print(f"Running the model on {device}", file=sys.stderr)
    # read greedily (temperature 0); with sampling, a Mac's GPU garbled pages in our test
    return DocumentParser(backend=Backend(model, processor), fullpage=GenerationSettings(temperature=0))


args = argparse.ArgumentParser()
args.add_argument("files", nargs="+")
args.add_argument("--server", help="a vLLM server, for example http://localhost:8000/v1")
args = args.parse_args()

parser = DocumentParser(base_url=args.server, model="nemotron-parse") if args.server else local_parser()
with parser:
    for name in args.files:
        path = Path(name)
        parts = []
        for number, page in enumerate(pages(path), start=1):
            start = time.perf_counter()
            try:
                parts.append(parser.parse(page, mode="single")["markdown"])
            except GenerationError:
                parts.append(f"<!-- page {number} could not be read completely -->")
            print(f"{path.name} page {number}: {time.perf_counter() - start:.1f} s", file=sys.stderr)
        markdown = "\n\n".join(parts)
        path.with_name(path.name + ".md").write_text(markdown, encoding="utf-8")
        print(markdown)