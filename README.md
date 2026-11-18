# openthai2.0-ocr

สคริปต์ Python สั้น ๆ สำหรับอ่านข้อความภาษาไทย/อังกฤษจากรูปภาพและไฟล์ PDF ด้วยโมเดล **OpenThai 2.0 OCR** จาก iApp

มี 2 โมเดล และ 3 โหมดการรัน:

| สคริปต์ | โมเดล | อุปกรณ์ | ผลลัพธ์ |
|---------|-------|---------|---------|
| `ocr_text.py` | [`iapp/OpenThai2.0-OCR-Text`](https://huggingface.co/iapp/OpenThai2.0-OCR-Text) — ONNX Runtime | CPU เท่านั้น ไม่ต้องใช้ GPU | ข้อความล้วน `<file>.txt` |
| `ocr_markdown.py` | [`iapp/OpenThai2.0-OCR-Markdown`](https://huggingface.co/iapp/OpenThai2.0-OCR-Markdown) — Transformers | **NVIDIA GPU (CUDA)** / Apple silicon (MPS) / CPU | Markdown `<file>.md` (ตารางเป็น HTML) |
| `ocr_markdown.py --server` | เดียวกัน แต่ยิงเข้า vLLM server | NVIDIA GPU ผ่าน vLLM | Markdown `<file>.md` |

ทั้งสองสคริปต์เรนเดอร์หน้า PDF เป็นภาพที่ **200 dpi** ด้วย `pypdfium2` ก่อนส่งเข้าโมเดล และรองรับไฟล์ภาพโดยตรง

---

## โครงสร้างโปรเจกต์

```
.
├── ocr_text.py              # OCR → ข้อความล้วน (.txt)
├── ocr_markdown.py          # OCR → Markdown (.md)
├── setup.txt                # คำสั่งติดตั้งแบบย่อ (CPU / GPU)
├── document.pdf             # ไฟล์ตัวอย่างสำหรับ ocr_text.py
└── document-markdown.pdf    # ไฟล์ตัวอย่างสำหรับ ocr_markdown.py
```

> โฟลเดอร์โมเดล `OpenThai2.0-OCR-Text/` และ `OpenThai2.0-OCR-Markdown/` ไม่ได้ถูก commit ไว้ ต้องดาวน์โหลดเองตามขั้นตอนด้านล่าง

---

## ความต้องการของระบบ

- Python **3.12**
- Hugging Face CLI (`hf`) สำหรับดาวน์โหลดโมเดล
- สำหรับโหมด GPU: NVIDIA GPU **สถาปัตยกรรม Ampere ขึ้นไป** (RTX 30-series / A-series ขึ้นไป) เพราะโค้ดใช้ `torch.bfloat16`

---

## 1) ติดตั้ง virtual environment

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

> ใน VS Code: `Ctrl + Shift + P` → **Python: Select Interpreter** → เลือก `.venv`

---

## 2) OCR Text (CPU)

```bash
pip install onnxruntime opencv-python-headless pyclipper numpy pillow pypdfium2 huggingface_hub
hf download iapp/OpenThai2.0-OCR-Text --local-dir OpenThai2.0-OCR-Text

python ocr_text.py document.pdf
python ocr_text.py scan1.png scan2.jpg report.pdf   # ใส่หลายไฟล์พร้อมกันได้
```

- บันทึกผลเป็น `document.pdf.txt` ข้าง ๆ ไฟล์ต้นฉบับ และพิมพ์ข้อความออก stdout
- ขึ้นบรรทัดสถานะ (จำนวนบรรทัด + เวลาที่ใช้ต่อหน้า) ออกทาง stderr
- ใช้ `LineOCR(..., threads=2)` — ปรับจำนวน thread ได้ในสคริปต์

---

## 3) OCR Markdown แบบ CPU

```bash
hf download iapp/OpenThai2.0-OCR-Markdown --local-dir OpenThai2.0-OCR-Markdown
pip install -e ./OpenThai2.0-OCR-Markdown

python ocr_markdown.py document-markdown.pdf
```

ช้ามาก เหมาะกับการทดสอบเท่านั้น

---

## 4) OCR Markdown แบบ GPU (Transformers + CUDA)

ไม่ต้องใช้ vLLM และไม่ต้องใช้ WSL — รันบน Windows ได้ตรง ๆ

### 4.1 ตรวจไดรเวอร์

```powershell
nvidia-smi
```

ดูเลข **CUDA Version** ที่มุมขวาบน ใช้เลือก wheel ให้ตรง

### 4.2 ติดตั้งโมเดลและแพ็กเกจ

```powershell
hf download iapp/OpenThai2.0-OCR-Markdown --local-dir OpenThai2.0-OCR-Markdown
pip install -e ./OpenThai2.0-OCR-Markdown
```

### 4.3 เปลี่ยน torch เป็น CUDA build (เวอร์ชันเดิม)

```powershell
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
```

⚠️ **ห้ามสั่ง `pip install torch torchvision` แบบไม่ระบุเวอร์ชัน** เพราะ `nemotron-parse 3.5.0.post7` pin ไว้ว่า:

| แพ็กเกจ | ช่วงที่ยอมรับ |
|---|---|
| `torch` | `>=2.11.0, <2.14` |
| `torchvision` | `>=0.26.0, <0.29` |
| `numpy` | `==2.2.6` (pin ตายตัว) |
| `Pillow` | `>=12.2.0, <13` |

ถ้าไม่ระบุเวอร์ชัน pip จะดึง torch 2.15 มา แล้วลาก `numpy` และ `Pillow` ให้เกิน pin ตามไปด้วย

เลือก index ให้ตรงกับไดรเวอร์:

| CUDA | index-url |
|---|---|
| 12.6 | `https://download.pytorch.org/whl/cu126` |
| 12.8 | `https://download.pytorch.org/whl/cu128` |
| 13.0 | `https://download.pytorch.org/whl/cu130` |
| 13.2 | `https://download.pytorch.org/whl/cu132` |

### 4.4 ตรวจสอบ

```powershell
pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -c "from nemotron_parse.inference import DocumentParser; print('ok')"
```

ควรได้ `No broken requirements found.` / `2.13.0+cu132 13.2 True True` / `ok`

### 4.5 รัน

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python ocr_markdown.py document-markdown.pdf
```

บรรทัดแรกใน stderr ต้องขึ้น **`Running the model on cuda`** ถ้าขึ้น `cpu` แปลว่า wheel ผิด

### 4.6 ใช้หลาย GPU

โค้ดใช้ `.to("cuda")` จึงจับแค่ `cuda:0` ใบเดียว ถ้ามีหลายใบ ให้เปิดหลาย terminal แล้วแบ่งไฟล์กัน:

```powershell
# terminal 1
$env:CUDA_VISIBLE_DEVICES="0"; python ocr_markdown.py batch_a\*.pdf

# terminal 2
$env:CUDA_VISIBLE_DEVICES="1"; python ocr_markdown.py batch_b\*.pdf
```

---

## 5) OCR Markdown ผ่าน vLLM server (ทางเลือก, เร็วที่สุด)

เป็นวิธีที่ iApp ใช้รัน hosted API จริง แต่ **vLLM ไม่รองรับ Windows native** ต้องใช้ WSL 2

```bash
pip install vllm==0.26.0
vllm serve ./OpenThai2.0-OCR-Markdown --served-model-name nemotron-parse --trust-remote-code
```

```bash
python ocr_markdown.py document-markdown.pdf --server http://localhost:8000/v1
```

- ชื่อ `--served-model-name nemotron-parse` ต้องตรงเป๊ะ เพราะสคริปต์ฮาร์ดโค้ด `model="nemotron-parse"` ไว้
- โดยดีฟอลต์ vLLM จอง **90% ของ VRAM** — บนการ์ดเล็กควรตั้ง `--gpu-memory-utilization` เองและลด `--max-num-seqs`

---

## ความเร็วที่วัดได้

| อุปกรณ์ | Backend | เวลา/หน้า |
|---|---|---|
| NVIDIA H100 | vLLM | 1.6 s (ค่าที่ iApp ประกาศ) |
| **NVIDIA RTX A5000** | **Transformers + CUDA (bf16)** | **8.7 s** (หน้าแรก รวม warm-up) |
| Mac M3 Max | Transformers + MPS | 15.5 s (หน้าตาราง) / 26.7 s (หน้าข้อความหนาแน่น) |

คะแนนความแม่นยำที่ iApp ประกาศทั้งหมดวัดด้วย vLLM บน NVIDIA GPU — เส้นทาง Transformers ใช้ PyTorch คนละ stack ผลลัพธ์หน้าเดียวกันอาจต่างกันเล็กน้อย

---

## หมายเหตุ / ข้อควรรู้

- **dtype:** CUDA ใช้ `bfloat16` (ต้องการ Ampere ขึ้นไป), MPS และ CPU ใช้ `float32` — การ์ดเก่ากว่า Ampere ต้องแก้สคริปต์เป็น `torch.float16`
- **`trust_remote_code=True`:** โมเดล Markdown โหลดโค้ดที่มากับ repo ของโมเดล ควรดาวน์โหลดจากแหล่งที่เชื่อถือได้เท่านั้น
- **ถอดรหัสแบบ greedy** (`temperature=0`) เพราะการ sampling ทำให้ผลลัพธ์บน Mac GPU ผิดเพี้ยนในการทดสอบของต้นทาง
- **หน้าที่อ่านไม่สำเร็จ** (`GenerationError`) จะใส่ `<!-- page N could not be read completely -->` แทน แล้วทำหน้าถัดไปต่อ
- **ไฟล์ผลลัพธ์:** ใช้ `path.name + ".md"/".txt"` จึงได้ชื่อแบบ `document.pdf.md` (คงนามสกุลเดิมไว้) และจะ **เขียนทับ** ไฟล์เดิมถ้ามีอยู่แล้ว
- **ตาราง** ในผลลัพธ์ออกมาเป็น HTML พร้อม `rowspan`/`colspan` ไม่ใช่ pipe table
- **ไม่มี batching:** สคริปต์วนทีละหน้าแบบ synchronous ถ้าต้องการ throughput สูงควรใช้ vLLM แล้วส่งหลายหน้าพร้อมกัน

---

## ที่มาของโมเดล

- OpenThai 2.0 OCR Text — https://huggingface.co/iapp/OpenThai2.0-OCR-Text (Apache-2.0)
- OpenThai 2.0 OCR Markdown — https://huggingface.co/iapp/OpenThai2.0-OCR-Markdown (OpenMDW-1.1)
- คู่มือทางการ — https://iapp.co.th/en/blog/openthai2p0-ocr-how-to-run

กรุณาตรวจสอบเงื่อนไขการใช้งาน (license) ของแต่ละโมเดลบน Hugging Face ก่อนนำไปใช้งานจริง
