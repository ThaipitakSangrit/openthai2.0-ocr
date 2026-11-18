# openthai2.0-ocr

สคริปต์ Python สั้น ๆ สำหรับอ่านข้อความภาษาไทย/อังกฤษจากรูปภาพและไฟล์ PDF ด้วยโมเดล **OpenThai 2.0 OCR** จาก iApp

มี 2 โหมดการทำงาน:

| สคริปต์ | โมเดล | ผลลัพธ์ | เหมาะกับ |
|---------|-------|---------|----------|
| `ocr_text.py` | [`iapp/OpenThai2.0-OCR-Text`](https://huggingface.co/iapp/OpenThai2.0-OCR-Text) (ONNX Runtime, CPU) | ข้อความล้วน `<file>.txt` | ดึงข้อความเร็ว ๆ ไม่ต้องใช้ GPU |
| `ocr_markdown.py` | [`iapp/OpenThai2.0-OCR-Markdown`](https://huggingface.co/iapp/OpenThai2.0-OCR-Markdown) (Transformers / vLLM) | Markdown `<file>.md` (ตารางเป็น HTML) | รักษาโครงสร้างเอกสาร ตาราง หัวข้อ |

ทั้งสองสคริปต์เรนเดอร์หน้า PDF เป็นภาพที่ **200 dpi** ด้วย `pypdfium2` ก่อนส่งเข้าโมเดล และรองรับไฟล์ภาพโดยตรง

---

## โครงสร้างโปรเจกต์

```
.
├── ocr_text.py              # OCR → ข้อความล้วน (.txt)
├── ocr_markdown.py          # OCR → Markdown (.md)
├── setup.txt                # คำสั่งติดตั้งแบบย่อ
├── document.pdf             # ไฟล์ตัวอย่างสำหรับ ocr_text.py
└── document-markdown.pdf    # ไฟล์ตัวอย่างสำหรับ ocr_markdown.py
```

> โฟลเดอร์โมเดล `OpenThai2.0-OCR-Text/` และ `OpenThai2.0-OCR-Markdown/` ไม่ได้ถูก commit ไว้ ต้องดาวน์โหลดเองตามขั้นตอนด้านล่าง

---

## ความต้องการของระบบ

- Python **3.12**
- Hugging Face CLI (`hf`) สำหรับดาวน์โหลดโมเดล
- GPU ไม่จำเป็น — `ocr_text.py` ใช้ CPU, `ocr_markdown.py` เลือกอุปกรณ์อัตโนมัติ (CUDA → MPS → CPU)

---

## การติดตั้ง

### 1) สร้าง virtual environment

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

> ใน VS Code: `Ctrl + Shift + P` → **Python: Select Interpreter** → เลือก `.venv`

### 2) ติดตั้งโมเดล Text (ข้อความล้วน)

```bash
pip install onnxruntime opencv-python-headless pyclipper numpy pillow pypdfium2 huggingface_hub
hf download iapp/OpenThai2.0-OCR-Text --local-dir OpenThai2.0-OCR-Text
```

### 3) ติดตั้งโมเดล Markdown

```bash
hf download iapp/OpenThai2.0-OCR-Markdown --local-dir OpenThai2.0-OCR-Markdown
pip install -e ./OpenThai2.0-OCR-Markdown
```

ขั้นตอนนี้จะติดตั้งแพ็กเกจ `nemotron_parse` ซึ่งสคริปต์ `ocr_markdown.py` เรียกใช้

---

## วิธีใช้งาน

### OCR เป็นข้อความล้วน

```bash
python ocr_text.py document.pdf
python ocr_text.py scan1.png scan2.jpg report.pdf   # ใส่หลายไฟล์พร้อมกันได้
```

- บันทึกผลเป็น `document.pdf.txt` ไว้ข้าง ๆ ไฟล์ต้นฉบับ และพิมพ์ข้อความออก stdout
- ขึ้นบรรทัดสถานะ (จำนวนบรรทัด + เวลาที่ใช้ต่อหน้า) ออกทาง stderr
- ใช้ `LineOCR(..., threads=2)` — ปรับจำนวน thread ได้ในสคริปต์

### OCR เป็น Markdown

```bash
# รันบนเครื่องตัวเอง (CUDA / MPS / CPU อัตโนมัติ)
python ocr_markdown.py document-markdown.pdf

# รันผ่าน vLLM server (เร็วที่สุด)
python ocr_markdown.py document-markdown.pdf --server http://localhost:8000/v1
```

- บันทึกผลเป็น `document-markdown.pdf.md` และพิมพ์ Markdown ออก stdout
- ถอดรหัสแบบ greedy (`temperature=0`) เพราะการ sampling ทำให้ผลลัพธ์บน Mac GPU (MPS) ผิดเพี้ยนในการทดสอบ
- ถ้าหน้าใดอ่านไม่สำเร็จ (`GenerationError`) จะใส่ `<!-- page N could not be read completely -->` แทน แล้วทำหน้าถัดไปต่อ
- ตารางในผลลัพธ์จะออกมาเป็น HTML ไม่ใช่ pipe table

---

## หมายเหตุ / ข้อควรรู้

- **การเลือก dtype:** CUDA ใช้ `bfloat16`, MPS และ CPU ใช้ `float32` — รันบน CPU ล้วนจะช้ามาก แนะนำใช้ `ocr_text.py` หรือ vLLM server แทน
- **`trust_remote_code=True`:** โมเดล Markdown โหลดโค้ดที่มากับ repo ของโมเดล ควรดาวน์โหลดจากแหล่งที่เชื่อถือได้เท่านั้น
- **ชื่อหัวข้อใน `setup.txt`:** บล็อกที่สามเขียนว่า "Run OpenThai2.0-OCR-Text with CPU" แต่จริง ๆ เป็นขั้นตอนของโมเดล **Markdown** (ยึดตาม README นี้)
- **ไฟล์ผลลัพธ์:** สคริปต์ใช้ `path.name + ".md"/".txt"` จึงได้ชื่อแบบ `document.pdf.md` (คงนามสกุลเดิมไว้) และจะ **เขียนทับ** ไฟล์เดิมถ้ามีอยู่แล้ว
- **ความเร็ว:** เวลาประมวลผลต่อหน้าแสดงทาง stderr ใช้เทียบผลระหว่างเครื่อง/โหมดได้

---

## ที่มาของโมเดล

- OpenThai 2.0 OCR Text — https://huggingface.co/iapp/OpenThai2.0-OCR-Text
- OpenThai 2.0 OCR Markdown — https://huggingface.co/iapp/OpenThai2.0-OCR-Markdown

กรุณาตรวจสอบเงื่อนไขการใช้งาน (license) ของแต่ละโมเดลบน Hugging Face ก่อนนำไปใช้งานจริง
