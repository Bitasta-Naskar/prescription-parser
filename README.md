# AI-Powered Prescription Parser

An OCR & NLP pipeline built in Python to parse handwritten doctor notes, extract dosage instructions, and map drug entities to standard generic medicine databases (RxNorm / DrugBank).

---

## 4-Stage Architecture

```
[ Prescription Image ] (Scan or Mobile Camera Photo)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Image Preprocessing (OpenCV)                  │  <-- [Week 1 Complete]
│  • Illumination & shadow removal (Morphological/CLAHE)  │
│  • Orientation & deskewing (Projection profiles/Hough) │
│  • Denoising & Sauvola adaptive binarization           │
│  • Line & word morphological segmentation              │
└─────────────────────────┬──────────────────────────────┘
                          │ (Line crops & cleaned binary)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Text Recognition / OCR                        │  <-- [Week 1-2 Next]
│  • Baseline: Tesseract OCR (benchmarking challenges)   │
│  • Advanced: TrOCR (Transformer handwriting model)     │
└─────────────────────────┬──────────────────────────────┘
                          │ (Raw noisy transcription)
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: NLP & Generic Drug Mapping                    │  <-- [Week 3]
│  • RapidFuzz / Levenshtein & embedding matching        │
│  • Brand-name to Generic mapping (RxNorm API)          │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Dosage Parsing & Structured Output            │  <-- [Week 3-4]
│  • Medical shorthand regex ("1-0-1", "BD", "TDS")      │
│  • Structured JSON export & Streamlit Web UI           │
└────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Prescription-parser/
├── data/
│   ├── raw/                  # Input prescriptions (scans & mobile photos)
│   ├── processed/            # Cleaned, deskewed, and binarized diagnostic stages
│   ├── segments/             # Isolated line strips cropped for OCR models
│   └── synthetic/            # Synthetic benchmark samples
├── src/
│   ├── preprocessing/
│   │   ├── illumination.py   # Morphological background division & CLAHE
│   │   ├── deskew.py         # Projection profile variance & Hough deskew
│   │   ├── binarization.py   # Bilateral denoise, Sauvola, Otsu, speckle filter
│   │   ├── segmentation.py   # Line & word morphological smearing
│   │   └── pipeline.py       # Unified PreprocessingPipeline orchestrator
│   ├── ocr/                  # (Week 1-2: Tesseract & TrOCR inference)
│   ├── nlp/                  # (Week 3: RapidFuzz & RxNorm drug mapping)
│   └── utils/
│       └── visualizer.py     # 4-stage diagnostic plotting & bounding boxes
├── notebooks/
│   └── 01_week1_preprocessing_walkthrough.ipynb  # Interactive visual guide
├── scripts/
│   ├── generate_synthetic_prescription.py        # Generates photo/scan test pairs
│   └── run_preprocessing_demo.py                 # Runs pipeline & benchmarks
├── tests/
│   └── test_preprocessing.py # Unit tests for all Stage 1 modules
└── requirements.txt
```

---

## Quick Start (Week 1)

### 1. Requirements
Ensure Python 3.10+ is installed:
```powershell
py -m pip install -r requirements.txt
```

### 2. Generate Test Prescriptions
If you don't have doctor prescription photos yet, generate realistic synthetic pairs (a clean scan and a tilted mobile photo with overhead lamp shadows):
```powershell
py scripts/generate_synthetic_prescription.py
```

### 3. Run Preprocessing Demo
Execute the full preprocessing pipeline on all images in `data/raw/`:
```powershell
py scripts/run_preprocessing_demo.py
```
This generates:
- Diagnostic 4-stage plots in `data/processed/*_stages.png`
- Binarized and deskewed images in `data/processed/`
- Cropped individual line strips in `data/segments/`

### 4. Run Unit Tests
```powershell
py -m unittest tests/test_preprocessing.py
```

---

## Roadmap

- [x] **Week 1**: Image Preprocessing Pipeline (OpenCV deskew, shadow removal, Sauvola binarization, line segmentation)
- [ ] **Week 1-2**: Install Tesseract OCR on Windows, build baseline OCR wrapper, extract raw text
- [ ] **Week 2**: Hugging Face TrOCR (`microsoft/trocr-base-handwritten`) inference on line segments
- [ ] **Week 3**: Drug matching engine (RapidFuzz + RxNorm API integration)
- [ ] **Week 3-4**: Regex dosage parsing engine & interactive Streamlit web dashboard
- [ ] **Week 4+**: CER/WER evaluation benchmark & embedding-based matching
