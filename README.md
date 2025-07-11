# PDF Table Extraction System

A professional system for extracting tables from PDF documents using DocLayout-YOLO for detection and Camelot for content extraction.

## Overview

This system processes PDF documents to automatically detect tables and extract their content into CSV files with comprehensive logging and metadata tracking.

## Installation

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y poppler-utils
```

**macOS:**
```bash
brew install poppler
```

### Python Environment

```bash
# Create conda environment
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_system.py
```

## Workflow

The extraction process follows four main stages:

```
PDF Document
     |
     v
[1] PDF to Images ─────────┐
     |                     │
     v                     │
[2] Table Detection        │
     |                     │
     v                     │
[3] Table Extraction       │ ── Memory Management
     |                     │   - Page-by-page processing
     v                     │   - Garbage collection
[4] CSV Export & Logging ──┘   - Temporary file cleanup
     |
     v
Results Directory
```

### Stage Details

**1. PDF to Images**
- Convert PDF pages to images using pdf2image
- Process at 200 DPI for memory efficiency
- Handle one page at a time

**2. Table Detection**
- Apply DocLayout-YOLO model to each page
- Filter detections for table class only
- Use confidence threshold (default: 0.2)
- Generate bounding boxes for detected tables

**3. Table Extraction**
- Process only pages containing detected tables
- Use Camelot library for table content extraction
- Generate structured table data

**4. CSV Export & Logging**
- Export tables to CSV with naming convention: `Page{XXX}_Table{XX}.csv`
- Create annotated images showing detected table boundaries
- Generate comprehensive logs and metadata

## Usage

### Basic Usage

```bash
# Run with default settings
python pdf_table_extractor.py
```

### Custom Usage

```python
from pdf_table_extractor import PDFTableExtractor

# Initialize extractor
extractor = PDFTableExtractor(confidence_threshold=0.2)

# Process PDF
results = extractor.process_pdf("document.pdf")

# Check results
print(f"Tables extracted: {results['total_tables_extracted']}")
```

## Output Structure

```
table_extraction_results/
├── csv_files/
│   ├── Page001_Table01.csv
│   ├── Page012_Table01.csv
│   └── ...
├── logs/
│   ├── extraction_metadata.json
│   └── extraction_summary.txt
└── annotated_images/
    ├── page_001_annotated.jpg
    └── ...
```

## Requirements

### Core Dependencies
- opencv-python >= 4.12.0
- pandas >= 2.3.0
- pdf2image >= 1.17.0
- huggingface-hub >= 0.33.0
- camelot-py[cv] >= 1.0.0
- numpy >= 2.2.0
- pillow >= 11.0.0
- doclayout-yolo

### System Requirements
- Python 3.10+
- poppler-utils (for PDF processing)
- 4GB+ RAM recommended
- Internet connection (for model download)

## Configuration

Key parameters can be adjusted in the extractor initialization:

```python
extractor = PDFTableExtractor(
    confidence_threshold=0.2,  # Detection sensitivity
    output_dir="results"       # Custom output directory
)
```

## Logging and Metadata

The system generates comprehensive logs including:
- Processing timestamps
- Detection results per page
- Extraction accuracy metrics
- Table dimensions and locations
- Complete metadata in JSON format

## Performance Notes

- Memory efficient: processes one page at a time
- Automatic cleanup of temporary files
- Suitable for large PDF documents
- Processing speed depends on document complexity

## Testing

Verify system functionality:

```bash
python test_system.py
```

This validates all dependencies and core functionality.

## Troubleshooting

**Common Issues:**

1. **PDF Conversion Error**: Ensure poppler-utils is installed
2. **Memory Issues**: Reduce PDF resolution or close other applications
3. **Import Errors**: Verify all dependencies are installed correctly
4. **Model Download Fails**: Check internet connection

## Technical Details

- **Detection Model**: DocLayout-YOLO (DocStructBench variant)
- **Extraction Engine**: Camelot with lattice method
- **Image Processing**: OpenCV for annotations
- **Memory Management**: Garbage collection and temporary file cleanup

## License

This project utilizes open-source libraries under their respective licenses:
- DocLayout-YOLO: Apache License 2.0
- Camelot: MIT License
- OpenCV: Apache License 2.0
