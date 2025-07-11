import cv2
import os
import json
import pandas as pd
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
import camelot
import numpy as np
from PIL import Image
import logging
from datetime import datetime
import gc

class PDFTableExtractor:
    def __init__(self, model_path=None, confidence_threshold=0.2):
        """
        Initialize PDF table extractor with DocLayout-YOLO model
        
        Args:
            model_path: Path to model file (optional)
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.setup_logging()
        
        if model_path is None:
            self.logger.info("Downloading pre-trained model...")
            self.model_path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench", 
                filename="doclayout_yolo_docstructbench_imgsz1024.pt"
            )
        else:
            self.model_path = model_path
            
        self.logger.info(f"Loading model: {self.model_path}")
        self.model = YOLOv10(self.model_path)
        
        # Create output directories
        self.output_dir = "table_extraction_results"
        self.csv_dir = os.path.join(self.output_dir, "csv_files")
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.img_dir = os.path.join(self.output_dir, "annotated_images")
        
        for dir_path in [self.output_dir, self.csv_dir, self.log_dir, self.img_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize metadata storage
        self.extraction_metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'pages_processed': 0,
            'total_tables_detected': 0,
            'total_tables_extracted': 0,
            'page_results': []
        }
        
    def setup_logging(self):
        """
        Setup logging configuration
        """
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join('logs', 'table_extraction.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_tables_in_page(self, page_image_path):
        """
        Detect tables in a single page image
        
        Args:
            page_image_path: Path to the page image
            
        Returns:
            List of table bounding boxes and confidences
        """
        try:
            # Perform prediction
            det_res = self.model.predict(
                page_image_path,
                imgsz=1024,
                conf=self.confidence_threshold,
                device="cpu"
            )
            
            table_boxes = []
            table_confidences = []
            
            if len(det_res) > 0:
                result = det_res[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Filter for table class only
                    for i, cls in enumerate(boxes.cls):
                        class_name = self.model.names[int(cls)]
                        if class_name.lower() == 'table':
                            bbox = boxes.xyxy[i].tolist()
                            confidence = boxes.conf[i].item()
                            table_boxes.append(bbox)
                            table_confidences.append(confidence)
            
            return table_boxes, table_confidences
            
        except Exception as e:
            self.logger.error(f"Error detecting tables in page: {e}")
            return [], []
    
    def extract_table_region(self, page_image, bbox, page_num, table_num):
        """
        Extract table region from page image
        
        Args:
            page_image: PIL Image object
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            page_num: Page number
            table_num: Table number within the page
            
        Returns:
            Extracted table region as numpy array
        """
        # Convert PIL to OpenCV format
        page_array = np.array(page_image)
        page_cv = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
        
        # Extract coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        height, width = page_cv.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Extract table region
        table_region = page_cv[y1:y2, x1:x2]
        
        return table_region
    
    def create_annotated_image(self, page_image, table_boxes, page_num):
        """
        Create annotated image with table bounding boxes
        
        Args:
            page_image: PIL Image object
            table_boxes: List of table bounding boxes
            page_num: Page number
            
        Returns:
            Path to saved annotated image
        """
        # Convert to OpenCV format
        page_array = np.array(page_image)
        annotated = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for i, bbox in enumerate(table_boxes):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            label = f"Table {i+1}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save annotated image
        output_path = os.path.join(self.img_dir, f"page_{page_num:03d}_annotated.jpg")
        cv2.imwrite(output_path, annotated)
        
        return output_path
    
    def extract_tables_with_camelot(self, pdf_path, pages_with_tables):
        """
        Extract table content using Camelot for specific pages
        
        Args:
            pdf_path: Path to PDF file
            pages_with_tables: List of page numbers that contain tables
            
        Returns:
            List of extracted table information
        """
        extracted_tables = []
        
        for page_num in pages_with_tables:
            try:
                self.logger.info(f"Extracting tables from page {page_num} using Camelot...")
                
                # Extract tables from specific page
                tables = camelot.read_pdf(pdf_path, pages=str(page_num))
                
                for table_idx, table in enumerate(tables):
                    # Generate table name and filename
                    table_name = f"Page{page_num:03d}_Table{table_idx+1:02d}"
                    csv_filename = f"{table_name}.csv"
                    csv_path = os.path.join(self.csv_dir, csv_filename)
                    
                    # Save table as CSV
                    table.to_csv(csv_path)
                    
                    # Store metadata
                    table_info = {
                        'page_num': page_num,
                        'table_num': table_idx + 1,
                        'table_name': table_name,
                        'csv_path': csv_path,
                        'csv_filename': csv_filename,
                        'shape': table.df.shape,
                        'accuracy': getattr(table, 'accuracy', 0.0)
                    }
                    extracted_tables.append(table_info)
                    
                    self.logger.info(f"Extracted table: {table_name} -> {csv_filename}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract tables from page {page_num}: {e}")
                continue
        
        return extracted_tables
    
    def process_pdf(self, pdf_path):
        """
        Process entire PDF and extract all tables
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        self.logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Convert PDF to images
        self.logger.info("Converting PDF to images...")
        try:
            pages = convert_from_path(pdf_path, dpi=200)  # Lower DPI for memory efficiency
            self.logger.info(f"Successfully converted {len(pages)} pages")
        except Exception as e:
            self.logger.error(f"Failed to convert PDF: {e}")
            return None
        
        pages_with_tables = []
        all_detection_results = []
        
        # Process each page for table detection
        for page_num, page_image in enumerate(pages, 1):
            self.logger.info(f"Processing page {page_num}/{len(pages)}")
            
            # Save page image temporarily
            temp_page_path = os.path.join(self.img_dir, f"temp_page_{page_num}.jpg")
            page_image.save(temp_page_path)
            
            # Detect tables
            table_boxes, table_confidences = self.detect_tables_in_page(temp_page_path)
            
            if table_boxes:
                self.logger.info(f"Page {page_num}: Found {len(table_boxes)} tables")
                pages_with_tables.append(page_num)
                
                # Create annotated image
                annotated_path = self.create_annotated_image(page_image, table_boxes, page_num)
                
                # Store detection results
                page_result = {
                    'page_num': page_num,
                    'table_count': len(table_boxes),
                    'table_boxes': table_boxes,
                    'table_confidences': table_confidences,
                    'annotated_image': annotated_path
                }
                all_detection_results.append(page_result)
                
                # Log table details
                for i, (bbox, conf) in enumerate(zip(table_boxes, table_confidences)):
                    self.logger.info(f"  Table {i+1}: confidence={conf:.3f}, bbox={bbox}")
                    
            else:
                self.logger.info(f"Page {page_num}: No tables detected")
            
            # Clean up temporary file
            os.remove(temp_page_path)
            
            # Force garbage collection to manage memory
            gc.collect()
        
        # Extract tables using Camelot
        self.logger.info(f"Extracting tables from {len(pages_with_tables)} pages with detected tables")
        extracted_tables = self.extract_tables_with_camelot(pdf_path, pages_with_tables)
        
        # Update metadata
        self.extraction_metadata.update({
            'pages_processed': len(pages),
            'total_tables_detected': sum(len(result['table_boxes']) for result in all_detection_results),
            'total_tables_extracted': len(extracted_tables),
            'page_results': all_detection_results,
            'extracted_tables': extracted_tables
        })
        
        # Save detailed logs
        self.save_extraction_logs()
        
        self.logger.info("PDF processing completed successfully")
        return self.extraction_metadata
    
    def save_extraction_logs(self):
        """
        Save detailed extraction logs and metadata
        """
        # Save JSON metadata
        metadata_path = os.path.join(self.log_dir, "extraction_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_metadata, f, indent=2, ensure_ascii=False)
        
        # Save human-readable log
        log_path = os.path.join(self.log_dir, "extraction_summary.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=== PDF Table Extraction Summary ===\n\n")
            
            # General statistics
            f.write(f"Processing timestamp: {self.extraction_metadata['processing_timestamp']}\n")
            f.write(f"Total pages processed: {self.extraction_metadata['pages_processed']}\n")
            f.write(f"Total tables detected: {self.extraction_metadata['total_tables_detected']}\n")
            f.write(f"Total tables extracted: {self.extraction_metadata['total_tables_extracted']}\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\n\n")
            
            # Page-by-page results
            f.write("=== Page-by-Page Detection Results ===\n")
            for page_result in self.extraction_metadata['page_results']:
                f.write(f"\nPage {page_result['page_num']}:\n")
                f.write(f"  Tables detected: {page_result['table_count']}\n")
                f.write(f"  Annotated image: {page_result['annotated_image']}\n")
                
                for i, (bbox, conf) in enumerate(zip(page_result['table_boxes'], page_result['table_confidences'])):
                    f.write(f"  Table {i+1}:\n")
                    f.write(f"    Confidence: {conf:.3f}\n")
                    f.write(f"    Bbox: {bbox}\n")
            
            # Extracted tables
            f.write("\n=== Extracted Tables ===\n")
            for table_info in self.extraction_metadata['extracted_tables']:
                f.write(f"\nTable: {table_info['table_name']}\n")
                f.write(f"  Page: {table_info['page_num']}\n")
                f.write(f"  CSV file: {table_info['csv_filename']}\n")
                f.write(f"  Shape: {table_info['shape']}\n")
                f.write(f"  Accuracy: {table_info['accuracy']:.3f}\n")
        
        self.logger.info(f"Extraction logs saved to: {self.log_dir}")

def main():
    """
    Main function to run PDF table extraction
    """
    # Configuration
    pdf_path = "/home/xcs/intern/DocLayout-YOLO/tsmc2024yearlyreport.pdf"
    confidence_threshold = 0.2  # Lower threshold to catch more tables
    
    # Create extractor
    extractor = PDFTableExtractor(confidence_threshold=confidence_threshold)
    
    # Process PDF
    results = extractor.process_pdf(pdf_path)
    
    if results:
        print(f"\n=== Extraction Complete ===")
        print(f"Pages processed: {results['pages_processed']}")
        print(f"Tables detected: {results['total_tables_detected']}")
        print(f"Tables extracted: {results['total_tables_extracted']}")
        print(f"Results saved to: {extractor.output_dir}")
        print(f"CSV files saved to: {extractor.csv_dir}")
        print(f"Logs saved to: {extractor.log_dir}")
    else:
        print("Extraction failed!")

if __name__ == "__main__":
    main() 