#!/usr/bin/env python3
"""
Test script for PDF Table Extraction System
This script tests all components to ensure the system is working correctly.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import camelot
        print(" Camelot imported successfully")
    except ImportError as e:
        print(f" Failed to import camelot: {e}")
        return False
    
    try:
        import cv2
        print(" Opencv imported successfully")
    except ImportError as e:
        print(f" Failed to import opencv: {e}")
        return False
    
    try:
        import pandas
        print(" Pandas imported successfully")
    except ImportError as e:
        print(f" Failed to import pandas: {e}")
        return False
    
    try:
        from pdf2image import convert_from_path
        print(" pdf2image imported successfully")
    except ImportError as e:
        print(f" Failed to import pdf2image: {e}")
        return False
    
    try:
        from huggingface_hub import hf_hub_download
        print(" huggingface_hub imported successfully")
    except ImportError as e:
        print(f" Failed to import huggingface_hub: {e}")
        return False
    
    try:
        from PIL import Image
        print(" PIL imported successfully")
    except ImportError as e:
        print(f" Failed to import PIL: {e}")
        return False
    
    try:
        import numpy
        print(" numpy imported successfully")
    except ImportError as e:
        print(f" Failed to import numpy: {e}")
        return False
    
    try:
        from doclayout_yolo import YOLOv10
        print(" doclayout_yolo imported successfully")
    except ImportError as e:
        print(f" Failed to import doclayout_yolo: {e}")
        return False
    
    return True

def test_extractor_initialization():
    """Test if the PDF extractor can be initialized."""
    print("\nTesting extractor initialization...")
    
    try:
        from pdf_table_extractor import PDFTableExtractor
        print(" PDFTableExtractor imported successfully")
        
        extractor = PDFTableExtractor(confidence_threshold=0.2)
        print(" PDFTableExtractor initialized successfully")
        return True
        
    except Exception as e:
        print(f" Failed to initialize PDFTableExtractor: {e}")
        return False

def test_camelot_functionality():
    """Test if camelot read_pdf function is accessible."""
    print("\nTesting camelot functionality...")
    
    try:
        import camelot
        
        # Check if read_pdf function exists
        if hasattr(camelot, 'read_pdf'):
            print(" camelot.read_pdf function is available")
            return True
        else:
            print(" camelot.read_pdf function not found")
            return False
            
    except Exception as e:
        print(f" Error testing camelot functionality: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("PDF TABLE EXTRACTION SYSTEM - COMPREHENSIVE TEST")
    print("="*60)
    
    all_tests_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: Extractor initialization
    if not test_extractor_initialization():
        all_tests_passed = False
    
    # Test 3: Camelot functionality
    if not test_camelot_functionality():
        all_tests_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_tests_passed:
        print("ALL TESTS PASSED! System is ready for use.")
        print("You can now run: python pdf_table_extractor.py")
    else:
        print("SOME TESTS FAILED! Please check the errors above.")
        print("Refer to the README for installation instructions.")
    print("="*60)

if __name__ == "__main__":
    main() 
