import re
from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except FileNotFoundError:
        return f"Error: The file at {image_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"

def extract_information(extracted_text):
    brand_pattern = r"\bCOFFEE\b"
    manufacture_pattern = r"Date of Manufacture\s*:\s*([0-9]{1,2}/[0-9]{2})"
    expiry_pattern = r"Expiry Date\s*:\s*([0-9]{1,2}/[0-9]{2})"

    brand = re.search(brand_pattern, extracted_text)
    manufacture_date = re.search(manufacture_pattern, extracted_text)
    expiry_date = re.search(expiry_pattern, extracted_text)

    if brand and manufacture_date and expiry_date:
        print(f"Brand: {brand.group()}")
        print(f"Manufacturing Date: {manufacture_date.group(1)}")
        print(f"Expiry Date: {expiry_date.group(1)}")
    else:
        print(extracted_text)

image_path ="Path to any image"

if os.path.exists(image_path):
    extracted_text = perform_ocr(image_path)
    extract_information(extracted_text)
else:
    print(f"Error: Image file '{image_path}' does not exist.")