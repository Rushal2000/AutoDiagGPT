import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import os
import json
from config import IMAGE_DIR, OCR_FILE


def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image for better OCR accuracy."""
    img = img.convert("L")  # Grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    img = img.filter(ImageFilter.SHARPEN)  # Sharpen
    width, height = img.size
    if width < 1000:
        scale = 1000 / width
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return img


def ocr_single_image(img_path: str) -> str:
    """Run OCR on a single image."""
    img = Image.open(img_path)
    img = preprocess_image(img)
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text.strip()


def run_ocr():
    """Run OCR on all extracted images."""
    print("=" * 60)
    print("  AutoDiagGPT — OCR on Diagram Images")
    print("=" * 60)

    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: No image directory found: {IMAGE_DIR}")
        print("   Run extract_images.py first!")
        return

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    if not image_files:
        print("Warning: No images found to OCR")
        return

    ocr_results = []

    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            text = ocr_single_image(img_path)
            if text and len(text) > 15:
                parts = os.path.splitext(img_file)[0].split("_page")
                source_pdf = parts[0] if parts else "unknown"
                page = "?"
                if len(parts) > 1:
                    page_part = parts[1].split("_img")[0]
                    page = page_part

                ocr_results.append({
                    "source_image": img_file,
                    "source_pdf": source_pdf,
                    "page": page,
                    "ocr_text": text,
                    "content_type": "diagram_ocr"
                })
                print(f"OK {img_file}: {len(text)} chars extracted")
            else:
                print(f"Skip {img_file}: No meaningful text found")
        except Exception as e:
            print(f"Error {img_file}: {e}")

    with open(OCR_FILE, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal images with text: {len(ocr_results)}")
    print(f"   Saved to: {OCR_FILE}")


if __name__ == "__main__":
    run_ocr()