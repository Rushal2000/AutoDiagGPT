import fitz  
fitz.TOOLS.set_icc(False)
fitz.TOOLS.mupdf_display_errors(False)
import os
from config import DATA_DIR, IMAGE_DIR


def extract_images_from_pdf(pdf_path: str) -> int:
    """Extract all images from a single PDF."""
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Skip very small images 
                if len(image_bytes) < 1000:  
                    continue

                image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(IMAGE_DIR, image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_count += 1
            except Exception as e:
                print(f"   Warning: Could not extract image {img_index+1} on page {page_num+1}: {e}")

    doc.close()
    return image_count


def run_extraction():
    """Extract images from all PDFs in the data directory."""
    print("=" * 60)
    print("  AutoDiagGPT — Image Extraction from PDFs")
    print("=" * 60)

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"Warning: No PDF files found in {DATA_DIR}")
        return

    total_images = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing: {pdf_file}")
        count = extract_images_from_pdf(pdf_path)
        print(f"   Extracted {count} images")
        total_images += count

    print(f"\n Total images extracted: {total_images}")
    print(f"   Saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    run_extraction()