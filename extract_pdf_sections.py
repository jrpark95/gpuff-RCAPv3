import PyPDF2
import pdfplumber
import os
import re

def extract_pdf_sections(pdf_path, output_dir="pdf_sections"):
    """Extract PDF content and split into manageable sections"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Keywords to search for
    keywords = {
        'cloudshine': ['cloudshine', 'cloud shine', 'immersion dose', 'external dose', 'gamma dose from airborne', 'finite cloud', 'semi-infinite'],
        'groundshine': ['groundshine', 'ground shine', 'ground deposition', 'surface contamination', 'deposited activity', 'ground dose'],
        'inhalation': ['inhalation', 'breathing rate', 'internal dose', 'committed dose', 'DCF', 'dose conversion factor', 'intake', 'CEDE'],
        'dispersion': ['dispersion', 'Gaussian plume', 'stability class', 'sigma', 'coordinate', 'polar', 'cylindrical'],
        'coefficients': ['dose coefficient', 'conversion factor', 'FGR', 'mrem', 'Sv', 'dose rate constant'],
        'integration': ['integration', 'integral', 'numerical method', 'approximation', 'grid', 'mesh', 'volume integral'],
        'shielding': ['shielding', 'protection factor', 'building', 'indoor', 'outdoor', 'roughness', 'attenuation']
    }

    # Try with pdfplumber first (better for complex PDFs)
    try:
        print(f"Processing PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")

            # Extract text from each page
            all_text = []
            page_texts = {}

            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        all_text.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
                        page_texts[i+1] = text

                        # Save individual pages for the most relevant sections
                        if i % 10 == 0:
                            print(f"Processed {i+1}/{total_pages} pages...")
                except Exception as e:
                    print(f"Error extracting page {i+1}: {e}")
                    continue

            # Save full text
            full_text_path = os.path.join(output_dir, "full_text.txt")
            with open(full_text_path, 'w', encoding='utf-8') as f:
                f.write(''.join(all_text))
            print(f"Saved full text to {full_text_path}")

            # Extract sections by keywords
            for category, terms in keywords.items():
                relevant_pages = []
                for page_num, text in page_texts.items():
                    if text:
                        text_lower = text.lower()
                        if any(term.lower() in text_lower for term in terms):
                            relevant_pages.append((page_num, text))

                if relevant_pages:
                    output_file = os.path.join(output_dir, f"{category}_sections.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== {category.upper()} RELATED SECTIONS ===\n")
                        f.write(f"Found in {len(relevant_pages)} pages\n\n")
                        for page_num, text in relevant_pages:
                            f.write(f"\n\n--- PAGE {page_num} ---\n\n")
                            f.write(text)
                    print(f"Saved {category} sections to {output_file}")

    except Exception as e:
        print(f"Error with pdfplumber, trying PyPDF2: {e}")

        # Fallback to PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                print(f"Total pages: {total_pages}")

                all_text = []
                for i in range(total_pages):
                    try:
                        page = pdf_reader.pages[i]
                        text = page.extract_text()
                        all_text.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
                        if i % 10 == 0:
                            print(f"Processed {i+1}/{total_pages} pages...")
                    except Exception as e:
                        print(f"Error extracting page {i+1}: {e}")
                        continue

                # Save full text
                full_text_path = os.path.join(output_dir, "full_text_pypdf2.txt")
                with open(full_text_path, 'w', encoding='utf-8') as f:
                    f.write(''.join(all_text))
                print(f"Saved full text to {full_text_path}")

        except Exception as e:
            print(f"Error with PyPDF2: {e}")

if __name__ == "__main__":
    pdf_path = r"docs\ML072480633-RASCAL Code Manual.pdf"
    extract_pdf_sections(pdf_path)
    print("\nExtraction completed!")