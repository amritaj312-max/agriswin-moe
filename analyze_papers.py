import pymupdf
import os
import re

PAPERS_DIR = r"c:\Users\AMRITA JENA\Plant Disease Detection\research_papers"
OUTPUT_FILE = r"c:\Users\AMRITA JENA\Plant Disease Detection\research_papers\paper_analysis.txt"

# Keywords to find relevant sections
SECTION_KEYWORDS = [
    'limitation', 'drawback', 'shortcoming', 'weakness',
    'future work', 'future direction', 'future research', 'future scope',
    'conclusion', 'concluding', 'summary and conclusion',
    'discussion', 'challenges', 'open issues'
]

def extract_last_pages_text(pdf_path, num_pages=4):
    """Extract text from the last N pages of a PDF."""
    try:
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        start_page = max(0, total_pages - num_pages)
        text = ""
        for page_num in range(start_page, total_pages):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text, total_pages
    except Exception as e:
        return f"ERROR reading {pdf_path}: {e}", 0

def extract_relevant_sections(text):
    """Extract paragraphs containing limitation/future work keywords."""
    lines = text.split('\n')
    relevant = []
    capture = False
    buffer = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line is a section header containing our keywords
        is_header = any(kw in line_lower for kw in SECTION_KEYWORDS)
        
        if is_header and len(line.strip()) < 120:
            if buffer:
                relevant.append('\n'.join(buffer))
            buffer = [line.strip()]
            capture = True
        elif capture:
            buffer.append(line.strip())
    
    if buffer:
        relevant.append('\n'.join(buffer))
    
    return relevant

results = []

pdf_files = sorted([f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')])

for pdf_file in pdf_files:
    pdf_path = os.path.join(PAPERS_DIR, pdf_file)
    print(f"Processing: {pdf_file[:80]}...")
    
    text, total_pages = extract_last_pages_text(pdf_path)
    
    # Extract paper number and clean title from filename
    parts = pdf_file.split('_', 2)
    paper_num = parts[0] if parts else "?"
    year = parts[1] if len(parts) > 1 else "?"
    
    # Get relevant sections
    sections = extract_relevant_sections(text)
    
    # Also do a broader keyword search in the full last-pages text
    keyword_passages = []
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in ['limitation', 'future work', 'future direction', 
                                              'drawback', 'future scope', 'remain', 'challenge',
                                              'not address', 'does not consider', 'lacks',
                                              'could be improved', 'further research',
                                              'real-world', 'field condition', 'generali']):
            if len(para.strip()) > 30:
                keyword_passages.append(para.strip())
    
    result = f"\n{'='*80}\n"
    result += f"PAPER {paper_num} ({year}): {pdf_file}\n"
    result += f"Total Pages: {total_pages}\n"
    result += f"{'='*80}\n\n"
    
    if sections:
        result += "--- IDENTIFIED SECTIONS ---\n"
        for sec in sections:
            # Trim to reasonable length
            trimmed = sec[:2000] if len(sec) > 2000 else sec
            result += trimmed + "\n\n"
    
    if keyword_passages:
        result += "--- KEY PASSAGES (limitations/future work keywords) ---\n"
        seen = set()
        for passage in keyword_passages[:10]:
            # Deduplicate
            short = passage[:100]
            if short not in seen:
                seen.add(short)
                trimmed = passage[:1000] if len(passage) > 1000 else passage
                result += f"  > {trimmed}\n\n"
    
    if not sections and not keyword_passages:
        # Fallback: just show last page text
        result += "--- LAST PAGE TEXT (no specific sections found) ---\n"
        result += text[-2000:] + "\n"
    
    results.append(result)
    print(f"  Found {len(sections)} sections, {len(keyword_passages)} keyword passages")

# Write all results
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("RESEARCH PAPER ANALYSIS: LIMITATIONS & FUTURE WORK\n")
    f.write(f"Total Papers Analyzed: {len(pdf_files)}\n")
    f.write("=" * 80 + "\n\n")
    for r in results:
        f.write(r)

print(f"\nAnalysis complete! Written to {OUTPUT_FILE}")
print(f"Analyzed {len(pdf_files)} papers total.")
