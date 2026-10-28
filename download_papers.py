import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import re

OUTPUT_DIR = os.path.join(os.getcwd(), "research_papers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of highly relevant, influential search queries on arXiv to get a diverse set of the best 20 papers
queries = [
    'ti:"plant disease" AND (abs:transformer OR abs:attention OR abs:vit OR abs:swin)',
    'abs:"plant disease" AND abs:"knowledge distillation"',
    'ti:"crop disease" AND (abs:mobile OR abs:edge OR abs:lightweight)',
    'all:"plant disease detection" AND (all:"transformer" OR all:"deep learning")',
    'abs:"plant disease" AND (abs:"detection transformer" OR abs:"detr" OR abs:"yolo")',
    'all:"crop disease" AND (all:"multimodal" OR all:"edge" OR all:"offline")'
]

papers = []
seen_ids = set()

headers = {'User-Agent': 'ResearchPaperDownloader/1.0 (academic research; contact@example.edu)'}

print("Searching arXiv for recent research papers (last 5 years: 2021-2026)...")

for q in queries:
    encoded_q = urllib.parse.quote(q)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=15&sortBy=submittedDate&sortOrder=descending"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            arxiv_id_elem = entry.find('atom:id', ns)
            if arxiv_id_elem is None:
                continue
            raw_id = arxiv_id_elem.text.strip()
            # extract id part like 2305.12345
            match = re.search(r'arxiv\.org/abs/(.+)$', raw_id)
            if not match:
                continue
            paper_id = match.group(1).split('v')[0] # base id without version
            if paper_id in seen_ids:
                continue

            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Untitled"
            title = re.sub(r'\s+', ' ', title)

            published_elem = entry.find('atom:published', ns)
            published = published_elem.text[:10] if published_elem is not None else "Unknown"

            # Check if published between 2021 and 2026
            year = int(published[:4]) if published[:4].isdigit() else 2020
            if year < 2021:
                continue

            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""

            # Check PDF link
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name_elem = author_elem.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())

            seen_ids.add(paper_id)
            papers.append({
                'id': paper_id,
                'title': title,
                'year': year,
                'published': published,
                'authors': authors[:3],
                'pdf_url': pdf_url,
                'summary': summary
            })
    except Exception as e:
        print(f"Error querying {q}: {e}")
    time.sleep(1)

print(f"Total candidate papers retrieved: {len(papers)}")

# Sort by relevance / recency and pick top 20
# Let's prioritize papers specifically focusing on plant/crop disease with transformers, distillation, lightweight/mobile, detection
def score_paper(p):
    score = 0
    t = (p['title'] + " " + p['summary']).lower()
    if 'plant disease' in t or 'crop disease' in t or 'leaf disease' in t:
        score += 10
    if 'transformer' in t or 'swin' in t or 'vit' in t or 'attention' in t:
        score += 5
    if 'distillation' in t:
        score += 5
    if 'lightweight' in t or 'mobile' in t or 'edge' in t:
        score += 5
    if 'detr' in t or 'yolo' in t or 'detection' in t:
        score += 4
    if p['year'] >= 2023:
        score += 3
    elif p['year'] >= 2021:
        score += 1
    return score

papers.sort(key=score_paper, reverse=True)
top_20 = papers[:20]

print(f"\nSelected Top 20 Papers to download:")
for idx, p in enumerate(top_20, 1):
    print(f"{idx}. [{p['published']}] {p['title']}")

print("\n--- Starting PDF Downloads ---")
downloaded = []
for idx, p in enumerate(top_20, 1):
    # sanitize title for filename
    clean_title = re.sub(r'[\\/*?:"<>|]', "", p['title'])
    clean_title = clean_title.replace(" ", "_")[:60]
    filename = f"{idx:02d}_{p['year']}_{clean_title}_{p['id']}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(f"[{idx}/20] Downloading {p['title'][:50]}... from {p['pdf_url']}")
    success = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(p['pdf_url'], headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                content = resp.read()
                if len(content) > 50000: # at least 50KB to be a valid PDF
                    with open(filepath, "wb") as f:
                        f.write(content)
                    print(f"    Saved: {filename} ({len(content)//1024} KB)")
                    success = True
                    downloaded.append({
                        'index': idx,
                        'title': p['title'],
                        'authors': p['authors'],
                        'published': p['published'],
                        'year': p['year'],
                        'id': p['id'],
                        'file': filename,
                        'size_kb': len(content) // 1024
                    })
                    break
                else:
                    print(f"    Warning: downloaded file suspiciously small ({len(content)} bytes), retrying...")
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    if not success:
        print(f"    Failed to download {p['title']}")
    time.sleep(1.5)

# Save an index file of downloaded papers
index_path = os.path.join(OUTPUT_DIR, "papers_index.md")
with open(index_path, "w", encoding="utf-8") as f:
    f.write("# Top 20 Recent Research Papers (2021-2026)\n\n")
    f.write("Field: Plant Disease Detection with Deep Learning, Vision Transformers, Knowledge Distillation, and Mobile/Edge Deployment\n\n")
    f.write("| # | Year | Title | Authors | arXiv ID | File |\n")
    f.write("|---|---|---|---|---|---|\n")
    for d in downloaded:
        auth = ", ".join(d['authors']) if d['authors'] else "N/A"
        f.write(f"| {d['index']} | {d['year']} | {d['title']} | {auth} | [{d['id']}](https://arxiv.org/abs/{d['id']}) | `{d['file']}` ({d['size_kb']} KB) |\n")

print(f"\nCompleted! Successfully downloaded {len(downloaded)} papers into {OUTPUT_DIR}")
