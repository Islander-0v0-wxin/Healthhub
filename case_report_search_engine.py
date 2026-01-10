"""
Case Report Search Engine with Deduplication (PubMed + PMC)

Enhanced version with multiple search modes:
1. Search by disease keyword
2. Search by custom query
3. Search by direct PMIDs or PMCIDs
4. Download full-text XML
5. Optional: Download media files (figures, images)
6. Build relational database with deduplication
"""

from Bio import Entrez
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import sqlite3
import requests
import time
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import os

# Configuration
PMC_ARTICLE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
DEFAULT_PAGE_SIZE = 500
api_key = os.environ.get('NCBI_API_KEY', None)

MONTHS = {
    'jan': '01', 'january': '01',
    'feb': '02', 'february': '02',
    'mar': '03', 'march': '03',
    'apr': '04', 'april': '04',
    'may': '05',
    'jun': '06', 'june': '06',
    'jul': '07', 'july': '07',
    'aug': '08', 'august': '08',
    'sep': '09', 'september': '09',
    'oct': '10', 'october': '10',
    'nov': '11', 'november': '11',
    'dec': '12', 'december': '12'
}


def get_existing_pmcids(db_path):
    """Load existing PMCIDs from database to avoid reprocessing."""
    db_path = Path(db_path)

    if not db_path.exists():
        return set()

    try:
        if db_path.suffix == '.db':
            conn = sqlite3.connect(str(db_path))
            query = "SELECT DISTINCT pmcid FROM articles WHERE pmcid IS NOT NULL"
            df = pd.read_sql_query(query, conn)
            conn.close()
            existing = set(df['pmcid'].dropna().tolist())
            print(f"Found {len(existing):,} existing PMCIDs in database")
            return existing
        elif db_path.suffix == '.csv':
            df = pd.read_csv(db_path)
            if 'pmcid' in df.columns:
                existing = set(df['pmcid'].dropna().tolist())
                print(f"Found {len(existing):,} existing PMCIDs in CSV")
                return existing
            else:
                print("Warning: 'pmcid' column not found in CSV")
                return set()
        else:
            print(f"Warning: Unsupported file format: {db_path.suffix}")
            return set()
    except Exception as e:
        print(f"Warning: Could not load existing PMCIDs: {e}")
        return set()


def sanitize_filename(name):
    """Create safe filename from disease query."""
    safe = re.sub(r'[^\w\s-]', '', name.lower())
    safe = re.sub(r'[-\s]+', '_', safe)
    return safe.strip('_')


def build_case_query(disease_query: str, database: str = "pubmed") -> str:
    """Build query with open-access filter."""
    base_query = (
        f'({disease_query}) AND ('
        f'"case reports"[Publication Type] OR '
        f'"case report"[Title/Abstract] OR '
        f'"case reports"[Title/Abstract] OR '
        f'"case series"[Title/Abstract])'
    )

    if database == "pmc":
        return f'{base_query} AND open access[filter]'
    elif database == "pubmed":
        return f'{base_query} AND free full text[filter]'
    return base_query


def search_pmc_with_history(query, email, mindate=None, maxdate=None):
    """Search PMC using history server (no 10k limit)."""
    Entrez.email = email

    if api_key:
        Entrez.api_key = api_key

    # Add date filter if provided
    if mindate and maxdate:
        query = f'{query} AND "{mindate}"[PDAT] : "{maxdate}"[PDAT]'
    elif mindate:
        query = f'{query} AND "{mindate}"[PDAT] : "3000"[PDAT]'

    print(f"\nSearching PMC with query: {query[:100]}...")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("PMC search query is empty. Ensure a non-empty 'query' was provided.")
    handle = Entrez.esearch(db="pmc", term=str(query), retmax=0, usehistory="y", retmode="xml")
    results = Entrez.read(handle)
    handle.close()

    count = int(results["Count"])
    webenv = results["WebEnv"]
    query_key = results["QueryKey"]

    print(f"Found {count:,} PMC articles")

    if count == 0:
        return []

    # Fetch in batches
    batch_size = 200
    pmcids = []

    for start in tqdm(range(0, count, batch_size), desc="Fetching PMC IDs"):
        handle = Entrez.esearch(
            db="pmc",
            term=str(query),
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
            retmode="xml",
            usehistory="y",
        )
        batch_results = Entrez.read(handle)
        handle.close()
        pmcids.extend(batch_results["IdList"])
        time.sleep(0.34)

    return [f"PMC{pid}" for pid in pmcids]


def search_pubmed_with_history(query, email, mindate=None, maxdate=None):
    """Search PubMed using history server (no 10k limit)."""
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Add date filter
    if mindate and maxdate:
        query = f'{query} AND ("{mindate}"[PDAT] : "{maxdate}"[PDAT])'
    elif mindate:
        query = f'{query} AND "{mindate}"[PDAT] : "3000"[PDAT]'

    if not isinstance(query, str) or not query.strip():
        raise ValueError("PubMed search query is empty. Ensure a non-empty 'query' was provided.")
    print(f"\nSearching PubMed with query: {query[:100]}...")

    handle = None
    results = None
    try:
        handle = Entrez.esearch(db="pubmed", term=str(query), retmax=0, usehistory="y", retmode="xml")
        results = Entrez.read(handle)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    count = int(results["Count"])
    webenv = results["WebEnv"]
    query_key = results["QueryKey"]

    print(f"Found {count:,} PubMed articles")

    if count == 0:
        return []

    # Fetch in batches
    batch_size = 200
    pmids = []

    for start in tqdm(range(0, count, batch_size), desc="Fetching PubMed IDs"):
        handle = None
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=str(query),
                retstart=start,
                retmax=batch_size,
                webenv=webenv,
                query_key=query_key,
                usehistory="y",
                retmode="xml",
            )
            batch_results = Entrez.read(handle)
            ids = batch_results.get("IdList", [])
            if ids:
                pmids.extend(ids)
            time.sleep(0.34)
        except Exception as e:
            print(f"Error fetching PubMed IDs at start {start}: {e}")
            time.sleep(1)
            continue
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
    return pmids


def pmids_to_pmcids(pmids, email):
    """Map PubMed PMIDs to PMCIDs using Entrez.elink."""
    if not pmids:
        return []

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    pmcids = []

    # Process in chunks
    chunk_size = 100
    for i in tqdm(range(0, len(pmids), chunk_size), desc="Mapping PMIDs to PMCIDs"):
        chunk = pmids[i:i + chunk_size]
        handle = None
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=chunk,
                linkname="pubmed_pmc"
            )
            results = Entrez.read(handle)
            for record in results:
                if record.get("LinkSetDb"):
                    for link in record["LinkSetDb"][0]["Link"]:
                        pmcids.append(f"PMC{link['Id']}")
            time.sleep(0.34)
        except Exception as e:
            print(f"Error mapping PMIDs to PMCIDs in chunk starting at index {i}: {e}")
            time.sleep(1)
            continue
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

        time.sleep(0.34)

    return list(set(pmcids))


def normalize_pmcid(pmcid_str):
    """Normalize PMCID to standard format (PMC123456)."""
    s = str(pmcid_str).strip().upper().replace(" ", "")
    # Remove "PMC" prefix if present
    s = s.replace("PMC", "")
    # Extract numeric part
    numeric_id = re.sub(r"\D+", "", s)
    if numeric_id and numeric_id.isdigit():
        return f"PMC{numeric_id}"
    return None


def read_ids_from_file(file_path):
    """Read IDs from a text file (one per line)."""
    ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                ids.append(line)
    return ids


def fetch_pmc_xml(pmcids, out_dir, email):
    """Fetch PMC full-text XML using Entrez.efetch."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    saved = []

    batch_size = 100
    for i in tqdm(range(0, len(pmcids), batch_size), desc="Downloading PMC XML"):
        # Normalize: extract numeric IDs only (remove "PMC" prefix)
        chunk = []
        for pid in pmcids[i:i + batch_size]:
            numeric_id = re.sub(r"\D+", "", str(pid).strip().upper().replace(" ", ""))
            if numeric_id and numeric_id.isdigit():
                chunk.append(numeric_id)

        if not chunk:
            continue

        # Use EPost to upload IDs, then EFetch via history
        post = None
        handle = None
        try:
            post = Entrez.epost(db="pmc", id=",".join(chunk))
            post_res = Entrez.read(post)
            webenv = post_res["WebEnv"]
            query_key = post_res["QueryKey"]
        except Exception as e:
            print(f"Error posting PMC IDs in chunk starting at index {i}: {e}")
            time.sleep(1)
            continue
        finally:
            if post is not None:
                try:
                    post.close()
                except Exception:
                    pass
        try:
            handle = Entrez.efetch(
                db="pmc",
                webenv=webenv,
                query_key=query_key,
                rettype="full",
                retmode="xml",
            )
            xml_content = handle.read()
        except Exception as e:
            print(f"Error fetching batch starting at index {i}: {e}")
            time.sleep(1)
            continue
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        soup = BeautifulSoup(xml_content, "lxml-xml")
        for art in soup.find_all("article"):
            pmcid_tag = art.find('article-id', {'pub-id-type': ['pmcid', 'pmc']})
            if pmcid_tag:
                pmcid = f"PMC{pmcid_tag.get_text().replace('PMC', '')}"
                if not pmcid.upper().startswith("PMC"):
                    pmcid = "PMC" + pmcid
                out_file = Path(out_dir) / f"{pmcid}.nxml"
                out_file.write_text(str(art), encoding='utf-8')
                saved.append(pmcid)

        time.sleep(0.5)
    return list(dict.fromkeys(saved))


def _join_date_parts(year, month, day):
    """Join date parts into ISO format."""
    y = year.text.strip() if year is not None and year.text else None
    m = month.text.strip() if month is not None and month.text else None
    d = day.text.strip() if day is not None and day.text else None

    if m and not m.isdigit():
        m = MONTHS.get(m.lower()[:4], MONTHS.get(m.lower()[:3], m))
    if d and d.isdigit():
        d = d.zfill(2)
    if y and y.isdigit():
        y = y.zfill(4)

    parts = [p for p in (y, m, d) if p]
    return "-".join(parts) if parts else None


def _get_text(elem, separator=" "):
    """Recursively extract all text from an element."""
    if elem is None:
        return None

    text_parts = []
    if elem.text:
        text_parts.append(elem.text)

    for child in elem:
        child_text = _get_text(child, separator)
        if child_text:
            text_parts.append(child_text)
        if child.tail:
            text_parts.append(child.tail)

    return separator.join(text_parts).strip() if text_parts else None


def process_pmc_xml(xml_path):
    """Parse PMC XML file and extract article metadata."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract IDs
        pmcid = None
        pmid = None
        for article_id in root.findall(".//article-id"):
            pid_type = (article_id.get('pub-id-type') or "").lower()
            if pid_type in ("pmcid", "pmc"):
                id_text = article_id.text or ""
                pmcid = f"PMC{id_text.replace('PMC', '')}"
            elif pid_type == "pmid":
                pmid = article_id.text

        if not pmcid:
            return None

        # Extract metadata
        title_elem = root.find(".//article-title")
        title = _get_text(title_elem) if title_elem is not None else None

        journal_elem = root.find(".//journal-title")
        journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else None

        # Fall back to bs4 if ET-derived values are missing
        if title is None or journal is None:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            soup = BeautifulSoup(xml_content, "lxml-xml")
            title_elem = soup.find("article-title")
            journal_elem = soup.find("journal-title")
            if title is None:
                title = title_elem.get_text(" ", strip=True) if title_elem else None
            if journal is None:
                journal = journal_elem.get_text(" ", strip=True) if journal_elem else None

        # Extract publication date
        pub_date_elem = (root.find(".//pub-date[@pub-type='epub']") or
                        root.find(".//pub-date[@pub-type='ppub']") or
                        root.find(".//pub-date"))

        publication_date = None
        if pub_date_elem is not None:
            publication_date = _join_date_parts(
                pub_date_elem.find('year'),
                pub_date_elem.find('month'),
                pub_date_elem.find('day')
            )

        # Extract text
        abstract_elem = root.find(".//abstract")
        abstract = _get_text(abstract_elem, separator=" ") if abstract_elem is not None else None

        body_elem = root.find(".//body")
        full_text = _get_text(body_elem, separator="\n") if body_elem is not None else None

        # Extract article type
        article_type_attr = root.get("article-type") or root.get("article_type")
        article_type = article_type_attr.lower() if article_type_attr else None

        article = {
            'pmcid': pmcid,
            'pmid': pmid,
            'title': title,
            'journal': journal,
            'article_link': PMC_ARTICLE_URL.format(pmcid=pmcid),
            "article_type": article_type,
            'publication_date': publication_date,
            'abstract': abstract,
            'full_text': full_text,
        }

        return article

    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return None


def download_media_files(pmcids, output_dir, email):
    """Download media files (figures, images) for PMC articles."""
    from pmc_media_downloader import fetch_pmc_media_urls

    media_dir = Path(output_dir) / "media_files"
    media_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DOWNLOADING MEDIA FILES")
    print("=" * 70)

    results = {}

    for pmcid in tqdm(pmcids, desc="Downloading media"):
        try:
            # Fetch media URLs
            media_urls = fetch_pmc_media_urls(pmcid)

            if not media_urls:
                results[pmcid] = {"status": "success", "message": "No media files found", "files": []}
                continue

            # Create subdirectory for this article
            article_media_dir = media_dir / pmcid
            article_media_dir.mkdir(exist_ok=True)

            downloaded_files = []
            for media_url in media_urls:
                filename = os.path.basename(media_url)
                filepath = article_media_dir / filename

                try:
                    response = requests.get(media_url, stream=True, timeout=30)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded_files.append(str(filepath))
                except Exception as e:
                    print(f"Failed to download {filename} from {pmcid}: {e}")

            results[pmcid] = {
                "status": "success",
                "files": downloaded_files
            }

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            results[pmcid] = {
                "status": "failed",
                "error": str(e),
                "files": []
            }

    # Print summary
    total_files = sum(len(r.get("files", [])) for r in results.values())
    print(f"\nMedia download complete: {total_files} files from {len(pmcids)} articles")

    return results


def create_csv_files(articles_df, output_dir):
    """Create CSV file for articles."""
    print("\n" + "=" * 70)
    print("CREATING CSV FILES")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    articles_df.to_csv(output_path / 'articles.csv', index=False)
    print(f"✓ articles.csv ({len(articles_df):,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Build case report database with multiple search modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SEARCH MODES:
  1. By disease keyword:     --disease "rheumatoid arthritis"
  2. By custom query:        --custom-query "diabetes[MeSH] AND case report[pt]"
  3. By PMIDs:               --pmids 12345678,87654321 or --pmids-file ids.txt
  4. By PMCIDs:              --pmcids PMC123456,PMC654321 or --pmcids-file ids.txt

EXAMPLES:
  # Search by disease keyword
  python case_report_search_engine.py --disease "lupus" --start-date 2020/01/01 --email your@email.com

  # Search by custom query
  python case_report_search_engine.py --custom-query "diabetes[MeSH] AND case report[pt]" --start-date 2020/01/01 --email your@email.com

  # Search by PMIDs (comma-separated)
  python case_report_search_engine.py --pmids 23256146,21876761 --email your@email.com

  # Search by PMCIDs from file
  python case_report_search_engine.py --pmcids-file pmcids.txt --email your@email.com --download-media
        """
    )

    # Search mode options (mutually exclusive)
    search_group = parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument('--disease', type=str,
                             help='Disease query term (e.g., "rheumatoid arthritis")')
    search_group.add_argument('--custom-query', type=str,
                             help='Custom PubMed/PMC query')
    search_group.add_argument('--pmids', type=str,
                             help='Comma-separated list of PMIDs (e.g., "12345678,87654321")')
    search_group.add_argument('--pmcids', type=str,
                             help='Comma-separated list of PMCIDs (e.g., "PMC123456,PMC654321")')
    search_group.add_argument('--pmids-file', type=str,
                             help='File containing PMIDs (one per line)')
    search_group.add_argument('--pmcids-file', type=str,
                             help='File containing PMCIDs (one per line)')

    # Date filters (optional for ID-based searches)
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY/MM/DD) - optional for ID-based searches')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY/MM/DD)')

    # Required
    parser.add_argument('--email', type=str, required=True,
                       help='Your email (required by NCBI)')

    # Optional
    parser.add_argument('--output-dir', type=str, default='case_report_output',
                       help='Output directory (default: case_report_output)')
    parser.add_argument('--existing-db', type=str, default=None,
                       help='Path to existing database to update')
    parser.add_argument('--db-name', type=str, default=None,
                       help='Custom database filename')
    parser.add_argument('--download-media', action='store_true',
                       help='Download media files (figures, images)')

    args = parser.parse_args()

    # Determine search mode
    search_mode = None
    all_pmcids = []

    if args.disease:
        search_mode = "disease"
        search_term = args.disease
    elif args.custom_query:
        search_mode = "custom_query"
        search_term = args.custom_query
    elif args.pmids:
        search_mode = "pmids"
        pmids = [p.strip() for p in args.pmids.split(',')]
        print(f"\n{'='*70}")
        print(f"SEARCH MODE: Direct PMID Input")
        print(f"{'='*70}")
        print(f"Processing {len(pmids)} PMIDs...")
        all_pmcids = pmids_to_pmcids(pmids, args.email)
        search_term = "direct_pmids"
    elif args.pmcids:
        search_mode = "pmcids"
        all_pmcids = [normalize_pmcid(p.strip()) for p in args.pmcids.split(',')]
        all_pmcids = [p for p in all_pmcids if p]  # Remove None values
        print(f"\n{'='*70}")
        print(f"SEARCH MODE: Direct PMCID Input")
        print(f"{'='*70}")
        print(f"Processing {len(all_pmcids)} PMCIDs...")
        search_term = "direct_pmcids"
    elif args.pmids_file:
        search_mode = "pmids_file"
        pmids = read_ids_from_file(args.pmids_file)
        print(f"\n{'='*70}")
        print(f"SEARCH MODE: PMIDs from File")
        print(f"{'='*70}")
        print(f"Loaded {len(pmids)} PMIDs from {args.pmids_file}")
        all_pmcids = pmids_to_pmcids(pmids, args.email)
        search_term = Path(args.pmids_file).stem
    elif args.pmcids_file:
        search_mode = "pmcids_file"
        pmcids_raw = read_ids_from_file(args.pmcids_file)
        all_pmcids = [normalize_pmcid(p) for p in pmcids_raw]
        all_pmcids = [p for p in all_pmcids if p]
        print(f"\n{'='*70}")
        print(f"SEARCH MODE: PMCIDs from File")
        print(f"{'='*70}")
        print(f"Loaded {len(all_pmcids)} PMCIDs from {args.pmcids_file}")
        search_term = Path(args.pmcids_file).stem

    # Determine database path
    if args.existing_db:
        db_path = Path(args.existing_db)
        output_dir = db_path.parent
        print(f"\n✓ Using existing database: {db_path}")
    else:
        search_term_safe = sanitize_filename(search_term[:30])
        output_dir = Path(args.output_dir) / search_term_safe
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.db_name:
            db_filename = args.db_name if args.db_name.endswith('.db') else f"{args.db_name}.db"
        else:
            db_filename = f"{search_term_safe}.db"

        db_path = output_dir / db_filename

    # Load existing PMCIDs
    existing_pmcids = get_existing_pmcids(db_path)

    # Perform search if not using direct IDs
    if search_mode in ["disease", "custom_query"]:
        # Validate date requirement for query searches
        if not args.start_date:
            parser.error("--start-date is required for disease/custom-query searches")

        print("\n" + "=" * 70)
        print("SEARCHING NCBI DATABASES")
        print("=" * 70)

        if search_mode == "disease":
            print(f"Disease query: {args.disease}")
            pmc_query = build_case_query(args.disease, "pmc")
            pubmed_query = build_case_query(args.disease, "pubmed")
        else:
            print(f"Custom query: {args.custom_query[:100]}...")
            pmc_query = args.custom_query
            pubmed_query = args.custom_query

        # Search PMC
        print("\n[1/3] Searching PMC Open Access...")
        pmcids_from_pmc = search_pmc_with_history(
            pmc_query, args.email,
            mindate=args.start_date,
            maxdate=args.end_date
        )

        # Search PubMed
        print("\n[2/3] Searching PubMed with free full text filter...")
        pmids_from_pubmed = search_pubmed_with_history(
            pubmed_query, args.email,
            mindate=args.start_date,
            maxdate=args.end_date
        )

        # Map PMIDs to PMCIDs
        print("\n[3/3] Mapping PubMed PMIDs to PMCIDs...")
        pmcids_from_pubmed = pmids_to_pmcids(pmids_from_pubmed, args.email)

        # Combine and deduplicate
        print("\n" + "=" * 70)
        print("DEDUPLICATION")
        print("=" * 70)
        print(f"PMC direct search:              {len(pmcids_from_pmc):,}")
        print(f"PubMed mapped to PMC:           {len(pmcids_from_pubmed):,}")

        all_pmcids = list(set(pmcids_from_pmc + pmcids_from_pubmed))
        duplicates = len(pmcids_from_pmc) + len(pmcids_from_pubmed) - len(all_pmcids)
        print(f"Duplicates removed:             {duplicates:,}")
        print(f"Unique PMCIDs from search:      {len(all_pmcids):,}")

    # Filter out existing PMCIDs
    if existing_pmcids:
        new_pmcids = [pmcid for pmcid in all_pmcids if pmcid not in existing_pmcids]
        already_processed = len(all_pmcids) - len(new_pmcids)
        print(f"Already in database (skipped):  {already_processed:,}")
        print(f"New PMCIDs to process:          {len(new_pmcids):,}")
        all_pmcids = new_pmcids
    else:
        print(f"New database - processing all:  {len(all_pmcids):,}")

    if not all_pmcids:
        print("\n⚠ No new articles to process!")
        return

    # Fetch XML files
    print("\n" + "=" * 70)
    print("DOWNLOADING FULL TEXT")
    print("=" * 70)
    xml_dir = output_dir / "pmc_xml"
    fetch_pmc_xml(all_pmcids, xml_dir, args.email)

    # Download media files if requested
    if args.download_media:
        download_media_files(all_pmcids, output_dir, args.email)

    # Process XML files
    print("\n" + "=" * 70)
    print("PARSING XML FILES")
    print("=" * 70)
    articles = []
    xml_files = list(Path(xml_dir).glob("*.nxml"))

    if not xml_files:
        print("No XML files found to process!")
        return

    print(f"Processing {len(xml_files):,} XML files...")

    for xml_file in tqdm(xml_files, desc="Processing XML"):
        art = process_pmc_xml(xml_file)
        if art:
            articles.append(art)

    # Create DataFrame
    articles_df = pd.DataFrame(articles)

    # Create CSV files
    csv_dir = output_dir / 'csv_files'
    create_csv_files(articles_df, csv_dir)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"Output location: {output_dir.absolute()}")
    print(f"\nFinal Statistics:")
    print(f"  Total articles processed:    {len(articles_df):,}")
    print(f"  Database: {db_path.name}")
    if args.download_media:
        print(f"  Media files: {output_dir / 'media_files'}")


if __name__ == "__main__":
    main()
