"""
PMC Media Downloader
Downloads media files from PubMed Central using FTP Open Access packages.
Supports both PMID and PMCID identifiers.

This approach uses the PMC Open Access FTP archive which is more reliable
than the OAI API endpoint.
"""

import os
import re
import time
import ftplib
import tarfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

# PMC Open Access package filelist
OA_PACKAGE_FILELIST_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/filelist.csv"

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_BASE = "/pub/pmc"

# File extensions to remove after extraction (keep only media files)
UNWANTED_EXTS = (".pdf", ".nxml", ".doc", ".xlsx", ".xls", ".txt", ".zip", ".ps", ".csv", ".tiff")


def _mkdir(p: Path):
    """Create directory and parents if needed."""
    p.mkdir(parents=True, exist_ok=True)


def _reconnect_ftp(ftp: Optional[ftplib.FTP] = None) -> ftplib.FTP:
    """Reconnect to FTP server."""
    try:
        if ftp:
            try:
                ftp.quit()
            except Exception:
                pass
    finally:
        f = ftplib.FTP(FTP_HOST, timeout=60)
        f.login("anonymous", "")
        return f


def _download_filelist_csv(dst: Path):
    """Download the OA package filelist CSV if not already present."""
    _mkdir(dst.parent)
    if dst.exists():
        return
    r = requests.get(OA_PACKAGE_FILELIST_URL, timeout=60)
    r.raise_for_status()
    dst.write_bytes(r.content)


def _load_filelist(dst: Path) -> pd.DataFrame:
    """Load the filelist CSV and extract PMCID from filenames."""
    df = pd.read_csv(dst, usecols=["File"])
    df["id"] = df["File"].str.rsplit("/", n=1).str[-1].str.replace(".tar.gz", "", regex=False)
    return df


def _normalize_pmcid(x: str) -> str:
    """Normalize PMCID to standard format (PMC followed by numbers)."""
    s = str(x).strip().upper()
    m = re.search(r"PMC\d+", s)
    if m:
        return m.group(0)
    return "PMC" + re.sub(r"\D+", "", s)


def _pmcids_to_remote_paths(pmcids: List[str], df: pd.DataFrame) -> List[str]:
    """Convert list of PMCIDs to their remote FTP paths."""
    ids = [_normalize_pmcid(x) for x in pmcids]
    sub = df[df["id"].isin(ids)]
    return (FTP_BASE + "/" + sub["File"]).tolist()


def _safe_extract(tar_path: Path, extract_to: Path):
    """Safely extract tar.gz file, preventing path traversal attacks."""
    def _within(base: Path, target: Path) -> bool:
        return str(target.resolve()).startswith(str(base.resolve()))

    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            t = extract_to / m.name
            if not _within(extract_to, t):
                raise RuntimeError(f"path traversal blocked: {m.name}")
        tar.extractall(path=extract_to)


def _delete_unwanted(pdir: Path):
    """Delete unwanted file types from extracted directory."""
    if not pdir.is_dir():
        return
    for ext in UNWANTED_EXTS:
        for fp in pdir.glob(f"*{ext}"):
            try:
                fp.unlink()
            except Exception:
                pass


def _download_one_tar(ftp: ftplib.FTP, remote_path: str, out_dir: Path,
                      keep_compressed: bool, pbar: Optional[tqdm] = None) -> Path:
    """Download and optionally extract a single tar.gz package."""
    out_dir = Path(out_dir)
    _mkdir(out_dir)
    fname = os.path.basename(remote_path)
    local_tar = out_dir / fname
    pmcid = re.search(r"(PMC\d+)\.tar\.gz$", fname).group(1)
    pmcid_dir = out_dir / pmcid

    if not keep_compressed and pmcid_dir.exists():
        print(f"[skip] already extracted: {pmcid_dir}")
        return pmcid_dir
    if keep_compressed and local_tar.exists():
        print(f"[skip] already downloaded: {local_tar}")
        return local_tar

    tmp = local_tar.with_suffix(local_tar.suffix + ".part")
    tries = 0
    while True:
        try:
            with tmp.open("wb") as fh:
                ftp.retrbinary(f"RETR {remote_path}", fh.write)
            tmp.replace(local_tar)
            print(f"[ok] {remote_path} -> {local_tar}")
            break
        except Exception as e:
            tries += 1
            if tries > 3:
                raise
            print(f"[retry {tries}/3] {e}")
            time.sleep(3)
            ftp = _reconnect_ftp(ftp)

    if not keep_compressed:
        _safe_extract(local_tar, out_dir)
        try:
            local_tar.unlink()
        except Exception:
            pass
        _delete_unwanted(pmcid_dir)
        return pmcid_dir
    else:
        return local_tar


def convert_pmid_to_pmcid(pmid: str) -> Optional[str]:
    """
    Convert PMID to PMCID using NCBI ID Converter API.

    Args:
        pmid: PubMed ID

    Returns:
        PMCID if available, None otherwise
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'records' in data and len(data['records']) > 0:
            record = data['records'][0]
            if 'pmcid' in record:
                return record['pmcid']
    except Exception as e:
        print(f"Error converting PMID {pmid} to PMCID: {e}")

    return None


def download_pmc_media(
    identifiers: List[str],
    output_dir: str = "pmc_media",
    id_type: str = "pmcid",
    keep_compressed: bool = False
) -> Dict[str, Any]:
    """
    Download media files from PubMed Central articles using FTP.

    This function downloads the Open Access package for each article,
    extracts media files (images, figures), and removes unwanted files
    like PDFs, XMLs, etc.

    Args:
        identifiers: List of PubMed IDs (PMID) or PubMed Central IDs (PMCID)
        output_dir: Directory to save downloaded media files
        id_type: Type of identifier - either "pmid" or "pmcid" (default: "pmcid")
        keep_compressed: If True, keep the .tar.gz files instead of extracting

    Returns:
        Dictionary with download results for each identifier

    Note:
        - If id_type is "pmid", the function will attempt to convert PMIDs to PMCIDs
        - Not all PubMed articles are in PMC Open Access; download may fail for some
        - Only articles in PMC Open Access subset have downloadable packages
    """
    if id_type not in ["pmid", "pmcid"]:
        raise ValueError("id_type must be either 'pmid' or 'pmcid'")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Convert PMIDs to PMCIDs if needed
    pmcids_to_process = []
    for identifier in identifiers:
        if id_type == "pmid":
            print(f"Converting PMID {identifier} to PMCID...")
            pmcid = convert_pmid_to_pmcid(identifier)
            if not pmcid:
                results[identifier] = {
                    "status": "failed",
                    "error": "No PMCID found - article may not be in PMC",
                    "files": []
                }
                print(f"  [fail] No PMCID found for PMID {identifier}")
                continue
            print(f"  -> {pmcid}")
            pmcids_to_process.append((identifier, pmcid))
        else:
            pmcid = _normalize_pmcid(identifier)
            pmcids_to_process.append((identifier, pmcid))

    if not pmcids_to_process:
        return results

    # Check for already downloaded
    pmcids_to_download = []
    for orig_id, pmcid in pmcids_to_process:
        pmcid_dir = out_dir / pmcid
        if pmcid_dir.exists() and list(pmcid_dir.glob("*.*")):
            print(f"[skip] already exists: {pmcid_dir}")
            results[orig_id] = {
                "status": "success",
                "pmcid": pmcid,
                "files": [str(f) for f in pmcid_dir.glob("*.*")],
                "message": "Already downloaded"
            }
        else:
            pmcids_to_download.append((orig_id, pmcid))

    if not pmcids_to_download:
        return results

    # Download filelist and find remote paths
    filelist_csv = out_dir.parent / "oa_package_filelist.csv"
    print("Downloading OA package filelist...")
    _download_filelist_csv(filelist_csv)
    df = _load_filelist(filelist_csv)

    # Map original IDs to their remote paths
    pmcid_list = [pmcid for _, pmcid in pmcids_to_download]
    remote_paths = _pmcids_to_remote_paths(pmcid_list, df)

    if not remote_paths:
        for orig_id, pmcid in pmcids_to_download:
            results[orig_id] = {
                "status": "failed",
                "pmcid": pmcid,
                "error": "Not found in PMC Open Access subset",
                "files": []
            }
        return results

    # Download via FTP
    ftp = _reconnect_ftp()
    try:
        with tqdm(total=len(remote_paths), desc="Downloading PMC files") as pbar:
            for rp in remote_paths:
                try:
                    result_path = _download_one_tar(ftp, rp, out_dir, keep_compressed, pbar)

                    # Find which original ID this corresponds to
                    pmcid_match = re.search(r"(PMC\d+)", str(result_path))
                    if pmcid_match:
                        pmcid = pmcid_match.group(1)
                        for orig_id, orig_pmcid in pmcids_to_download:
                            if orig_pmcid == pmcid:
                                if result_path.is_dir():
                                    files = [str(f) for f in result_path.glob("*.*")]
                                else:
                                    files = [str(result_path)]
                                results[orig_id] = {
                                    "status": "success",
                                    "pmcid": pmcid,
                                    "files": files
                                }
                                break
                    pbar.update(1)
                except Exception as e:
                    print(f"[error] {rp}: {e}")
                    pbar.update(1)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    # Mark any remaining as failed
    for orig_id, pmcid in pmcids_to_download:
        if orig_id not in results:
            results[orig_id] = {
                "status": "failed",
                "pmcid": pmcid,
                "error": "Download failed or not found in OA subset",
                "files": []
            }

    return results


def download_pmc_media_for_pmcids(
    pmcids: List[str],
    out_dir: Path,
    keep_compressed: bool = False
) -> List[Path]:
    """
    Download PMC media for a list of PMCIDs.

    Args:
        pmcids: List of PMCIDs like ["PMC7450322", "PMC9989416"] (numbers-only also ok)
        out_dir: Output directory path
        keep_compressed: Keep .tar.gz instead of extracting

    Returns:
        List of paths to downloaded directories/files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded for each pmcid
    already_downloaded = []
    pmcids_to_download = []

    for pmcid in tqdm(pmcids, desc="Checking existing files"):
        normalized_pmcid = _normalize_pmcid(pmcid)
        pmcid_dir = out_dir / normalized_pmcid
        if pmcid_dir.exists() and list(pmcid_dir.glob("*.*")):
            print(f"[skip] already exists: {pmcid_dir}")
            already_downloaded.append(pmcid_dir)
        else:
            pmcids_to_download.append(pmcid)

    # If all are already downloaded, return early
    if not pmcids_to_download:
        return already_downloaded

    # Download filelist and process
    filelist_csv = out_dir.parent / "oa_package_filelist.csv"
    _download_filelist_csv(filelist_csv)
    df = _load_filelist(filelist_csv)

    remote_paths = _pmcids_to_remote_paths(pmcids_to_download, df)
    if not remote_paths:
        print(f"Warning: No OA package entries found for {pmcids_to_download}")
        return already_downloaded

    ftp = _reconnect_ftp()
    results = already_downloaded.copy()
    try:
        with tqdm(total=len(remote_paths), desc="Downloading PMC files") as pbar:
            for rp in remote_paths:
                p = _download_one_tar(ftp, rp, out_dir, keep_compressed, pbar)
                results.append(p)
                pbar.update(1)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass
    return results


# ============================================================================
# XML Processing Functions for extracting metadata
# ============================================================================

def _safe_filename(name: str) -> str:
    """Convert filename to safe format."""
    p = Path(name).name
    return re.sub(r'[^A-Za-z0-9._-]+', '_', p)


def _get_graphic_href(elem):
    """Extract href from a graphic element."""
    if elem is None:
        return None

    # Try common href attributes
    for attr in ['{http://www.w3.org/1999/xlink}href', 'href']:
        val = elem.get(attr)
        if val:
            return val

    # Check all attributes for anything containing 'href'
    for key, val in elem.attrib.items():
        if 'href' in key.lower():
            return val

    return None


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


def process_pmc_xml(xml_path) -> tuple:
    """
    Parse PMC XML file and extract tables and images metadata.

    Args:
        xml_path: Path to the XML file

    Returns:
        Tuple of (images_list, tables_list)
    """
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
            return [], []

        # Extract images metadata
        images = []
        fig_counter = 0
        for fig in root.findall(".//fig"):
            label_elem = fig.find(".//label")
            label = label_elem.text.strip() if label_elem is not None and label_elem.text else None

            caption_elem = fig.find(".//caption")
            caption = _get_text(caption_elem, separator=" ") if caption_elem is not None else None

            for g in fig.findall(".//*[@{http://www.w3.org/1999/xlink}href]") + fig.findall(".//*[@href]"):
                if g.tag not in ['graphic', 'inline-graphic', '{http://www.w3.org/1999/xlink}graphic']:
                    continue

                href = _get_graphic_href(g)
                if href:
                    fig_counter += 1
                    fname = _safe_filename(href)
                    images.append({
                        'image_id': f"{pmcid}_img_{fig_counter}",
                        'pmcid': pmcid,
                        'figure_id': fig.get('id', f'fig_{fig_counter}'),
                        'figure_label': label or f'Figure {fig_counter}',
                        'figure_number': fig_counter,
                        'caption': caption,
                        'local_path': f"images/{pmcid}/{fname}",
                        'filename': fname,
                    })

        # Extract tables metadata
        tables = []
        table_counter = 0
        for table_wrap in root.findall(".//table-wrap"):
            table_counter += 1

            label_elem = table_wrap.find(".//label")
            label = label_elem.text.strip() if label_elem is not None and label_elem.text else None

            caption_elem = table_wrap.find(".//caption")
            caption = _get_text(caption_elem, separator=" ") if caption_elem is not None else None

            # Check for table graphics (some tables are images)
            for g in table_wrap.findall(".//*[@{http://www.w3.org/1999/xlink}href]") + table_wrap.findall(".//*[@href]"):
                if g.tag not in ['graphic', 'inline-graphic', '{http://www.w3.org/1999/xlink}graphic']:
                    continue

                href = _get_graphic_href(g)
                if href:
                    fname = _safe_filename(href)
                    tables.append({
                        'table_id': f"{pmcid}_tbl_{table_counter}",
                        'pmcid': pmcid,
                        'table_wrap_id': table_wrap.get('id', f'table_{table_counter}'),
                        'table_label': label or f'Table {table_counter}',
                        'table_number': table_counter,
                        'caption': caption,
                        'local_path': f"tables/{pmcid}/{fname}",
                        'filename': fname,
                        'is_graphic': True
                    })

        return images, tables
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return [], []


def extract_metadata_from_xml(df: pd.DataFrame, xml_dir: str) -> tuple:
    """
    Extract metadata from XML files for all PMCIDs in dataframe.

    Args:
        df: DataFrame with 'pmcid' column (and optionally 'xml_path')
        xml_dir: Directory containing XML files

    Returns:
        Tuple of (images_df, tables_df)
    """
    xml_dir = Path(xml_dir)

    all_images = []
    all_tables = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing XML files"):
        pmcid = row.get('pmcid')
        if pd.isna(pmcid):
            continue

        row_xml_path = row.get('xml_path')
        if pd.notna(row_xml_path) if 'xml_path' in row else False:
            xml_path = row_xml_path if os.path.exists(row_xml_path) else None
        else:
            path = xml_dir / f"{pmcid}.nxml"
            xml_path = str(path) if path.exists() else None

        if xml_path:
            images, tables = process_pmc_xml(xml_path)
            all_images.extend(images)
            all_tables.extend(tables)
        else:
            print(f"Warning: XML not found for {pmcid}")

    images_df = pd.DataFrame(all_images)
    tables_df = pd.DataFrame(all_tables)
    return images_df, tables_df


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PMC Media Downloader - Example Usage")
    print("=" * 60)

    # Example 1: Download using PMCIDs
    print("\nExample 1: Using PMCIDs directly")
    print("-" * 40)
    pmcids = ["PMC7450322", "PMC9989416"]
    results = download_pmc_media(
        identifiers=pmcids,
        output_dir="pmc_media_output",
        id_type="pmcid"
    )

    for identifier, result in results.items():
        status = result['status']
        num_files = len(result.get('files', []))
        print(f"  {identifier}: {status} - {num_files} files")

    # Example 2: Download using PMIDs (auto-converts to PMCIDs)
    print("\nExample 2: Using PMIDs (will convert to PMCIDs)")
    print("-" * 40)
    pmids = ["23256146", "21876761"]
    results = download_pmc_media(
        identifiers=pmids,
        output_dir="pmc_media_output",
        id_type="pmid"
    )

    for identifier, result in results.items():
        status = result['status']
        num_files = len(result.get('files', []))
        pmcid = result.get('pmcid', 'N/A')
        print(f"  PMID {identifier} ({pmcid}): {status} - {num_files} files")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
