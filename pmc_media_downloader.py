"""
PMC Media Downloader
Downloads media files from PubMed Central using either PMID or PMCID
"""

import requests
import os
from typing import List, Optional
from xml.etree import ElementTree as ET


def convert_pmid_to_pmcid(pmid: str) -> Optional[str]:
    """
    Convert PMID to PMCID using NCBI ID Converter API

    Args:
        pmid: PubMed ID

    Returns:
        PMCID if available, None otherwise
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'records' in data and len(data['records']) > 0:
            record = data['records'][0]
            if 'pmcid' in record:
                return record['pmcid']
    except Exception as e:
        print(f"Error converting PMID {pmid} to PMCID: {e}")

    return None


def download_pmc_media(identifiers: List[str],
                       output_dir: str = "pmc_media",
                       id_type: str = "pmcid") -> dict:
    """
    Download media files from PubMed Central articles

    Args:
        identifiers: List of PubMed IDs (PMID) or PubMed Central IDs (PMCID)
        output_dir: Directory to save downloaded media files
        id_type: Type of identifier - either "pmid" or "pmcid" (default: "pmcid")

    Returns:
        Dictionary with download results: {identifier: {"status": "success/failed", "files": [...]}}

    Note:
        - If id_type is "pmid", the function will attempt to convert PMIDs to PMCIDs
        - Not all articles have PMCIDs; conversion may fail for some PMIDs
        - Only articles in PMC (not just PubMed) have downloadable media
    """

    if id_type not in ["pmid", "pmcid"]:
        raise ValueError("id_type must be either 'pmid' or 'pmcid'")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for identifier in identifiers:
        print(f"\nProcessing {id_type.upper()}: {identifier}")

        # Convert PMID to PMCID if necessary
        if id_type == "pmid":
            pmcid = convert_pmid_to_pmcid(identifier)
            if not pmcid:
                results[identifier] = {
                    "status": "failed",
                    "error": "No PMCID found - article may not be in PMC",
                    "files": []
                }
                print(f"  ✗ No PMCID found for PMID {identifier}")
                continue
            print(f"  → Converted to {pmcid}")
        else:
            pmcid = identifier
            # Ensure PMCID has proper format (PMC prefix)
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"

        # Download media for this PMCID
        try:
            media_files = fetch_pmc_media_urls(pmcid)

            if not media_files:
                results[identifier] = {
                    "status": "success",
                    "message": "No media files found",
                    "files": []
                }
                print(f"  ℹ No media files found for {pmcid}")
                continue

            # Create subdirectory for this article
            article_dir = os.path.join(output_dir, pmcid)
            os.makedirs(article_dir, exist_ok=True)

            downloaded_files = []
            for media_url in media_files:
                filename = os.path.basename(media_url)
                filepath = os.path.join(article_dir, filename)

                try:
                    print(f"  ↓ Downloading: {filename}")
                    response = requests.get(media_url, stream=True)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded_files.append(filepath)
                    print(f"    ✓ Saved to: {filepath}")

                except Exception as e:
                    print(f"    ✗ Failed to download {filename}: {e}")

            results[identifier] = {
                "status": "success",
                "pmcid": pmcid,
                "files": downloaded_files
            }

        except Exception as e:
            results[identifier] = {
                "status": "failed",
                "error": str(e),
                "files": []
            }
            print(f"  ✗ Error processing {pmcid}: {e}")

    return results


def fetch_pmc_media_urls(pmcid: str) -> List[str]:
    """
    Fetch media file URLs from a PMC article

    Args:
        pmcid: PubMed Central ID (with or without PMC prefix)

    Returns:
        List of media file URLs
    """
    # Ensure PMCID has proper format
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    # Remove PMC prefix for API call
    pmc_number = pmcid.replace("PMC", "")

    # Fetch article XML from PMC OA service
    url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmc_number}&metadataPrefix=pmc"

    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse XML to find media files
        root = ET.fromstring(response.content)

        media_urls = []

        # Find all graphic elements (images, figures)
        for graphic in root.iter('{http://dtd.nlm.nih.gov/publishing/2.3}graphic'):
            href = graphic.get('{http://www.w3.org/1999/xlink}href')
            if href:
                # Construct full URL to the media file
                media_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{href}"
                media_urls.append(media_url)

        # Also check for supplementary material
        for media in root.iter('{http://dtd.nlm.nih.gov/publishing/2.3}media'):
            href = media.get('{http://www.w3.org/1999/xlink}href')
            if href:
                media_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{href}"
                media_urls.append(media_url)

        return media_urls

    except Exception as e:
        print(f"Error fetching media URLs for {pmcid}: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Example with PMCIDs
    print("=" * 60)
    print("Example 1: Using PMCIDs directly")
    print("=" * 60)
    pmcids = ["PMC3539452", "PMC3156691"]
    results_pmcid = download_pmc_media(pmcids, output_dir="media_output", id_type="pmcid")

    print("\n" + "=" * 60)
    print("Example 2: Using PMIDs (will convert to PMCIDs)")
    print("=" * 60)
    pmids = ["23256146", "21876761"]  # These should have corresponding PMCIDs
    results_pmid = download_pmc_media(pmids, output_dir="media_output", id_type="pmid")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for identifier, result in {**results_pmcid, **results_pmid}.items():
        status = result['status']
        num_files = len(result.get('files', []))
        print(f"{identifier}: {status} - {num_files} files downloaded")
