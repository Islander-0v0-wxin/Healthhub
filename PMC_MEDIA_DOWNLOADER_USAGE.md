# PMC Media Downloader Usage Guide

## Overview

The `pmc_media_downloader.py` script downloads media files (images, figures, supplementary materials) from PubMed Central (PMC) articles using either **PMID** (PubMed ID) or **PMCID** (PubMed Central ID).

This tool uses the **PMC Open Access FTP archive** which is more reliable than the OAI API endpoint.

## Key Features

- **Flexible Input**: Use either PMID or PMCID
- **Automatic Conversion**: PMIDs are automatically converted to PMCIDs when needed
- **FTP-based Download**: Uses reliable PMC Open Access FTP archive
- **Batch Processing**: Process multiple IDs at once
- **Smart Caching**: Skips already downloaded articles
- **Media-Only Extraction**: Removes PDFs, XMLs, and other unwanted files automatically
- **XML Metadata Extraction**: Parse article XMLs for figure/table metadata

## Installation

```bash
pip install requests pandas tqdm
```

## Basic Usage

### Using PMCIDs (Direct)

```python
from pmc_media_downloader import download_pmc_media

# Download media using PMCIDs
pmcids = ["PMC7450322", "PMC9989416"]
results = download_pmc_media(
    identifiers=pmcids,
    output_dir="my_media",
    id_type="pmcid"
)
```

### Using PMIDs (With Auto-Conversion)

```python
from pmc_media_downloader import download_pmc_media

# Download media using PMIDs (will convert to PMCIDs automatically)
pmids = ["23256146", "21876761"]
results = download_pmc_media(
    identifiers=pmids,
    output_dir="my_media",
    id_type="pmid"  # Will convert PMID → PMCID
)
```

### Using the Batch Function

```python
from pmc_media_downloader import download_pmc_media_for_pmcids
from pathlib import Path

# Download for a list of PMCIDs
pmcids = ["PMC7450322", "PMC9989416"]
paths = download_pmc_media_for_pmcids(
    pmcids=pmcids,
    out_dir=Path("./pmc_media"),
    keep_compressed=False  # Extract tar.gz files
)

for p in paths:
    print(f"Downloaded: {p}")
```

## Function Parameters

### `download_pmc_media()`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `identifiers` | List[str] | Yes | - | List of PMIDs or PMCIDs |
| `output_dir` | str | No | "pmc_media" | Directory to save downloaded files |
| `id_type` | str | No | "pmcid" | Type of identifier: "pmid" or "pmcid" |
| `keep_compressed` | bool | No | False | Keep .tar.gz files instead of extracting |

### `download_pmc_media_for_pmcids()`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pmcids` | List[str] | Yes | - | List of PMCIDs |
| `out_dir` | Path | Yes | - | Output directory path |
| `keep_compressed` | bool | No | False | Keep .tar.gz files instead of extracting |

## Return Value

`download_pmc_media()` returns a dictionary:

```python
{
    "identifier": {
        "status": "success" or "failed",
        "pmcid": "PMC123456",
        "files": ["/path/to/file1.jpg", "/path/to/file2.png"],
        "error": "error message"  # (if failed)
    }
}
```

`download_pmc_media_for_pmcids()` returns a list of Path objects pointing to downloaded directories/files.

## How It Works

1. **PMID Conversion** (if needed): Uses NCBI ID Converter API to convert PMIDs to PMCIDs
2. **Filelist Download**: Downloads the OA package filelist CSV from NCBI FTP
3. **FTP Download**: Downloads tar.gz packages via anonymous FTP from `ftp.ncbi.nlm.nih.gov`
4. **Extraction**: Extracts files and removes unwanted types (PDF, NXML, DOC, etc.)
5. **Caching**: Skips articles that have already been downloaded

## Important Notes

### Not All Articles Are in PMC Open Access

⚠️ **Important**: Only articles in the **PMC Open Access Subset** can be downloaded.

- **PubMed** = database of citations and abstracts
- **PubMed Central (PMC)** = repository of full-text articles
- **PMC Open Access** = subset of PMC with freely redistributable content

When an article is not in the Open Access subset, you'll see: "Not found in PMC Open Access subset"

### File Types Removed

The following file types are automatically deleted after extraction (keeping only media):
- `.pdf`, `.nxml`, `.doc`, `.xlsx`, `.xls`, `.txt`, `.zip`, `.ps`, `.csv`, `.tiff`

## Complete Example

```python
from pmc_media_downloader import download_pmc_media

pmids_to_process = [
    "23256146",   # Has PMCID in OA
    "21876761",   # Has PMCID in OA
    "99999999"    # May not be in OA
]

print("Downloading media files using PMIDs...")
results = download_pmc_media(
    identifiers=pmids_to_process,
    output_dir="downloaded_media",
    id_type="pmid"
)

# Process results
print("\n=== Download Summary ===")
for pmid, result in results.items():
    if result["status"] == "success":
        num_files = len(result.get("files", []))
        pmcid = result.get("pmcid", "N/A")
        print(f"✓ PMID {pmid} ({pmcid}): {num_files} files")
    else:
        error = result.get("error", "Unknown error")
        print(f"✗ PMID {pmid}: {error}")
```

## XML Metadata Extraction

Extract figure and table metadata from downloaded XML files:

```python
from pmc_media_downloader import process_pmc_xml, extract_metadata_from_xml
import pandas as pd

# Single XML file
images, tables = process_pmc_xml("/path/to/article.nxml")

# Multiple XMLs from a DataFrame
df = pd.DataFrame({'pmcid': ['PMC123456', 'PMC789012']})
images_df, tables_df = extract_metadata_from_xml(df, xml_dir="/path/to/xmls")
```

## Output Structure

Downloaded files are organized as follows:

```
output_dir/
├── PMC7450322/
│   ├── figure1.jpg
│   ├── figure2.png
│   └── table1.gif
├── PMC9989416/
│   ├── image1.jpg
│   └── supplementary.gif
└── ...
```

## Troubleshooting

### "No PMCID found - article may not be in PMC"

The PMID doesn't have a corresponding PMCID. The article is in PubMed but not in PubMed Central.

### "Not found in PMC Open Access subset"

The article has a PMCID but is not in the Open Access subset (it may be access-restricted).

### "Download failed or not found in OA subset"

FTP download failed or the article package is not available.

### FTP Connection Issues

The tool automatically retries up to 3 times with reconnection on FTP errors. If issues persist, check your network/firewall settings for FTP access.

## License

This tool uses public NCBI APIs and FTP services. Intended for research and educational purposes.
