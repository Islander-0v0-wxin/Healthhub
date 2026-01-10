# PMC Media Downloader Usage Guide

## Overview

The `pmc_media_downloader.py` script allows you to download media files (images, figures, supplementary materials) from PubMed Central (PMC) articles using either **PMID** (PubMed ID) or **PMCID** (PubMed Central ID).

## Key Features

- ✅ **Flexible Input**: Use either PMID or PMCID
- ✅ **Automatic Conversion**: PMIDs are automatically converted to PMCIDs when needed
- ✅ **Handles Missing PMCIDs**: Gracefully handles cases where articles don't have PMCIDs
- ✅ **Batch Processing**: Process multiple IDs at once
- ✅ **Organized Output**: Files are organized in subdirectories by PMCID

## Installation

```bash
pip install requests
```

## Basic Usage

### Using PMCIDs (Direct)

```python
from pmc_media_downloader import download_pmc_media

# Download media using PMCIDs
pmcids = ["PMC3539452", "PMC3156691"]
results = download_pmc_media(
    identifiers=pmcids,
    output_dir="my_media",
    id_type="pmcid"  # Use PMCID directly
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

## Function Parameters

### `download_pmc_media()`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `identifiers` | List[str] | Yes | - | List of PMIDs or PMCIDs |
| `output_dir` | str | No | "pmc_media" | Directory to save downloaded files |
| `id_type` | str | No | "pmcid" | Type of identifier: "pmid" or "pmcid" |

## Return Value

Returns a dictionary with download results:

```python
{
    "identifier": {
        "status": "success" or "failed",
        "pmcid": "PMC123456",  # (if applicable)
        "files": ["/path/to/file1.jpg", "/path/to/file2.png"],
        "error": "error message"  # (if failed)
    }
}
```

## Important Notes

### Not All Articles Have PMCIDs

⚠️ **Important**: Not all PubMed articles have PMCIDs. Only articles that are available in PubMed Central (PMC) have PMCIDs.

- **PubMed** is a database of citations and abstracts
- **PubMed Central (PMC)** is a repository of full-text articles
- Only articles in PMC can have media files downloaded

When using `id_type="pmid"`, the function will:
1. Attempt to convert the PMID to a PMCID
2. If no PMCID is found, it will return a "failed" status with the message: "No PMCID found - article may not be in PMC"

### Example: Handling Missing PMCIDs

```python
pmids = ["12345678", "98765432"]  # Some may not have PMCIDs
results = download_pmc_media(pmids, id_type="pmid")

for pmid, result in results.items():
    if result["status"] == "failed":
        print(f"PMID {pmid}: {result.get('error', 'Unknown error')}")
    else:
        print(f"PMID {pmid}: Downloaded {len(result['files'])} files")
```

## Complete Example

```python
from pmc_media_downloader import download_pmc_media

# Mix of PMIDs - some may have PMCIDs, some may not
pmids_to_process = [
    "23256146",   # Has PMCID
    "21876761",   # Has PMCID
    "99999999"    # Hypothetical - may not have PMCID
]

print("Downloading media files using PMIDs...")
results = download_pmc_media(
    identifiers=pmids_to_process,
    output_dir="downloaded_media",
    id_type="pmid"
)

# Process results
print("\n=== Download Summary ===")
successful = 0
failed = 0

for pmid, result in results.items():
    if result["status"] == "success":
        successful += 1
        num_files = len(result.get("files", []))
        pmcid = result.get("pmcid", "N/A")
        print(f"✓ PMID {pmid} ({pmcid}): {num_files} files")
    else:
        failed += 1
        error = result.get("error", "Unknown error")
        print(f"✗ PMID {pmid}: {error}")

print(f"\nTotal: {successful} successful, {failed} failed")
```

## Output Structure

Downloaded files are organized as follows:

```
output_dir/
├── PMC3539452/
│   ├── figure1.jpg
│   ├── figure2.png
│   └── table1.pdf
├── PMC3156691/
│   ├── image1.tif
│   └── supplementary_data.xlsx
└── ...
```

## Troubleshooting

### "No PMCID found - article may not be in PMC"

This means the PMID you provided doesn't have a corresponding PMCID. The article may be:
- In PubMed but not in PubMed Central
- Behind a paywall
- Not openly accessible

**Solution**: Use `id_type="pmcid"` and provide PMCIDs directly for articles you know are in PMC.

### "No media files found"

The article exists in PMC but doesn't have any downloadable media files (figures, images, etc.).

### Network Errors

Ensure you have internet connectivity and can access NCBI servers.

## License

This tool uses public NCBI APIs and is intended for research and educational purposes.
