# Case Report Search Engine - Usage Guide

## Overview

Enhanced case report search engine with **5 search modes**:
1. **Disease Keyword** - Search by disease name
2. **Custom Query** - Use custom PubMed/PMC queries
3. **Direct PMIDs** - Provide PMID list
4. **Direct PMCIDs** - Provide PMCID list
5. **ID Files** - Load IDs from text files

## Features

✅ Multiple search modes (keyword, query, IDs)
✅ Automatic PMID → PMCID conversion
✅ Deduplication of results
✅ Full-text XML download
✅ Optional media file download (figures, images)
✅ CSV output for easy analysis
✅ Incremental database updates

## Installation

```bash
pip install biopython beautifulsoup4 lxml pandas requests tqdm
```

## Required Parameters

- `--email`: Your email address (required by NCBI)
- One of the search mode parameters (see below)

## Search Modes

### 1. Search by Disease Keyword

Search for case reports about a specific disease.

```bash
python case_report_search_engine.py \
  --disease "rheumatoid arthritis" \
  --start-date 2020/01/01 \
  --end-date 2023/12/31 \
  --email your@email.com
```

**Required:**
- `--disease`: Disease name or medical term
- `--start-date`: Publication start date (YYYY/MM/DD)
- `--email`: Your email

**Optional:**
- `--end-date`: Publication end date

### 2. Search by Custom Query

Use your own PubMed/PMC search query.

```bash
python case_report_search_engine.py \
  --custom-query '(diabetes[MeSH]) AND ("case reports"[Publication Type])' \
  --start-date 2022/01/01 \
  --email your@email.com
```

**Use cases:**
- Complex Boolean searches
- MeSH term searches
- Specific journal filters
- Advanced PubMed queries

### 3. Search by PMIDs (Comma-Separated)

Provide a list of PubMed IDs directly.

```bash
python case_report_search_engine.py \
  --pmids 23256146,21876761,33445566 \
  --email your@email.com
```

**Features:**
- Automatically converts PMIDs to PMCIDs
- No date filter required
- Skips articles without PMCIDs

### 4. Search by PMCIDs (Comma-Separated)

Provide a list of PubMed Central IDs directly.

```bash
python case_report_search_engine.py \
  --pmcids PMC3539452,PMC3156691,PMC7654321 \
  --email your@email.com
```

**Features:**
- Direct access to PMC articles
- No conversion needed
- Faster than PMID input

### 5. Search by IDs from File

Load IDs from a text file (one ID per line).

**PMIDs from file:**
```bash
python case_report_search_engine.py \
  --pmids-file my_pmids.txt \
  --email your@email.com
```

**PMCIDs from file:**
```bash
python case_report_search_engine.py \
  --pmcids-file my_pmcids.txt \
  --email your@email.com
```

**File format (my_pmids.txt):**
```
23256146
21876761
33445566
# Comments start with #
12345678
```

## Optional Parameters

### Output Configuration

```bash
--output-dir custom_output    # Custom output directory (default: case_report_output)
--db-name mydb                # Custom database name (default: based on search term)
```

### Database Updates

```bash
--existing-db path/to/existing.db    # Update existing database (skips duplicates)
```

### Media Download

```bash
--download-media    # Download figures, images, supplementary materials
```

## Complete Examples

### Example 1: Disease Search with Media Download

```bash
python case_report_search_engine.py \
  --disease "systemic lupus erythematosus" \
  --start-date 2020/01/01 \
  --end-date 2024/12/31 \
  --email researcher@university.edu \
  --download-media \
  --output-dir lupus_cases
```

### Example 2: Custom Query for Specific Journal

```bash
python case_report_search_engine.py \
  --custom-query '("Case Reports in Rheumatology"[Journal]) AND (open access[filter])' \
  --start-date 2015/01/01 \
  --email researcher@university.edu \
  --db-name rheumatology_cases
```

### Example 3: Process List of PMIDs

```bash
python case_report_search_engine.py \
  --pmids 23256146,21876761,19876543,12345678 \
  --email researcher@university.edu \
  --download-media
```

### Example 4: Update Existing Database

```bash
python case_report_search_engine.py \
  --disease "diabetes mellitus" \
  --start-date 2024/01/01 \
  --email researcher@university.edu \
  --existing-db case_report_output/diabetes_mellitus/diabetes_mellitus.db
```

### Example 5: Batch Process from File

Create a file `pmcids.txt`:
```
PMC3539452
PMC3156691
PMC7654321
PMC8901234
```

Then run:
```bash
python case_report_search_engine.py \
  --pmcids-file pmcids.txt \
  --email researcher@university.edu \
  --download-media \
  --output-dir batch_download
```

## Output Structure

```
output_dir/
├── search_term/
│   ├── search_term.db              # SQLite database (if applicable)
│   ├── pmc_xml/                    # Full-text XML files
│   │   ├── PMC123456.nxml
│   │   ├── PMC654321.nxml
│   │   └── ...
│   ├── media_files/                # Media files (if --download-media used)
│   │   ├── PMC123456/
│   │   │   ├── figure1.jpg
│   │   │   └── figure2.png
│   │   └── PMC654321/
│   │       └── supplementary.pdf
│   └── csv_files/
│       └── articles.csv            # Article metadata
```

## Output Fields (articles.csv)

| Field | Description |
|-------|-------------|
| pmcid | PubMed Central ID |
| pmid | PubMed ID |
| title | Article title |
| journal | Journal name |
| article_link | Direct link to PMC article |
| article_type | Article type (case-reports, etc.) |
| publication_date | Publication date (YYYY-MM-DD) |
| abstract | Article abstract |
| full_text | Full article text |

## Search Mode Comparison

| Feature | Disease/Query | PMIDs | PMCIDs | ID Files |
|---------|---------------|-------|--------|----------|
| Requires dates | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Auto-conversion | N/A | ✅ Yes | N/A | ✅ (for PMIDs) |
| Deduplication | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Large batches | ✅ Yes | ⚠️ Limited | ⚠️ Limited | ✅ Yes |

## Important Notes

### PMID vs PMCID

⚠️ **Not all PubMed articles have PMCIDs!**

- **PubMed** = Citation database (all published articles)
- **PMC** = Full-text repository (subset of PubMed)
- Only PMC articles can have XML and media downloaded

When using `--pmids`, the tool will:
1. Attempt to convert each PMID to PMCID
2. Skip PMIDs that don't have PMCIDs
3. Only process articles available in PMC

### Rate Limiting

The tool respects NCBI rate limits:
- Without API key: 3 requests/second
- With API key: 10 requests/second

**To use an API key:**
```bash
export NCBI_API_KEY=your_api_key_here
python case_report_search_engine.py ...
```

Get a free API key: https://www.ncbi.nlm.nih.gov/account/settings/

### Incremental Updates

To avoid reprocessing articles:

```bash
# First run
python case_report_search_engine.py \
  --disease "arthritis" \
  --start-date 2020/01/01 \
  --email you@email.com

# Update with new articles
python case_report_search_engine.py \
  --disease "arthritis" \
  --start-date 2024/01/01 \
  --email you@email.com \
  --existing-db case_report_output/arthritis/arthritis.db
```

## Troubleshooting

### "No new articles to process"

**Causes:**
- All articles already in database
- PMIDs don't have corresponding PMCIDs
- Date range has no results

**Solutions:**
- Check date range
- Use `--pmcids` instead of `--pmids` for known PMC articles
- Verify search query

### "No XML files found"

**Causes:**
- Network issues during download
- Articles not in PMC Open Access
- Invalid PMCIDs

**Solutions:**
- Check internet connection
- Verify PMCIDs are correct and accessible
- Try again with smaller batches

### Memory Issues with Large Datasets

For very large downloads (>10,000 articles):
- Process in smaller date ranges
- Use `--pmcids-file` with batches of 1,000-2,000 IDs
- Close other applications

## Advanced Usage

### Combining with Other Tools

**Extract specific fields:**
```bash
# After running the search engine
python -c "
import pandas as pd
df = pd.read_csv('case_report_output/disease/csv_files/articles.csv')
print(df[['pmcid', 'title', 'publication_date']])
"
```

**Filter by journal:**
```python
import pandas as pd
df = pd.read_csv('case_report_output/disease/csv_files/articles.csv')
filtered = df[df['journal'].str.contains('Rheumatology', case=False)]
filtered.to_csv('rheumatology_only.csv', index=False)
```

### Automating Regular Updates

Create a script `update_database.sh`:
```bash
#!/bin/bash
python case_report_search_engine.py \
  --disease "lupus" \
  --start-date $(date -d "30 days ago" +%Y/%m/%d) \
  --end-date $(date +%Y/%m/%d) \
  --email your@email.com \
  --existing-db lupus_database/lupus.db \
  --download-media
```

Run weekly with cron:
```
0 0 * * 0 /path/to/update_database.sh
```

## Performance Tips

1. **Use NCBI API Key** - 3x faster
2. **Batch Processing** - Use ID files for large lists
3. **Incremental Updates** - Use `--existing-db` to skip duplicates
4. **Media Download** - Only use `--download-media` when needed
5. **Date Filters** - Narrow date ranges for faster searches

## License

This tool uses NCBI E-utilities and is subject to their terms of service.
For research and educational purposes only.
