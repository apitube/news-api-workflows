# Export and Analytics

Workflow for exporting news data and building analytics pipelines using the [APITube News API](https://apitube.io).

## Overview

The **Export and Analytics** workflow demonstrates how to collect large news datasets, export them to various formats (CSV, JSON, JSONL), and build analytics pipelines for reporting. Combine pagination, filtering, and sorting to create comprehensive news datasets for analysis.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `export`                      | string  | Export format (e.g., `csv`).                                        |
| `per_page`                    | integer | Number of results per page (default: 50).                           |
| `page`                        | integer | Page number for pagination.                                          |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `title`                       | string  | Filter by keywords in article title.                                |
| `topic.id`                    | string  | Filter by topic.                                                     |
| `organization.name`           | string  | Filter by organization name (e.g., `Tesla`).                        |
| `person.name`                 | string  | Filter by person name (e.g., `Elon Musk`).                          |
| `brand.name`                  | string  | Filter by brand name.                                                |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.domain`               | string  | Filter by source domain (comma-separated).                          |
| `source.country.code`         | string  | Filter by source country.                                            |
| `source.rank.opr.min`         | integer | Minimum source OPR rank (0–7).                                       |
| `language.code`               | string  | Filter by language code.                                             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |

## Quick Start

### cURL

```bash
# Export tech news as CSV
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=technology&language.code=en&export=csv&per_page=50" -o tech_news.csv

# Paginated JSON export
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=technology&language.code=en&per_page=50&page=1" -o page1.json
```

### Python

```python
import requests
import csv

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "topic.id": "technology",
    "language.code": "en",
    "per_page": 50,
})

articles = response.json().get("results", [])

with open("news.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["title", "source", "url", "published_at", "sentiment"])
    for a in articles:
        writer.writerow([
            a["title"], a["source"]["domain"], a["href"],
            a["published_at"], a["sentiment"]["overall"]["polarity"],
        ])

print(f"Exported {len(articles)} articles to news.csv")
```

### JavaScript (Node.js)

```javascript
import { writeFileSync } from "node:fs";

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const params = new URLSearchParams({
  api_key: API_KEY,
  "topic.id": "technology",
  "language.code": "en",
  per_page: "50",
});

const response = await fetch(`${BASE_URL}?${params}`);
const data = await response.json();

writeFileSync("news.json", JSON.stringify(data.results, null, 2));
console.log(`Exported ${data.results.length} articles to news.json`);
```

### PHP

```php
$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"  => $apiKey,
    "topic.id" => "technology",
    "language" => "en",
    "per_page" => 50,
]);

$data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$articles = $data["results"] ?? [];

$fp = fopen("news.csv", "w");
fputcsv($fp, ["title", "source", "url", "published_at", "sentiment"]);
foreach ($articles as $a) {
    fputcsv($fp, [
        $a["title"], $a["source"]["domain"], $a["href"],
        $a["published_at"], $a["sentiment"]["overall"]["polarity"],
    ]);
}
fclose($fp);

echo "Exported " . count($articles) . " articles to news.csv\n";
```

## Common Use Cases

- **Dataset creation** — build large news datasets for machine learning and NLP.
- **Periodic reporting** — schedule exports for daily/weekly news reports.
- **Data warehouse loading** — pipe news data into BigQuery, Snowflake, or Redshift.
- **Spreadsheet analysis** — export to CSV for analysis in Excel or Google Sheets.
- **Archival** — collect and store historical news data for research.

## See Also

- [examples.md](./examples.md) — detailed code examples for export and analytics workflows.
