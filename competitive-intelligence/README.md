# Competitive Intelligence

Workflow for monitoring competitors, comparing brand coverage, and analyzing market positioning using the [APITube News API](https://apitube.io).

## Overview

The **Competitive Intelligence** workflow combines entity tracking, sentiment analysis, source filtering, and topic filtering to build a comprehensive view of how companies and products are covered in the news. Track competitors side-by-side, detect PR crises early, and benchmark media presence.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by organization name (e.g., `Tesla`, `Google`).               |
| `person.name`                 | string  | Filter by person name (e.g., `Elon Musk`).                           |
| `brand.name`                  | string  | Filter by brand name (e.g., `Nike`, `Coca-Cola`).                    |
| `title`                       | string  | Filter by keywords in article title (comma-separated).              |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | float   | Minimum sentiment score (-1.0 to 1.0).                               |
| `sentiment.overall.score.max` | float   | Maximum sentiment score (-1.0 to 1.0).                               |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `topic.id`                    | string  | Filter by topic ID.                                                  |
| `source.domain`               | string  | Filter by source domain (comma-separated).                          |
| `source.country.code`         | string  | Filter by source country.                                            |
| `source.rank.opr.min`         | integer | Minimum source OPR rank (0–7).                                       |
| `language.code`               | string  | Filter by language code.                                             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `per_page`                    | integer | Number of results per page.                                          |
| `page`                        | integer | Page number for pagination.                                          |

## Quick Start

### cURL

```bash
# Compare coverage: Tesla vs Rivian
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Tesla&language.code=en&per_page=1" | jq '.results | length'
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Rivian&language.code=en&per_page=1" | jq '.results | length'
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

competitors = ["Tesla", "Rivian", "Lucid Motors"]

for company in competitors:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "language.code": "en",
        "per_page": 100,
    })
    results = response.json().get("results", [])
    print(f"{company}: {len(results)} articles (page 1)")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const competitors = ["Tesla", "Rivian", "Lucid Motors"];

for (const company of competitors) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": company,
    "language.code": "en",
    per_page: "100",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  console.log(`${company}: ${data.results.length} articles (page 1)`);
}
```

### PHP

```php
$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$competitors = ["Tesla", "Rivian", "Lucid Motors"];

foreach ($competitors as $company) {
    $query = http_build_query([
        "api_key"           => $apiKey,
        "organization.name" => $company,
        "language.code"     => "en",
        "per_page"          => 100,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    echo "{$company}: " . count($data['results']) . " articles (page 1)\n";
}
```

## Common Use Cases

- **Competitor benchmarking** — compare media coverage volume and sentiment across competitors.
- **PR crisis early warning** — detect negative sentiment spikes for your brand or competitors.
- **Market positioning** — understand how different companies are covered by top-tier media.
- **Share of voice analysis** — calculate relative media presence within an industry.
- **Executive reputation tracking** — monitor coverage of key executives across organizations.

## See Also

- [examples.md](./examples.md) — detailed code examples for competitive intelligence workflows.
