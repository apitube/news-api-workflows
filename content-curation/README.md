# Content Curation

Automated content curation workflow using the [APITube News API](https://apitube.io).

## Overview

The **Content Curation** workflow enables automated content curation with quality scoring, deduplication, and newsletter generation. Build curated feeds that filter by source authority, content quality, sentiment balance, and reading time. Generate newsletter-ready digests with intelligent article selection and ranking.

Use this workflow to:
- Score and rank articles by composite quality metrics
- Automatically generate newsletters from multiple topics
- Present balanced perspectives (positive/negative sentiment pairing)
- Build executive briefings from high-authority sources
- Analyze topic trends for content calendar planning

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY
```

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `topic.id` | string | Filter by topic ID (e.g., `technology`, `climate_change`). |
| `category.id` | string | Filter by category ID. |
| `language` | string | Filter by language code (e.g., `en`, `fr`, `de`). |
| `source.rankings.opr.min` | integer | Minimum source authority rank (1-10). |
| `source.domain` | string | Filter by source domain (e.g., `techcrunch.com`). |
| `is_duplicate` | boolean | Filter duplicates (`false` = exclude duplicates). |
| `is_free` | boolean | Filter paywall content (`true` = free articles only). |
| `is_breaking` | boolean | Filter breaking news (`true` = breaking only). |
| `sort.by` | string | Sort field: `published_at`, `relevance`. |
| `sort.order` | string | Sort order: `asc`, `desc`. |
| `per_page` | integer | Results per page (max 100). |
| `page` | integer | Page number for pagination. |
| `published_at.start` | string | Start date in ISO 8601 format. |
| `published_at.end` | string | End date in ISO 8601 format. |
| `sentiment.overall.polarity` | string | Filter by sentiment: `positive`, `negative`, `neutral`. |
| `entity.name` | string | Filter by entity name (e.g., `Apple`, `Elon Musk`). |
| `title` | string | Search query for article title. |

## Quick Start

### cURL

```bash
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=technology&source.rankings.opr.min=5&is_duplicate=false&is_free=true&per_page=20"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "topic.id": "technology",
    "source.rankings.opr.min": 5,
    "is_duplicate": False,
    "is_free": True,
    "per_page": 20
})

results = response.json().get("results", [])
for article in results:
    print(f"{article['title']} - {article['source']['domain']}")
    print(f"Quality: OPR={article['source']['rankings']['opr']}, Words={article['words_count']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
    api_key: "YOUR_API_KEY",
    "topic.id": "technology",
    "source.rankings.opr.min": "5",
    is_duplicate: "false",
    is_free: "true",
    per_page: "20"
});

const response = await fetch(`https://api.apitube.io/v1/news/everything?${params}`);
const data = await response.json();

data.results.forEach(article => {
    console.log(`${article.title} - ${article.source.domain}`);
    console.log(`Quality: OPR=${article.source.rankings.opr}, Words=${article.words_count}`);
});
```

### PHP

```php
$url = "https://api.apitube.io/v1/news/everything?" . http_build_query([
    "api_key" => "YOUR_API_KEY",
    "topic.id" => "technology",
    "source.rankings.opr.min" => 5,
    "is_duplicate" => false,
    "is_free" => true,
    "per_page" => 20
]);

$response = json_decode(file_get_contents($url), true);

foreach ($response["results"] as $article) {
    echo "{$article['title']} - {$article['source']['domain']}\n";
    echo "Quality: OPR={$article['source']['rankings']['opr']}, Words={$article['words_count']}\n";
}
```

## Common Use Cases

- **Automated Newsletter Generation** — Build daily/weekly digests from multiple topics with intelligent article selection
- **Quality-Filtered News Feeds** — Rank articles by composite quality scores (source authority + content depth + freshness)
- **Balanced Perspective Curation** — Present multiple viewpoints by pairing positive/negative sentiment articles on the same topic
- **Executive Briefing Builder** — Generate concise summaries from high-authority sources for decision-makers
- **Content Marketing Feed** — Identify trending topics and optimal publishing times based on volume analysis

## See Also

- [examples.md](./examples.md) — detailed code examples with production-quality implementations in Python, JavaScript, and PHP.
