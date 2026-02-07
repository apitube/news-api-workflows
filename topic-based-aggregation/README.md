# Topic-Based Aggregation

Workflow for aggregating and analyzing news by topic using the [APITube News API](https://apitube.io).

## Overview

The **Topic-Based Aggregation** workflow lets you filter news by predefined topics, combine multiple topic filters, and perform cross-topic comparisons. Use this to build thematic news feeds, discover trending topics, and analyze topic-level coverage patterns.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Topic Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `topic.id`                    | string  | Filter by topic ID (e.g., `cryptocurrency`, `climate_change`).      |
| `category.id`                 | string  | Filter by IPTC category (e.g., `medtop:13000000` for science).      |
| `title`                       | string  | Filter by keywords in article title (comma-separated).              |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `source.domain`               | string  | Filter by source domain (comma-separated).                          |
| `source.country.code`         | string  | Filter by source country (comma-separated).                         |
| `language`                    | string  | Filter by language code.                                             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `per_page`                    | integer | Number of results per page (default: 50).                           |
| `page`                        | integer | Page number for pagination.                                         |

## Available Topics

Some commonly available topics include:

- `technology`, `artificial_intelligence`, `cryptocurrency`, `blockchain`
- `politics`, `elections`, `government`
- `finance`, `stock_market`, `banking`
- `health`, `covid_19`, `mental_health`
- `climate_change`, `environment`, `energy`
- `sports`, `football`, `basketball`, `tennis`
- `entertainment`, `movies`, `music`, `gaming`
- `science`, `space`, `education`

For the complete list, see the [APITube Topics Reference](https://docs.apitube.io/platform/news-api/list-of-topics).

## Quick Start

### cURL

```bash
# Get crypto news sorted by date
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=cryptocurrency&language=en&sort.by=published_at&sort.order=desc&per_page=10"

# Get AI news with positive sentiment
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=artificial_intelligence&sentiment.overall.polarity=positive&language=en&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "topic.id": "cryptocurrency",
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 10,
})

data = response.json()
for article in data["articles"]:
    print(f"[{article['published_at'][:10]}] {article['title']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  "topic.id": "cryptocurrency",
  language: "en",
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.articles.forEach((a) => {
  console.log(`[${a.published_at.slice(0, 10)}] ${a.title}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"    => "YOUR_API_KEY",
    "topic.id"   => "cryptocurrency",
    "language"   => "en",
    "sort.by"    => "published_at",
    "sort.order" => "desc",
    "per_page"   => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($data["articles"] as $article) {
    $date = substr($article["published_at"], 0, 10);
    echo "[{$date}] {$article['title']}\n";
}
```

## Common Use Cases

- **Thematic news digests** — create curated feeds by topic for newsletters and social media.
- **Trending topic discovery** — compare article volume across topics over time.
- **Cross-topic correlation** — analyze how topics co-occur in the news.
- **Topic sentiment tracking** — track sentiment shifts within specific topics.
- **Industry-specific monitoring** — build focused feeds for finance, tech, health, etc.

## See Also

- [examples.md](./examples.md) — detailed code examples for topic-based aggregation workflows.
