# Sentiment Analysis

Workflow for filtering and analyzing news articles by sentiment using the [APITube News API](https://apitube.io).

## Overview

The **Sentiment Analysis** workflow lets you filter news by emotional tone (positive, negative, neutral), sort by sentiment score, and build sentiment-driven dashboards. This is useful for brand monitoring, market research, and media analysis.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Sentiment Parameters

| Parameter                      | Type   | Description                                                        |
|-------------------------------|--------|--------------------------------------------------------------------|
| `api_key`                     | string | **Required.** Your API key.                                        |
| `sentiment.overall.polarity`  | string | Filter by polarity: `positive`, `negative`, or `neutral`.          |
| `sentiment.overall.score`     | number | Filter by numeric sentiment score (range roughly -1.0 to 1.0).    |
| `sort.by`                     | string | Sort field. Use `sentiment.overall.score` to sort by sentiment.    |
| `sort.order`                  | string | Sort direction: `asc` or `desc`.                                   |
| `title`                       | string | Filter by keywords in article title (comma-separated).             |
| `topic.id`                    | string | Filter by topic (e.g., `cryptocurrency`, `climate_change`).       |
| `source.domain`               | string | Filter by source domain (comma-separated).                         |
| `language`                    | string | Filter by language code (e.g., `en`, `fr`, `de`).                 |
| `published_at.start`          | string | Start date for filtering (ISO 8601 or `YYYY-MM-DD`).              |
| `published_at.end`            | string | End date for filtering (ISO 8601 or `YYYY-MM-DD`).                |
| `per_page`                    | integer| Number of results per page (default: 50).                          |
| `page`                        | integer| Page number for pagination.                                        |

## Quick Start

### cURL

```bash
# Get positive news about technology
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sentiment.overall.polarity=positive&topic.id=technology&language=en&per_page=10"

# Get negative news sorted by sentiment score (most negative first)
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sentiment.overall.polarity=negative&sort.by=sentiment.overall.score&sort.order=asc&language=en&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "sentiment.overall.polarity": "positive",
    "topic.id": "technology",
    "language": "en",
    "per_page": 10,
})

data = response.json()
for article in data["articles"]:
    print(f"[{article['sentiment']['overall']['polarity']}] {article['title']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  "sentiment.overall.polarity": "positive",
  "topic.id": "technology",
  language: "en",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.articles.forEach((article) => {
  console.log(`[${article.sentiment.overall.polarity}] ${article.title}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"                     => "YOUR_API_KEY",
    "sentiment.overall.polarity"  => "positive",
    "topic.id"                    => "technology",
    "language"                    => "en",
    "per_page"                    => 10,
]);

$response = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($response["articles"] as $article) {
    $polarity = $article["sentiment"]["overall"]["polarity"];
    echo "[{$polarity}] {$article['title']}\n";
}
```

## Common Use Cases

- **Brand reputation monitoring** — track positive and negative coverage of a brand or product.
- **Market sentiment dashboards** — visualize sentiment trends across financial topics.
- **Media bias analysis** — compare sentiment polarity across different news sources.
- **Crisis detection** — alert on spikes in negative sentiment for a given topic.

## See Also

- [examples.md](./examples.md) — detailed code examples for sentiment analysis workflows.
