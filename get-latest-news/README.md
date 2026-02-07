# Get Latest News

Workflow for fetching the latest news articles using the [APITube News API](https://apitube.io).

## Overview

The **Get Latest News** workflow retrieves the most recent news articles from thousands of sources worldwide. You can filter results by topic, language, country, source, and more.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY
```

## Parameters

| Parameter    | Type     | Description                                      |
|-------------|----------|--------------------------------------------------|
| `api_key`   | string   | **Required.** Your API key.                      |
| `limit`     | integer  | Number of articles to return (default: 50).      |
| `language`  | string   | Filter by language code (e.g., `en`, `fr`, `de`).|
| `country`   | string   | Filter by country code (e.g., `us`, `gb`, `de`). |
| `category`  | string   | Filter by category (e.g., `technology`, `sports`).|
| `query`     | string   | Search query to match in article title/body.     |
| `sort_by`   | string   | Sort order: `published_at` (default), `relevance`.|
| `from`      | string   | Start date in ISO 8601 format.                   |
| `to`        | string   | End date in ISO 8601 format.                     |

## Quick Start

### cURL

```bash
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language=en&limit=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "language": "en",
    "limit": 10
})

results = response.json().get("results", [])
for article in results:
    print(article["title"])
```

### JavaScript (Node.js)

```javascript
const response = await fetch(
  "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language=en&limit=10"
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(article.title);
});
```

### PHP

```php
$url = "https://api.apitube.io/v1/news/everything?" . http_build_query([
    "api_key"  => "YOUR_API_KEY",
    "language" => "en",
    "limit"    => 10,
]);

$response = json_decode(file_get_contents($url), true);

foreach ($response["results"] as $article) {
    echo $article["title"] . PHP_EOL;
}
```

## Response Example

```json
{
  "status": "ok",
  "page": 1,
  "path": "https://api.apitube.io/v1/news/everything?language=en&limit=1",
  "has_next_pages": true,
  "next_page": "https://api.apitube.io/v1/news/everything?language=en&limit=1&page=2",
  "has_previous_page": false,
  "previous_page": null,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "export": {
    "json": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=json",
    "xlsx": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=xlsx",
    "csv": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=csv",
    "tsv": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=tsv",
    "xml": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=xml"
  },
  "results": [
    {
      "id": "abc123def456",
      "title": "Breaking: Major Tech Company Announces New Product",
      "description": "A leading technology company unveiled its latest innovation today...",
      "content": "Full article content goes here...",
      "url": "https://example.com/article/12345",
      "image": "https://example.com/images/article.jpg",
      "published_at": "2026-02-07T10:30:00Z",
      "source": {
        "name": "Tech Daily",
        "domain": "example.com",
        "url": "https://example.com",
        "country": {
          "code": "us",
          "name": "United States"
        },
        "rank": {
          "opr": 0.82
        }
      },
      "language": {
        "code": "en",
        "name": "English"
      },
      "category": {
        "id": "medtop:04000000",
        "name": "economy, business and finance"
      },
      "topic": {
        "id": "technology",
        "name": "Technology"
      },
      "industry": {
        "id": "tech",
        "name": "Technology"
      },
      "sentiment": {
        "overall": {
          "score": 0.75,
          "polarity": "positive"
        }
      },
      "entities": [
        {
          "name": "Apple",
          "type": "organization"
        }
      ],
      "stories": [],
      "links": [],
      "media": [],
      "hashtags": [],
      "duplicate": false,
      "paywall": false,
      "breaking_news": true,
      "sentences": 24,
      "paragraphs": 8,
      "words": 520,
      "characters": 3100,
      "reading_time": 2.5
    }
  ]
}
```

For the complete data model reference, see [examples.md](./examples.md#response-data-model).

## Common Use Cases

- **News aggregators** — collect articles from multiple sources in one feed.
- **Monitoring dashboards** — track news about specific topics or companies.
- **Chatbots and assistants** — provide users with up-to-date news on demand.
- **Research and analytics** — gather large datasets of news articles for analysis.

## See Also

- [examples.md](./examples.md) — detailed code examples in PHP, Python, and JavaScript.
