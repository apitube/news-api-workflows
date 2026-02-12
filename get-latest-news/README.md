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

| Parameter             | Type     | Description                                                       |
|-----------------------|----------|-------------------------------------------------------------------|
| `api_key`             | string   | **Required.** Your API key.                                       |
| `per_page`            | integer  | Number of articles to return (default: 50, max: 100).             |
| `page`                | integer  | Page number for pagination.                                       |
| `language.code`       | string   | Filter by language code (e.g., `en`, `fr`, `de`).                 |
| `source.country.code` | string   | Filter by source country code (e.g., `us`, `gb`, `de`).           |
| `category.id`         | string   | Filter by IPTC category ID (e.g., `medtop:04000000` for business).|
| `title`               | string   | Search keywords in article title.                                 |
| `sort.by`             | string   | Sort field: `published_at` (default), `sentiment.overall.score`.  |
| `sort.order`          | string   | Sort direction: `asc` or `desc`.                                  |
| `published_at.start`  | string   | Start date (ISO 8601 or `YYYY-MM-DD`).                            |
| `published_at.end`    | string   | End date (ISO 8601 or `YYYY-MM-DD`).                              |

## Quick Start

### cURL

```bash
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language.code=en&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "language.code": "en",
    "per_page": 10
})

results = response.json().get("results", [])
for article in results:
    print(article["title"])
```

### JavaScript (Node.js)

```javascript
const response = await fetch(
  "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language.code=en&per_page=10"
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(article.title);
});
```

### PHP

```php
$url = "https://api.apitube.io/v1/news/everything?" . http_build_query([
    "api_key"       => "YOUR_API_KEY",
    "language.code" => "en",
    "per_page"      => 10,
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
  "limit": 1,
  "page": 1,
  "path": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1",
  "has_next_pages": true,
  "next_page": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&page=2",
  "has_previous_page": false,
  "previous_page": "",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "export": {
    "json": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=json",
    "xlsx": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=xlsx",
    "csv": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=csv",
    "tsv": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=tsv",
    "xml": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=xml",
    "rss": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=rss"
  },
  "results": [
    {
      "id": 12345678,
      "href": "https://example.com/article/12345",
      "published_at": "2026-02-07T10:30:00Z",
      "title": "Breaking: Major Tech Company Announces New Product",
      "description": "A leading technology company unveiled its latest innovation today...",
      "body": "Full article content goes here...",
      "body_html": "<p>Full article content goes here...</p>",
      "language": "en",
      "author": {
        "id": 5678,
        "name": "John Smith"
      },
      "image": "https://example.com/images/article.jpg",
      "categories": [
        {
          "id": 199,
          "name": "economy, business and finance",
          "score": 0.8,
          "taxonomy": "iptc_mediatopics",
          "links": {
            "self": "https://api.apitube.io/v1/news/category/iptc_mediatopics/medtop:04000000"
          }
        }
      ],
      "topics": [
        {
          "id": "technology",
          "name": "Technology",
          "score": 0.9,
          "links": {
            "self": "https://api.apitube.io/v1/news/topic/technology"
          }
        }
      ],
      "industries": [
        {
          "id": 456,
          "name": "Consumer Electronics",
          "links": {
            "self": "https://api.apitube.io/v1/news/industry/456"
          }
        }
      ],
      "entities": [
        {
          "id": 789,
          "name": "Apple",
          "type": "organization",
          "links": {
            "self": "https://api.apitube.io/v1/news/entity/789",
            "wikipedia": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "wikidata": "https://www.wikidata.org/wiki/Q312"
          },
          "frequency": 3,
          "title": {
            "pos": [{ "start": 4, "end": 9 }]
          },
          "body": {
            "pos": [{ "start": 10, "end": 15 }, { "start": 120, "end": 125 }]
          },
          "metadata": {
            "name": "Apple",
            "type": "business",
            "country": { "code": "US", "name": "United States" },
            "description": "Technology company"
          }
        }
      ],
      "source": {
        "id": 4232,
        "domain": "example.com",
        "home_page_url": "https://example.com",
        "type": "news",
        "bias": "center",
        "rankings": {
          "opr": 5
        },
        "location": {
          "country_name": "United States",
          "country_code": "us"
        },
        "favicon": "https://www.google.com/s2/favicons?domain=https://example.com"
      },
      "sentiment": {
        "overall": {
          "score": 0.75,
          "polarity": "positive"
        },
        "title": {
          "score": 0.60,
          "polarity": "positive"
        },
        "body": {
          "score": 0.80,
          "polarity": "positive"
        }
      },
      "summary": [
        {
          "sentence": "A leading technology company unveiled its latest innovation today.",
          "sentiment": {
            "score": 0.6,
            "polarity": "positive"
          }
        }
      ],
      "keywords": ["technology", "innovation", "product launch"],
      "links": [
        {
          "url": "https://example.com/related-article",
          "type": "link"
        }
      ],
      "media": [
        {
          "url": "https://example.com/images/article.jpg",
          "type": "image"
        }
      ],
      "story": {
        "id": 9876,
        "uri": "https://api.apitube.io/v1/news/story/9876"
      },
      "is_duplicate": false,
      "is_free": true,
      "is_breaking": true,
      "read_time": 2,
      "sentences_count": 24,
      "paragraphs_count": 8,
      "words_count": 520,
      "characters_count": 3100
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
