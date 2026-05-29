# Raw News Firehose

Low-latency ingestion of the raw discovery feed before NLP enrichment using the [APITube News API](https://apitube.io).

## Overview

The **Raw News Firehose** workflow consumes articles straight from the discovery stage, before HTML parsing and NLP enrichment run. This is the fastest way to learn that an article exists: items appear within seconds of being discovered in RSS feeds, sitemaps, and Google News. The trade-off is that raw items carry no enrichment fields (no `language`, `categories`, `topics`, `entities`, `sentiment`, or `summary`), and the staging feed rotates quickly: rows are continuously consumed by the pipeline and live for roughly one day. If you need a permanent archive or enriched data, pair this with `/v1/news/everything`.

## API Endpoint

```
GET  https://api.apitube.io/v1/news/raw
POST https://api.apitube.io/v1/news/raw
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/news/raw?api_key=YOUR_API_KEY
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `page` | integer | Page number for pagination. Default `1`. |
| `per_page` | integer | Results per page. Default `100`, maximum `250`. |
| `source.id` | string | Comma-separated sitemap source IDs to include (max 3). Example: `123` or `123,456`. |
| `ignore.source.id` | string | Comma-separated source IDs to exclude (max 3). |
| `published_at` | string | Single-day filter (creates a 24-hour range). ISO 8601, `YYYY-MM-DD`, or relative. |
| `published_at.start` | string | Start of the publication range. |
| `published_at.end` | string | End of the publication range. |
| `sort.by` | string | Sort field: `id`, `published_at`, or `created_at`. Default `id`. |
| `sort.order` | string | Sort direction: `asc` or `desc`. Default `desc`. |

## Quick Start

### cURL

```bash
# Newest raw items, 250 per page
curl -s "https://api.apitube.io/v1/news/raw?api_key=YOUR_API_KEY&per_page=250&sort.by=published_at&sort.order=desc"

# Only two specific sources, newest first
curl -s "https://api.apitube.io/v1/news/raw?api_key=YOUR_API_KEY&source.id=123,456&sort.by=published_at&sort.order=desc"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/raw", params={
    "api_key": "YOUR_API_KEY",
    "per_page": 250,
    "sort.by": "published_at",
    "sort.order": "desc",
})
response.raise_for_status()

data = response.json()
for article in data["results"]:
    print(f"{article['created_at']}  {article['title']}")
    print(f"  {article['href']}  ({article['source']['domain']})")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  per_page: "250",
  "sort.by": "published_at",
  "sort.order": "desc",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/raw?${params}`
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(`${article.created_at}  ${article.title}`);
  console.log(`  ${article.href}  (${article.source.domain})`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"    => "YOUR_API_KEY",
    "per_page"   => 250,
    "sort.by"    => "published_at",
    "sort.order" => "desc",
]);

$response = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/raw?{$query}"
), true);

foreach ($response["results"] as $article) {
    echo "{$article['created_at']}  {$article['title']}\n";
    echo "  {$article['href']}  ({$article['source']['domain']})\n";
}
```

## Response Example

```json
{
  "status": "ok",
  "limit": 100,
  "page": 1,
  "has_next_pages": true,
  "next_page": "https://api.apitube.io/v1/news/raw?per_page=100&page=2",
  "has_previous_page": false,
  "previous_page": "",
  "request_id": "a1b2c3d4-...",
  "results": [
    {
      "id": 987654321,
      "title": "Raw headline straight from the RSS feed",
      "href": "https://example.com/article/123",
      "created_at": "2026-05-27T08:15:00",
      "description": "Short description from the RSS item (HTML stripped).",
      "body": "Article body with HTML stripped (plain text).",
      "body_html": "<p>Article body as received from RSS (HTML).</p>",
      "author": "Jane Doe",
      "keywords": ["politics", "economy"],
      "source": {
        "id": 4232,
        "domain": "example.com",
        "home_page_url": "https://example.com",
        "type": "news",
        "bias": "center",
        "rankings": { "opr": 5 },
        "location": { "country_name": "United States", "country_code": "us" },
        "favicon": "https://www.google.com/s2/favicons?domain=https://example.com"
      }
    }
  ]
}
```

Note: `created_at` may be `null`, and `keywords` may be `null`. There is no `export` block on this endpoint. On the free plan, body and domain fields are truncated with a `...[Upgrade subscription plan]` suffix.

## Common Use Cases

- **Continuous raw ingestion** — poll the newest page repeatedly and deduplicate by `href` to build a real-time stream of just-discovered articles.
- **Source monitoring** — narrow the feed to up to three `source.id` values to watch specific publishers as they break stories.
- **Daily backfill** — replay one calendar day with `published_at.start`/`published_at.end` while the data is still in the ~1-day retention window.
- **Noise reduction** — exclude low-value publishers with `ignore.source.id` so the firehose carries only the sources you care about.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
- [../raw-to-enriched-pipeline](../raw-to-enriched-pipeline) — combine the raw feed with enriched data from `/v1/news/everything`.
