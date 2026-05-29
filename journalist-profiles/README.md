# Journalist Profiles

Build a journalist directory and per-author profile pages using the [APITube News API](https://apitube.io).

## Overview

The **Journalist Profiles** workflow lets you browse a normalized directory of journalists (one record per author name, even when they write for multiple outlets) and pull a detailed profile for any single journalist. The list endpoint powers search and pagination; the profile endpoint returns the journalist object plus a `coverage` block (topics, entities, countries, languages, sentiment, momentum, timeline) and up to five `recent_articles`. From any journalist you can jump straight to their articles via `author.id` on `/v1/news/everything`.

## API Endpoint

```
GET https://api.apitube.io/v1/journalists
GET https://api.apitube.io/v1/journalists/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/journalists?api_key=YOUR_API_KEY
```

## Parameters

### List — `/v1/journalists`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `name` | string | Filter by journalist name (partial match). |
| `page` | integer | Page number for pagination. |
| `per_page` | integer | Number of results per page. |

### Profile — `/v1/journalists/:id`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `coverage` | boolean | Set to `false` to omit the `coverage` block. Defaults to enabled. |

Returns `404` (`ER0151`) if the journalist is not found.

## Quick Start

### cURL

```bash
# Search the directory by name
curl -s "https://api.apitube.io/v1/journalists?api_key=YOUR_API_KEY&name=Jane&per_page=10"

# Fetch a single journalist profile with coverage and recent articles
curl -s "https://api.apitube.io/v1/journalists/88123?api_key=YOUR_API_KEY"

# Fetch a profile without the coverage block (lighter payload)
curl -s "https://api.apitube.io/v1/journalists/88123?api_key=YOUR_API_KEY&coverage=false"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/journalists", params={
    "api_key": "YOUR_API_KEY",
    "name": "Jane",
    "per_page": 10,
})
response.raise_for_status()

data = response.json()
for journalist in data["results"]:
    outlets = ", ".join(o["name"] for o in journalist["outlets"])
    print(f"#{journalist['id']} {journalist['name']} -> {outlets} ({journalist['outlet_count']} outlets)")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  name: "Jane",
  per_page: "10",
});

const response = await fetch(`https://api.apitube.io/v1/journalists?${params}`);
const data = await response.json();

data.results.forEach((journalist) => {
  const outlets = journalist.outlets.map((o) => o.name).join(", ");
  console.log(`#${journalist.id} ${journalist.name} -> ${outlets} (${journalist.outlet_count} outlets)`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"  => "YOUR_API_KEY",
    "name"     => "Jane",
    "per_page" => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/journalists?{$query}"
), true);

foreach ($data["results"] as $journalist) {
    $outlets = implode(", ", array_column($journalist["outlets"], "name"));
    echo "#{$journalist['id']} {$journalist['name']} -> {$outlets} ({$journalist['outlet_count']} outlets)\n";
}
```

## Response Example

### List — `/v1/journalists?name=Jane`

```json
{
  "status": "ok",
  "limit": 10,
  "page": 1,
  "has_next_pages": false,
  "results": [
    {
      "id": 88123,
      "name": "Jane Doe",
      "outlets": [
        { "id": 4232, "name": "Example News", "domain": "example.com" },
        { "id": 771, "name": "Daily Wire", "domain": "dailywire.com" }
      ],
      "outlet_count": 2,
      "links": {
        "self": "https://api.apitube.io/v1/journalists/88123",
        "articles": "https://api.apitube.io/v1/news/everything?author.id=88123"
      }
    }
  ]
}
```

### Profile — `/v1/journalists/88123`

```json
{
  "id": 88123,
  "name": "Jane Doe",
  "outlets": [
    { "id": 4232, "name": "Example News", "domain": "example.com" },
    { "id": 771, "name": "Daily Wire", "domain": "dailywire.com" }
  ],
  "outlet_count": 2,
  "links": {
    "self": "https://api.apitube.io/v1/journalists/88123",
    "articles": "https://api.apitube.io/v1/news/everything?author.id=88123"
  },
  "coverage": {
    "article_count": 1820,
    "first_seen": "2018-09-04",
    "last_seen": "2026-05-28",
    "sentiment": { "positive": 600, "neutral": 900, "negative": 320 },
    "momentum": { "last_30_days": 42, "previous_30_days": 51, "change_pct": -18 },
    "timeline": [ { "period": "2024-06-01", "count": 38 } ],
    "top_topics": [ { "id": "politics", "name": "Politics", "count": 410 } ],
    "top_entities": [ { "id": 5021, "name": "Elon Musk", "count": 120 } ],
    "top_countries": [ { "id": 840, "name": "United States", "code": "us", "count": 1500 } ],
    "top_languages": [ { "id": 1, "name": "English", "code": "en", "count": 1700 } ]
  },
  "recent_articles": []
}
```

Note that `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (e.g. no articles in the previous 30-day window), and the entire `coverage` block may be `null` when analytics are unavailable — guard for these before formatting.

## Common Use Cases

- **Author directory** — build a searchable index of journalists with the outlets they write for.
- **Profile pages** — render a journalist's coverage stats, top topics, and recent articles on a single page.
- **Byline lookup** — resolve a name from an article byline to a journalist record and their full archive.
- **Drill-through to articles** — use `links.articles` (or `author.id` on `/v1/news/everything`) to load everything an author has written.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
