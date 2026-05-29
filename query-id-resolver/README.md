# Query ID Resolver

Turn human-readable input into precise filter IDs, then search articles with them, using the [APITube News API](https://apitube.io).

## Overview

The **Query ID Resolver** workflow connects the typeahead suggest endpoints to the main article search. A user types free text (for example "tech" or "appl"); a suggest endpoint returns matching candidates with their canonical IDs; you take an `id` and pass it to `/v1/news/everything` as `topic.id`, `category.id`, `industry.id`, or `entity.id` for an exact, ID-based query. This avoids ambiguous keyword matching and powers autocomplete-driven search UIs.

## API Endpoint

```
GET https://api.apitube.io/v1/suggest/topics
GET https://api.apitube.io/v1/suggest/categories
GET https://api.apitube.io/v1/suggest/industries
GET https://api.apitube.io/v1/suggest/entities
GET https://api.apitube.io/v1/news/everything
```

The four suggest endpoints each return a **flat JSON array** (not wrapped in `results`) and require a `prefix` parameter. Omitting `prefix` returns error `ER0346` ("`Prefix` in query required.").

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/suggest/topics?api_key=YOUR_API_KEY&prefix=tech
```

## Parameters

### Suggest — `/v1/suggest/{topics,categories,industries,entities}`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `prefix` | string | **Required.** Name prefix to autocomplete. Missing prefix returns `ER0346`. |

### Search — `/v1/news/everything`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `topic.id` | string | Filter by a topic ID resolved from `/v1/suggest/topics`. |
| `category.id` | string | Filter by a category ID resolved from `/v1/suggest/categories`. |
| `industry.id` | string | Filter by an industry ID resolved from `/v1/suggest/industries`. |
| `entity.id` | string | Filter by an entity ID resolved from `/v1/suggest/entities`. |
| `language.code` | string | Filter by language code (e.g., `en`). |
| `per_page` | integer | Number of results per page. |
| `sort.by` | string | Sort field (e.g., `published_at`). |
| `sort.order` | string | Sort direction: `asc` or `desc`. |

## Quick Start

### cURL

```bash
# 1. Resolve "tech" to topic candidates
curl -s "https://api.apitube.io/v1/suggest/topics?api_key=YOUR_API_KEY&prefix=tech"

# 2. Use the resolved id to search articles
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=technology&language.code=en&per_page=10"
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"

# 1. Resolve prefix -> topic id
suggest = requests.get("https://api.apitube.io/v1/suggest/topics", params={
    "api_key": API_KEY,
    "prefix": "tech",
})
suggest.raise_for_status()
candidates = suggest.json()  # flat array
topic_id = candidates[0]["id"]
print(f"Resolved 'tech' -> {topic_id} ({candidates[0]['name']})")

# 2. Search with the resolved id
articles = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": API_KEY,
    "topic.id": topic_id,
    "language.code": "en",
    "per_page": 10,
})
articles.raise_for_status()
for article in articles.json()["results"]:
    print(f"  {article['title']}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";

// 1. Resolve prefix -> topic id
const suggestParams = new URLSearchParams({ api_key: API_KEY, prefix: "tech" });
const suggestResp = await fetch(`https://api.apitube.io/v1/suggest/topics?${suggestParams}`);
const candidates = await suggestResp.json(); // flat array
const topicId = candidates[0].id;
console.log(`Resolved 'tech' -> ${topicId} (${candidates[0].name})`);

// 2. Search with the resolved id
const searchParams = new URLSearchParams({
  api_key: API_KEY,
  "topic.id": topicId,
  "language.code": "en",
  per_page: "10",
});
const articlesResp = await fetch(`https://api.apitube.io/v1/news/everything?${searchParams}`);
const data = await articlesResp.json();
data.results.forEach((article) => console.log(`  ${article.title}`));
```

### PHP

```php
$apiKey = "YOUR_API_KEY";

// 1. Resolve prefix -> topic id
$suggestQuery = http_build_query(["api_key" => $apiKey, "prefix" => "tech"]);
$candidates = json_decode(file_get_contents(
    "https://api.apitube.io/v1/suggest/topics?{$suggestQuery}"
), true); // flat array
$topicId = $candidates[0]["id"];
echo "Resolved 'tech' -> {$topicId} ({$candidates[0]['name']})\n";

// 2. Search with the resolved id
$searchQuery = http_build_query([
    "api_key"       => $apiKey,
    "topic.id"      => $topicId,
    "language.code" => "en",
    "per_page"      => 10,
]);
$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$searchQuery}"
), true);
foreach ($data["results"] as $article) {
    echo "  {$article['title']}\n";
}
```

## Response Example

### Suggest — `/v1/suggest/topics?prefix=tech` (flat array)

```json
[
  { "id": "technology", "name": "Technology", "links": { "self": "https://api.apitube.io/v1/news/topic/technology" } },
  { "id": "techstartups", "name": "Tech Startups", "links": { "self": "https://api.apitube.io/v1/news/topic/techstartups" } }
]
```

### Suggest — `/v1/suggest/entities?prefix=appl` (flat array)

```json
[
  {
    "id": 312,
    "name": "Apple",
    "type": "organization",
    "links": {
      "self": "https://api.apitube.io/v1/news/entity/312",
      "wikipedia": "https://en.wikipedia.org/wiki/Apple_Inc.",
      "wikidata": "https://www.wikidata.org/wiki/Q312"
    }
  }
]
```

### Search — `/v1/news/everything?topic.id=technology`

```json
{
  "status": "ok",
  "results": [
    { "id": 123456, "title": "An article about technology", "published_at": "2026-05-29T10:00:00Z" }
  ]
}
```

## Common Use Cases

- **Resolve-then-search** — convert a typed prefix into a canonical ID before querying articles.
- **Autocomplete UIs** — power a search box where selecting a suggestion runs an exact ID query.
- **Multi-filter queries** — resolve several inputs and combine `topic.id`, `entity.id`, etc. in one search.
- **Disambiguation** — present multiple suggest candidates and let the user pick the right ID.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
