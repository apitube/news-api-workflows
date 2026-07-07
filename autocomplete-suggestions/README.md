# Autocomplete Suggestions

Build a typeahead/autocomplete UI for a news search box using the five suggest endpoints of the [APITube News API](https://apitube.io).

## Overview

The **Autocomplete Suggestions** workflow powers a search-as-you-type dropdown. As the user types, you query one or more of the suggest endpoints with a `prefix` and resolve free text into concrete IDs (`entity.id`, `category.id`, `topic.id`, `industry.id`, `source.id`) that can then be passed to `/v1/news/everything`. There are five suggest endpoints, one per dimension: entities, categories, topics, industries, and sources. Each returns a **flat array** (the results are not wrapped in a `results` object), and every endpoint **requires** the `prefix` parameter. Omitting `prefix` returns error `ER0346` ("`Prefix` in query required.").

## API Endpoints

```
GET https://api.apitube.io/v1/suggest/entities
GET https://api.apitube.io/v1/suggest/categories
GET https://api.apitube.io/v1/suggest/topics
GET https://api.apitube.io/v1/suggest/industries
GET https://api.apitube.io/v1/suggest/sources
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/suggest/topics?api_key=YOUR_API_KEY&prefix=tech
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `prefix` | string | **Required.** Name prefix to match. Missing prefix returns error `ER0346`. |

## Quick Start

### cURL

```bash
# Topic suggestions for "tech"
curl -s "https://api.apitube.io/v1/suggest/topics?api_key=YOUR_API_KEY&prefix=tech"

# Entity suggestions for "appl"
curl -s "https://api.apitube.io/v1/suggest/entities?api_key=YOUR_API_KEY&prefix=appl"

# Category and industry suggestions
curl -s "https://api.apitube.io/v1/suggest/categories?api_key=YOUR_API_KEY&prefix=econ"
curl -s "https://api.apitube.io/v1/suggest/industries?api_key=YOUR_API_KEY&prefix=semi"

# Source suggestions for "bbc" (resolves to source.id)
curl -s "https://api.apitube.io/v1/suggest/sources?api_key=YOUR_API_KEY&prefix=bbc"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/suggest/topics", params={
    "api_key": "YOUR_API_KEY",
    "prefix": "tech",
})
response.raise_for_status()

# Response is a flat array, NOT wrapped in {"results": [...]}
for item in response.json():
    print(f"{item['id']:<16} {item['name']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  prefix: "tech",
});

const response = await fetch(
  `https://api.apitube.io/v1/suggest/topics?${params}`
);
const items = await response.json(); // flat array

items.forEach((item) => {
  console.log(`${item.id.padEnd(16)} ${item.name}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key" => "YOUR_API_KEY",
    "prefix"  => "tech",
]);

// Response is a flat array, decoded directly
$items = json_decode(file_get_contents(
    "https://api.apitube.io/v1/suggest/topics?{$query}"
), true);

foreach ($items as $item) {
    printf("%-16s %s\n", $item["id"], $item["name"]);
}
```

## Response Examples

`/v1/suggest/topics?prefix=tech` — flat array of topics:

```json
[
  { "id": "technology", "name": "Technology", "links": { "self": "https://api.apitube.io/v1/news/topic/technology" } },
  { "id": "techstartups", "name": "Tech Startups", "links": { "self": "https://api.apitube.io/v1/news/topic/techstartups" } }
]
```

`/v1/suggest/entities?prefix=appl` — flat array of entity objects (each has `id`, `name`, `type`, `links`):

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

`/v1/suggest/categories?prefix=econ` returns `{ id, name, taxonomy, links: { self } }` objects, and `/v1/suggest/industries?prefix=semi` returns `{ id, name, links: { self } }` objects, both as flat arrays.

## Common Use Cases

- **Single-type suggest** — one helper function that takes a type and prefix, hits the matching endpoint, and returns the flat array.
- **Multi-type autocomplete** — query several suggest endpoints in parallel and merge the results into one dropdown grouped by type.
- **Debounced search box** — wait for typing to pause before firing requests, reducing call volume on every keystroke.
- **Resolve to filters** — map a chosen suggestion's `id` onto `topic.id` / `category.id` / `industry.id` / `entity.id` for a follow-up `/v1/news/everything` query.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
