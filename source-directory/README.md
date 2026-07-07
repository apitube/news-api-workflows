# Source Directory

Workflow for resolving publisher names into source IDs and pivoting to their articles using the [APITube News API](https://apitube.io).

## Overview

The **Source Directory** workflow lets you turn a publisher name typed by a user into a concrete `source.id`, then pivot straight to that publisher's articles. It uses the autocomplete endpoint `/v1/suggest/sources`, which matches a **name or domain prefix** and returns catalog records (id, name, domain, resource type, political bias). From any result you pivot to its coverage through `links.self`, which targets `/v1/news/everything` with `source.id`.

> **Note:** APITube does not expose a full publisher-listing or per-source profile endpoint. `/v1/suggest/sources` is a **prefix search** (you must pass a `prefix`), not a browsable catalog, and there is no source-level `coverage` block. To measure a publisher's volume, pivot to `/v1/news/everything` (or `/v1/news/count`) with its `source.id`.

## API Endpoints

```
GET https://api.apitube.io/v1/suggest/sources     (find sources by name/domain prefix)
GET https://api.apitube.io/v1/news/everything      (for the source -> articles pivot)
```

## Authentication

All requests require an API key passed via the `api_key` query parameter (or the `X-API-Key` header):

```
https://api.apitube.io/v1/suggest/sources?api_key=YOUR_API_KEY&prefix=guardian
```

## Parameters

### `GET /v1/suggest/sources`

| Parameter   | Type    | Description                                                              |
|-------------|---------|-------------------------------------------------------------------------|
| `api_key`   | string  | **Required.** Your API key.                                             |
| `prefix`    | string  | **Required.** Name or domain prefix to match. Missing prefix returns `400 ER0346`. |

The endpoint returns a **flat array** (results are not wrapped in a `results` object).

## Quick Start

### cURL

```bash
# Find sources whose name/domain starts with "guardian"
curl -s "https://api.apitube.io/v1/suggest/sources?api_key=YOUR_API_KEY&prefix=guardian"

# Pivot: load the matched source's latest articles by source.id
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&source.id=4232&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/suggest/sources", params={
    "api_key": "YOUR_API_KEY",
    "prefix": "guardian",
})
response.raise_for_status()

# The response is a flat array of source objects
for source in response.json():
    print(f"{source['id']:>8}  {source['name']} ({source['domain']}) bias={source.get('bias')}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  prefix: "guardian",
});

const response = await fetch(`https://api.apitube.io/v1/suggest/sources?${params}`);
const sources = await response.json();

sources.forEach((source) => {
  console.log(`${source.id}  ${source.name} (${source.domain}) bias=${source.bias}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key" => "YOUR_API_KEY",
    "prefix"  => "guardian",
]);

$sources = json_decode(file_get_contents(
    "https://api.apitube.io/v1/suggest/sources?{$query}"
), true);

foreach ($sources as $source) {
    printf("%8d  %s (%s) bias=%s\n",
        $source["id"], $source["name"], $source["domain"], $source["bias"] ?? "");
}
```

## Response Example

### `GET /v1/suggest/sources?prefix=guardian`

```json
[
  {
    "id": 4232,
    "name": "The Guardian",
    "domain": "theguardian.com",
    "type": "news",
    "bias": "center",
    "links": {
      "self": "https://api.apitube.io/v1/news/everything?source.id=4232"
    }
  }
]
```

Each object carries `id` (use it in the `source.id` filter), `name`, `domain` (use it in the `source.domain` filter), `type` (may be `null`), `bias` (`left` / `center` / `right`, may be `null`), and `links.self` (a ready-made `/v1/news/everything` URL scoped to that source).

## Common Use Cases

- **Publisher lookup** — resolve a publisher name or domain typed by a user into its source ID, domain, and bias.
- **Source-to-articles pivot** — follow `links.self` (or pass `source.id` to `/v1/news/everything`) to load a publisher's latest stories.
- **Publisher volume** — pass `source.id` to [`/v1/news/count`](../../README.md) to get the number of matching articles for a date range, since there is no per-source coverage endpoint.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
