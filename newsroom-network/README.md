# Newsroom Network

Map the relationships between journalists and the outlets they write for using the [APITube News API](https://apitube.io).

## Overview

The **Newsroom Network** workflow builds a "who writes where" graph from the journalist directory. Each journalist record carries an `outlets` array (every publication that has run their byline) and an `outlet_count`. By walking the list endpoint you can build journalist-to-outlet edges, list every author attached to a given outlet, and surface cross-outlet contributors (`outlet_count > 1`). The profile endpoint adds the same `outlets` field plus coverage when you want to drill into a single node.

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
| `coverage` | boolean | Set to `false` to omit the `coverage` block. The `outlets` field is always present. |

Each journalist object exposes the network fields:

| Field | Description |
|-------|-------------|
| `id` | Journalist ID. |
| `name` | Journalist name. |
| `outlets` | Array of `{ id, name, domain }` for every outlet the author writes for. |
| `outlet_count` | Number of outlets (a value greater than 1 means a cross-outlet contributor). |
| `links.articles` | Link to the author's articles on `/v1/news/everything?author.id=...`. |

## Quick Start

### cURL

```bash
# Page through the directory to collect journalist -> outlet edges
curl -s "https://api.apitube.io/v1/journalists?api_key=YOUR_API_KEY&per_page=50&page=1"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/journalists", params={
    "api_key": "YOUR_API_KEY",
    "per_page": 50,
})
response.raise_for_status()

for journalist in response.json()["results"]:
    for outlet in journalist["outlets"]:
        print(f"{journalist['name']} -> {outlet['name']} ({outlet['domain']})")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({ api_key: "YOUR_API_KEY", per_page: "50" });
const response = await fetch(`https://api.apitube.io/v1/journalists?${params}`);
const data = await response.json();

data.results.forEach((journalist) => {
  journalist.outlets.forEach((outlet) => {
    console.log(`${journalist.name} -> ${outlet.name} (${outlet.domain})`);
  });
});
```

### PHP

```php
$query = http_build_query(["api_key" => "YOUR_API_KEY", "per_page" => 50]);
$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/journalists?{$query}"
), true);

foreach ($data["results"] as $journalist) {
    foreach ($journalist["outlets"] as $outlet) {
        echo "{$journalist['name']} -> {$outlet['name']} ({$outlet['domain']})\n";
    }
}
```

## Response Example

```json
{
  "status": "ok",
  "limit": 50,
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

## Common Use Cases

- **Outlet roster** — list every journalist whose byline appears at a given publication.
- **Network graph** — emit journalist-to-outlet edges for visualization or graph databases.
- **Cross-outlet contributors** — find freelancers and syndicated authors via `outlet_count > 1`.
- **Outlet overlap** — discover which publications share contributors.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
