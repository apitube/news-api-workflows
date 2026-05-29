# Raw to Enriched Pipeline

A two-stage workflow that catches headlines instantly from the raw feed and then backfills NLP-enriched data, using the [APITube News API](https://apitube.io).

## Overview

The **Raw to Enriched Pipeline** combines two endpoints to balance speed against completeness. Stage one polls `/v1/news/raw` to learn about articles within seconds of discovery, before any parsing or NLP runs. Stage two queries `/v1/news/everything` for the same sources and date range to attach enriched fields (`sentiment`, `entities`, `categories`, `topics`, `language`, `summary`) once the pipeline has processed them. You decide per article whether to act on the fast-but-bare raw signal or wait for the slower-but-complete enriched record. Matching between the two stages is done on the article URL: the raw `href` corresponds to the enriched article `href`.

## API Endpoints

```
GET https://api.apitube.io/v1/news/raw
GET https://api.apitube.io/v1/news/everything
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/news/raw?api_key=YOUR_API_KEY
https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY
```

## Parameters

Shared between both endpoints:

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `per_page` | integer | Results per page. Raw: default `100`, max `250`. |
| `page` | integer | Page number for pagination. |
| `source.id` | string | Comma-separated source IDs. Raw allows max 3. |
| `published_at.start` | string | Start of the publication range (ISO 8601 or `YYYY-MM-DD`). |
| `published_at.end` | string | End of the publication range. |
| `sort.by` | string | Sort field (e.g. `published_at`). |
| `sort.order` | string | Sort direction: `asc` or `desc`. |

Only on `/v1/news/everything`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `language.code` | string | Filter by language code (e.g. `en`). |
| `sentiment.overall.polarity` | string | `positive`, `negative`, or `neutral`. |
| `category.id` | string | Filter by category. |
| `topic.id` | string | Filter by topic. |
| `entity.id` | string | Filter by entity. |

## Quick Start

### cURL

```bash
# Stage 1: catch the latest raw headlines from a source (fast, no enrichment)
curl -s "https://api.apitube.io/v1/news/raw?api_key=YOUR_API_KEY&source.id=4232&per_page=50&sort.by=published_at&sort.order=desc"

# Stage 2: pull the same source enriched (sentiment, entities, categories)
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&source.id=4232&per_page=50&sort.by=published_at&sort.order=desc"
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"

raw = requests.get("https://api.apitube.io/v1/news/raw", params={
    "api_key": API_KEY, "source.id": "4232", "per_page": 50,
    "sort.by": "published_at", "sort.order": "desc",
})
raw.raise_for_status()
print(f"Raw stage: {len(raw.json()['results'])} headlines (no enrichment yet)")

enriched = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": API_KEY, "source.id": "4232", "per_page": 50,
    "sort.by": "published_at", "sort.order": "desc",
})
enriched.raise_for_status()
print(f"Enriched stage: {len(enriched.json()['results'])} articles with sentiment/entities")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";

const rawParams = new URLSearchParams({
  api_key: API_KEY, "source.id": "4232", per_page: "50",
  "sort.by": "published_at", "sort.order": "desc",
});
const raw = await (await fetch(`https://api.apitube.io/v1/news/raw?${rawParams}`)).json();
console.log(`Raw stage: ${raw.results.length} headlines (no enrichment yet)`);

const richParams = new URLSearchParams({
  api_key: API_KEY, "source.id": "4232", per_page: "50",
  "sort.by": "published_at", "sort.order": "desc",
});
const enriched = await (await fetch(`https://api.apitube.io/v1/news/everything?${richParams}`)).json();
console.log(`Enriched stage: ${enriched.results.length} articles with sentiment/entities`);
```

### PHP

```php
$apiKey = "YOUR_API_KEY";

$rawQuery = http_build_query([
    "api_key" => $apiKey, "source.id" => "4232", "per_page" => 50,
    "sort.by" => "published_at", "sort.order" => "desc",
]);
$raw = json_decode(file_get_contents("https://api.apitube.io/v1/news/raw?{$rawQuery}"), true);
echo "Raw stage: " . count($raw["results"]) . " headlines (no enrichment yet)\n";

$richQuery = http_build_query([
    "api_key" => $apiKey, "source.id" => "4232", "per_page" => 50,
    "sort.by" => "published_at", "sort.order" => "desc",
]);
$enriched = json_decode(file_get_contents("https://api.apitube.io/v1/news/everything?{$richQuery}"), true);
echo "Enriched stage: " . count($enriched["results"]) . " articles with sentiment/entities\n";
```

## Response Example

Raw stage (`/v1/news/raw`) — note the absence of enrichment fields:

```json
{
  "status": "ok",
  "results": [
    {
      "id": 987654321,
      "title": "Raw headline straight from the RSS feed",
      "href": "https://example.com/article/123",
      "created_at": "2026-05-27T08:15:00",
      "author": "Jane Doe",
      "source": { "id": 4232, "domain": "example.com" }
    }
  ]
}
```

Enriched stage (`/v1/news/everything`) — the same article once processed:

```json
{
  "status": "ok",
  "results": [
    {
      "id": 987654321,
      "title": "Headline straight from the RSS feed",
      "href": "https://example.com/article/123",
      "language": "en",
      "sentiment": { "overall": { "polarity": "neutral", "score": 0.05 } },
      "categories": [ { "id": 199, "name": "Politics", "taxonomy": "iptc_mediatopics" } ],
      "topics": [ { "id": "elections", "name": "Elections" } ],
      "entities": [ { "id": 5021, "name": "Elon Musk", "type": "person" } ],
      "source": { "id": 4232, "domain": "example.com" }
    }
  ]
}
```

## Common Use Cases

- **Early breaking detection** — react to a raw headline immediately, then enrich it once `/v1/news/everything` has the sentiment and entities.
- **Latency comparison** — measure how long an article takes to move from the raw feed to the enriched index for a given source.
- **Unified stream** — emit a single record per article that starts bare from raw and gets upgraded in place when enrichment arrives.
- **Speed vs completeness routing** — send time-critical alerts off raw, route analytics off enriched.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
- [../raw-news-firehose](../raw-news-firehose) — the raw feed on its own.
- [../sentiment-analysis](../sentiment-analysis) — enriched sentiment filtering on `/v1/news/everything`.
