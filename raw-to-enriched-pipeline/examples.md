# Raw to Enriched Pipeline — Code Examples

Detailed examples for combining the raw discovery feed with NLP-enriched articles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

Stage one (`/v1/news/raw`) gives headlines fast but with no enrichment. Stage two (`/v1/news/everything`) attaches `sentiment`, `entities`, `categories`, `topics`, and `language`. The two stages are joined on the article `href`.

---

## Python

### Early Breaking Detection, Then Enrich

```python
import requests

API_KEY = "YOUR_API_KEY"
RAW_URL = "https://api.apitube.io/v1/news/raw"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"

SOURCE_ID = "4232"

# Stage 1: get raw headlines immediately
raw = requests.get(RAW_URL, params={
    "api_key": API_KEY, "source.id": SOURCE_ID, "per_page": 50,
    "sort.by": "published_at", "sort.order": "desc",
})
raw.raise_for_status()
raw_items = raw.json()["results"]
print(f"Caught {len(raw_items)} raw headlines\n")

# Stage 2: pull enriched articles for the same source, index by href
enriched = requests.get(EVERYTHING_URL, params={
    "api_key": API_KEY, "source.id": SOURCE_ID, "per_page": 50,
    "sort.by": "published_at", "sort.order": "desc",
})
enriched.raise_for_status()
by_href = {a["href"]: a for a in enriched.json()["results"]}

for item in raw_items:
    match = by_href.get(item["href"])
    if match:
        polarity = match["sentiment"]["overall"]["polarity"]
        print(f"  [ENRICHED {polarity:>8}] {item['title']}")
    else:
        print(f"  [RAW ONLY        ] {item['title']}")
```

### Latency Comparison Between Stages

```python
import requests

API_KEY = "YOUR_API_KEY"
RAW_URL = "https://api.apitube.io/v1/news/raw"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"

SOURCE_ID = "4232"

def fetch_hrefs(url, extra=None):
    params = {
        "api_key": API_KEY, "source.id": SOURCE_ID, "per_page": 250,
        "sort.by": "published_at", "sort.order": "desc",
    }
    if extra:
        params.update(extra)
    response = requests.get(url, params=params)
    response.raise_for_status()
    return {a["href"] for a in response.json()["results"]}

raw_hrefs = fetch_hrefs(RAW_URL)
enriched_hrefs = fetch_hrefs(EVERYTHING_URL)

pending = raw_hrefs - enriched_hrefs
both = raw_hrefs & enriched_hrefs

print("Pipeline lag snapshot:\n")
print(f"  In raw feed:            {len(raw_hrefs)}")
print(f"  Already enriched:       {len(both)}")
print(f"  Awaiting enrichment:    {len(pending)}")
print(f"  Enrichment coverage:    {len(both) / (len(raw_hrefs) or 1):.1%}")
```

### Unified Stream (Upgrade In Place)

```python
import requests

API_KEY = "YOUR_API_KEY"
RAW_URL = "https://api.apitube.io/v1/news/raw"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"

SOURCE_ID = "4232"

def build_unified(source_id):
    stream = {}

    raw = requests.get(RAW_URL, params={
        "api_key": API_KEY, "source.id": source_id, "per_page": 100,
        "sort.by": "published_at", "sort.order": "desc",
    })
    raw.raise_for_status()
    for a in raw.json()["results"]:
        stream[a["href"]] = {
            "title": a["title"],
            "stage": "raw",
            "sentiment": None,
            "entities": [],
        }

    enriched = requests.get(EVERYTHING_URL, params={
        "api_key": API_KEY, "source.id": source_id, "per_page": 100,
        "sort.by": "published_at", "sort.order": "desc",
    })
    enriched.raise_for_status()
    for a in enriched.json()["results"]:
        record = stream.setdefault(a["href"], {"title": a["title"], "entities": []})
        record["stage"] = "enriched"
        record["title"] = a["title"]
        record["sentiment"] = a["sentiment"]["overall"]["polarity"]
        record["entities"] = [e["name"] for e in a.get("entities", [])]

    return stream

unified = build_unified(SOURCE_ID)
for href, rec in unified.items():
    tag = rec["stage"].upper()
    extra = f" sentiment={rec['sentiment']}" if rec["stage"] == "enriched" else ""
    print(f"  [{tag:>8}]{extra}  {rec['title']}")
```

### Route by Speed vs Completeness

```python
import requests

API_KEY = "YOUR_API_KEY"
RAW_URL = "https://api.apitube.io/v1/news/raw"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"

SOURCE_ID = "4232"
BREAKING_KEYWORDS = ["breaking", "urgent", "live"]

# Fast path: scan raw titles for time-critical signals
raw = requests.get(RAW_URL, params={
    "api_key": API_KEY, "source.id": SOURCE_ID, "per_page": 100,
    "sort.by": "published_at", "sort.order": "desc",
})
raw.raise_for_status()

alerts = []
for a in raw.json()["results"]:
    title = a["title"].lower()
    if any(kw in title for kw in BREAKING_KEYWORDS):
        alerts.append(a)
        print(f"  [ALERT -> raw path] {a['title']}")

# Slow path: everything else gets enriched analytics
enriched = requests.get(EVERYTHING_URL, params={
    "api_key": API_KEY, "source.id": SOURCE_ID, "per_page": 100,
    "sentiment.overall.polarity": "negative",
    "language.code": "en",
    "sort.by": "published_at", "sort.order": "desc",
})
enriched.raise_for_status()
print(f"\n  Analytics path: {len(enriched.json()['results'])} negative enriched articles")
```

---

## JavaScript

### Early Breaking Detection, Then Enrich

```javascript
const API_KEY = "YOUR_API_KEY";
const RAW_URL = "https://api.apitube.io/v1/news/raw";
const EVERYTHING_URL = "https://api.apitube.io/v1/news/everything";

const SOURCE_ID = "4232";

async function fetchJson(url, params) {
  const response = await fetch(`${url}?${new URLSearchParams(params)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const raw = await fetchJson(RAW_URL, {
  api_key: API_KEY, "source.id": SOURCE_ID, per_page: "50",
  "sort.by": "published_at", "sort.order": "desc",
});
console.log(`Caught ${raw.results.length} raw headlines\n`);

const enriched = await fetchJson(EVERYTHING_URL, {
  api_key: API_KEY, "source.id": SOURCE_ID, per_page: "50",
  "sort.by": "published_at", "sort.order": "desc",
});
const byHref = new Map(enriched.results.map((a) => [a.href, a]));

for (const item of raw.results) {
  const match = byHref.get(item.href);
  if (match) {
    const polarity = match.sentiment.overall.polarity;
    console.log(`  [ENRICHED ${polarity.padStart(8)}] ${item.title}`);
  } else {
    console.log(`  [RAW ONLY        ] ${item.title}`);
  }
}
```

### Latency Comparison Between Stages

```javascript
const API_KEY = "YOUR_API_KEY";
const RAW_URL = "https://api.apitube.io/v1/news/raw";
const EVERYTHING_URL = "https://api.apitube.io/v1/news/everything";

const SOURCE_ID = "4232";

async function fetchHrefs(url) {
  const params = new URLSearchParams({
    api_key: API_KEY, "source.id": SOURCE_ID, per_page: "250",
    "sort.by": "published_at", "sort.order": "desc",
  });
  const data = await (await fetch(`${url}?${params}`)).json();
  return new Set(data.results.map((a) => a.href));
}

const rawHrefs = await fetchHrefs(RAW_URL);
const enrichedHrefs = await fetchHrefs(EVERYTHING_URL);

const both = [...rawHrefs].filter((h) => enrichedHrefs.has(h));
const pending = [...rawHrefs].filter((h) => !enrichedHrefs.has(h));

console.log("Pipeline lag snapshot:\n");
console.log(`  In raw feed:            ${rawHrefs.size}`);
console.log(`  Already enriched:       ${both.length}`);
console.log(`  Awaiting enrichment:    ${pending.length}`);
console.log(`  Enrichment coverage:    ${((both.length / (rawHrefs.size || 1)) * 100).toFixed(1)}%`);
```

### Unified Stream (Upgrade In Place)

```javascript
const API_KEY = "YOUR_API_KEY";
const RAW_URL = "https://api.apitube.io/v1/news/raw";
const EVERYTHING_URL = "https://api.apitube.io/v1/news/everything";

const SOURCE_ID = "4232";

async function fetchJson(url) {
  const params = new URLSearchParams({
    api_key: API_KEY, "source.id": SOURCE_ID, per_page: "100",
    "sort.by": "published_at", "sort.order": "desc",
  });
  return (await fetch(`${url}?${params}`)).json();
}

async function buildUnified() {
  const stream = new Map();

  const raw = await fetchJson(RAW_URL);
  for (const a of raw.results) {
    stream.set(a.href, { title: a.title, stage: "raw", sentiment: null, entities: [] });
  }

  const enriched = await fetchJson(EVERYTHING_URL);
  for (const a of enriched.results) {
    const record = stream.get(a.href) || { entities: [] };
    record.title = a.title;
    record.stage = "enriched";
    record.sentiment = a.sentiment.overall.polarity;
    record.entities = (a.entities || []).map((e) => e.name);
    stream.set(a.href, record);
  }

  return stream;
}

const unified = await buildUnified();
for (const rec of unified.values()) {
  const tag = rec.stage.toUpperCase().padStart(8);
  const extra = rec.stage === "enriched" ? ` sentiment=${rec.sentiment}` : "";
  console.log(`  [${tag}]${extra}  ${rec.title}`);
}
```

### Route by Speed vs Completeness

```javascript
const API_KEY = "YOUR_API_KEY";
const RAW_URL = "https://api.apitube.io/v1/news/raw";
const EVERYTHING_URL = "https://api.apitube.io/v1/news/everything";

const SOURCE_ID = "4232";
const BREAKING_KEYWORDS = ["breaking", "urgent", "live"];

const rawParams = new URLSearchParams({
  api_key: API_KEY, "source.id": SOURCE_ID, per_page: "100",
  "sort.by": "published_at", "sort.order": "desc",
});
const raw = await (await fetch(`${RAW_URL}?${rawParams}`)).json();

for (const a of raw.results) {
  const title = a.title.toLowerCase();
  if (BREAKING_KEYWORDS.some((kw) => title.includes(kw))) {
    console.log(`  [ALERT -> raw path] ${a.title}`);
  }
}

const richParams = new URLSearchParams({
  api_key: API_KEY, "source.id": SOURCE_ID, per_page: "100",
  "sentiment.overall.polarity": "negative", "language.code": "en",
  "sort.by": "published_at", "sort.order": "desc",
});
const enriched = await (await fetch(`${EVERYTHING_URL}?${richParams}`)).json();
console.log(`\n  Analytics path: ${enriched.results.length} negative enriched articles`);
```

---

## PHP

### Early Breaking Detection, Then Enrich

```php
<?php

$apiKey        = "YOUR_API_KEY";
$rawUrl        = "https://api.apitube.io/v1/news/raw";
$everythingUrl = "https://api.apitube.io/v1/news/everything";

$sourceId = "4232";

$rawQuery = http_build_query([
    "api_key" => $apiKey, "source.id" => $sourceId, "per_page" => 50,
    "sort.by" => "published_at", "sort.order" => "desc",
]);
$raw = json_decode(file_get_contents("{$rawUrl}?{$rawQuery}"), true);
echo "Caught " . count($raw["results"]) . " raw headlines\n\n";

$richQuery = http_build_query([
    "api_key" => $apiKey, "source.id" => $sourceId, "per_page" => 50,
    "sort.by" => "published_at", "sort.order" => "desc",
]);
$enriched = json_decode(file_get_contents("{$everythingUrl}?{$richQuery}"), true);

$byHref = [];
foreach ($enriched["results"] as $a) {
    $byHref[$a["href"]] = $a;
}

foreach ($raw["results"] as $item) {
    if (isset($byHref[$item["href"]])) {
        $polarity = $byHref[$item["href"]]["sentiment"]["overall"]["polarity"];
        printf("  [ENRICHED %8s] %s\n", $polarity, $item["title"]);
    } else {
        printf("  [RAW ONLY        ] %s\n", $item["title"]);
    }
}
```

### Latency Comparison Between Stages

```php
<?php

$apiKey        = "YOUR_API_KEY";
$rawUrl        = "https://api.apitube.io/v1/news/raw";
$everythingUrl = "https://api.apitube.io/v1/news/everything";

$sourceId = "4232";

function fetchHrefs(string $url): array
{
    global $apiKey, $sourceId;

    $query = http_build_query([
        "api_key" => $apiKey, "source.id" => $sourceId, "per_page" => 250,
        "sort.by" => "published_at", "sort.order" => "desc",
    ]);
    $data  = json_decode(file_get_contents("{$url}?{$query}"), true);

    $hrefs = [];
    foreach ($data["results"] as $a) {
        $hrefs[$a["href"]] = true;
    }
    return $hrefs;
}

$rawHrefs      = fetchHrefs($rawUrl);
$enrichedHrefs = fetchHrefs($everythingUrl);

$both    = array_intersect_key($rawHrefs, $enrichedHrefs);
$pending = array_diff_key($rawHrefs, $enrichedHrefs);
$rawTotal = count($rawHrefs) ?: 1;

echo "Pipeline lag snapshot:\n\n";
printf("  In raw feed:            %d\n", count($rawHrefs));
printf("  Already enriched:       %d\n", count($both));
printf("  Awaiting enrichment:    %d\n", count($pending));
printf("  Enrichment coverage:    %.1f%%\n", count($both) / $rawTotal * 100);
```

### Unified Stream (Upgrade In Place)

```php
<?php

$apiKey        = "YOUR_API_KEY";
$rawUrl        = "https://api.apitube.io/v1/news/raw";
$everythingUrl = "https://api.apitube.io/v1/news/everything";

$sourceId = "4232";

function fetchResults(string $url): array
{
    global $apiKey, $sourceId;

    $query = http_build_query([
        "api_key" => $apiKey, "source.id" => $sourceId, "per_page" => 100,
        "sort.by" => "published_at", "sort.order" => "desc",
    ]);
    $data = json_decode(file_get_contents("{$url}?{$query}"), true);
    return $data["results"];
}

$stream = [];
foreach (fetchResults($rawUrl) as $a) {
    $stream[$a["href"]] = [
        "title"     => $a["title"],
        "stage"     => "raw",
        "sentiment" => null,
        "entities"  => [],
    ];
}

foreach (fetchResults($everythingUrl) as $a) {
    $stream[$a["href"]] = [
        "title"     => $a["title"],
        "stage"     => "enriched",
        "sentiment" => $a["sentiment"]["overall"]["polarity"],
        "entities"  => array_map(fn($e) => $e["name"], $a["entities"] ?? []),
    ];
}

foreach ($stream as $rec) {
    $tag   = strtoupper($rec["stage"]);
    $extra = $rec["stage"] === "enriched" ? " sentiment={$rec['sentiment']}" : "";
    printf("  [%8s]%s  %s\n", $tag, $extra, $rec["title"]);
}
```

### Route by Speed vs Completeness

```php
<?php

$apiKey        = "YOUR_API_KEY";
$rawUrl        = "https://api.apitube.io/v1/news/raw";
$everythingUrl = "https://api.apitube.io/v1/news/everything";

$sourceId         = "4232";
$breakingKeywords = ["breaking", "urgent", "live"];

$rawQuery = http_build_query([
    "api_key" => $apiKey, "source.id" => $sourceId, "per_page" => 100,
    "sort.by" => "published_at", "sort.order" => "desc",
]);
$raw = json_decode(file_get_contents("{$rawUrl}?{$rawQuery}"), true);

foreach ($raw["results"] as $a) {
    $title = strtolower($a["title"]);
    foreach ($breakingKeywords as $kw) {
        if (str_contains($title, $kw)) {
            printf("  [ALERT -> raw path] %s\n", $a["title"]);
            break;
        }
    }
}

$richQuery = http_build_query([
    "api_key"                    => $apiKey,
    "source.id"                  => $sourceId,
    "per_page"                   => 100,
    "sentiment.overall.polarity" => "negative",
    "language.code"              => "en",
    "sort.by"                    => "published_at",
    "sort.order"                 => "desc",
]);
$enriched = json_decode(file_get_contents("{$everythingUrl}?{$richQuery}"), true);
printf("\n  Analytics path: %d negative enriched articles\n", count($enriched["results"]));
```
