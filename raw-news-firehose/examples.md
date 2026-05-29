# Raw News Firehose — Code Examples

Detailed examples for low-latency ingestion of the raw discovery feed using the APITube News API in **Python**, **JavaScript**, and **PHP**.

The `/v1/news/raw` endpoint returns articles before NLP enrichment, so the response contains no `language`, `categories`, `topics`, `entities`, or `sentiment` fields. The feed rotates quickly (roughly one day of retention), which makes deduplication and prompt consumption the central concerns of every example below.

---

## Python

### Continuous Firehose with href Deduplication

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/raw"

POLL_INTERVAL = 30  # seconds
seen_hrefs = set()

def fetch_newest(per_page=250):
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "per_page": per_page,
        "sort.by": "published_at",
        "sort.order": "desc",
    })
    response.raise_for_status()
    return response.json().get("results", [])

print("Streaming raw news (Ctrl+C to stop)...\n")

while True:
    fresh = 0
    for article in fetch_newest():
        href = article["href"]
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        fresh += 1
        print(f"  {article['created_at']}  [{article['source']['domain']}] {article['title']}")

    print(f"  -- {fresh} new items, {len(seen_hrefs)} tracked --\n")
    time.sleep(POLL_INTERVAL)
```

### Monitor Specific Sources

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/raw"

# Up to 3 sitemap source IDs
WATCHED_SOURCES = "123,456,789"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "source.id": WATCHED_SOURCES,
    "per_page": 100,
    "sort.by": "published_at",
    "sort.order": "desc",
})
response.raise_for_status()
data = response.json()

by_source = {}
for article in data["results"]:
    domain = article["source"]["domain"]
    by_source.setdefault(domain, []).append(article)

print("Latest raw items per watched source:\n")
for domain, articles in by_source.items():
    print(f"  {domain} ({len(articles)} items)")
    for article in articles[:5]:
        print(f"    {article['created_at']}  {article['title']}")
    print()
```

### Daily Backfill with Pagination

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/raw"

def backfill_day(day, per_page=250):
    """Pull every raw item published on a single day, paging until exhausted."""
    page = 1
    collected = []

    while True:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "published_at.start": f"{day}T00:00:00Z",
            "published_at.end": f"{day}T23:59:59Z",
            "per_page": per_page,
            "page": page,
            "sort.by": "published_at",
            "sort.order": "asc",
        })
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        collected.extend(results)
        print(f"  page {page}: +{len(results)} (total {len(collected)})")

        if not data.get("has_next_pages") or not results:
            break
        page += 1

    return collected

items = backfill_day("2026-05-28")
print(f"\nBackfilled {len(items)} raw items for 2026-05-28")
```

### Excluding Noisy Sources

```python
import requests
from collections import Counter

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/raw"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "ignore.source.id": "111,222",   # drop up to 3 low-value publishers
    "per_page": 250,
    "sort.by": "published_at",
    "sort.order": "desc",
})
response.raise_for_status()
data = response.json()

domains = Counter(a["source"]["domain"] for a in data["results"])

print("Source distribution after exclusions:\n")
for domain, count in domains.most_common():
    bar = "#" * count
    print(f"  {domain:<28} {count:>3} {bar}")
```

---

## JavaScript

### Continuous Firehose with href Deduplication

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/raw";

const POLL_INTERVAL = 30000; // ms
const seenHrefs = new Set();

async function fetchNewest(perPage = 250) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    per_page: String(perPage),
    "sort.by": "published_at",
    "sort.order": "desc",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).results || [];
}

async function streamForever() {
  console.log("Streaming raw news (Ctrl+C to stop)...\n");

  for (;;) {
    let fresh = 0;
    for (const article of await fetchNewest()) {
      if (seenHrefs.has(article.href)) continue;
      seenHrefs.add(article.href);
      fresh++;
      console.log(`  ${article.created_at}  [${article.source.domain}] ${article.title}`);
    }
    console.log(`  -- ${fresh} new items, ${seenHrefs.size} tracked --\n`);
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));
  }
}

streamForever();
```

### Monitor Specific Sources

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/raw";

async function monitorSources(sourceIds) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "source.id": sourceIds, // up to 3, comma-separated
    per_page: "100",
    "sort.by": "published_at",
    "sort.order": "desc",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  const bySource = {};
  for (const article of data.results) {
    const domain = article.source.domain;
    (bySource[domain] ||= []).push(article);
  }

  console.log("Latest raw items per watched source:\n");
  for (const [domain, articles] of Object.entries(bySource)) {
    console.log(`  ${domain} (${articles.length} items)`);
    articles.slice(0, 5).forEach((a) => {
      console.log(`    ${a.created_at}  ${a.title}`);
    });
    console.log();
  }
}

monitorSources("123,456,789");
```

### Daily Backfill with Pagination

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/raw";

async function backfillDay(day, perPage = 250) {
  let page = 1;
  const collected = [];

  for (;;) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "published_at.start": `${day}T00:00:00Z`,
      "published_at.end": `${day}T23:59:59Z`,
      per_page: String(perPage),
      page: String(page),
      "sort.by": "published_at",
      "sort.order": "asc",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const results = data.results || [];
    collected.push(...results);
    console.log(`  page ${page}: +${results.length} (total ${collected.length})`);

    if (!data.has_next_pages || results.length === 0) break;
    page++;
  }

  return collected;
}

const items = await backfillDay("2026-05-28");
console.log(`\nBackfilled ${items.length} raw items for 2026-05-28`);
```

### Excluding Noisy Sources

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/raw";

async function sourceDistribution() {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "ignore.source.id": "111,222", // drop up to 3 publishers
    per_page: "250",
    "sort.by": "published_at",
    "sort.order": "desc",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  const domains = {};
  for (const a of data.results) {
    domains[a.source.domain] = (domains[a.source.domain] || 0) + 1;
  }

  console.log("Source distribution after exclusions:\n");
  Object.entries(domains)
    .sort((a, b) => b[1] - a[1])
    .forEach(([domain, count]) => {
      console.log(`  ${domain.padEnd(28)} ${String(count).padStart(3)} ${"#".repeat(count)}`);
    });
}

sourceDistribution();
```

---

## PHP

### Continuous Firehose with href Deduplication

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/raw";

$pollInterval = 30; // seconds
$seenHrefs    = [];

function fetchNewest(int $perPage = 250): array
{
    global $apiKey, $baseUrl;

    $query = http_build_query([
        "api_key"    => $apiKey,
        "per_page"   => $perPage,
        "sort.by"    => "published_at",
        "sort.order" => "desc",
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    return $data["results"] ?? [];
}

echo "Streaming raw news (Ctrl+C to stop)...\n\n";

while (true) {
    $fresh = 0;
    foreach (fetchNewest() as $article) {
        $href = $article["href"];
        if (isset($seenHrefs[$href])) {
            continue;
        }
        $seenHrefs[$href] = true;
        $fresh++;
        printf("  %s  [%s] %s\n", $article["created_at"], $article["source"]["domain"], $article["title"]);
    }

    printf("  -- %d new items, %d tracked --\n\n", $fresh, count($seenHrefs));
    sleep($pollInterval);
}
```

### Monitor Specific Sources

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/raw";

$query = http_build_query([
    "api_key"    => $apiKey,
    "source.id"  => "123,456,789", // up to 3 source IDs
    "per_page"   => 100,
    "sort.by"    => "published_at",
    "sort.order" => "desc",
]);

$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

$bySource = [];
foreach ($data["results"] as $article) {
    $bySource[$article["source"]["domain"]][] = $article;
}

echo "Latest raw items per watched source:\n\n";
foreach ($bySource as $domain => $articles) {
    printf("  %s (%d items)\n", $domain, count($articles));
    foreach (array_slice($articles, 0, 5) as $article) {
        printf("    %s  %s\n", $article["created_at"], $article["title"]);
    }
    echo "\n";
}
```

### Daily Backfill with Pagination

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/raw";

function backfillDay(string $day, int $perPage = 250): array
{
    global $apiKey, $baseUrl;

    $page      = 1;
    $collected = [];

    while (true) {
        $query = http_build_query([
            "api_key"            => $apiKey,
            "published_at.start" => "{$day}T00:00:00Z",
            "published_at.end"   => "{$day}T23:59:59Z",
            "per_page"           => $perPage,
            "page"               => $page,
            "sort.by"            => "published_at",
            "sort.order"         => "asc",
        ]);

        $data    = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $results = $data["results"] ?? [];
        $collected = array_merge($collected, $results);
        printf("  page %d: +%d (total %d)\n", $page, count($results), count($collected));

        if (empty($data["has_next_pages"]) || count($results) === 0) {
            break;
        }
        $page++;
    }

    return $collected;
}

$items = backfillDay("2026-05-28");
printf("\nBackfilled %d raw items for 2026-05-28\n", count($items));
```

### Excluding Noisy Sources

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/raw";

$query = http_build_query([
    "api_key"          => $apiKey,
    "ignore.source.id" => "111,222", // drop up to 3 publishers
    "per_page"         => 250,
    "sort.by"          => "published_at",
    "sort.order"       => "desc",
]);

$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

$domains = [];
foreach ($data["results"] as $article) {
    $domain = $article["source"]["domain"];
    $domains[$domain] = ($domains[$domain] ?? 0) + 1;
}

arsort($domains);

echo "Source distribution after exclusions:\n\n";
foreach ($domains as $domain => $count) {
    printf("  %-28s %3d %s\n", $domain, $count, str_repeat("#", $count));
}
```
