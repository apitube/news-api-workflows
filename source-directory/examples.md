# Source Directory — Code Examples

Detailed examples for browsing the publisher catalog and inspecting source profiles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Search Sources by Name

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/sources"

def search_sources(name, per_page=10):
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "name": name,
        "per_page": per_page,
    })
    response.raise_for_status()
    return response.json()

data = search_sources("guardian")
print(f"Found {len(data['results'])} matching sources\n")

for source in data["results"]:
    opr = source["rank"]["opr"]
    print(f"  [{source['id']:>7}] {source['name']}")
    print(f"           domain={source['domain']}  bias={source['bias']}  opr={opr}")
    print(f"           articles -> {source['links']['articles']}")
    print()
```

### Browse Sources by Country with Pagination

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/sources"

def iter_sources_by_country(country_id, per_page=50, max_pages=5):
    page = 1
    while page <= max_pages:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "country": country_id,
            "per_page": per_page,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for source in data["results"]:
            yield source

        if not data.get("has_next_pages"):
            break
        page += 1

# Walk every US publisher (country_id 840)
count = 0
for source in iter_sources_by_country(840):
    count += 1
    print(f"  {source['name']:<30} {source['domain']:<25} {source['resource_type']}")

print(f"\nCollected {count} sources for country 840")
```

### Render a Source Profile Card

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_source_profile(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

source = get_source_profile(4232)
cov = source["coverage"] or {}
sent = cov["sentiment"]
total = sent["positive"] + sent["neutral"] + sent["negative"] or 1
m = cov["momentum"]
change_str = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"

print(f"{source['name']} ({source['domain']})")
print(f"  Type:    {source['resource_type']}")
print(f"  Bias:    {source['bias']}")
print(f"  OPR:     {source['rank']['opr']}")
print(f"  Website: {source['links']['website']}")
print()
print(f"  Articles tracked: {cov['article_count']:,}")
print(f"  Active:           {cov['first_seen'] or 'n/a'} -> {cov['last_seen'] or 'n/a'}")
print(f"  Momentum (30d):   {change_str} "
      f"({m['previous_30_days']} -> {m['last_30_days']})")
print("  Sentiment mix:")
for label in ("positive", "neutral", "negative"):
    pct = sent[label] / total * 100
    bar = "#" * int(pct / 2)
    print(f"    {label:>8}: {pct:5.1f}% {bar}")
```

### Pivot from a Source to Its Latest Articles

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_source_profile(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

# 1. Resolve the source
source = get_source_profile(4232)
print(f"Latest articles from {source['name']} ({source['domain']}):\n")

# 2. Use source.id with /v1/news/everything (same target as links.articles)
response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": API_KEY,
    "source.id": source["id"],
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 10,
})
response.raise_for_status()

for article in response.json()["results"]:
    print(f"  - {article['title']}")
```

---

## JavaScript

### Search Sources by Name

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/sources";

async function searchSources(name, perPage = 10) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    name,
    per_page: String(perPage),
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await searchSources("guardian");
console.log(`Found ${data.results.length} matching sources\n`);

data.results.forEach((source) => {
  console.log(`  [${source.id}] ${source.name}`);
  console.log(`        domain=${source.domain}  bias=${source.bias}  opr=${source.rank.opr}`);
  console.log(`        articles -> ${source.links.articles}\n`);
});
```

### Browse Sources by Country with Pagination

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/sources";

async function listSourcesByCountry(countryId, perPage = 50, maxPages = 5) {
  const collected = [];
  let page = 1;

  while (page <= maxPages) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      country: String(countryId),
      per_page: String(perPage),
      page: String(page),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    collected.push(...data.results);

    if (!data.has_next_pages) break;
    page++;
  }

  return collected;
}

const sources = await listSourcesByCountry(840);
sources.forEach((source) => {
  console.log(
    `  ${source.name.padEnd(30)} ${source.domain.padEnd(25)} ${source.resource_type}`
  );
});
console.log(`\nCollected ${sources.length} sources for country 840`);
```

### Render a Source Profile Card

```javascript
const API_KEY = "YOUR_API_KEY";

async function getSourceProfile(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(
    `https://api.apitube.io/v1/sources/${sourceId}?${params}`
  );
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const source = await getSourceProfile(4232);
const cov = source.coverage ?? {};
const sent = cov.sentiment;
const total = sent.positive + sent.neutral + sent.negative || 1;
const m = cov.momentum;
const changeStr = m.change_pct == null ? "n/a" : `${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%`;

console.log(`${source.name} (${source.domain})`);
console.log(`  Type:    ${source.resource_type}`);
console.log(`  Bias:    ${source.bias}`);
console.log(`  OPR:     ${source.rank.opr}`);
console.log(`  Website: ${source.links.website}\n`);
console.log(`  Articles tracked: ${cov.article_count.toLocaleString()}`);
console.log(`  Active:           ${cov.first_seen ?? "n/a"} -> ${cov.last_seen ?? "n/a"}`);
console.log(
  `  Momentum (30d):   ${changeStr} ` +
  `(${m.previous_30_days} -> ${m.last_30_days})`
);
console.log("  Sentiment mix:");
["positive", "neutral", "negative"].forEach((label) => {
  const pct = (sent[label] / total) * 100;
  const bar = "#".repeat(Math.round(pct / 2));
  console.log(`    ${label.padStart(8)}: ${pct.toFixed(1).padStart(5)}% ${bar}`);
});
```

### Pivot from a Source to Its Latest Articles

```javascript
const API_KEY = "YOUR_API_KEY";

async function getSourceProfile(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(
    `https://api.apitube.io/v1/sources/${sourceId}?${params}`
  );
  return response.json();
}

// 1. Resolve the source
const source = await getSourceProfile(4232);
console.log(`Latest articles from ${source.name} (${source.domain}):\n`);

// 2. Use source.id with /v1/news/everything (same target as links.articles)
const params = new URLSearchParams({
  api_key: API_KEY,
  "source.id": String(source.id),
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(`  - ${article.title}`);
});
```

---

## PHP

### Search Sources by Name

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/sources";

function searchSources(string $name, int $perPage = 10): array
{
    global $apiKey, $baseUrl;

    $query = http_build_query([
        "api_key"  => $apiKey,
        "name"     => $name,
        "per_page" => $perPage,
    ]);

    return json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
}

$data = searchSources("guardian");
echo "Found " . count($data["results"]) . " matching sources\n\n";

foreach ($data["results"] as $source) {
    $opr = $source["rank"]["opr"];
    printf("  [%7d] %s\n", $source["id"], $source["name"]);
    printf("           domain=%s  bias=%s  opr=%d\n", $source["domain"], $source["bias"], $opr);
    printf("           articles -> %s\n\n", $source["links"]["articles"]);
}
```

### Browse Sources by Country with Pagination

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/sources";

function listSourcesByCountry(int $countryId, int $perPage = 50, int $maxPages = 5): array
{
    global $apiKey, $baseUrl;

    $collected = [];
    $page      = 1;

    while ($page <= $maxPages) {
        $query = http_build_query([
            "api_key"  => $apiKey,
            "country"  => $countryId,
            "per_page" => $perPage,
            "page"     => $page,
        ]);

        $data      = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $collected = array_merge($collected, $data["results"]);

        if (empty($data["has_next_pages"])) {
            break;
        }
        $page++;
    }

    return $collected;
}

$sources = listSourcesByCountry(840);
foreach ($sources as $source) {
    printf("  %-30s %-25s %s\n",
        $source["name"], $source["domain"], $source["resource_type"]);
}

echo "\nCollected " . count($sources) . " sources for country 840\n";
```

### Render a Source Profile Card

```php
<?php

$apiKey = "YOUR_API_KEY";

function getSourceProfile(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

$source = getSourceProfile(4232);
$cov    = $source["coverage"] ?? [];
$sent   = $cov["sentiment"];
$total  = ($sent["positive"] + $sent["neutral"] + $sent["negative"]) ?: 1;
$m      = $cov["momentum"];
$changeStr = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);

printf("%s (%s)\n", $source["name"], $source["domain"]);
printf("  Type:    %s\n", $source["resource_type"]);
printf("  Bias:    %s\n", $source["bias"]);
printf("  OPR:     %d\n", $source["rank"]["opr"]);
printf("  Website: %s\n\n", $source["links"]["website"]);
printf("  Articles tracked: %s\n", number_format($cov["article_count"]));
printf("  Active:           %s -> %s\n", $cov["first_seen"] ?? "n/a", $cov["last_seen"] ?? "n/a");
printf("  Momentum (30d):   %s (%d -> %d)\n",
    $changeStr,
    $m["previous_30_days"],
    $m["last_30_days"]);
echo "  Sentiment mix:\n";
foreach (["positive", "neutral", "negative"] as $label) {
    $pct = $sent[$label] / $total * 100;
    $bar = str_repeat("#", (int) ($pct / 2));
    printf("    %8s: %5.1f%% %s\n", $label, $pct, $bar);
}
```

### Pivot from a Source to Its Latest Articles

```php
<?php

$apiKey = "YOUR_API_KEY";

function getSourceProfile(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

// 1. Resolve the source
$source = getSourceProfile(4232);
printf("Latest articles from %s (%s):\n\n", $source["name"], $source["domain"]);

// 2. Use source.id with /v1/news/everything (same target as links.articles)
$query = http_build_query([
    "api_key"    => $apiKey,
    "source.id"  => $source["id"],
    "sort.by"    => "published_at",
    "sort.order" => "desc",
    "per_page"   => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($data["results"] as $article) {
    echo "  - {$article['title']}\n";
}
```
