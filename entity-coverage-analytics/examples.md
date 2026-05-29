# Entity Coverage Analytics — Code Examples

Detailed examples for analyzing the `coverage` block of a person or company using the APITube News API in **Python**, **JavaScript**, and **PHP**.

All examples work against both `/v1/people/:id` and `/v1/companies/:id` because both return the same ENTITY-form coverage object.

---

## Python

### Fetch Coverage for a Person or Company

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(entity_type, entity_id):
    # entity_type is "people" or "companies"
    url = f"https://api.apitube.io/v1/{entity_type}/{entity_id}"
    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    data = response.json()
    return data["name"], data["coverage"]

name, coverage = get_coverage("people", 5021)
if not coverage:
    raise SystemExit(f"No coverage available for {name}")

print(f"{name}: {coverage['article_count']:,} articles "
      f"({coverage['first_seen'] or 'n/a'} -> {coverage['last_seen'] or 'n/a'})")
```

### Monthly Trend Chart from Timeline

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(entity_type, entity_id):
    url = f"https://api.apitube.io/v1/{entity_type}/{entity_id}"
    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    data = response.json()
    return data["name"], data["coverage"]

name, coverage = get_coverage("companies", 312)
timeline = coverage["timeline"]
peak = max((p["count"] for p in timeline), default=1) or 1

print(f"Monthly coverage trend for {name}:\n")
for point in timeline:
    bar = "#" * round(point["count"] / peak * 40)
    print(f"  {point['period']}  {point['count']:>6} {bar}")
```

### Sentiment Breakdown

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(entity_type, entity_id):
    url = f"https://api.apitube.io/v1/{entity_type}/{entity_id}"
    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    data = response.json()
    return data["name"], data["coverage"]

name, coverage = get_coverage("people", 5021)
sentiment = coverage["sentiment"]
total = sum(sentiment.values()) or 1

print(f"Sentiment breakdown for {name} ({total:,} classified articles):\n")
for label in ("positive", "neutral", "negative"):
    count = sentiment[label]
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    print(f"  {label:>8}: {count:>7,} ({pct:5.1f}%) {bar}")

net = (sentiment["positive"] - sentiment["negative"]) / total * 100
print(f"\n  Net sentiment: {net:+.1f}%")

m = coverage["momentum"]
change = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"
print(f"  Momentum: {m['previous_30_days']} -> {m['last_30_days']} ({change})")
```

### Share of Coverage by Country and Source

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(entity_type, entity_id):
    url = f"https://api.apitube.io/v1/{entity_type}/{entity_id}"
    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    data = response.json()
    return data["name"], data["coverage"]

def share_table(title, items, label_key):
    total = sum(i["count"] for i in items) or 1
    print(f"{title}:")
    for item in sorted(items, key=lambda i: i["count"], reverse=True):
        share = item["count"] / total * 100
        bar = "#" * int(share / 2)
        print(f"  {item[label_key]:<22} {item['count']:>7,} ({share:5.1f}%) {bar}")
    print()

name, coverage = get_coverage("companies", 312)
print(f"Share of coverage for {name}:\n")
share_table("By country", coverage["top_countries"], "name")
share_table("By source", coverage["top_sources"], "name")
```

### Related Entities as a Co-Mention Graph

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(entity_type, entity_id):
    url = f"https://api.apitube.io/v1/{entity_type}/{entity_id}"
    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    data = response.json()
    return data["name"], data["coverage"]

name, coverage = get_coverage("people", 5021)
related = coverage["related_entities"]
strongest = max((e["count"] for e in related), default=1) or 1

print(f"Co-mention graph for {name}:\n")
for entity in sorted(related, key=lambda e: e["count"], reverse=True):
    weight = entity["count"] / strongest
    edge = "=" * round(weight * 30)
    print(f"  {name} --{edge}--> {entity['name']} ({entity['count']:,})")
```

---

## JavaScript

### Fetch Coverage for a Person or Company

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(entityType, entityId) {
  // entityType is "people" or "companies"
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${entityType}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return { name: data.name, coverage: data.coverage };
}

const { name, coverage } = await getCoverage("people", 5021);
if (!coverage) throw new Error(`No coverage available for ${name}`);
console.log(`${name}: ${coverage.article_count.toLocaleString()} articles (${coverage.first_seen ?? "n/a"} -> ${coverage.last_seen ?? "n/a"})`);
```

### Monthly Trend Chart from Timeline

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(entityType, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${entityType}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return { name: data.name, coverage: data.coverage };
}

const { name, coverage } = await getCoverage("companies", 312);
const timeline = coverage.timeline;
const peak = Math.max(...timeline.map((p) => p.count), 1);

console.log(`Monthly coverage trend for ${name}:\n`);
timeline.forEach((point) => {
  const bar = "#".repeat(Math.round((point.count / peak) * 40));
  console.log(`  ${point.period}  ${String(point.count).padStart(6)} ${bar}`);
});
```

### Sentiment Breakdown

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(entityType, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${entityType}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return { name: data.name, coverage: data.coverage };
}

const { name, coverage } = await getCoverage("people", 5021);
const sentiment = coverage.sentiment;
const total = Object.values(sentiment).reduce((a, b) => a + b, 0) || 1;

console.log(`Sentiment breakdown for ${name} (${total.toLocaleString()} classified articles):\n`);
for (const label of ["positive", "neutral", "negative"]) {
  const count = sentiment[label];
  const pct = (count / total) * 100;
  const bar = "#".repeat(Math.round(pct / 2));
  console.log(`  ${label.padStart(8)}: ${String(count).padStart(7)} (${pct.toFixed(1).padStart(5)}%) ${bar}`);
}

const net = ((sentiment.positive - sentiment.negative) / total) * 100;
console.log(`\n  Net sentiment: ${net >= 0 ? "+" : ""}${net.toFixed(1)}%`);

const m = coverage.momentum;
const change = m.change_pct == null ? "n/a" : `${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%`;
console.log(`  Momentum: ${m.previous_30_days} -> ${m.last_30_days} (${change})`);
```

### Share of Coverage by Country and Source

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(entityType, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${entityType}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return { name: data.name, coverage: data.coverage };
}

function shareTable(title, items, labelKey) {
  const total = items.reduce((sum, i) => sum + i.count, 0) || 1;
  console.log(`${title}:`);
  items
    .slice()
    .sort((a, b) => b.count - a.count)
    .forEach((item) => {
      const share = (item.count / total) * 100;
      const bar = "#".repeat(Math.round(share / 2));
      console.log(`  ${String(item[labelKey]).padEnd(22)} ${String(item.count).padStart(7)} (${share.toFixed(1).padStart(5)}%) ${bar}`);
    });
  console.log();
}

const { name, coverage } = await getCoverage("companies", 312);
console.log(`Share of coverage for ${name}:\n`);
shareTable("By country", coverage.top_countries, "name");
shareTable("By source", coverage.top_sources, "name");
```

### Related Entities as a Co-Mention Graph

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(entityType, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${entityType}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return { name: data.name, coverage: data.coverage };
}

const { name, coverage } = await getCoverage("people", 5021);
const related = coverage.related_entities;
const strongest = Math.max(...related.map((e) => e.count), 1);

console.log(`Co-mention graph for ${name}:\n`);
related
  .slice()
  .sort((a, b) => b.count - a.count)
  .forEach((entity) => {
    const weight = entity.count / strongest;
    const edge = "=".repeat(Math.round(weight * 30));
    console.log(`  ${name} --${edge}--> ${entity.name} (${entity.count.toLocaleString()})`);
  });
```

---

## PHP

### Fetch Coverage for a Person or Company

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(string $entityType, int $entityId): array
{
    global $apiKey;

    // $entityType is "people" or "companies"
    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$entityType}/{$entityId}?{$query}"
    ), true);
    return [$data["name"], $data["coverage"]];
}

[$name, $coverage] = getCoverage("people", 5021);
if (!$coverage) {
    exit("No coverage available for {$name}\n");
}
printf("%s: %s articles (%s -> %s)\n",
    $name, number_format($coverage["article_count"]),
    $coverage["first_seen"] ?? "n/a", $coverage["last_seen"] ?? "n/a");
```

### Monthly Trend Chart from Timeline

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(string $entityType, int $entityId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$entityType}/{$entityId}?{$query}"
    ), true);
    return [$data["name"], $data["coverage"]];
}

[$name, $coverage] = getCoverage("companies", 312);
$timeline = $coverage["timeline"];
$peak     = max(array_column($timeline, "count")) ?: 1;

echo "Monthly coverage trend for {$name}:\n\n";
foreach ($timeline as $point) {
    $bar = str_repeat("#", (int) round($point["count"] / $peak * 40));
    printf("  %s  %6d %s\n", $point["period"], $point["count"], $bar);
}
```

### Sentiment Breakdown

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(string $entityType, int $entityId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$entityType}/{$entityId}?{$query}"
    ), true);
    return [$data["name"], $data["coverage"]];
}

[$name, $coverage] = getCoverage("people", 5021);
$sentiment = $coverage["sentiment"];
$total     = array_sum($sentiment) ?: 1;

printf("Sentiment breakdown for %s (%s classified articles):\n\n", $name, number_format($total));
foreach (["positive", "neutral", "negative"] as $label) {
    $count = $sentiment[$label];
    $pct   = $count / $total * 100;
    $bar   = str_repeat("#", (int) ($pct / 2));
    printf("  %8s: %7s (%5.1f%%) %s\n", $label, number_format($count), $pct, $bar);
}

$net = ($sentiment["positive"] - $sentiment["negative"]) / $total * 100;
printf("\n  Net sentiment: %+.1f%%\n", $net);

$m      = $coverage["momentum"];
$change = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
printf("  Momentum: %d -> %d (%s)\n",
    $m["previous_30_days"], $m["last_30_days"], $change);
```

### Share of Coverage by Country and Source

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(string $entityType, int $entityId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$entityType}/{$entityId}?{$query}"
    ), true);
    return [$data["name"], $data["coverage"]];
}

function shareTable(string $title, array $items, string $labelKey): void
{
    $total = array_sum(array_column($items, "count")) ?: 1;
    usort($items, fn($a, $b) => $b["count"] <=> $a["count"]);

    echo "{$title}:\n";
    foreach ($items as $item) {
        $share = $item["count"] / $total * 100;
        $bar   = str_repeat("#", (int) ($share / 2));
        printf("  %-22s %7s (%5.1f%%) %s\n",
            $item[$labelKey], number_format($item["count"]), $share, $bar);
    }
    echo "\n";
}

[$name, $coverage] = getCoverage("companies", 312);
echo "Share of coverage for {$name}:\n\n";
shareTable("By country", $coverage["top_countries"], "name");
shareTable("By source", $coverage["top_sources"], "name");
```

### Related Entities as a Co-Mention Graph

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(string $entityType, int $entityId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$entityType}/{$entityId}?{$query}"
    ), true);
    return [$data["name"], $data["coverage"]];
}

[$name, $coverage] = getCoverage("people", 5021);
$related   = $coverage["related_entities"];
$strongest = max(array_column($related, "count")) ?: 1;

usort($related, fn($a, $b) => $b["count"] <=> $a["count"]);

echo "Co-mention graph for {$name}:\n\n";
foreach ($related as $entity) {
    $weight = $entity["count"] / $strongest;
    $edge   = str_repeat("=", (int) round($weight * 30));
    printf("  %s --%s--> %s (%s)\n",
        $name, $edge, $entity["name"], number_format($entity["count"]));
}
```
