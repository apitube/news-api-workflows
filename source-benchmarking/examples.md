# Source Benchmarking — Code Examples

Detailed examples for comparing publisher coverage using the APITube News API in **Python**, **JavaScript**, and **PHP**.

Every example polls `/v1/sources/:id` once per source ID and works only with the source summary coverage fields: `article_count`, `first_seen`, `last_seen`, `sentiment`, `momentum`, and `timeline`.

---

## Python

### Benchmark Table

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]

def fetch_coverage(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

sources = [fetch_coverage(sid) for sid in SOURCE_IDS]

print(f"{'Source':<22} {'Articles':>12} {'First seen':>12} {'30d':>8} {'Change':>8}")
print("-" * 64)

for source in sources:
    cov = source["coverage"] or {}
    change = cov["momentum"]["change_pct"]
    change_str = "n/a" if change is None else f"{change:>+7d}%"
    print(f"{source['name']:<22} {cov['article_count']:>12,} "
          f"{(cov['first_seen'] or 'n/a'):>12} {cov['momentum']['last_30_days']:>8} "
          f"{change_str:>8}")
```

### Sentiment Profile of Each Source

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]

def fetch_coverage(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

print("Sentiment balance per source:\n")

for source_id in SOURCE_IDS:
    source = fetch_coverage(source_id)
    sent = source["coverage"]["sentiment"]
    total = sent["positive"] + sent["neutral"] + sent["negative"] or 1

    print(f"  {source['name']} ({source['domain']})")
    for label in ("positive", "neutral", "negative"):
        pct = sent[label] / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:>8}: {sent[label]:>10,} ({pct:5.1f}%) {bar}")
    print()
```

### Rank Sources by Volume and by Positivity

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]

def fetch_coverage(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

sources = [fetch_coverage(sid) for sid in SOURCE_IDS]

def positive_share(source):
    sent = source["coverage"]["sentiment"]
    total = sent["positive"] + sent["neutral"] + sent["negative"] or 1
    return sent["positive"] / total

by_volume = sorted(sources, key=lambda s: s["coverage"]["article_count"], reverse=True)
by_positivity = sorted(sources, key=positive_share, reverse=True)

print("Ranked by article volume:\n")
for rank, source in enumerate(by_volume, 1):
    print(f"  {rank}. {source['name']:<22} {source['coverage']['article_count']:>12,}")

print("\nRanked by positive share:\n")
for rank, source in enumerate(by_positivity, 1):
    print(f"  {rank}. {source['name']:<22} {positive_share(source):6.1%}")
```

### Find Publishers Gaining Momentum

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]

def fetch_coverage(source_id):
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

sources = [fetch_coverage(sid) for sid in SOURCE_IDS]
sources.sort(
    key=lambda s: s["coverage"]["momentum"]["change_pct"] if s["coverage"]["momentum"]["change_pct"] is not None else 0,
    reverse=True,
)

print("Publishing momentum (last 30 days vs previous 30 days):\n")
print(f"{'Source':<22} {'Prev':>8} {'Last':>8} {'Change':>8}")
print("-" * 48)

for source in sources:
    m = source["coverage"]["momentum"]
    change = m["change_pct"] if m["change_pct"] is not None else 0
    trend = "up" if change > 0 else "down" if change < 0 else "flat"
    change_str = "n/a" if m["change_pct"] is None else f"{m['change_pct']:>+6d}%"
    print(f"{source['name']:<22} {m['previous_30_days']:>8} "
          f"{m['last_30_days']:>8} {change_str:>7}  {trend}")
```

---

## JavaScript

### Benchmark Table

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function fetchCoverage(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/sources/${sourceId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const sources = await Promise.all(SOURCE_IDS.map(fetchCoverage));

console.log(
  `${"Source".padEnd(22)} ${"Articles".padStart(12)} ${"First seen".padStart(12)} ` +
  `${"30d".padStart(8)} ${"Change".padStart(8)}`
);
console.log("-".repeat(64));

sources.forEach((source) => {
  const cov = source.coverage ?? {};
  const change = cov.momentum.change_pct == null
    ? "n/a"
    : `${cov.momentum.change_pct >= 0 ? "+" : ""}${cov.momentum.change_pct}%`;
  console.log(
    `${source.name.padEnd(22)} ${cov.article_count.toLocaleString().padStart(12)} ` +
    `${(cov.first_seen ?? "n/a").padStart(12)} ${String(cov.momentum.last_30_days).padStart(8)} ` +
    `${change.padStart(8)}`
  );
});
```

### Sentiment Profile of Each Source

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function fetchCoverage(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/sources/${sourceId}?${params}`);
  return response.json();
}

console.log("Sentiment balance per source:\n");

for (const id of SOURCE_IDS) {
  const source = await fetchCoverage(id);
  const sent = source.coverage.sentiment;
  const total = sent.positive + sent.neutral + sent.negative || 1;

  console.log(`  ${source.name} (${source.domain})`);
  for (const label of ["positive", "neutral", "negative"]) {
    const pct = (sent[label] / total) * 100;
    const bar = "#".repeat(Math.round(pct / 2));
    console.log(
      `    ${label.padStart(8)}: ${sent[label].toLocaleString().padStart(10)} ` +
      `(${pct.toFixed(1).padStart(5)}%) ${bar}`
    );
  }
  console.log();
}
```

### Rank Sources by Volume and by Positivity

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function fetchCoverage(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/sources/${sourceId}?${params}`);
  return response.json();
}

function positiveShare(source) {
  const sent = source.coverage.sentiment;
  const total = sent.positive + sent.neutral + sent.negative || 1;
  return sent.positive / total;
}

const sources = await Promise.all(SOURCE_IDS.map(fetchCoverage));

const byVolume = [...sources].sort(
  (a, b) => b.coverage.article_count - a.coverage.article_count
);
const byPositivity = [...sources].sort(
  (a, b) => positiveShare(b) - positiveShare(a)
);

console.log("Ranked by article volume:\n");
byVolume.forEach((source, i) => {
  console.log(
    `  ${i + 1}. ${source.name.padEnd(22)} ${source.coverage.article_count.toLocaleString().padStart(12)}`
  );
});

console.log("\nRanked by positive share:\n");
byPositivity.forEach((source, i) => {
  console.log(`  ${i + 1}. ${source.name.padEnd(22)} ${(positiveShare(source) * 100).toFixed(1)}%`);
});
```

### Find Publishers Gaining Momentum

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function fetchCoverage(sourceId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/sources/${sourceId}?${params}`);
  return response.json();
}

const sources = await Promise.all(SOURCE_IDS.map(fetchCoverage));
sources.sort((a, b) => (b.coverage.momentum.change_pct ?? 0) - (a.coverage.momentum.change_pct ?? 0));

console.log("Publishing momentum (last 30 days vs previous 30 days):\n");
console.log(`${"Source".padEnd(22)} ${"Prev".padStart(8)} ${"Last".padStart(8)} ${"Change".padStart(8)}`);
console.log("-".repeat(48));

sources.forEach((source) => {
  const m = source.coverage.momentum;
  const changePct = m.change_pct ?? 0;
  const trend = changePct > 0 ? "up" : changePct < 0 ? "down" : "flat";
  const change = m.change_pct == null ? "n/a" : `${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%`;
  console.log(
    `${source.name.padEnd(22)} ${String(m.previous_30_days).padStart(8)} ` +
    `${String(m.last_30_days).padStart(8)} ${change.padStart(7)}  ${trend}`
  );
});
```

---

## PHP

### Benchmark Table

```php
<?php

$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

function fetchCoverage(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

$sources = array_map("fetchCoverage", $sourceIds);

printf("%-22s %12s %12s %8s %8s\n", "Source", "Articles", "First seen", "30d", "Change");
echo str_repeat("-", 64) . "\n";

foreach ($sources as $source) {
    $cov = $source["coverage"] ?? [];
    $changeStr = $cov["momentum"]["change_pct"] === null
        ? "n/a"
        : sprintf("%+d%%", $cov["momentum"]["change_pct"]);
    printf("%-22s %12s %12s %8d %8s\n",
        $source["name"],
        number_format($cov["article_count"]),
        $cov["first_seen"] ?? "n/a",
        $cov["momentum"]["last_30_days"],
        $changeStr);
}
```

### Sentiment Profile of Each Source

```php
<?php

$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

function fetchCoverage(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

echo "Sentiment balance per source:\n\n";

foreach ($sourceIds as $id) {
    $source = fetchCoverage($id);
    $sent   = $source["coverage"]["sentiment"];
    $total  = ($sent["positive"] + $sent["neutral"] + $sent["negative"]) ?: 1;

    printf("  %s (%s)\n", $source["name"], $source["domain"]);
    foreach (["positive", "neutral", "negative"] as $label) {
        $pct = $sent[$label] / $total * 100;
        $bar = str_repeat("#", (int) ($pct / 2));
        printf("    %8s: %10s (%5.1f%%) %s\n", $label, number_format($sent[$label]), $pct, $bar);
    }
    echo "\n";
}
```

### Rank Sources by Volume and by Positivity

```php
<?php

$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

function fetchCoverage(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

function positiveShare(array $source): float
{
    $sent  = $source["coverage"]["sentiment"];
    $total = ($sent["positive"] + $sent["neutral"] + $sent["negative"]) ?: 1;
    return $sent["positive"] / $total;
}

$sources = array_map("fetchCoverage", $sourceIds);

$byVolume = $sources;
usort($byVolume, fn($a, $b) => $b["coverage"]["article_count"] <=> $a["coverage"]["article_count"]);

$byPositivity = $sources;
usort($byPositivity, fn($a, $b) => positiveShare($b) <=> positiveShare($a));

echo "Ranked by article volume:\n\n";
foreach ($byVolume as $i => $source) {
    printf("  %d. %-22s %12s\n",
        $i + 1, $source["name"], number_format($source["coverage"]["article_count"]));
}

echo "\nRanked by positive share:\n\n";
foreach ($byPositivity as $i => $source) {
    printf("  %d. %-22s %5.1f%%\n", $i + 1, $source["name"], positiveShare($source) * 100);
}
```

### Find Publishers Gaining Momentum

```php
<?php

$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

function fetchCoverage(int $sourceId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$sourceId}?{$query}"
    ), true);
}

$sources = array_map("fetchCoverage", $sourceIds);
usort($sources, fn($a, $b) =>
    ($b["coverage"]["momentum"]["change_pct"] ?? 0) <=> ($a["coverage"]["momentum"]["change_pct"] ?? 0));

echo "Publishing momentum (last 30 days vs previous 30 days):\n\n";
printf("%-22s %8s %8s %8s\n", "Source", "Prev", "Last", "Change");
echo str_repeat("-", 48) . "\n";

foreach ($sources as $source) {
    $m       = $source["coverage"]["momentum"];
    $change  = $m["change_pct"] ?? 0;
    $trend   = $change > 0 ? "up" : ($change < 0 ? "down" : "flat");
    $changeStr = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
    printf("%-22s %8d %8d %7s  %s\n",
        $source["name"], $m["previous_30_days"], $m["last_30_days"], $changeStr, $trend);
}
```
