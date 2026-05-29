# Executive Watchlist — Code Examples

Detailed examples for building and polling an executive watchlist using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Build a Watchlist (Resolve Names to Stable IDs)

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/people"

# Each entry can be resolved by wikidata_id (exact) or name (fuzzy)
EXECUTIVES = [
    {"wikidata_id": "Q317521"},      # Elon Musk
    {"name": "tim cook"},
    {"name": "satya nadella"},
]

def resolve(entry):
    params = {"api_key": API_KEY, **entry}
    response = requests.get(LIST_URL, params=params)
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None

watchlist = []
for entry in EXECUTIVES:
    person = resolve(entry)
    if person:
        watchlist.append({"id": person["id"], "name": person["name"]})
        print(f"  resolved {entry} -> [{person['id']}] {person['name']}")
    else:
        print(f"  could not resolve {entry}")

print(f"\nWatchlist contains {len(watchlist)} executives")
```

### Comparison Table (Momentum and Sentiment)

```python
import requests

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

WATCHLIST = [
    {"id": 5021, "name": "Elon Musk"},
    {"id": 6110, "name": "Tim Cook"},
    {"id": 6231, "name": "Satya Nadella"},
]

def fetch_coverage(person_id):
    response = requests.get(PROFILE_URL.format(person_id), params={"api_key": API_KEY})
    response.raise_for_status()
    return response.json()["coverage"]

rows = []
for exec_ in WATCHLIST:
    cov = fetch_coverage(exec_["id"])
    if not cov:
        continue
    s = cov["sentiment"]
    total = sum(s.values()) or 1
    rows.append({
        "name": exec_["name"],
        "last_30": cov["momentum"]["last_30_days"],
        "change": cov["momentum"]["change_pct"],
        "neg_pct": s["negative"] / total * 100,
    })

print(f"{'Executive':<18} {'30d':>6} {'Change':>8} {'Neg%':>7}")
print("-" * 41)
for r in rows:
    change = "n/a" if r["change"] is None else f"{r['change']:+d}%"
    print(f"{r['name']:<18} {r['last_30']:>6} {change:>8} {r['neg_pct']:>6.1f}%")
```

### Rank by Coverage Spike

```python
import requests

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

WATCHLIST = [
    {"id": 5021, "name": "Elon Musk"},
    {"id": 6110, "name": "Tim Cook"},
    {"id": 6231, "name": "Satya Nadella"},
]

def fetch_momentum(person_id):
    response = requests.get(PROFILE_URL.format(person_id), params={"api_key": API_KEY})
    response.raise_for_status()
    return response.json()["coverage"]["momentum"]

ranked = []
for exec_ in WATCHLIST:
    m = fetch_momentum(exec_["id"])
    ranked.append((exec_["name"], m["change_pct"], m["last_30_days"]))

ranked.sort(key=lambda r: (r[1] if r[1] is not None else 0), reverse=True)

print("Coverage spikes (ranked by 30-day change):\n")
for name, change, last_30 in ranked:
    value = change if change is not None else 0
    direction = "up" if value >= 0 else "down"
    bar = "#" * min(abs(value), 50)
    change_str = "n/a" if change is None else f"{change:+d}%"
    print(f"  {name:<18} {change_str:>5} {direction:<4} ({last_30} articles) {bar}")
```

### Negative Sentiment Alert (Polling Loop)

```python
import requests
import time
from datetime import datetime

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

WATCHLIST = [
    {"id": 5021, "name": "Elon Musk"},
    {"id": 6110, "name": "Tim Cook"},
]

NEG_THRESHOLD = 25.0   # alert when negative share exceeds this percent
POLL_INTERVAL = 3600   # seconds

def negative_share(person_id):
    response = requests.get(PROFILE_URL.format(person_id), params={"api_key": API_KEY})
    response.raise_for_status()
    s = response.json()["coverage"]["sentiment"]
    total = sum(s.values()) or 1
    return s["negative"] / total * 100

print("Monitoring negative sentiment across watchlist...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for exec_ in WATCHLIST:
        neg = negative_share(exec_["id"])
        status = "ALERT" if neg >= NEG_THRESHOLD else "OK"
        print(f"  [{timestamp}] {exec_['name']:<18} negative={neg:5.1f}% [{status}]")

    print()
    time.sleep(POLL_INTERVAL)
```

---

## JavaScript

### Build a Watchlist (Resolve Names to Stable IDs)

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/people";

const EXECUTIVES = [
  { wikidata_id: "Q317521" }, // Elon Musk
  { name: "tim cook" },
  { name: "satya nadella" },
];

async function resolve(entry) {
  const params = new URLSearchParams({ api_key: API_KEY, ...entry });
  const response = await fetch(`${LIST_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  return data.results?.[0] || null;
}

const watchlist = [];
for (const entry of EXECUTIVES) {
  const person = await resolve(entry);
  if (person) {
    watchlist.push({ id: person.id, name: person.name });
    console.log(`  resolved ${JSON.stringify(entry)} -> [${person.id}] ${person.name}`);
  } else {
    console.log(`  could not resolve ${JSON.stringify(entry)}`);
  }
}

console.log(`\nWatchlist contains ${watchlist.length} executives`);
```

### Comparison Table (Momentum and Sentiment)

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  { id: 5021, name: "Elon Musk" },
  { id: 6110, name: "Tim Cook" },
  { id: 6231, name: "Satya Nadella" },
];

async function fetchCoverage(personId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/people/${personId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).coverage;
}

const rows = [];
for (const exec of WATCHLIST) {
  const cov = await fetchCoverage(exec.id);
  if (!cov) continue;
  const s = cov.sentiment;
  const total = Object.values(s).reduce((a, b) => a + b, 0) || 1;
  rows.push({
    name: exec.name,
    last30: cov.momentum.last_30_days,
    change: cov.momentum.change_pct,
    negPct: (s.negative / total) * 100,
  });
}

console.log(`${"Executive".padEnd(18)} ${"30d".padStart(6)} ${"Change".padStart(8)} ${"Neg%".padStart(7)}`);
console.log("-".repeat(41));
rows.forEach((r) => {
  const change = r.change == null ? "n/a" : `${r.change >= 0 ? "+" : ""}${r.change}%`;
  console.log(`${r.name.padEnd(18)} ${String(r.last30).padStart(6)} ${change.padStart(8)} ${(r.negPct.toFixed(1) + "%").padStart(7)}`);
});
```

### Rank by Coverage Spike

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  { id: 5021, name: "Elon Musk" },
  { id: 6110, name: "Tim Cook" },
  { id: 6231, name: "Satya Nadella" },
];

async function fetchMomentum(personId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/people/${personId}?${params}`);
  return (await response.json()).coverage.momentum;
}

const ranked = [];
for (const exec of WATCHLIST) {
  const m = await fetchMomentum(exec.id);
  ranked.push({ name: exec.name, change: m.change_pct, last30: m.last_30_days });
}

ranked.sort((a, b) => (b.change ?? 0) - (a.change ?? 0));

console.log("Coverage spikes (ranked by 30-day change):\n");
ranked.forEach(({ name, change, last30 }) => {
  const value = change ?? 0;
  const direction = value >= 0 ? "up" : "down";
  const bar = "#".repeat(Math.min(Math.abs(value), 50));
  const pct = change == null ? "n/a" : `${change >= 0 ? "+" : ""}${change}%`;
  console.log(`  ${name.padEnd(18)} ${pct.padStart(5)} ${direction.padEnd(4)} (${last30} articles) ${bar}`);
});
```

### Negative Sentiment Alert (Polling Loop)

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  { id: 5021, name: "Elon Musk" },
  { id: 6110, name: "Tim Cook" },
];

const NEG_THRESHOLD = 25.0;
const POLL_INTERVAL = 3600 * 1000;

async function negativeShare(personId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/people/${personId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const s = (await response.json()).coverage.sentiment;
  const total = Object.values(s).reduce((a, b) => a + b, 0) || 1;
  return (s.negative / total) * 100;
}

async function poll() {
  const timestamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  for (const exec of WATCHLIST) {
    const neg = await negativeShare(exec.id);
    const status = neg >= NEG_THRESHOLD ? "ALERT" : "OK";
    console.log(`  [${timestamp}] ${exec.name.padEnd(18)} negative=${neg.toFixed(1).padStart(5)}% [${status}]`);
  }
  console.log();
}

console.log("Monitoring negative sentiment across watchlist...\n");
await poll();
setInterval(poll, POLL_INTERVAL);
```

---

## PHP

### Build a Watchlist (Resolve Names to Stable IDs)

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/people";

$executives = [
    ["wikidata_id" => "Q317521"], // Elon Musk
    ["name" => "tim cook"],
    ["name" => "satya nadella"],
];

function resolve(array $entry): ?array
{
    global $apiKey, $listUrl;

    $query = http_build_query(array_merge(["api_key" => $apiKey], $entry));
    $data  = json_decode(file_get_contents("{$listUrl}?{$query}"), true);
    return $data["results"][0] ?? null;
}

$watchlist = [];
foreach ($executives as $entry) {
    $person = resolve($entry);
    if ($person) {
        $watchlist[] = ["id" => $person["id"], "name" => $person["name"]];
        echo "  resolved " . json_encode($entry) . " -> [{$person['id']}] {$person['name']}\n";
    } else {
        echo "  could not resolve " . json_encode($entry) . "\n";
    }
}

echo "\nWatchlist contains " . count($watchlist) . " executives\n";
```

### Comparison Table (Momentum and Sentiment)

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["id" => 5021, "name" => "Elon Musk"],
    ["id" => 6110, "name" => "Tim Cook"],
    ["id" => 6231, "name" => "Satya Nadella"],
];

function fetchCoverage(int $personId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/people/{$personId}?{$query}"
    ), true);
    return $data["coverage"];
}

$rows = [];
foreach ($watchlist as $exec) {
    $cov = fetchCoverage($exec["id"]);
    if (!$cov) {
        continue;
    }
    $s     = $cov["sentiment"];
    $total = array_sum($s) ?: 1;
    $rows[] = [
        "name"    => $exec["name"],
        "last_30" => $cov["momentum"]["last_30_days"],
        "change"  => $cov["momentum"]["change_pct"],
        "neg_pct" => $s["negative"] / $total * 100,
    ];
}

printf("%-18s %6s %8s %7s\n", "Executive", "30d", "Change", "Neg%");
echo str_repeat("-", 41) . "\n";
foreach ($rows as $r) {
    $change = $r["change"] === null ? "n/a" : sprintf("%+d%%", $r["change"]);
    printf("%-18s %6d %8s %6.1f%%\n",
        $r["name"], $r["last_30"], $change, $r["neg_pct"]);
}
```

### Rank by Coverage Spike

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["id" => 5021, "name" => "Elon Musk"],
    ["id" => 6110, "name" => "Tim Cook"],
    ["id" => 6231, "name" => "Satya Nadella"],
];

function fetchMomentum(int $personId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/people/{$personId}?{$query}"
    ), true);
    return $data["coverage"]["momentum"];
}

$ranked = [];
foreach ($watchlist as $exec) {
    $m = fetchMomentum($exec["id"]);
    $ranked[] = ["name" => $exec["name"], "change" => $m["change_pct"], "last_30" => $m["last_30_days"]];
}

usort($ranked, fn($a, $b) => ($b["change"] ?? 0) <=> ($a["change"] ?? 0));

echo "Coverage spikes (ranked by 30-day change):\n\n";
foreach ($ranked as $r) {
    $value     = $r["change"] ?? 0;
    $direction = $value >= 0 ? "up" : "down";
    $bar       = str_repeat("#", min(abs($value), 50));
    $changeStr = $r["change"] === null ? "n/a" : sprintf("%+d%%", $r["change"]);
    printf("  %-18s %5s %-4s (%d articles) %s\n",
        $r["name"], $changeStr, $direction, $r["last_30"], $bar);
}
```

### Negative Sentiment Alert (Polling Loop)

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["id" => 5021, "name" => "Elon Musk"],
    ["id" => 6110, "name" => "Tim Cook"],
];

$negThreshold = 25.0;   // alert when negative share exceeds this percent
$pollInterval = 3600;   // seconds

function negativeShare(int $personId): float
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/people/{$personId}?{$query}"
    ), true);
    $s     = $data["coverage"]["sentiment"];
    $total = array_sum($s) ?: 1;
    return $s["negative"] / $total * 100;
}

echo "Monitoring negative sentiment across watchlist...\n\n";

while (true) {
    $timestamp = gmdate("Y-m-d H:i:s");
    foreach ($watchlist as $exec) {
        $neg    = negativeShare($exec["id"]);
        $status = $neg >= $negThreshold ? "ALERT" : "OK";
        printf("  [%s] %-18s negative=%5.1f%% [%s]\n",
            $timestamp, $exec["name"], $neg, $status);
    }
    echo "\n";
    sleep($pollInterval);
}
```
