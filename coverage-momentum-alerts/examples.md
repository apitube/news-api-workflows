# Coverage Momentum Alerts — Code Examples

Detailed examples for detecting coverage surges and declines using the APITube News API in **Python**, **JavaScript**, and **PHP**.

Each example reads `coverage.momentum` (`last_30_days`, `previous_30_days`, `change_pct`) from `GET /v1/companies/:id` or `GET /v1/people/:id`. A watchlist mixes both entity kinds.

---

## Python

### Momentum Monitoring Loop

```python
import requests
import time
from datetime import datetime

API_KEY = "YOUR_API_KEY"

WATCHLIST = [
    ("companies", 312),
    ("companies", 4501),
    ("people", 5021),
]
SURGE_THRESHOLD = 25       # percent
DECLINE_THRESHOLD = -25    # percent
POLL_INTERVAL = 3600       # seconds

def fetch_momentum(kind, entity_id):
    response = requests.get(
        f"https://api.apitube.io/v1/{kind}/{entity_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], (profile.get("coverage") or {}).get("momentum")

def classify(change_pct):
    if change_pct is None:
        return "stable"
    if change_pct >= SURGE_THRESHOLD:
        return "SURGE"
    if change_pct <= DECLINE_THRESHOLD:
        return "DECLINE"
    return "stable"

print("Monitoring coverage momentum...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]")

    for kind, entity_id in WATCHLIST:
        name, m = fetch_momentum(kind, entity_id)
        if m is None:
            print(f"  {name:<22} no coverage")
            continue
        status = classify(m["change_pct"])
        flag = "  <-- ALERT" if status != "stable" else ""
        change_str = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"
        print(f"  {name:<22} {m['last_30_days']:>5} vs {m['previous_30_days']:>5} "
              f"({change_str}) [{status}]{flag}")

    print()
    time.sleep(POLL_INTERVAL)
```

### Acceleration Ranking

```python
import requests

API_KEY = "YOUR_API_KEY"

WATCHLIST = [
    ("companies", 312),
    ("companies", 4501),
    ("companies", 7720),
    ("people", 5021),
    ("people", 6033),
]

def fetch_momentum(kind, entity_id):
    response = requests.get(
        f"https://api.apitube.io/v1/{kind}/{entity_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], (profile.get("coverage") or {}).get("momentum")

rows = [(name, m) for name, m in (fetch_momentum(kind, eid) for kind, eid in WATCHLIST) if m]
rows.sort(key=lambda r: r[1]["change_pct"] if r[1]["change_pct"] is not None else 0, reverse=True)

print(f"{'Entity':<22} {'30d':>6} {'Prev':>6} {'Change':>8}  Trend")
print("-" * 58)

for name, m in rows:
    change = m["change_pct"]
    direction = "n/a" if change is None else ("accelerating" if change > 0 else ("cooling" if change < 0 else "flat"))
    change_str = "n/a" if change is None else f"{change:+d}%"
    print(f"{name:<22} {m['last_30_days']:>6} {m['previous_30_days']:>6} "
          f"{change_str:>8}  {direction}")
```

### Negative Coverage Surge Alert

```python
import requests

API_KEY = "YOUR_API_KEY"

WATCHLIST = [
    ("companies", 312),
    ("people", 5021),
]
SURGE_THRESHOLD = 25  # percent

def fetch_coverage(kind, entity_id):
    response = requests.get(
        f"https://api.apitube.io/v1/{kind}/{entity_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

print("Scanning for negative coverage surges...\n")

for kind, entity_id in WATCHLIST:
    profile = fetch_coverage(kind, entity_id)
    cov = profile.get("coverage")
    if not cov:
        print(f"{profile['name']}: no coverage\n")
        continue
    m = cov["momentum"]
    s = cov["sentiment"]

    total = s["positive"] + s["neutral"] + s["negative"] or 1
    negative_share = s["negative"] / total

    change = m["change_pct"] if m["change_pct"] is not None else 0
    change_str = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"
    surging = change >= SURGE_THRESHOLD
    negative_leaning = negative_share > (s["positive"] / total)

    print(f"{profile['name']}:")
    print(f"  momentum     : {change_str} ({m['previous_30_days']} -> {m['last_30_days']})")
    print(f"  negative share: {negative_share:.0%}")

    if surging and negative_leaning:
        print(f"  ALERT: negative coverage surge (volume up {change_str}, "
              f"negative-leaning tone)")
    print()
```

---

## JavaScript

### Momentum Monitoring Loop

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  ["companies", 312],
  ["companies", 4501],
  ["people", 5021],
];
const SURGE_THRESHOLD = 25;
const DECLINE_THRESHOLD = -25;
const POLL_INTERVAL = 3600 * 1000;

async function fetchMomentum(kind, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${kind}/${entityId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const profile = await response.json();
  return { name: profile.name, momentum: profile.coverage?.momentum };
}

function classify(changePct) {
  if (changePct == null) return "stable";
  if (changePct >= SURGE_THRESHOLD) return "SURGE";
  if (changePct <= DECLINE_THRESHOLD) return "DECLINE";
  return "stable";
}

async function poll() {
  const timestamp = new Date().toISOString().replace("T", " ").slice(0, 19);
  console.log(`[${timestamp}]`);

  for (const [kind, entityId] of WATCHLIST) {
    const { name, momentum: m } = await fetchMomentum(kind, entityId);
    if (!m) {
      console.log(`  ${name.padEnd(22)} no coverage`);
      continue;
    }
    const status = classify(m.change_pct);
    const flag = status !== "stable" ? "  <-- ALERT" : "";
    const change = m.change_pct == null ? "n/a" : `${m.change_pct > 0 ? "+" : ""}${m.change_pct}%`;
    console.log(
      `  ${name.padEnd(22)} ${String(m.last_30_days).padStart(5)} vs ` +
      `${String(m.previous_30_days).padStart(5)} (${change}) [${status}]${flag}`
    );
  }
  console.log();
}

console.log("Monitoring coverage momentum...\n");
await poll();
setInterval(poll, POLL_INTERVAL);
```

### Acceleration Ranking

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  ["companies", 312],
  ["companies", 4501],
  ["companies", 7720],
  ["people", 5021],
  ["people", 6033],
];

async function fetchMomentum(kind, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${kind}/${entityId}?${params}`);
  const profile = await response.json();
  return { name: profile.name, momentum: profile.coverage?.momentum };
}

const rows = (await Promise.all(WATCHLIST.map(([k, id]) => fetchMomentum(k, id)))).filter((r) => r.momentum);
rows.sort((a, b) => (b.momentum.change_pct ?? 0) - (a.momentum.change_pct ?? 0));

console.log(`${"Entity".padEnd(22)} ${"30d".padStart(6)} ${"Prev".padStart(6)} ${"Change".padStart(8)}  Trend`);
console.log("-".repeat(58));

rows.forEach(({ name, momentum: m }) => {
  const direction = m.change_pct == null ? "n/a" : m.change_pct > 0 ? "accelerating" : m.change_pct < 0 ? "cooling" : "flat";
  const change = m.change_pct == null ? "n/a" : `${m.change_pct > 0 ? "+" : ""}${m.change_pct}%`;
  console.log(
    `${name.padEnd(22)} ${String(m.last_30_days).padStart(6)} ` +
    `${String(m.previous_30_days).padStart(6)} ${change.padStart(8)}  ${direction}`
  );
});
```

### Negative Coverage Surge Alert

```javascript
const API_KEY = "YOUR_API_KEY";

const WATCHLIST = [
  ["companies", 312],
  ["people", 5021],
];
const SURGE_THRESHOLD = 25;

async function fetchCoverage(kind, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${kind}/${entityId}?${params}`);
  return response.json();
}

console.log("Scanning for negative coverage surges...\n");

for (const [kind, entityId] of WATCHLIST) {
  const profile = await fetchCoverage(kind, entityId);
  if (!profile.coverage) {
    console.log(`${profile.name}: no coverage\n`);
    continue;
  }
  const { momentum: m, sentiment: s } = profile.coverage;

  const total = s.positive + s.neutral + s.negative || 1;
  const negativeShare = s.negative / total;

  const changeStr = m.change_pct == null ? "n/a" : `${m.change_pct > 0 ? "+" : ""}${m.change_pct}%`;
  const surging = (m.change_pct ?? 0) >= SURGE_THRESHOLD;
  const negativeLeaning = negativeShare > s.positive / total;

  console.log(`${profile.name}:`);
  console.log(`  momentum      : ${changeStr} (${m.previous_30_days} -> ${m.last_30_days})`);
  console.log(`  negative share: ${Math.round(negativeShare * 100)}%`);

  if (surging && negativeLeaning) {
    console.log(`  ALERT: negative coverage surge (volume up ${changeStr}, negative-leaning tone)`);
  }
  console.log();
}
```

---

## PHP

### Momentum Monitoring Loop

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["companies", 312],
    ["companies", 4501],
    ["people", 5021],
];
$surgeThreshold   = 25;
$declineThreshold = -25;
$pollInterval     = 3600;

function fetchMomentum(string $kind, int $entityId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$kind}/{$entityId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "momentum" => $profile["coverage"]["momentum"] ?? null];
}

function classify(?int $changePct): string
{
    global $surgeThreshold, $declineThreshold;

    if ($changePct === null) {
        return "stable";
    }
    if ($changePct >= $surgeThreshold) {
        return "SURGE";
    }
    if ($changePct <= $declineThreshold) {
        return "DECLINE";
    }
    return "stable";
}

echo "Monitoring coverage momentum...\n\n";

while (true) {
    echo "[" . gmdate("Y-m-d H:i:s") . "]\n";

    foreach ($watchlist as [$kind, $entityId]) {
        ["name" => $name, "momentum" => $m] = fetchMomentum($kind, $entityId);
        if ($m === null) {
            printf("  %-22s no coverage\n", $name);
            continue;
        }
        $status    = classify($m["change_pct"]);
        $flag      = $status !== "stable" ? "  <-- ALERT" : "";
        $changeStr = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
        printf("  %-22s %5d vs %5d (%s) [%s]%s\n",
            $name, $m["last_30_days"], $m["previous_30_days"], $changeStr, $status, $flag);
    }

    echo "\n";
    sleep($pollInterval);
}
```

### Acceleration Ranking

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["companies", 312],
    ["companies", 4501],
    ["companies", 7720],
    ["people", 5021],
    ["people", 6033],
];

function fetchMomentum(string $kind, int $entityId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$kind}/{$entityId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "momentum" => $profile["coverage"]["momentum"] ?? null];
}

$rows = array_values(array_filter(array_map(fn($e) => fetchMomentum($e[0], $e[1]), $watchlist), fn($r) => $r["momentum"]));
usort($rows, fn($a, $b) => ($b["momentum"]["change_pct"] ?? 0) - ($a["momentum"]["change_pct"] ?? 0));

printf("%-22s %6s %6s %8s  Trend\n", "Entity", "30d", "Prev", "Change");
echo str_repeat("-", 58) . "\n";

foreach ($rows as $row) {
    $m         = $row["momentum"];
    $change    = $m["change_pct"];
    $direction = $change === null ? "n/a" : ($change > 0 ? "accelerating" : ($change < 0 ? "cooling" : "flat"));
    $changeStr = $change === null ? "n/a" : sprintf("%+d%%", $change);
    printf("%-22s %6d %6d %8s  %s\n",
        $row["name"], $m["last_30_days"], $m["previous_30_days"], $changeStr, $direction);
}
```

### Negative Coverage Surge Alert

```php
<?php

$apiKey = "YOUR_API_KEY";

$watchlist = [
    ["companies", 312],
    ["people", 5021],
];
$surgeThreshold = 25;

function fetchCoverage(string $kind, int $entityId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$kind}/{$entityId}?{$query}"
    ), true);
}

echo "Scanning for negative coverage surges...\n\n";

foreach ($watchlist as [$kind, $entityId]) {
    $profile = fetchCoverage($kind, $entityId);
    $cov     = $profile["coverage"] ?? null;
    if (!$cov) {
        echo "{$profile['name']}: no coverage\n\n";
        continue;
    }
    $m       = $cov["momentum"];
    $s       = $cov["sentiment"];

    $total          = $s["positive"] + $s["neutral"] + $s["negative"] ?: 1;
    $negativeShare  = $s["negative"] / $total;

    $change         = $m["change_pct"] ?? 0;
    $changeStr      = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
    $surging        = $change >= $surgeThreshold;
    $negativeLeaning = $negativeShare > ($s["positive"] / $total);

    echo "{$profile['name']}:\n";
    printf("  momentum      : %s (%d -> %d)\n",
        $changeStr, $m["previous_30_days"], $m["last_30_days"]);
    printf("  negative share: %d%%\n", round($negativeShare * 100));

    if ($surging && $negativeLeaning) {
        printf("  ALERT: negative coverage surge (volume up %s, negative-leaning tone)\n",
            $changeStr);
    }
    echo "\n";
}
```
