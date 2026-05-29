# Share of Voice — Code Examples

Detailed examples for measuring share of voice across competing companies using the APITube News API in **Python**, **JavaScript**, and **PHP**.

Each example polls `GET /v1/companies/:id` for a set of competitor entity IDs and derives metrics from the `coverage` block (`article_count`, `sentiment`, `momentum`).

---

## Python

### Share of Voice Table

```python
import requests

API_KEY = "YOUR_API_KEY"
COMPETITORS = [312, 4501, 7720, 9013]

def fetch_coverage(company_id):
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], profile["coverage"]

rows = [(name, cov) for name, cov in (fetch_coverage(cid) for cid in COMPETITORS) if cov]
total = sum(cov["article_count"] for _, cov in rows) or 1

print(f"{'Company':<20} {'Articles':>10} {'SOV':>7}  Share of voice")
print("-" * 60)

for name, cov in sorted(rows, key=lambda r: r[1]["article_count"], reverse=True):
    sov = cov["article_count"] / total
    bar = "#" * int(sov * 40)
    print(f"{name:<20} {cov['article_count']:>10,} {sov:>6.1%}  {bar}")
```

### Sentiment Benchmark

```python
import requests

API_KEY = "YOUR_API_KEY"
COMPETITORS = [312, 4501, 7720]

def fetch_coverage(company_id):
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], (profile.get("coverage") or {}).get("sentiment")

print(f"{'Company':<20} {'Pos%':>6} {'Neu%':>6} {'Neg%':>6} {'Net':>6}")
print("-" * 48)

for cid in COMPETITORS:
    name, s = fetch_coverage(cid)
    if not s:
        print(f"{name:<20} no coverage")
        continue
    total = s["positive"] + s["neutral"] + s["negative"] or 1
    pos = s["positive"] / total
    neg = s["negative"] / total
    neu = s["neutral"] / total
    net = pos - neg
    print(f"{name:<20} {pos:>5.0%} {neu:>5.0%} {neg:>5.0%} {net:>+5.0%}")
```

### Momentum Leaderboard (Gaining vs Losing)

```python
import requests

API_KEY = "YOUR_API_KEY"
COMPETITORS = [312, 4501, 7720, 9013]

def fetch_coverage(company_id):
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], (profile.get("coverage") or {}).get("momentum")

rows = [(name, m) for name, m in (fetch_coverage(cid) for cid in COMPETITORS) if m]
rows.sort(key=lambda r: r[1]["change_pct"] if r[1]["change_pct"] is not None else 0, reverse=True)

print(f"{'Company':<20} {'30d':>6} {'Prev':>6} {'Change':>8}  Trend")
print("-" * 56)

for name, m in rows:
    change = m["change_pct"]
    trend = "n/a" if change is None else ("gaining" if change > 0 else ("losing" if change < 0 else "flat"))
    change_str = "n/a" if change is None else f"{change:+d}%"
    print(f"{name:<20} {m['last_30_days']:>6} {m['previous_30_days']:>6} "
          f"{change_str:>8}  {trend}")
```

---

## JavaScript

### Share of Voice Table

```javascript
const API_KEY = "YOUR_API_KEY";
const COMPETITORS = [312, 4501, 7720, 9013];

async function fetchCoverage(companyId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${companyId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const profile = await response.json();
  return { name: profile.name, coverage: profile.coverage };
}

const rows = (await Promise.all(COMPETITORS.map(fetchCoverage))).filter((r) => r.coverage);
const total = rows.reduce((sum, r) => sum + r.coverage.article_count, 0) || 1;

console.log(`${"Company".padEnd(20)} ${"Articles".padStart(10)} ${"SOV".padStart(7)}  Share of voice`);
console.log("-".repeat(60));

rows
  .sort((a, b) => b.coverage.article_count - a.coverage.article_count)
  .forEach(({ name, coverage }) => {
    const sov = coverage.article_count / total;
    const bar = "#".repeat(Math.round(sov * 40));
    console.log(
      `${name.padEnd(20)} ${coverage.article_count.toLocaleString().padStart(10)} ` +
      `${(sov * 100).toFixed(1).padStart(6)}%  ${bar}`
    );
  });
```

### Sentiment Benchmark

```javascript
const API_KEY = "YOUR_API_KEY";
const COMPETITORS = [312, 4501, 7720];

async function fetchSentiment(companyId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${companyId}?${params}`);
  const profile = await response.json();
  return { name: profile.name, sentiment: profile.coverage?.sentiment };
}

console.log(`${"Company".padEnd(20)} ${"Pos%".padStart(6)} ${"Neu%".padStart(6)} ${"Neg%".padStart(6)} ${"Net".padStart(6)}`);
console.log("-".repeat(48));

for (const id of COMPETITORS) {
  const { name, sentiment: s } = await fetchSentiment(id);
  if (!s) {
    console.log(`${name.padEnd(20)} no coverage`);
    continue;
  }
  const total = s.positive + s.neutral + s.negative || 1;
  const pos = s.positive / total;
  const neg = s.negative / total;
  const neu = s.neutral / total;
  const net = pos - neg;
  console.log(
    `${name.padEnd(20)} ${(pos * 100).toFixed(0).padStart(5)}% ` +
    `${(neu * 100).toFixed(0).padStart(5)}% ${(neg * 100).toFixed(0).padStart(5)}% ` +
    `${(net >= 0 ? "+" : "") + (net * 100).toFixed(0)}%`.padStart(6)
  );
}
```

### Momentum Leaderboard (Gaining vs Losing)

```javascript
const API_KEY = "YOUR_API_KEY";
const COMPETITORS = [312, 4501, 7720, 9013];

async function fetchMomentum(companyId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${companyId}?${params}`);
  const profile = await response.json();
  return { name: profile.name, momentum: profile.coverage?.momentum };
}

const rows = (await Promise.all(COMPETITORS.map(fetchMomentum))).filter((r) => r.momentum);
rows.sort((a, b) => (b.momentum.change_pct ?? 0) - (a.momentum.change_pct ?? 0));

console.log(`${"Company".padEnd(20)} ${"30d".padStart(6)} ${"Prev".padStart(6)} ${"Change".padStart(8)}  Trend`);
console.log("-".repeat(56));

rows.forEach(({ name, momentum: m }) => {
  const trend = m.change_pct == null ? "n/a" : m.change_pct > 0 ? "gaining" : m.change_pct < 0 ? "losing" : "flat";
  const change = m.change_pct == null ? "n/a" : `${m.change_pct > 0 ? "+" : ""}${m.change_pct}%`;
  console.log(
    `${name.padEnd(20)} ${String(m.last_30_days).padStart(6)} ` +
    `${String(m.previous_30_days).padStart(6)} ${change.padStart(8)}  ${trend}`
  );
});
```

---

## PHP

### Share of Voice Table

```php
<?php

$apiKey      = "YOUR_API_KEY";
$competitors = [312, 4501, 7720, 9013];

function fetchCoverage(int $companyId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$companyId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "coverage" => $profile["coverage"] ?? null];
}

$rows  = array_values(array_filter(array_map("fetchCoverage", $competitors), fn($r) => $r["coverage"]));
$total = array_sum(array_map(fn($r) => $r["coverage"]["article_count"], $rows)) ?: 1;

usort($rows, fn($a, $b) => $b["coverage"]["article_count"] - $a["coverage"]["article_count"]);

printf("%-20s %10s %7s  Share of voice\n", "Company", "Articles", "SOV");
echo str_repeat("-", 60) . "\n";

foreach ($rows as $row) {
    $sov = $row["coverage"]["article_count"] / $total;
    $bar = str_repeat("#", (int) ($sov * 40));
    printf("%-20s %10s %6.1f%%  %s\n",
        $row["name"], number_format($row["coverage"]["article_count"]), $sov * 100, $bar);
}
```

### Sentiment Benchmark

```php
<?php

$apiKey      = "YOUR_API_KEY";
$competitors = [312, 4501, 7720];

function fetchSentiment(int $companyId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$companyId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "sentiment" => $profile["coverage"]["sentiment"] ?? null];
}

printf("%-20s %6s %6s %6s %6s\n", "Company", "Pos%", "Neu%", "Neg%", "Net");
echo str_repeat("-", 48) . "\n";

foreach ($competitors as $id) {
    $row   = fetchSentiment($id);
    $s     = $row["sentiment"];
    if (!$s) {
        printf("%-20s %s\n", $row["name"], "no coverage");
        continue;
    }
    $total = $s["positive"] + $s["neutral"] + $s["negative"] ?: 1;
    $pos   = $s["positive"] / $total;
    $neg   = $s["negative"] / $total;
    $neu   = $s["neutral"] / $total;
    $net   = $pos - $neg;
    printf("%-20s %5.0f%% %5.0f%% %5.0f%% %+5.0f%%\n",
        $row["name"], $pos * 100, $neu * 100, $neg * 100, $net * 100);
}
```

### Momentum Leaderboard (Gaining vs Losing)

```php
<?php

$apiKey      = "YOUR_API_KEY";
$competitors = [312, 4501, 7720, 9013];

function fetchMomentum(int $companyId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$companyId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "momentum" => $profile["coverage"]["momentum"] ?? null];
}

$rows = array_values(array_filter(array_map("fetchMomentum", $competitors), fn($r) => $r["momentum"]));
usort($rows, fn($a, $b) => ($b["momentum"]["change_pct"] ?? 0) - ($a["momentum"]["change_pct"] ?? 0));

printf("%-20s %6s %6s %8s  Trend\n", "Company", "30d", "Prev", "Change");
echo str_repeat("-", 56) . "\n";

foreach ($rows as $row) {
    $m      = $row["momentum"];
    $change = $m["change_pct"];
    $trend  = $change === null ? "n/a" : ($change > 0 ? "gaining" : ($change < 0 ? "losing" : "flat"));
    $changeStr = $change === null ? "n/a" : sprintf("%+d%%", $change);
    printf("%-20s %6d %6d %8s  %s\n",
        $row["name"], $m["last_30_days"], $m["previous_30_days"], $changeStr, $trend);
}
```
