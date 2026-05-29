# Company Profiles — Code Examples

Detailed examples for searching the company directory and reading per-company coverage profiles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Search a Company by Name

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/companies"

def search_companies(name, per_page=10):
    response = requests.get(LIST_URL, params={
        "api_key": API_KEY,
        "name": name,
        "per_page": per_page,
    })
    response.raise_for_status()
    return response.json()

data = search_companies("apple")
print(f"Found {len(data['results'])} matches for 'apple':\n")

for company in data["results"]:
    profile = company.get("profile", {})
    country = profile.get("country", {}).get("name", "—")
    print(f"  [{company['id']:>6}] {company['name']} ({company['type']}) — {country}")
    print(f"          {company['links']['articles']}")
```

### Paginate the Full Directory

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/companies"

def iterate_companies(name=None, per_page=50, max_pages=5):
    page = 1
    while page <= max_pages:
        response = requests.get(LIST_URL, params={
            "api_key": API_KEY,
            "name": name,
            "per_page": per_page,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for company in data["results"]:
            yield company

        if not data.get("has_next_pages"):
            break
        page += 1

count = 0
for company in iterate_companies(name="bank"):
    count += 1
    print(f"  {company['id']:>6}  {company['name']}")

print(f"\nTotal companies listed: {count}")
```

### Profile with Media Metrics

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_profile(company_id):
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

profile = get_profile(312)
cov = profile.get("coverage")
if not cov:
    print(f"{profile['name']} ({profile['type']}) — no coverage available")
    raise SystemExit

total = cov["article_count"] or 1
pos = cov["sentiment"]["positive"]
neg = cov["sentiment"]["negative"]
neu = cov["sentiment"]["neutral"]
change = cov["momentum"]["change_pct"]
change_str = "n/a" if change is None else f"{change:+d}%"

print(f"{profile['name']} ({profile['type']})")
print(f"  Description : {profile['profile'].get('description', '—')}")
print(f"  Articles    : {cov['article_count']:,}")
print(f"  First seen  : {cov['first_seen'] or 'n/a'}")
print(f"  Last seen   : {cov['last_seen'] or 'n/a'}")
print(f"  Sentiment   : +{pos / total:.0%} positive / {neu / total:.0%} neutral / {neg / total:.0%} negative")
print(f"  Momentum    : {cov['momentum']['last_30_days']} (last 30d) vs "
      f"{cov['momentum']['previous_30_days']} (prev 30d), {change_str}")

print("\n  Top topics:")
for topic in cov["top_topics"]:
    print(f"    {topic['name']:<24} {topic['count']:>6}")

print("\n  Top sources:")
for source in cov["top_sources"]:
    print(f"    {source['name']:<24} {source['domain']:<24} {source['count']:>6}")
```

### Related Entities (Partners, Competitors, People)

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_profile(company_id):
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

profile = get_profile(312)
related = (profile.get("coverage") or {}).get("related_entities", [])

print(f"Entities most frequently co-mentioned with {profile['name']}:\n")

max_count = max((e["count"] for e in related), default=1)
for entity in related:
    bar = "#" * int(entity["count"] / max_count * 40)
    print(f"  {entity['name']:<28} {entity['count']:>6}  {bar}")

print("\n  Recent coverage:")
for article in profile.get("recent_articles", [])[:5]:
    polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "—")
    print(f"    [{polarity:>8}] {article['title']}")
```

---

## JavaScript

### Search a Company by Name

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/companies";

async function searchCompanies(name, perPage = 10) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    name,
    per_page: String(perPage),
  });

  const response = await fetch(`${LIST_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await searchCompanies("apple");
console.log(`Found ${data.results.length} matches for 'apple':\n`);

data.results.forEach((company) => {
  const country = company.profile?.country?.name || "—";
  console.log(`  [${String(company.id).padStart(6)}] ${company.name} (${company.type}) — ${country}`);
  console.log(`          ${company.links.articles}`);
});
```

### Paginate the Full Directory

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/companies";

async function* iterateCompanies(name, perPage = 50, maxPages = 5) {
  let page = 1;
  while (page <= maxPages) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      per_page: String(perPage),
      page: String(page),
    });
    if (name) params.set("name", name);

    const response = await fetch(`${LIST_URL}?${params}`);
    const data = await response.json();

    for (const company of data.results) yield company;

    if (!data.has_next_pages) break;
    page += 1;
  }
}

let count = 0;
for await (const company of iterateCompanies("bank")) {
  count += 1;
  console.log(`  ${String(company.id).padStart(6)}  ${company.name}`);
}

console.log(`\nTotal companies listed: ${count}`);
```

### Profile with Media Metrics

```javascript
const API_KEY = "YOUR_API_KEY";

async function getProfile(companyId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${companyId}?${params}`);
  return response.json();
}

const profile = await getProfile(312);
const cov = profile.coverage;
if (!cov) {
  console.log(`${profile.name} (${profile.type}) — no coverage available`);
} else {
const total = cov.article_count || 1;
const { positive, neutral, negative } = cov.sentiment;
const changePct = cov.momentum.change_pct;
const changeStr = changePct == null ? "n/a" : `${changePct > 0 ? "+" : ""}${changePct}%`;

console.log(`${profile.name} (${profile.type})`);
console.log(`  Description : ${profile.profile?.description || "—"}`);
console.log(`  Articles    : ${cov.article_count.toLocaleString()}`);
console.log(`  First seen  : ${cov.first_seen ?? "n/a"}`);
console.log(`  Last seen   : ${cov.last_seen ?? "n/a"}`);
console.log(
  `  Sentiment   : +${Math.round((positive / total) * 100)}% positive / ` +
  `${Math.round((neutral / total) * 100)}% neutral / ` +
  `${Math.round((negative / total) * 100)}% negative`
);
console.log(
  `  Momentum    : ${cov.momentum.last_30_days} (last 30d) vs ` +
  `${cov.momentum.previous_30_days} (prev 30d), ${changeStr}`
);

console.log("\n  Top topics:");
cov.top_topics.forEach((t) => {
  console.log(`    ${t.name.padEnd(24)} ${String(t.count).padStart(6)}`);
});

console.log("\n  Top sources:");
cov.top_sources.forEach((s) => {
  console.log(`    ${s.name.padEnd(24)} ${s.domain.padEnd(24)} ${String(s.count).padStart(6)}`);
});
}
```

### Related Entities (Partners, Competitors, People)

```javascript
const API_KEY = "YOUR_API_KEY";

async function getProfile(companyId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${companyId}?${params}`);
  return response.json();
}

const profile = await getProfile(312);
const related = profile.coverage?.related_entities ?? [];

console.log(`Entities most frequently co-mentioned with ${profile.name}:\n`);

const maxCount = Math.max(...related.map((e) => e.count), 1);
related.forEach((entity) => {
  const bar = "#".repeat(Math.round((entity.count / maxCount) * 40));
  console.log(`  ${entity.name.padEnd(28)} ${String(entity.count).padStart(6)}  ${bar}`);
});

console.log("\n  Recent coverage:");
(profile.recent_articles || []).slice(0, 5).forEach((article) => {
  const polarity = article.sentiment?.overall?.polarity || "—";
  console.log(`    [${polarity.padStart(8)}] ${article.title}`);
});
```

---

## PHP

### Search a Company by Name

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/companies";

$query = http_build_query([
    "api_key"  => $apiKey,
    "name"     => "apple",
    "per_page" => 10,
]);

$data = json_decode(file_get_contents("{$listUrl}?{$query}"), true);

echo "Found " . count($data["results"]) . " matches for 'apple':\n\n";

foreach ($data["results"] as $company) {
    $country = $company["profile"]["country"]["name"] ?? "—";
    printf("  [%6d] %s (%s) — %s\n", $company["id"], $company["name"], $company["type"], $country);
    echo "          {$company['links']['articles']}\n";
}
```

### Paginate the Full Directory

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/companies";

function iterateCompanies(string $name, int $perPage = 50, int $maxPages = 5): array
{
    global $apiKey, $listUrl;

    $all  = [];
    $page = 1;

    while ($page <= $maxPages) {
        $query = http_build_query([
            "api_key"  => $apiKey,
            "name"     => $name,
            "per_page" => $perPage,
            "page"     => $page,
        ]);

        $data = json_decode(file_get_contents("{$listUrl}?{$query}"), true);
        $all  = array_merge($all, $data["results"]);

        if (empty($data["has_next_pages"])) {
            break;
        }
        $page++;
    }

    return $all;
}

$companies = iterateCompanies("bank");

foreach ($companies as $company) {
    printf("  %6d  %s\n", $company["id"], $company["name"]);
}

echo "\nTotal companies listed: " . count($companies) . "\n";
```

### Profile with Media Metrics

```php
<?php

$apiKey = "YOUR_API_KEY";

function getProfile(int $companyId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$companyId}?{$query}"
    ), true);
}

$profile = getProfile(312);
$cov     = $profile["coverage"] ?? null;

if (!$cov) {
    echo "{$profile['name']} ({$profile['type']}) — no coverage available\n";
    exit;
}

$total   = $cov["article_count"] ?: 1;
$pos     = $cov["sentiment"]["positive"];
$neg     = $cov["sentiment"]["negative"];
$neu     = $cov["sentiment"]["neutral"];
$change  = $cov["momentum"]["change_pct"];
$changeStr = $change === null ? "n/a" : sprintf("%+d%%", $change);

echo "{$profile['name']} ({$profile['type']})\n";
echo "  Description : " . ($profile["profile"]["description"] ?? "—") . "\n";
printf("  Articles    : %s\n", number_format($cov["article_count"]));
echo "  First seen  : " . ($cov["first_seen"] ?? "n/a") . "\n";
echo "  Last seen   : " . ($cov["last_seen"] ?? "n/a") . "\n";
printf("  Sentiment   : +%d%% positive / %d%% neutral / %d%% negative\n",
    round($pos / $total * 100), round($neu / $total * 100), round($neg / $total * 100));
printf("  Momentum    : %d (last 30d) vs %d (prev 30d), %s\n",
    $cov["momentum"]["last_30_days"], $cov["momentum"]["previous_30_days"], $changeStr);

echo "\n  Top topics:\n";
foreach ($cov["top_topics"] as $topic) {
    printf("    %-24s %6d\n", $topic["name"], $topic["count"]);
}

echo "\n  Top sources:\n";
foreach ($cov["top_sources"] as $source) {
    printf("    %-24s %-24s %6d\n", $source["name"], $source["domain"], $source["count"]);
}
```

### Related Entities (Partners, Competitors, People)

```php
<?php

$apiKey = "YOUR_API_KEY";

function getProfile(int $companyId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$companyId}?{$query}"
    ), true);
}

$profile = getProfile(312);
$related = $profile["coverage"]["related_entities"] ?? [];

echo "Entities most frequently co-mentioned with {$profile['name']}:\n\n";

$maxCount = max(array_map(fn($e) => $e["count"], $related) ?: [1]);
foreach ($related as $entity) {
    $bar = str_repeat("#", (int) ($entity["count"] / $maxCount * 40));
    printf("  %-28s %6d  %s\n", $entity["name"], $entity["count"], $bar);
}

echo "\n  Recent coverage:\n";
foreach (array_slice($profile["recent_articles"] ?? [], 0, 5) as $article) {
    $polarity = $article["sentiment"]["overall"]["polarity"] ?? "—";
    printf("    [%8s] %s\n", $polarity, $article["title"]);
}
```
