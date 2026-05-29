# People Profiles — Code Examples

Detailed examples for searching public figures and retrieving their coverage profiles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Search People by Name

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/people"

def search_people(name, per_page=20):
    response = requests.get(LIST_URL, params={
        "api_key": API_KEY,
        "name": name,
        "per_page": per_page,
    })
    response.raise_for_status()
    return response.json()

data = search_people("musk")
print(f"Found {len(data['results'])} matches for 'musk':\n")

for person in data["results"]:
    profile = person.get("profile", {})
    country = profile.get("country", {}).get("name", "n/a")
    desc = profile.get("description", "")
    print(f"  [{person['id']}] {person['name']} - {country}")
    if desc:
        print(f"        {desc}")
```

### Resolve a Person by Wikidata ID

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/people"

def resolve_by_wikidata(wikidata_id):
    response = requests.get(LIST_URL, params={
        "api_key": API_KEY,
        "wikidata_id": wikidata_id,
    })
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None

person = resolve_by_wikidata("Q317521")
if person:
    print(f"Resolved {person['name']} to stable id {person['id']}")
    print(f"  Wikipedia: {person['links']['wikipedia']}")
else:
    print("No person matched that Wikidata ID")
```

### Print a Profile with Coverage Dynamics

```python
import requests

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

def get_profile(person_id):
    response = requests.get(PROFILE_URL.format(person_id), params={
        "api_key": API_KEY,
    })
    response.raise_for_status()
    return response.json()

person = get_profile(5021)
coverage = person["coverage"]
if not coverage:
    raise SystemExit(f"No coverage available for {person['name']}")

print(f"Profile: {person['name']} ({person['type']})")
print(f"  Active: {coverage['first_seen'] or 'n/a'} -> {coverage['last_seen'] or 'n/a'}")
print(f"  Total articles: {coverage['article_count']:,}\n")

sentiment = coverage["sentiment"]
total = sum(sentiment.values()) or 1
print("Sentiment split:")
for label in ("positive", "neutral", "negative"):
    count = sentiment[label]
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    print(f"  {label:>8}: {count:>7,} ({pct:5.1f}%) {bar}")

momentum = coverage["momentum"]
change = momentum["change_pct"]
arrow = "up" if (change if change is not None else 0) >= 0 else "down"
change_str = "n/a" if change is None else f"{change:+d}%"
print(f"\nMomentum: {momentum['previous_30_days']} -> {momentum['last_30_days']} "
      f"({change_str} {arrow})")

print("\nTop topics:")
for topic in coverage["top_topics"]:
    print(f"  {topic['name']:<20} {topic['count']:>6,}")
```

### Traverse Related Entities

```python
import requests

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

def get_profile(person_id):
    response = requests.get(PROFILE_URL.format(person_id), params={
        "api_key": API_KEY,
    })
    response.raise_for_status()
    return response.json()

root = get_profile(5021)
related = root["coverage"]["related_entities"]

print(f"Entities most co-mentioned with {root['name']}:\n")
total = sum(e["count"] for e in related) or 1

for entity in sorted(related, key=lambda e: e["count"], reverse=True):
    share = entity["count"] / total * 100
    print(f"  {entity['name']:<24} {entity['count']:>6,} co-mentions ({share:4.1f}%)")
```

### Fetch a Profile Without Coverage

```python
import requests

API_KEY = "YOUR_API_KEY"
PROFILE_URL = "https://api.apitube.io/v1/people/{}"

# coverage=false skips the analytics block for a faster, lighter response
response = requests.get(PROFILE_URL.format(5021), params={
    "api_key": API_KEY,
    "coverage": "false",
})
response.raise_for_status()
person = response.json()

print(f"{person['name']} - {person['profile'].get('description', '')}")
print(f"Coverage block present: {'coverage' in person}")
print(f"Recent articles: {len(person.get('recent_articles', []))}")

for article in person.get("recent_articles", []):
    print(f"  - {article['title']}")
```

---

## JavaScript

### Search People by Name

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/people";

async function searchPeople(name, perPage = 20) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    name,
    per_page: String(perPage),
  });

  const response = await fetch(`${LIST_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await searchPeople("musk");
console.log(`Found ${data.results.length} matches for 'musk':\n`);

data.results.forEach((person) => {
  const profile = person.profile || {};
  const country = profile.country?.name || "n/a";
  console.log(`  [${person.id}] ${person.name} - ${country}`);
  if (profile.description) console.log(`        ${profile.description}`);
});
```

### Resolve a Person by Wikidata ID

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/people";

async function resolveByWikidata(wikidataId) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    wikidata_id: wikidataId,
  });

  const response = await fetch(`${LIST_URL}?${params}`);
  const data = await response.json();
  return data.results?.[0] || null;
}

const person = await resolveByWikidata("Q317521");
if (person) {
  console.log(`Resolved ${person.name} to stable id ${person.id}`);
  console.log(`  Wikipedia: ${person.links.wikipedia}`);
} else {
  console.log("No person matched that Wikidata ID");
}
```

### Print a Profile with Coverage Dynamics

```javascript
const API_KEY = "YOUR_API_KEY";

async function getProfile(personId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/people/${personId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const person = await getProfile(5021);
const coverage = person.coverage;
if (!coverage) throw new Error(`No coverage available for ${person.name}`);

console.log(`Profile: ${person.name} (${person.type})`);
console.log(`  Active: ${coverage.first_seen ?? "n/a"} -> ${coverage.last_seen ?? "n/a"}`);
console.log(`  Total articles: ${coverage.article_count.toLocaleString()}\n`);

const sentiment = coverage.sentiment;
const total = Object.values(sentiment).reduce((a, b) => a + b, 0) || 1;
console.log("Sentiment split:");
for (const label of ["positive", "neutral", "negative"]) {
  const count = sentiment[label];
  const pct = (count / total) * 100;
  const bar = "#".repeat(Math.round(pct / 2));
  console.log(`  ${label.padStart(8)}: ${String(count).padStart(7)} (${pct.toFixed(1).padStart(5)}%) ${bar}`);
}

const m = coverage.momentum;
const arrow = (m.change_pct ?? 0) >= 0 ? "up" : "down";
const changeStr = m.change_pct == null ? "n/a" : `${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%`;
console.log(`\nMomentum: ${m.previous_30_days} -> ${m.last_30_days} (${changeStr} ${arrow})`);

console.log("\nTop topics:");
coverage.top_topics.forEach((topic) => {
  console.log(`  ${topic.name.padEnd(20)} ${String(topic.count).padStart(6)}`);
});
```

### Traverse Related Entities

```javascript
const API_KEY = "YOUR_API_KEY";

async function getProfile(personId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/people/${personId}?${params}`);
  return response.json();
}

const root = await getProfile(5021);
const related = root.coverage.related_entities;

console.log(`Entities most co-mentioned with ${root.name}:\n`);
const total = related.reduce((sum, e) => sum + e.count, 0) || 1;

related
  .sort((a, b) => b.count - a.count)
  .forEach((entity) => {
    const share = (entity.count / total) * 100;
    console.log(`  ${entity.name.padEnd(24)} ${String(entity.count).padStart(6)} co-mentions (${share.toFixed(1)}%)`);
  });
```

### Fetch a Profile Without Coverage

```javascript
const API_KEY = "YOUR_API_KEY";

const params = new URLSearchParams({ api_key: API_KEY, coverage: "false" });
const response = await fetch(`https://api.apitube.io/v1/people/5021?${params}`);
if (!response.ok) throw new Error(`HTTP ${response.status}`);
const person = await response.json();

console.log(`${person.name} - ${person.profile?.description || ""}`);
console.log(`Coverage block present: ${"coverage" in person}`);
console.log(`Recent articles: ${(person.recent_articles || []).length}`);

(person.recent_articles || []).forEach((article) => {
  console.log(`  - ${article.title}`);
});
```

---

## PHP

### Search People by Name

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/people";

function searchPeople(string $name, int $perPage = 20): array
{
    global $apiKey, $listUrl;

    $query = http_build_query([
        "api_key"  => $apiKey,
        "name"     => $name,
        "per_page" => $perPage,
    ]);

    return json_decode(file_get_contents("{$listUrl}?{$query}"), true);
}

$data = searchPeople("musk");
echo "Found " . count($data["results"]) . " matches for 'musk':\n\n";

foreach ($data["results"] as $person) {
    $profile = $person["profile"] ?? [];
    $country = $profile["country"]["name"] ?? "n/a";
    echo "  [{$person['id']}] {$person['name']} - {$country}\n";
    if (!empty($profile["description"])) {
        echo "        {$profile['description']}\n";
    }
}
```

### Resolve a Person by Wikidata ID

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/people";

function resolveByWikidata(string $wikidataId): ?array
{
    global $apiKey, $listUrl;

    $query = http_build_query([
        "api_key"     => $apiKey,
        "wikidata_id" => $wikidataId,
    ]);

    $data = json_decode(file_get_contents("{$listUrl}?{$query}"), true);
    return $data["results"][0] ?? null;
}

$person = resolveByWikidata("Q317521");
if ($person) {
    echo "Resolved {$person['name']} to stable id {$person['id']}\n";
    echo "  Wikipedia: {$person['links']['wikipedia']}\n";
} else {
    echo "No person matched that Wikidata ID\n";
}
```

### Print a Profile with Coverage Dynamics

```php
<?php

$apiKey = "YOUR_API_KEY";

function getProfile(int $personId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/people/{$personId}?{$query}"
    ), true);
}

$person   = getProfile(5021);
$coverage = $person["coverage"];
if (!$coverage) {
    exit("No coverage available for {$person['name']}\n");
}

echo "Profile: {$person['name']} ({$person['type']})\n";
echo "  Active: " . ($coverage["first_seen"] ?? "n/a") . " -> " . ($coverage["last_seen"] ?? "n/a") . "\n";
printf("  Total articles: %s\n\n", number_format($coverage["article_count"]));

$sentiment = $coverage["sentiment"];
$total     = array_sum($sentiment) ?: 1;
echo "Sentiment split:\n";
foreach (["positive", "neutral", "negative"] as $label) {
    $count = $sentiment[$label];
    $pct   = $count / $total * 100;
    $bar   = str_repeat("#", (int) ($pct / 2));
    printf("  %8s: %7s (%5.1f%%) %s\n", $label, number_format($count), $pct, $bar);
}

$m         = $coverage["momentum"];
$change    = $m["change_pct"] ?? 0;
$arrow     = $change >= 0 ? "up" : "down";
$changeStr = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
printf("\nMomentum: %d -> %d (%s %s)\n",
    $m["previous_30_days"], $m["last_30_days"], $changeStr, $arrow);

echo "\nTop topics:\n";
foreach ($coverage["top_topics"] as $topic) {
    printf("  %-20s %6s\n", $topic["name"], number_format($topic["count"]));
}
```

### Traverse Related Entities

```php
<?php

$apiKey = "YOUR_API_KEY";

function getProfile(int $personId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/people/{$personId}?{$query}"
    ), true);
}

$root    = getProfile(5021);
$related = $root["coverage"]["related_entities"];

echo "Entities most co-mentioned with {$root['name']}:\n\n";
$total = array_sum(array_column($related, "count")) ?: 1;

usort($related, fn($a, $b) => $b["count"] <=> $a["count"]);

foreach ($related as $entity) {
    $share = $entity["count"] / $total * 100;
    printf("  %-24s %6s co-mentions (%4.1f%%)\n",
        $entity["name"], number_format($entity["count"]), $share);
}
```

### Fetch a Profile Without Coverage

```php
<?php

$apiKey = "YOUR_API_KEY";

$query = http_build_query([
    "api_key"  => $apiKey,
    "coverage" => "false",
]);

$person = json_decode(file_get_contents(
    "https://api.apitube.io/v1/people/5021?{$query}"
), true);

echo "{$person['name']} - " . ($person["profile"]["description"] ?? "") . "\n";
echo "Coverage block present: " . (isset($person["coverage"]) ? "yes" : "no") . "\n";
echo "Recent articles: " . count($person["recent_articles"] ?? []) . "\n";

foreach ($person["recent_articles"] ?? [] as $article) {
    echo "  - {$article['title']}\n";
}
```
