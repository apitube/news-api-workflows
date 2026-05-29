# Query ID Resolver — Code Examples

Detailed examples for resolving human-readable input into filter IDs and searching articles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Resolve-Then-Search (Topic)

```python
import requests

API_KEY = "YOUR_API_KEY"

def resolve_topic(prefix):
    response = requests.get("https://api.apitube.io/v1/suggest/topics", params={
        "api_key": API_KEY,
        "prefix": prefix,
    })
    response.raise_for_status()
    return response.json()  # flat array of { id, name, links }

candidates = resolve_topic("clim")
if not candidates:
    raise SystemExit("No topic matched 'clim'")

topic = candidates[0]
print(f"Resolved 'clim' -> {topic['id']} ({topic['name']})\n")

articles = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": API_KEY,
    "topic.id": topic["id"],
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 10,
})
articles.raise_for_status()

print(f"Latest articles for topic '{topic['name']}':")
for article in articles.json()["results"]:
    print(f"  {article['title']}")
```

### Multi-Filter Search from Several Resolved IDs

```python
import requests

API_KEY = "YOUR_API_KEY"

def resolve_first(kind, prefix):
    response = requests.get(f"https://api.apitube.io/v1/suggest/{kind}", params={
        "api_key": API_KEY,
        "prefix": prefix,
    })
    response.raise_for_status()
    items = response.json()
    return items[0] if items else None

topic = resolve_first("topics", "tech")
entity = resolve_first("entities", "appl")
industry = resolve_first("industries", "semi")

params = {"api_key": API_KEY, "language.code": "en", "per_page": 10}
if topic:
    params["topic.id"] = topic["id"]
if entity:
    params["entity.id"] = entity["id"]
if industry:
    params["industry.id"] = industry["id"]

print("Resolved filters:")
for label, item in [("topic", topic), ("entity", entity), ("industry", industry)]:
    if item:
        print(f"  {label:<9} {item['id']} ({item['name']})")

articles = requests.get("https://api.apitube.io/v1/news/everything", params=params)
articles.raise_for_status()

print("\nMatching articles:")
for article in articles.json()["results"]:
    print(f"  {article['title']}")
```

### Interactive Disambiguation

```python
import requests

API_KEY = "YOUR_API_KEY"

def suggest_entities(prefix):
    response = requests.get("https://api.apitube.io/v1/suggest/entities", params={
        "api_key": API_KEY,
        "prefix": prefix,
    })
    response.raise_for_status()
    return response.json()

prefix = "appl"
candidates = suggest_entities(prefix)

print(f"Did you mean (prefix '{prefix}')?\n")
for i, entity in enumerate(candidates, 1):
    print(f"  [{i}] {entity['name']} ({entity['type']})  -> entity.id={entity['id']}")

# Simulate the user picking option 1
choice = candidates[0]
print(f"\nSelected: {choice['name']} (id={choice['id']})\n")

articles = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": API_KEY,
    "entity.id": choice["id"],
    "language.code": "en",
    "per_page": 10,
})
articles.raise_for_status()

print(f"Articles mentioning {choice['name']}:")
for article in articles.json()["results"]:
    print(f"  {article['title']}")
```

### Category Resolver with Fallback

```python
import requests

API_KEY = "YOUR_API_KEY"

def resolve_category(prefix):
    response = requests.get("https://api.apitube.io/v1/suggest/categories", params={
        "api_key": API_KEY,
        "prefix": prefix,
    })
    response.raise_for_status()
    return response.json()

for prefix in ["econ", "sportz", "pol"]:
    candidates = resolve_category(prefix)
    if not candidates:
        print(f"'{prefix}': no match")
        continue

    top = candidates[0]
    count_resp = requests.get("https://api.apitube.io/v1/news/everything", params={
        "api_key": API_KEY,
        "category.id": top["id"],
        "language.code": "en",
        "per_page": 1,
    })
    count_resp.raise_for_status()
    sample = count_resp.json().get("results", [])
    state = "articles available" if sample else "no recent articles"
    print(f"'{prefix}' -> {top['name']} (id={top['id']}, taxonomy={top.get('taxonomy')}): {state}")
```

---

## JavaScript

### Resolve-Then-Search (Topic)

```javascript
const API_KEY = "YOUR_API_KEY";

async function resolveTopic(prefix) {
  const params = new URLSearchParams({ api_key: API_KEY, prefix });
  const response = await fetch(`https://api.apitube.io/v1/suggest/topics?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json(); // flat array of { id, name, links }
}

const candidates = await resolveTopic("clim");
if (candidates.length === 0) throw new Error("No topic matched 'clim'");

const topic = candidates[0];
console.log(`Resolved 'clim' -> ${topic.id} (${topic.name})\n`);

const searchParams = new URLSearchParams({
  api_key: API_KEY,
  "topic.id": topic.id,
  "language.code": "en",
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "10",
});
const articlesResp = await fetch(`https://api.apitube.io/v1/news/everything?${searchParams}`);
const data = await articlesResp.json();

console.log(`Latest articles for topic '${topic.name}':`);
data.results.forEach((article) => console.log(`  ${article.title}`));
```

### Multi-Filter Search from Several Resolved IDs

```javascript
const API_KEY = "YOUR_API_KEY";

async function resolveFirst(kind, prefix) {
  const params = new URLSearchParams({ api_key: API_KEY, prefix });
  const response = await fetch(`https://api.apitube.io/v1/suggest/${kind}?${params}`);
  const items = await response.json();
  return items[0] ?? null;
}

const topic = await resolveFirst("topics", "tech");
const entity = await resolveFirst("entities", "appl");
const industry = await resolveFirst("industries", "semi");

const params = new URLSearchParams({ api_key: API_KEY, "language.code": "en", per_page: "10" });
if (topic) params.set("topic.id", topic.id);
if (entity) params.set("entity.id", String(entity.id));
if (industry) params.set("industry.id", String(industry.id));

console.log("Resolved filters:");
[["topic", topic], ["entity", entity], ["industry", industry]].forEach(([label, item]) => {
  if (item) console.log(`  ${label.padEnd(9)} ${item.id} (${item.name})`);
});

const articlesResp = await fetch(`https://api.apitube.io/v1/news/everything?${params}`);
const data = await articlesResp.json();

console.log("\nMatching articles:");
data.results.forEach((article) => console.log(`  ${article.title}`));
```

### Interactive Disambiguation

```javascript
const API_KEY = "YOUR_API_KEY";

async function suggestEntities(prefix) {
  const params = new URLSearchParams({ api_key: API_KEY, prefix });
  const response = await fetch(`https://api.apitube.io/v1/suggest/entities?${params}`);
  return response.json();
}

const prefix = "appl";
const candidates = await suggestEntities(prefix);

console.log(`Did you mean (prefix '${prefix}')?\n`);
candidates.forEach((entity, i) => {
  console.log(`  [${i + 1}] ${entity.name} (${entity.type})  -> entity.id=${entity.id}`);
});

// Simulate the user picking option 1
const choice = candidates[0];
console.log(`\nSelected: ${choice.name} (id=${choice.id})\n`);

const searchParams = new URLSearchParams({
  api_key: API_KEY,
  "entity.id": String(choice.id),
  "language.code": "en",
  per_page: "10",
});
const articlesResp = await fetch(`https://api.apitube.io/v1/news/everything?${searchParams}`);
const data = await articlesResp.json();

console.log(`Articles mentioning ${choice.name}:`);
data.results.forEach((article) => console.log(`  ${article.title}`));
```

### Category Resolver with Fallback

```javascript
const API_KEY = "YOUR_API_KEY";

async function resolveCategory(prefix) {
  const params = new URLSearchParams({ api_key: API_KEY, prefix });
  const response = await fetch(`https://api.apitube.io/v1/suggest/categories?${params}`);
  return response.json();
}

for (const prefix of ["econ", "sportz", "pol"]) {
  const candidates = await resolveCategory(prefix);
  if (candidates.length === 0) {
    console.log(`'${prefix}': no match`);
    continue;
  }

  const top = candidates[0];
  const countParams = new URLSearchParams({
    api_key: API_KEY,
    "category.id": String(top.id),
    "language.code": "en",
    per_page: "1",
  });
  const countResp = await fetch(`https://api.apitube.io/v1/news/everything?${countParams}`);
  const sample = (await countResp.json()).results ?? [];
  const state = sample.length ? "articles available" : "no recent articles";
  console.log(`'${prefix}' -> ${top.name} (id=${top.id}, taxonomy=${top.taxonomy}): ${state}`);
}
```

---

## PHP

### Resolve-Then-Search (Topic)

```php
<?php

$apiKey = "YOUR_API_KEY";

function resolveTopic(string $prefix): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey, "prefix" => $prefix]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/suggest/topics?{$query}"
    ), true); // flat array of { id, name, links }
}

$candidates = resolveTopic("clim");
if (empty($candidates)) {
    exit("No topic matched 'clim'\n");
}

$topic = $candidates[0];
echo "Resolved 'clim' -> {$topic['id']} ({$topic['name']})\n\n";

$searchQuery = http_build_query([
    "api_key"       => $apiKey,
    "topic.id"      => $topic["id"],
    "language.code" => "en",
    "sort.by"       => "published_at",
    "sort.order"    => "desc",
    "per_page"      => 10,
]);
$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$searchQuery}"
), true);

echo "Latest articles for topic '{$topic['name']}':\n";
foreach ($data["results"] as $article) {
    echo "  {$article['title']}\n";
}
```

### Multi-Filter Search from Several Resolved IDs

```php
<?php

$apiKey = "YOUR_API_KEY";

function resolveFirst(string $kind, string $prefix): ?array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey, "prefix" => $prefix]);
    $items = json_decode(file_get_contents(
        "https://api.apitube.io/v1/suggest/{$kind}?{$query}"
    ), true);
    return $items[0] ?? null;
}

$topic    = resolveFirst("topics", "tech");
$entity   = resolveFirst("entities", "appl");
$industry = resolveFirst("industries", "semi");

$params = ["api_key" => $apiKey, "language.code" => "en", "per_page" => 10];
if ($topic) {
    $params["topic.id"] = $topic["id"];
}
if ($entity) {
    $params["entity.id"] = $entity["id"];
}
if ($industry) {
    $params["industry.id"] = $industry["id"];
}

echo "Resolved filters:\n";
foreach (["topic" => $topic, "entity" => $entity, "industry" => $industry] as $label => $item) {
    if ($item) {
        printf("  %-9s %s (%s)\n", $label, $item["id"], $item["name"]);
    }
}

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?" . http_build_query($params)
), true);

echo "\nMatching articles:\n";
foreach ($data["results"] as $article) {
    echo "  {$article['title']}\n";
}
```

### Interactive Disambiguation

```php
<?php

$apiKey = "YOUR_API_KEY";

function suggestEntities(string $prefix): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey, "prefix" => $prefix]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/suggest/entities?{$query}"
    ), true);
}

$prefix     = "appl";
$candidates = suggestEntities($prefix);

echo "Did you mean (prefix '{$prefix}')?\n\n";
foreach ($candidates as $i => $entity) {
    printf("  [%d] %s (%s)  -> entity.id=%s\n", $i + 1, $entity["name"], $entity["type"], $entity["id"]);
}

// Simulate the user picking option 1
$choice = $candidates[0];
echo "\nSelected: {$choice['name']} (id={$choice['id']})\n\n";

$searchQuery = http_build_query([
    "api_key"       => $apiKey,
    "entity.id"     => $choice["id"],
    "language.code" => "en",
    "per_page"      => 10,
]);
$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$searchQuery}"
), true);

echo "Articles mentioning {$choice['name']}:\n";
foreach ($data["results"] as $article) {
    echo "  {$article['title']}\n";
}
```

### Category Resolver with Fallback

```php
<?php

$apiKey = "YOUR_API_KEY";

function resolveCategory(string $prefix): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey, "prefix" => $prefix]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/suggest/categories?{$query}"
    ), true);
}

foreach (["econ", "sportz", "pol"] as $prefix) {
    $candidates = resolveCategory($prefix);
    if (empty($candidates)) {
        echo "'{$prefix}': no match\n";
        continue;
    }

    $top   = $candidates[0];
    $query = http_build_query([
        "api_key"       => $apiKey,
        "category.id"   => $top["id"],
        "language.code" => "en",
        "per_page"      => 1,
    ]);
    $sample = json_decode(file_get_contents(
        "https://api.apitube.io/v1/news/everything?{$query}"
    ), true)["results"] ?? [];

    $state = $sample ? "articles available" : "no recent articles";
    printf("'%s' -> %s (id=%s, taxonomy=%s): %s\n",
        $prefix, $top["name"], $top["id"], $top["taxonomy"] ?? "", $state);
}
```
