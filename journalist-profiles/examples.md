# Journalist Profiles — Code Examples

Detailed examples for searching journalists and rendering author profiles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Search Journalists by Name

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

def search_journalists(name, per_page=20):
    response = requests.get(LIST_URL, params={
        "api_key": API_KEY,
        "name": name,
        "per_page": per_page,
    })
    response.raise_for_status()
    return response.json()["results"]

results = search_journalists("Jane")
print(f"Found {len(results)} journalists matching 'Jane':\n")

for journalist in results:
    outlets = ", ".join(o["name"] for o in journalist["outlets"])
    print(f"  #{journalist['id']:>7}  {journalist['name']:<24} {outlets}")
```

### Paginate Through the Full Directory

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

def iter_journalists(per_page=50, max_pages=10):
    page = 1
    while page <= max_pages:
        response = requests.get(LIST_URL, params={
            "api_key": API_KEY,
            "per_page": per_page,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for journalist in data["results"]:
            yield journalist

        if not data.get("has_next_pages"):
            break
        page += 1

count = 0
for journalist in iter_journalists():
    count += 1

print(f"Walked {count} journalist records across the directory")
```

### Render a Journalist Profile

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_profile(journalist_id):
    response = requests.get(
        f"https://api.apitube.io/v1/journalists/{journalist_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

profile = get_profile(88123)
coverage = profile["coverage"] or {}
change = coverage["momentum"]["change_pct"]
change_str = "n/a" if change is None else f"{change:+d}%"

print(f"{profile['name']}  (#{profile['id']})")
print(f"Writes for: {', '.join(o['name'] for o in profile['outlets'])}")
print(f"Articles:   {coverage['article_count']:,}  "
      f"({coverage['first_seen'] or 'n/a'} -> {coverage['last_seen'] or 'n/a'})")
print(f"Momentum:   {change_str} "
      f"(last 30d: {coverage['momentum']['last_30_days']})\n")

print("Top topics:")
for topic in coverage["top_topics"]:
    print(f"  {topic['name']:<20} {topic['count']:>5}")

print("\nTop entities covered:")
for entity in coverage["top_entities"]:
    print(f"  {entity['name']:<20} {entity['count']:>5}")
```

### Profile Without Coverage, Then Jump to Articles

```python
import requests

API_KEY = "YOUR_API_KEY"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"

# Lightweight fetch: skip the coverage block
response = requests.get(
    "https://api.apitube.io/v1/journalists/88123",
    params={"api_key": API_KEY, "coverage": "false"},
)
response.raise_for_status()
profile = response.json()

print(f"{profile['name']} writes for {profile['outlet_count']} outlets\n")

# Drill through to the author's latest articles
articles = requests.get(EVERYTHING_URL, params={
    "api_key": API_KEY,
    "author.id": profile["id"],
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 10,
})
articles.raise_for_status()

print(f"Latest articles by {profile['name']}:")
for article in articles.json()["results"]:
    print(f"  {article['published_at'][:10]}  {article['title']}")
```

---

## JavaScript

### Search Journalists by Name

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

async function searchJournalists(name, perPage = 20) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    name,
    per_page: String(perPage),
  });

  const response = await fetch(`${LIST_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()).results;
}

const results = await searchJournalists("Jane");
console.log(`Found ${results.length} journalists matching 'Jane':\n`);

results.forEach((journalist) => {
  const outlets = journalist.outlets.map((o) => o.name).join(", ");
  console.log(`  #${String(journalist.id).padStart(7)}  ${journalist.name.padEnd(24)} ${outlets}`);
});
```

### Paginate Through the Full Directory

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

async function* iterJournalists(perPage = 50, maxPages = 10) {
  let page = 1;
  while (page <= maxPages) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      per_page: String(perPage),
      page: String(page),
    });

    const response = await fetch(`${LIST_URL}?${params}`);
    const data = await response.json();

    for (const journalist of data.results) {
      yield journalist;
    }

    if (!data.has_next_pages) break;
    page += 1;
  }
}

let count = 0;
for await (const _journalist of iterJournalists()) {
  count += 1;
}

console.log(`Walked ${count} journalist records across the directory`);
```

### Render a Journalist Profile

```javascript
const API_KEY = "YOUR_API_KEY";

async function getProfile(journalistId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(
    `https://api.apitube.io/v1/journalists/${journalistId}?${params}`
  );
  return response.json();
}

const profile = await getProfile(88123);
const coverage = profile.coverage ?? {};
const changeStr = coverage.momentum.change_pct == null
  ? "n/a"
  : `${coverage.momentum.change_pct >= 0 ? "+" : ""}${coverage.momentum.change_pct}%`;

console.log(`${profile.name}  (#${profile.id})`);
console.log(`Writes for: ${profile.outlets.map((o) => o.name).join(", ")}`);
console.log(`Articles:   ${coverage.article_count.toLocaleString()}  (${coverage.first_seen ?? "n/a"} -> ${coverage.last_seen ?? "n/a"})`);
console.log(`Momentum:   ${changeStr} (last 30d: ${coverage.momentum.last_30_days})\n`);

console.log("Top topics:");
coverage.top_topics.forEach((topic) => {
  console.log(`  ${topic.name.padEnd(20)} ${String(topic.count).padStart(5)}`);
});

console.log("\nTop entities covered:");
coverage.top_entities.forEach((entity) => {
  console.log(`  ${entity.name.padEnd(20)} ${String(entity.count).padStart(5)}`);
});
```

### Profile Without Coverage, Then Jump to Articles

```javascript
const API_KEY = "YOUR_API_KEY";
const EVERYTHING_URL = "https://api.apitube.io/v1/news/everything";

// Lightweight fetch: skip the coverage block
const profileParams = new URLSearchParams({ api_key: API_KEY, coverage: "false" });
const profileResp = await fetch(
  `https://api.apitube.io/v1/journalists/88123?${profileParams}`
);
const profile = await profileResp.json();

console.log(`${profile.name} writes for ${profile.outlet_count} outlets\n`);

// Drill through to the author's latest articles
const articleParams = new URLSearchParams({
  api_key: API_KEY,
  "author.id": String(profile.id),
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "10",
});

const articlesResp = await fetch(`${EVERYTHING_URL}?${articleParams}`);
const articles = await articlesResp.json();

console.log(`Latest articles by ${profile.name}:`);
articles.results.forEach((article) => {
  console.log(`  ${article.published_at.slice(0, 10)}  ${article.title}`);
});
```

---

## PHP

### Search Journalists by Name

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

$query = http_build_query([
    "api_key"  => $apiKey,
    "name"     => "Jane",
    "per_page" => 20,
]);

$data = json_decode(file_get_contents("{$listUrl}?{$query}"), true);
$results = $data["results"];

echo "Found " . count($results) . " journalists matching 'Jane':\n\n";

foreach ($results as $journalist) {
    $outlets = implode(", ", array_column($journalist["outlets"], "name"));
    printf("  #%7d  %-24s %s\n", $journalist["id"], $journalist["name"], $outlets);
}
```

### Paginate Through the Full Directory

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

function iterJournalists(int $perPage = 50, int $maxPages = 10): array
{
    global $apiKey, $listUrl;

    $all  = [];
    $page = 1;

    while ($page <= $maxPages) {
        $query = http_build_query([
            "api_key"  => $apiKey,
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

$journalists = iterJournalists();
echo "Walked " . count($journalists) . " journalist records across the directory\n";
```

### Render a Journalist Profile

```php
<?php

$apiKey = "YOUR_API_KEY";

function getProfile(int $journalistId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/journalists/{$journalistId}?{$query}"
    ), true);
}

$profile  = getProfile(88123);
$coverage = $profile["coverage"] ?? [];
$changeStr = $coverage["momentum"]["change_pct"] === null
    ? "n/a"
    : sprintf("%+d%%", $coverage["momentum"]["change_pct"]);

printf("%s  (#%d)\n", $profile["name"], $profile["id"]);
echo "Writes for: " . implode(", ", array_column($profile["outlets"], "name")) . "\n";
printf("Articles:   %s  (%s -> %s)\n",
    number_format($coverage["article_count"]), $coverage["first_seen"] ?? "n/a", $coverage["last_seen"] ?? "n/a");
printf("Momentum:   %s (last 30d: %d)\n\n",
    $changeStr, $coverage["momentum"]["last_30_days"]);

echo "Top topics:\n";
foreach ($coverage["top_topics"] as $topic) {
    printf("  %-20s %5d\n", $topic["name"], $topic["count"]);
}

echo "\nTop entities covered:\n";
foreach ($coverage["top_entities"] as $entity) {
    printf("  %-20s %5d\n", $entity["name"], $entity["count"]);
}
```

### Profile Without Coverage, Then Jump to Articles

```php
<?php

$apiKey        = "YOUR_API_KEY";
$everythingUrl = "https://api.apitube.io/v1/news/everything";

// Lightweight fetch: skip the coverage block
$profileQuery = http_build_query(["api_key" => $apiKey, "coverage" => "false"]);
$profile = json_decode(file_get_contents(
    "https://api.apitube.io/v1/journalists/88123?{$profileQuery}"
), true);

echo "{$profile['name']} writes for {$profile['outlet_count']} outlets\n\n";

// Drill through to the author's latest articles
$articleQuery = http_build_query([
    "api_key"    => $apiKey,
    "author.id"  => $profile["id"],
    "sort.by"    => "published_at",
    "sort.order" => "desc",
    "per_page"   => 10,
]);

$articles = json_decode(file_get_contents("{$everythingUrl}?{$articleQuery}"), true);

echo "Latest articles by {$profile['name']}:\n";
foreach ($articles["results"] as $article) {
    echo "  " . substr($article["published_at"], 0, 10) . "  {$article['title']}\n";
}
```
