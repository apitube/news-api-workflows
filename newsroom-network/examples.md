# Newsroom Network — Code Examples

Detailed examples for mapping journalist-to-outlet relationships using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### List Authors for a Given Outlet

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

def authors_at_outlet(domain, max_pages=20):
    matches, page = [], 1
    while page <= max_pages:
        response = requests.get(LIST_URL, params={
            "api_key": API_KEY,
            "per_page": 50,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for journalist in data["results"]:
            if any(o["domain"] == domain for o in journalist["outlets"]):
                matches.append(journalist["name"])

        if not data.get("has_next_pages"):
            break
        page += 1

    return matches

names = authors_at_outlet("example.com")
print(f"Authors writing for example.com ({len(names)}):\n")
for name in sorted(names):
    print(f"  {name}")
```

### Build a Journalist -> Outlets Graph

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

def build_edges(max_pages=10):
    edges, page = [], 1
    while page <= max_pages:
        response = requests.get(LIST_URL, params={
            "api_key": API_KEY,
            "per_page": 50,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for journalist in data["results"]:
            for outlet in journalist["outlets"]:
                edges.append((journalist["name"], outlet["name"]))

        if not data.get("has_next_pages"):
            break
        page += 1

    return edges

edges = build_edges()
print(f"Collected {len(edges)} journalist -> outlet edges\n")
for journalist, outlet in edges[:20]:
    print(f'  "{journalist}" -> "{outlet}"')
```

### Top Cross-Outlet Journalists

```python
import requests

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

def multi_outlet_journalists(max_pages=10):
    found, page = [], 1
    while page <= max_pages:
        response = requests.get(LIST_URL, params={
            "api_key": API_KEY,
            "per_page": 50,
            "page": page,
        })
        response.raise_for_status()
        data = response.json()

        for journalist in data["results"]:
            if journalist["outlet_count"] > 1:
                found.append(journalist)

        if not data.get("has_next_pages"):
            break
        page += 1

    return sorted(found, key=lambda j: j["outlet_count"], reverse=True)

top = multi_outlet_journalists()
print("Journalists publishing across multiple outlets:\n")
print(f"{'Outlets':>7}  {'Journalist':<22} Publications")
print("-" * 60)

for journalist in top[:15]:
    outlets = ", ".join(o["name"] for o in journalist["outlets"])
    print(f"{journalist['outlet_count']:>7}  {journalist['name']:<22} {outlets}")
```

### Outlet Reach Ranking

```python
import requests
from collections import Counter

API_KEY = "YOUR_API_KEY"
LIST_URL = "https://api.apitube.io/v1/journalists"

reach = Counter()
page = 1

while page <= 10:
    response = requests.get(LIST_URL, params={
        "api_key": API_KEY,
        "per_page": 50,
        "page": page,
    })
    response.raise_for_status()
    data = response.json()

    for journalist in data["results"]:
        for outlet in journalist["outlets"]:
            reach[outlet["name"]] += 1

    if not data.get("has_next_pages"):
        break
    page += 1

print("Outlets ranked by number of distinct contributors:\n")
for outlet, count in reach.most_common(15):
    bar = "#" * count
    print(f"  {outlet:<24} {count:>4} {bar}")
```

---

## JavaScript

### List Authors for a Given Outlet

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

async function authorsAtOutlet(domain, maxPages = 20) {
  const matches = [];
  let page = 1;

  while (page <= maxPages) {
    const params = new URLSearchParams({ api_key: API_KEY, per_page: "50", page: String(page) });
    const response = await fetch(`${LIST_URL}?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    for (const journalist of data.results) {
      if (journalist.outlets.some((o) => o.domain === domain)) {
        matches.push(journalist.name);
      }
    }

    if (!data.has_next_pages) break;
    page += 1;
  }

  return matches;
}

const names = await authorsAtOutlet("example.com");
console.log(`Authors writing for example.com (${names.length}):\n`);
names.sort().forEach((name) => console.log(`  ${name}`));
```

### Build a Journalist -> Outlets Graph

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

async function buildEdges(maxPages = 10) {
  const edges = [];
  let page = 1;

  while (page <= maxPages) {
    const params = new URLSearchParams({ api_key: API_KEY, per_page: "50", page: String(page) });
    const response = await fetch(`${LIST_URL}?${params}`);
    const data = await response.json();

    for (const journalist of data.results) {
      for (const outlet of journalist.outlets) {
        edges.push([journalist.name, outlet.name]);
      }
    }

    if (!data.has_next_pages) break;
    page += 1;
  }

  return edges;
}

const edges = await buildEdges();
console.log(`Collected ${edges.length} journalist -> outlet edges\n`);
edges.slice(0, 20).forEach(([journalist, outlet]) => {
  console.log(`  "${journalist}" -> "${outlet}"`);
});
```

### Top Cross-Outlet Journalists

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

async function multiOutletJournalists(maxPages = 10) {
  const found = [];
  let page = 1;

  while (page <= maxPages) {
    const params = new URLSearchParams({ api_key: API_KEY, per_page: "50", page: String(page) });
    const response = await fetch(`${LIST_URL}?${params}`);
    const data = await response.json();

    for (const journalist of data.results) {
      if (journalist.outlet_count > 1) found.push(journalist);
    }

    if (!data.has_next_pages) break;
    page += 1;
  }

  return found.sort((a, b) => b.outlet_count - a.outlet_count);
}

const top = await multiOutletJournalists();
console.log("Journalists publishing across multiple outlets:\n");
console.log(`${"Outlets".padStart(7)}  ${"Journalist".padEnd(22)} Publications`);
console.log("-".repeat(60));

top.slice(0, 15).forEach((journalist) => {
  const outlets = journalist.outlets.map((o) => o.name).join(", ");
  console.log(`${String(journalist.outlet_count).padStart(7)}  ${journalist.name.padEnd(22)} ${outlets}`);
});
```

### Outlet Reach Ranking

```javascript
const API_KEY = "YOUR_API_KEY";
const LIST_URL = "https://api.apitube.io/v1/journalists";

const reach = new Map();
let page = 1;

while (page <= 10) {
  const params = new URLSearchParams({ api_key: API_KEY, per_page: "50", page: String(page) });
  const response = await fetch(`${LIST_URL}?${params}`);
  const data = await response.json();

  for (const journalist of data.results) {
    for (const outlet of journalist.outlets) {
      reach.set(outlet.name, (reach.get(outlet.name) || 0) + 1);
    }
  }

  if (!data.has_next_pages) break;
  page += 1;
}

console.log("Outlets ranked by number of distinct contributors:\n");
[...reach.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, 15)
  .forEach(([outlet, count]) => {
    const bar = "#".repeat(count);
    console.log(`  ${outlet.padEnd(24)} ${String(count).padStart(4)} ${bar}`);
  });
```

---

## PHP

### List Authors for a Given Outlet

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

function authorsAtOutlet(string $domain, int $maxPages = 20): array
{
    global $apiKey, $listUrl;

    $matches = [];
    $page    = 1;

    while ($page <= $maxPages) {
        $query = http_build_query(["api_key" => $apiKey, "per_page" => 50, "page" => $page]);
        $data  = json_decode(file_get_contents("{$listUrl}?{$query}"), true);

        foreach ($data["results"] as $journalist) {
            if (in_array($domain, array_column($journalist["outlets"], "domain"), true)) {
                $matches[] = $journalist["name"];
            }
        }

        if (empty($data["has_next_pages"])) {
            break;
        }
        $page++;
    }

    return $matches;
}

$names = authorsAtOutlet("example.com");
sort($names);
echo "Authors writing for example.com (" . count($names) . "):\n\n";
foreach ($names as $name) {
    echo "  {$name}\n";
}
```

### Build a Journalist -> Outlets Graph

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

function buildEdges(int $maxPages = 10): array
{
    global $apiKey, $listUrl;

    $edges = [];
    $page  = 1;

    while ($page <= $maxPages) {
        $query = http_build_query(["api_key" => $apiKey, "per_page" => 50, "page" => $page]);
        $data  = json_decode(file_get_contents("{$listUrl}?{$query}"), true);

        foreach ($data["results"] as $journalist) {
            foreach ($journalist["outlets"] as $outlet) {
                $edges[] = [$journalist["name"], $outlet["name"]];
            }
        }

        if (empty($data["has_next_pages"])) {
            break;
        }
        $page++;
    }

    return $edges;
}

$edges = buildEdges();
echo "Collected " . count($edges) . " journalist -> outlet edges\n\n";
foreach (array_slice($edges, 0, 20) as [$journalist, $outlet]) {
    echo "  \"{$journalist}\" -> \"{$outlet}\"\n";
}
```

### Top Cross-Outlet Journalists

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

function multiOutletJournalists(int $maxPages = 10): array
{
    global $apiKey, $listUrl;

    $found = [];
    $page  = 1;

    while ($page <= $maxPages) {
        $query = http_build_query(["api_key" => $apiKey, "per_page" => 50, "page" => $page]);
        $data  = json_decode(file_get_contents("{$listUrl}?{$query}"), true);

        foreach ($data["results"] as $journalist) {
            if ($journalist["outlet_count"] > 1) {
                $found[] = $journalist;
            }
        }

        if (empty($data["has_next_pages"])) {
            break;
        }
        $page++;
    }

    usort($found, fn($a, $b) => $b["outlet_count"] <=> $a["outlet_count"]);
    return $found;
}

$top = multiOutletJournalists();
echo "Journalists publishing across multiple outlets:\n\n";
printf("%7s  %-22s %s\n", "Outlets", "Journalist", "Publications");
echo str_repeat("-", 60) . "\n";

foreach (array_slice($top, 0, 15) as $journalist) {
    $outlets = implode(", ", array_column($journalist["outlets"], "name"));
    printf("%7d  %-22s %s\n", $journalist["outlet_count"], $journalist["name"], $outlets);
}
```

### Outlet Reach Ranking

```php
<?php

$apiKey  = "YOUR_API_KEY";
$listUrl = "https://api.apitube.io/v1/journalists";

$reach = [];
$page  = 1;

while ($page <= 10) {
    $query = http_build_query(["api_key" => $apiKey, "per_page" => 50, "page" => $page]);
    $data  = json_decode(file_get_contents("{$listUrl}?{$query}"), true);

    foreach ($data["results"] as $journalist) {
        foreach ($journalist["outlets"] as $outlet) {
            $reach[$outlet["name"]] = ($reach[$outlet["name"]] ?? 0) + 1;
        }
    }

    if (empty($data["has_next_pages"])) {
        break;
    }
    $page++;
}

arsort($reach);
echo "Outlets ranked by number of distinct contributors:\n\n";
foreach (array_slice($reach, 0, 15, true) as $outlet => $count) {
    $bar = str_repeat("#", $count);
    printf("  %-24s %4d %s\n", $outlet, $count, $bar);
}
```
