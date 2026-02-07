# Multi-Source Monitoring — Code Examples

Detailed examples for monitoring multiple news sources using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Custom Source Feed

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

SOURCES = [
    "reuters.com",
    "bbc.com",
    "bloomberg.com",
    "nytimes.com",
    "theguardian.com",
]

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "source.domain": ",".join(SOURCES),
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 30,
})
response.raise_for_status()

data = response.json()
print(f"Latest {len(data['results'])} articles from trusted sources:\n")

for article in data["results"]:
    date = article["published_at"][:10]
    print(f"  [{date}] [{article['source']['domain']}]")
    print(f"    {article['title']}")
    print(f"    {article['href']}\n")
```

### High-Authority Sources Only

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "source.rank.opr.min": 0.8,
    "topic.id": "finance",
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 20,
})
response.raise_for_status()

data = response.json()
print(f"Top-ranked finance news ({data['total_results']} total):\n")

for article in data["results"]:
    print(f"  {article['title']}")
    print(f"    Source: {article['source']['domain']} — {article['source']['home_page_url']}")
    print()
```

### Compare Coverage Across Sources

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def coverage_by_source(title_keyword, domains):
    results = {}
    for domain in domains:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": title_keyword,
            "source.domain": domain,
            "language": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        results[domain] = response.json().get("total_results", 0)
    return results

keyword = "AI"
sources = [
    "reuters.com", "bbc.com", "cnn.com", "foxnews.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bloomberg.com", "ft.com", "techcrunch.com",
]

coverage = coverage_by_source(keyword, sources)
max_val = max(coverage.values()) or 1

print(f"Coverage of '{keyword}' by source:\n")

for domain, count in sorted(coverage.items(), key=lambda x: -x[1]):
    bar = "█" * int(count / max_val * 30)
    print(f"  {domain:<25} {count:>5} {bar}")
```

### Source Country Comparison

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

countries = {
    "us": "United States",
    "gb": "United Kingdom",
    "de": "Germany",
    "fr": "France",
    "jp": "Japan",
    "au": "Australia",
}

topic = "climate_change"
print(f"Coverage of '{topic}' by source country:\n")

for code, name in countries.items():
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "source.country.code": code,
        "per_page": 1,
    })
    response.raise_for_status()
    count = response.json().get("total_results", 0)
    bar = "█" * (count // 100)
    print(f"  {name:<20} ({code}) {count:>6} articles {bar}")
```

### Source Rank Distribution

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topic = "technology"
ranges = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.4, "Low"),
    (0.4, 0.6, "Medium"),
    (0.6, 0.8, "High"),
    (0.8, 1.0, "Very High"),
]

print(f"Source rank distribution for '{topic}' news:\n")

for min_rank, max_rank, label in ranges:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "source.rank.opr.min": min_rank,
        "source.rank.opr.max": max_rank,
        "language": "en",
        "per_page": 1,
    })
    response.raise_for_status()
    count = response.json().get("total_results", 0)
    bar = "█" * (count // 50)
    print(f"  {label:<12} ({min_rank:.1f}-{max_rank:.1f}): {count:>6} {bar}")
```

### Multi-Country Source Feed with Pagination

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def fetch_multi_country_feed(country_codes, topic, max_per_country=50):
    all_articles = []

    for code in country_codes:
        page = 1
        country_articles = []

        while len(country_articles) < max_per_country:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "source.country.code": code,
                "topic.id": topic,
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 50,
                "page": page,
            })
            response.raise_for_status()
            articles = response.json().get("results", [])

            if not articles:
                break

            country_articles.extend(articles)
            page += 1

        print(f"  {code.upper()}: {len(country_articles)} articles")
        all_articles.extend(country_articles[:max_per_country])

    return all_articles

print("Fetching multi-country tech news feed:\n")
articles = fetch_multi_country_feed(
    ["us", "gb", "de", "fr", "jp"],
    "technology",
    max_per_country=20,
)

print(f"\nTotal: {len(articles)} articles")
```

---

## JavaScript

### Custom Source Feed

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const SOURCES = [
  "reuters.com",
  "bbc.com",
  "bloomberg.com",
  "nytimes.com",
  "theguardian.com",
];

const params = new URLSearchParams({
  api_key: API_KEY,
  "source.domain": SOURCES.join(","),
  language: "en",
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "30",
});

const response = await fetch(`${BASE_URL}?${params}`);
const data = await response.json();

console.log(`Latest ${data.results.length} articles from trusted sources:\n`);

data.results.forEach((article) => {
  const date = article.published_at.slice(0, 10);
  console.log(`  [${date}] [${article.source.domain}]`);
  console.log(`    ${article.title}`);
  console.log(`    ${article.href}\n`);
});
```

### Compare Coverage Across Sources

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function coverageBySource(titleKeyword, domains) {
  const results = {};

  for (const domain of domains) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      title: titleKeyword,
      "source.domain": domain,
      language: "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    results[domain] = data.total_results || 0;
  }

  return results;
}

const keyword = "AI";
const sources = [
  "reuters.com", "bbc.com", "cnn.com", "foxnews.com",
  "nytimes.com", "washingtonpost.com", "theguardian.com",
  "bloomberg.com", "ft.com", "techcrunch.com",
];

const coverage = await coverageBySource(keyword, sources);
const maxVal = Math.max(...Object.values(coverage)) || 1;

console.log(`Coverage of '${keyword}' by source:\n`);

Object.entries(coverage)
  .sort((a, b) => b[1] - a[1])
  .forEach(([domain, count]) => {
    const bar = "#".repeat(Math.round((count / maxVal) * 30));
    console.log(`  ${domain.padEnd(25)} ${String(count).padStart(5)} ${bar}`);
  });
```

### Source Country Comparison

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const countries = {
  us: "United States",
  gb: "United Kingdom",
  de: "Germany",
  fr: "France",
  jp: "Japan",
  au: "Australia",
};

const topic = "climate_change";
console.log(`Coverage of '${topic}' by source country:\n`);

for (const [code, name] of Object.entries(countries)) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": topic,
    "source.country.code": code,
    per_page: "1",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  const count = data.total_results || 0;
  const bar = "#".repeat(Math.floor(count / 100));
  console.log(
    `  ${name.padEnd(20)} (${code}) ${String(count).padStart(6)} articles ${bar}`
  );
}
```

### High-Authority Sources Only

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getHighRankNews(topic, minRank = 0.8, perPage = 20) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "source.rank.opr.min": String(minRank),
    "topic.id": topic,
    language: "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: String(perPage),
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await getHighRankNews("finance");
console.log(`Top-ranked finance news (${data.total_results} total):\n`);

data.results.forEach((article) => {
  console.log(`  ${article.title}`);
  console.log(`    Source: ${article.source.domain} — ${article.source.home_page_url}\n`);
});
```

---

## PHP

### Custom Source Feed

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$sources = [
    "reuters.com",
    "bbc.com",
    "bloomberg.com",
    "nytimes.com",
    "theguardian.com",
];

$query = http_build_query([
    "api_key"       => $apiKey,
    "source.domain" => implode(",", $sources),
    "language"      => "en",
    "sort.by"       => "published_at",
    "sort.order"    => "desc",
    "per_page"      => 30,
]);

$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

echo "Latest " . count($data["results"]) . " articles from trusted sources:\n\n";

foreach ($data["results"] as $article) {
    $date = substr($article["published_at"], 0, 10);
    echo "  [{$date}] [{$article['source']['domain']}]\n";
    echo "    {$article['title']}\n";
    echo "    {$article['href']}\n\n";
}
```

### Compare Coverage Across Sources

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function coverageBySource(string $titleKeyword, array $domains): array
{
    global $apiKey, $baseUrl;

    $results = [];
    foreach ($domains as $domain) {
        $query = http_build_query([
            "api_key"       => $apiKey,
            "title"         => $titleKeyword,
            "source.domain" => $domain,
            "language"      => "en",
            "per_page"      => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $results[$domain] = $data["total_results"] ?? 0;
    }

    return $results;
}

$keyword = "AI";
$sources = [
    "reuters.com", "bbc.com", "cnn.com", "foxnews.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bloomberg.com", "ft.com", "techcrunch.com",
];

$coverage = coverageBySource($keyword, $sources);
arsort($coverage);
$maxVal = max($coverage) ?: 1;

echo "Coverage of '{$keyword}' by source:\n\n";

foreach ($coverage as $domain => $count) {
    $bar = str_repeat("#", (int) round($count / $maxVal * 30));
    printf("  %-25s %5d %s\n", $domain, $count, $bar);
}
```

### Source Country Comparison

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$countries = [
    "us" => "United States",
    "gb" => "United Kingdom",
    "de" => "Germany",
    "fr" => "France",
    "jp" => "Japan",
    "au" => "Australia",
];

$topic = "climate_change";
echo "Coverage of '{$topic}' by source country:\n\n";

foreach ($countries as $code => $name) {
    $query = http_build_query([
        "api_key"              => $apiKey,
        "topic.id"             => $topic,
        "source.country.code"  => $code,
        "per_page"             => 1,
    ]);

    $data  = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $count = $data["total_results"] ?? 0;
    $bar   = str_repeat("#", intdiv($count, 100));
    printf("  %-20s (%s) %6d articles %s\n", $name, $code, $count, $bar);
}
```

### High-Authority Sources with cURL

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"              => $apiKey,
    "source.rank.opr.min"  => 0.8,
    "topic.id"             => "finance",
    "language"             => "en",
    "sort.by"              => "published_at",
    "sort.order"           => "desc",
    "per_page"             => 20,
]);

$ch = curl_init("{$baseUrl}?{$query}");
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => 10,
]);

$body = curl_exec($ch);
$code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($code !== 200) {
    echo "Error: HTTP {$code}\n";
    exit(1);
}

$data = json_decode($body, true);
echo "Top-ranked finance news ({$data['total_results']} total):\n\n";

foreach ($data["results"] as $article) {
    echo "  {$article['title']}\n";
    echo "    Source: {$article['source']['domain']} — {$article['source']['home_page_url']}\n\n";
}
```
