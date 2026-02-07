# Get Latest News — Code Examples

Detailed examples for fetching the latest news using the APITube News API in **PHP**, **Python**, and **JavaScript**.

---

## Response Data Model

Reference: [Response Structure](https://docs.apitube.io/platform/news-api/response-structure)

### Top-Level Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Request status (e.g., `"ok"`). |
| `page` | integer | Current page number. |
| `path` | string | The request URL path. |
| `has_next_pages` | boolean | Whether more pages of results exist. |
| `next_page` | string | URL for the next page of results. |
| `has_previous_page` | boolean | Whether a previous page exists. |
| `previous_page` | string | URL for the previous page of results. |
| `request_id` | string | Unique identifier for the request. |
| `export` | object | Download links in multiple formats (see below). |
| `results` | array | Array of article objects. |

#### `export` Object

| Field | Type | Description |
|-------|------|-------------|
| `json` | string | Export URL in JSON format. |
| `xlsx` | string | Export URL in XLSX format. |
| `csv` | string | Export URL in CSV format. |
| `tsv` | string | Export URL in TSV format. |
| `xml` | string | Export URL in XML format. |

### Article Object (each item in `results`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique article identifier. |
| `title` | string | The title of the news article. |
| `description` | string | A short description or summary of the article. |
| `content` | string | The full content of the news article. |
| `url` | string | The URL of the news article. |
| `image` | string | URL of the article's main image. |
| `published_at` | string | Publication date in ISO 8601 format. |
| `source` | object | Information about the article's source. |
| `language` | object | The language of the article. |
| `category` | object | The category of the article (IPTC media topic). |
| `topic` | object | The topic of the article. |
| `industry` | object | The industry classification of the article. |
| `sentiment` | object | Sentiment analysis results. |
| `entities` | array | Named entities extracted from the article. |
| `stories` | array | Story group identifiers (clusters related articles). |
| `links` | array | Links extracted from the article body. |
| `media` | array | Media items (images, videos) from the article. |
| `hashtags` | array | Hashtags extracted from the article. |
| `duplicate` | boolean | Whether the article is a detected duplicate. |
| `paywall` | boolean | Whether the article is behind a paywall. |
| `breaking_news` | boolean | Whether the article is flagged as breaking news. |
| `sentences` | integer | Number of sentences in the article. |
| `paragraphs` | integer | Number of paragraphs in the article. |
| `words` | integer | Number of words in the article. |
| `characters` | integer | Number of characters in the article. |
| `reading_time` | number | Estimated reading time in minutes. |

#### `source` Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Name of the source (e.g., `"TechCrunch"`). |
| `domain` | string | Domain of the source (e.g., `"techcrunch.com"`). |
| `url` | string | URL of the source. |
| `country` | object | Country of the source. |
| `country.code` | string | ISO country code (e.g., `"us"`). |
| `country.name` | string | Country name (e.g., `"United States"`). |
| `rank` | object | Source ranking information. |
| `rank.opr` | number | OPR (Open Page Rank) score, from `0` to `1`. |

#### `language` Object

| Field | Type | Description |
|-------|------|-------------|
| `code` | string | ISO 639-1 language code (e.g., `"en"`). |
| `name` | string | Language name (e.g., `"English"`). |

#### `category` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | IPTC media topic ID (e.g., `"medtop:04000000"`). |
| `name` | string | Category name (e.g., `"economy, business and finance"`). |

#### `topic` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Topic identifier (e.g., `"climate_change"`). |
| `name` | string | Topic name (e.g., `"Climate Change"`). |

#### `industry` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Industry identifier. |
| `name` | string | Industry name. |

#### `sentiment` Object

| Field | Type | Description |
|-------|------|-------------|
| `overall` | object | Overall sentiment analysis. |
| `overall.score` | number | Sentiment score (numeric value). |
| `overall.polarity` | string | Sentiment polarity: `"positive"`, `"negative"`, or `"neutral"`. |

#### `entities` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Entity name (e.g., `"Apple"`, `"Elon Musk"`). |
| `type` | string | Entity type (e.g., `"organization"`, `"person"`, `"location"`). |

### Example Response

```json
{
  "status": "ok",
  "page": 1,
  "path": "https://api.apitube.io/v1/news/everything?language=en&limit=1",
  "has_next_pages": true,
  "next_page": "https://api.apitube.io/v1/news/everything?language=en&limit=1&page=2",
  "has_previous_page": false,
  "previous_page": null,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "export": {
    "json": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=json",
    "xlsx": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=xlsx",
    "csv": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=csv",
    "tsv": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=tsv",
    "xml": "https://api.apitube.io/v1/news/everything?language=en&limit=1&format=xml"
  },
  "results": [
    {
      "id": "abc123def456",
      "title": "Breaking: Major Tech Company Announces New Product",
      "description": "A leading technology company unveiled its latest innovation today...",
      "content": "Full article content goes here...",
      "url": "https://example.com/article/12345",
      "image": "https://example.com/images/article.jpg",
      "published_at": "2026-02-07T10:30:00Z",
      "source": {
        "name": "Tech Daily",
        "domain": "example.com",
        "url": "https://example.com",
        "country": {
          "code": "us",
          "name": "United States"
        },
        "rank": {
          "opr": 0.82
        }
      },
      "language": {
        "code": "en",
        "name": "English"
      },
      "category": {
        "id": "medtop:04000000",
        "name": "economy, business and finance"
      },
      "topic": {
        "id": "technology",
        "name": "Technology"
      },
      "industry": {
        "id": "tech",
        "name": "Technology"
      },
      "sentiment": {
        "overall": {
          "score": 0.75,
          "polarity": "positive"
        }
      },
      "entities": [
        {
          "name": "Apple",
          "type": "organization"
        },
        {
          "name": "Tim Cook",
          "type": "person"
        }
      ],
      "stories": ["story_abc123"],
      "links": [
        "https://example.com/related-article"
      ],
      "media": [],
      "hashtags": ["#tech", "#innovation"],
      "duplicate": false,
      "paywall": false,
      "breaking_news": true,
      "sentences": 24,
      "paragraphs": 8,
      "words": 520,
      "characters": 3100,
      "reading_time": 2.5
    }
  ]
}
```

---

## Python

### Basic Request

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "language": "en",
    "limit": 10,
})
response.raise_for_status()

data = response.json()

for article in data["results"]:
    print(f"- {article['title']}")
    print(f"  Source: {article['source']['name']}")
    print(f"  URL: {article['url']}")
    print()
```

### Filter by Category and Country

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "category": "technology",
    "country": "us",
    "language": "en",
    "limit": 20,
    "sort_by": "published_at",
})
response.raise_for_status()

data = response.json()
for article in data["results"]:
    print(f"[{article['published_at']}] {article['title']}")
```

### Search by Keyword with Date Range

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

today = datetime.utcnow()
week_ago = today - timedelta(days=7)

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "query": "artificial intelligence",
    "language": "en",
    "from": week_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "to": today.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sort_by": "relevance",
    "limit": 25,
})
response.raise_for_status()

data = response.json()
print(f"Found {len(data['results'])} articles about AI in the last 7 days\n")

for i, article in enumerate(data["results"], 1):
    print(f"{i}. {article['title']}")
    print(f"   {article['description'][:120]}...")
    print(f"   {article['url']}\n")
```

### Pagination

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

page = 1
limit = 50
all_articles = []

while True:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "language": "en",
        "limit": limit,
        "page": page,
    })
    response.raise_for_status()

    data = response.json()
    articles = data.get("results", [])

    if not articles:
        break

    all_articles.extend(articles)
    print(f"Page {page}: fetched {len(articles)} articles")

    if len(all_articles) >= 200:
        break

    page += 1

print(f"\nTotal collected: {len(all_articles)} articles")
```

### Error Handling

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

MAX_RETRIES = 3

def fetch_news(params, retries=MAX_RETRIES):
    params["api_key"] = API_KEY

    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)

            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise Exception("Failed to fetch news after multiple retries")


data = fetch_news({"language": "en", "limit": 10})
for article in data["results"]:
    print(article["title"])
```

---

## JavaScript

### Basic Request (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getLatestNews() {
  const params = new URLSearchParams({
    api_key: API_KEY,
    language: "en",
    limit: "10",
  });

  const response = await fetch(`${BASE_URL}?${params}`);

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const data = await response.json();
  console.log(`Total results: ${data.results.length}`);

  data.results.forEach((article) => {
    console.log(`- ${article.title}`);
    console.log(`  Source: ${article.source.name}`);
    console.log(`  URL: ${article.url}\n`);
  });
}

getLatestNews();
```

### Filter by Category and Country

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getTechNews() {
  const params = new URLSearchParams({
    api_key: API_KEY,
    category: "technology",
    country: "us",
    language: "en",
    limit: "20",
    sort_by: "published_at",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  data.results.forEach((article) => {
    console.log(`[${article.published_at}] ${article.title}`);
  });
}

getTechNews();
```

### Search by Keyword with Date Range

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function searchNews(query, days = 7) {
  const now = new Date();
  const from = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);

  const params = new URLSearchParams({
    api_key: API_KEY,
    query,
    language: "en",
    from: from.toISOString(),
    to: now.toISOString(),
    sort_by: "relevance",
    limit: "25",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  console.log(
    `Found ${data.results.length} articles about "${query}" in the last ${days} days\n`
  );

  data.results.forEach((article, i) => {
    console.log(`${i + 1}. ${article.title}`);
    console.log(`   ${article.description?.slice(0, 120)}...`);
    console.log(`   ${article.url}\n`);
  });
}

searchNews("artificial intelligence");
```

### Pagination

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function fetchAllNews(maxArticles = 200) {
  const allArticles = [];
  let page = 1;
  const limit = 50;

  while (allArticles.length < maxArticles) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      language: "en",
      limit: String(limit),
      page: String(page),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results ?? [];

    if (articles.length === 0) break;

    allArticles.push(...articles);
    console.log(`Page ${page}: fetched ${articles.length} articles`);
    page++;
  }

  console.log(`\nTotal collected: ${allArticles.length} articles`);
  return allArticles;
}

fetchAllNews();
```

### Error Handling with Retry

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function fetchNews(params, retries = 3) {
  params.api_key = API_KEY;
  const qs = new URLSearchParams(params);

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10_000);

      const response = await fetch(`${BASE_URL}?${qs}`, {
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.status === 429) {
        const wait = 2 ** attempt * 1000;
        console.log(`Rate limited. Retrying in ${wait}ms...`);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      console.error(`Attempt ${attempt + 1} failed: ${err.message}`);
      if (attempt < retries - 1) {
        await new Promise((r) => setTimeout(r, 2 ** attempt * 1000));
      }
    }
  }

  throw new Error("Failed to fetch news after multiple retries");
}

const data = await fetchNews({ language: "en", limit: "10" });
data.results.forEach((a) => console.log(a.title));
```

---

## PHP

### Basic Request

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"  => $apiKey,
    "language" => "en",
    "limit"    => 10,
]);

$response = file_get_contents("{$baseUrl}?{$query}");
$data     = json_decode($response, true);

echo "Total results: {$len(data['results'])}\n\n";

foreach ($data["results"] as $article) {
    echo "- {$article['title']}\n";
    echo "  Source: {$article['source']['name']}\n";
    echo "  URL: {$article['url']}\n\n";
}
```

### Filter by Category and Country

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"  => $apiKey,
    "category" => "technology",
    "country"  => "us",
    "language" => "en",
    "limit"    => 20,
    "sort_by"  => "published_at",
]);

$response = file_get_contents("{$baseUrl}?{$query}");
$data     = json_decode($response, true);

foreach ($data["results"] as $article) {
    echo "[{$article['published_at']}] {$article['title']}\n";
}
```

### Search by Keyword with Date Range

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$now     = new DateTimeImmutable("now", new DateTimeZone("UTC"));
$weekAgo = $now->modify("-7 days");

$query = http_build_query([
    "api_key"  => $apiKey,
    "query"    => "artificial intelligence",
    "language" => "en",
    "from"     => $weekAgo->format("c"),
    "to"       => $now->format("c"),
    "sort_by"  => "relevance",
    "limit"    => 25,
]);

$response = file_get_contents("{$baseUrl}?{$query}");
$data     = json_decode($response, true);

$count = count($data["results"]);
echo "Found {$count} articles about AI in the last 7 days\n\n";

foreach ($data["results"] as $i => $article) {
    $num = $i + 1;
    $desc = mb_substr($article["description"], 0, 120);
    echo "{$num}. {$article['title']}\n";
    echo "   {$desc}...\n";
    echo "   {$article['url']}\n\n";
}
```

### Pagination

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$page        = 1;
$limit       = 50;
$maxArticles = 200;
$allArticles = [];

while (count($allArticles) < $maxArticles) {
    $query = http_build_query([
        "api_key"  => $apiKey,
        "language" => "en",
        "limit"    => $limit,
        "page"     => $page,
    ]);

    $response = file_get_contents("{$baseUrl}?{$query}");
    $data     = json_decode($response, true);
    $articles = $data["results"] ?? [];

    if (empty($articles)) {
        break;
    }

    $allArticles = array_merge($allArticles, $articles);
    echo "Page {$page}: fetched " . count($articles) . " articles\n";
    $page++;
}

echo "\nTotal collected: " . count($allArticles) . " articles\n";
```

### Error Handling with cURL and Retry

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function fetchNews(array $params, int $retries = 3): array
{
    global $apiKey, $baseUrl;

    $params["api_key"] = $apiKey;
    $url = $baseUrl . "?" . http_build_query($params);

    for ($attempt = 0; $attempt < $retries; $attempt++) {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 10,
            CURLOPT_FOLLOWLOCATION => true,
        ]);

        $body     = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error    = curl_error($ch);
        curl_close($ch);

        if ($body === false) {
            echo "Attempt " . ($attempt + 1) . " failed: {$error}\n";
        } elseif ($httpCode === 429) {
            $wait = pow(2, $attempt);
            echo "Rate limited. Retrying in {$wait}s...\n";
            sleep($wait);
            continue;
        } elseif ($httpCode >= 200 && $httpCode < 300) {
            return json_decode($body, true);
        } else {
            echo "Attempt " . ($attempt + 1) . " failed: HTTP {$httpCode}\n";
        }

        if ($attempt < $retries - 1) {
            sleep(pow(2, $attempt));
        }
    }

    throw new RuntimeException("Failed to fetch news after {$retries} retries");
}

$data = fetchNews(["language" => "en", "limit" => 10]);

foreach ($data["results"] as $article) {
    echo $article["title"] . "\n";
}
```
