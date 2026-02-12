# Get Latest News — Code Examples

Detailed examples for fetching the latest news using the APITube News API in **PHP**, **Python**, and **JavaScript**.

---

## Response Data Model

Reference: [Response Structure](https://docs.apitube.io/platform/news-api/response-structure)

### Top-Level Response

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Request status (e.g., `"ok"`). |
| `limit` | integer | Number of results per page. |
| `page` | integer | Current page number. |
| `path` | string | The full request URL. |
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
| `rss` | string | Export URL in RSS format. |

### Article Object (each item in `results`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique article identifier. |
| `href` | string | URL of the original article. |
| `published_at` | string | Publication date in ISO 8601 format. |
| `title` | string | The title of the news article. |
| `description` | string | A short description or summary of the article. |
| `body` | string | The full article text without HTML tags. |
| `body_html` | string | The full article text with HTML formatting. |
| `language` | string | Article language in ISO 639-1 format (e.g., `"en"`). |
| `author` | object | Information about the article author. |
| `image` | string | URL of the article's main image. |
| `categories` | array | Array of categories with relevance scores. |
| `topics` | array | Array of topics with relevance scores. |
| `industries` | array | Array of industry classifications. |
| `entities` | array | Named entities extracted from the article. |
| `source` | object | Information about the article's source. |
| `sentiment` | object | Sentiment analysis results (overall, title, body). |
| `summary` | array | Key sentences from the article with sentiment. |
| `keywords` | array | Keywords extracted from the article. |
| `links` | array | Links extracted from the article body. |
| `media` | array | Media items (images, videos) from the article. |
| `story` | object | Story group identifier (clusters related articles). |
| `is_duplicate` | boolean | Whether the article is a detected duplicate. |
| `is_free` | boolean | Whether the article is freely available (not behind paywall). |
| `is_breaking` | boolean | Whether the article is flagged as breaking news. |
| `read_time` | integer | Estimated reading time in minutes. |
| `sentences_count` | integer | Number of sentences in the article. |
| `paragraphs_count` | integer | Number of paragraphs in the article. |
| `words_count` | integer | Number of words in the article. |
| `characters_count` | integer | Number of characters in the article. |

#### `author` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Author identifier (may be `null`). |
| `name` | string | Author name (may be empty). |

#### `source` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique source identifier. |
| `domain` | string | Domain of the source (e.g., `"techcrunch.com"`). |
| `home_page_url` | string | URL of the source homepage. |
| `type` | string | Source type (e.g., `"news"`, `"blog"`). |
| `bias` | string | Source bias (e.g., `"left"`, `"right"`, `"center"`). |
| `rankings.opr` | integer | Open Page Rank score. |
| `location.country_name` | string | Country name of the source. |
| `location.country_code` | string | ISO 3166-1 alpha-2 country code. |
| `favicon` | string | URL of the source favicon. |

#### `categories` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Category identifier. |
| `name` | string | Category name. |
| `score` | float | Category relevance score (0 to 1). |
| `taxonomy` | string | Taxonomy name (e.g., `"iptc_mediatopics"`). |
| `links.self` | string | URL to get information about the category. |

#### `topics` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Topic identifier (e.g., `"climate_change"`). |
| `name` | string | Topic name (e.g., `"Climate Change"`). |
| `score` | float | Topic relevance score (0 to 1). |
| `links.self` | string | URL to get information about the topic. |

#### `industries` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Industry identifier. |
| `name` | string | Industry name. |
| `links.self` | string | URL to get information about the industry. |

#### `sentiment` Object

| Field | Type | Description |
|-------|------|-------------|
| `overall.score` | float | Overall sentiment score (-1 to 1). |
| `overall.polarity` | string | Overall polarity: `"positive"`, `"negative"`, or `"neutral"`. |
| `title.score` | float | Title sentiment score (-1 to 1). |
| `title.polarity` | string | Title sentiment polarity. |
| `body.score` | float | Body sentiment score (-1 to 1). |
| `body.polarity` | string | Body sentiment polarity. |

#### `entities` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique entity identifier. |
| `name` | string | Entity name (e.g., `"Apple"`, `"Elon Musk"`). |
| `type` | string | Entity type (e.g., `"organization"`, `"person"`, `"location"`, `"brand"`). |
| `links.self` | string | URL to get information about the entity. |
| `links.wikipedia` | string | Wikipedia article URL. |
| `links.wikidata` | string | Wikidata entity URL. |
| `frequency` | integer | Frequency of entity mentions in the article. |
| `title.pos` | array | Array of entity mention positions in the title. |
| `body.pos` | array | Array of entity mention positions in the body. |
| `metadata` | object | Additional entity information (varies by entity type). |

#### `story` Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique story identifier. |
| `uri` | string | URL to get information about the story. |

#### `summary` Array Items

| Field | Type | Description |
|-------|------|-------------|
| `sentence` | string | Key sentence from the article. |
| `sentiment.score` | float | Sentence sentiment score. |
| `sentiment.polarity` | string | Sentence sentiment polarity. |

### Example Response

```json
{
  "status": "ok",
  "per_page": 1,
  "page": 1,
  "path": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1",
  "has_next_pages": true,
  "next_page": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&page=2",
  "has_previous_page": false,
  "previous_page": "",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "export": {
    "json": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=json",
    "xlsx": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=xlsx",
    "csv": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=csv",
    "tsv": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=tsv",
    "xml": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=xml",
    "rss": "https://api.apitube.io/v1/news/everything?language.code=en&per_page=1&export=rss"
  },
  "results": [
    {
      "id": 12345678,
      "href": "https://example.com/article/12345",
      "published_at": "2026-02-07T10:30:00Z",
      "title": "Breaking: Major Tech Company Announces New Product",
      "description": "A leading technology company unveiled its latest innovation today...",
      "body": "Full article content goes here...",
      "body_html": "<p>Full article content goes here...</p>",
      "language": "en",
      "author": {
        "id": 5678,
        "name": "John Smith"
      },
      "image": "https://example.com/images/article.jpg",
      "categories": [
        {
          "id": 199,
          "name": "economy, business and finance",
          "score": 0.8,
          "taxonomy": "iptc_mediatopics",
          "links": {
            "self": "https://api.apitube.io/v1/news/category/iptc_mediatopics/medtop:04000000"
          }
        }
      ],
      "topics": [
        {
          "id": "technology",
          "name": "Technology",
          "score": 0.9,
          "links": {
            "self": "https://api.apitube.io/v1/news/topic/technology"
          }
        }
      ],
      "industries": [
        {
          "id": 456,
          "name": "Consumer Electronics",
          "links": {
            "self": "https://api.apitube.io/v1/news/industry/456"
          }
        }
      ],
      "entities": [
        {
          "id": 789,
          "name": "Apple",
          "type": "organization",
          "links": {
            "self": "https://api.apitube.io/v1/news/entity/789",
            "wikipedia": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "wikidata": "https://www.wikidata.org/wiki/Q312"
          },
          "frequency": 3,
          "title": { "pos": [{ "start": 4, "end": 9 }] },
          "body": { "pos": [{ "start": 10, "end": 15 }] },
          "metadata": {
            "name": "Apple",
            "type": "business"
          }
        }
      ],
      "source": {
        "id": 4232,
        "domain": "example.com",
        "home_page_url": "https://example.com",
        "type": "news",
        "bias": "center",
        "rankings": { "opr": 5 },
        "location": {
          "country_name": "United States",
          "country_code": "us"
        },
        "favicon": "https://www.google.com/s2/favicons?domain=https://example.com"
      },
      "sentiment": {
        "overall": { "score": 0.75, "polarity": "positive" },
        "title": { "score": 0.60, "polarity": "positive" },
        "body": { "score": 0.80, "polarity": "positive" }
      },
      "summary": [
        {
          "sentence": "A leading technology company unveiled its latest innovation today.",
          "sentiment": { "score": 0.6, "polarity": "positive" }
        }
      ],
      "keywords": ["technology", "innovation", "product launch"],
      "links": [{ "url": "https://example.com/related-article", "type": "link" }],
      "media": [{ "url": "https://example.com/images/article.jpg", "type": "image" }],
      "story": { "id": 9876, "uri": "https://api.apitube.io/v1/news/story/9876" },
      "is_duplicate": false,
      "is_free": true,
      "is_breaking": true,
      "read_time": 2,
      "sentences_count": 24,
      "paragraphs_count": 8,
      "words_count": 520,
      "characters_count": 3100
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
    "language.code": "en",
    "per_page": 10,
})
response.raise_for_status()

data = response.json()

for article in data["results"]:
    print(f"- {article['title']}")
    print(f"  Source: {article['source']['domain']}")
    print(f"  URL: {article['href']}")
    print()
```

### Filter by Category and Country

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "category.id": "medtop:13000000",
    "source.country.code": "us",
    "language.code": "en",
    "per_page": 20,
    "sort.by": "published_at",
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
    "title": "artificial intelligence",
    "language.code": "en",
    "published_at.start": week_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "published_at.end": today.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sort.by": "published_at",
    "per_page": 25,
})
response.raise_for_status()

data = response.json()
print(f"Found {len(data['results'])} articles about AI in the last 7 days\n")

for i, article in enumerate(data["results"], 1):
    print(f"{i}. {article['title']}")
    print(f"   {article['description'][:120]}...")
    print(f"   {article['href']}\n")
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
        "language.code": "en",
        "per_page": limit,
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


data = fetch_news({"language.code": "en", "per_page": 10})
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
    "language.code": "en",
    per_page: "10",
  });

  const response = await fetch(`${BASE_URL}?${params}`);

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const data = await response.json();
  console.log(`Total results: ${data.results.length}`);

  data.results.forEach((article) => {
    console.log(`- ${article.title}`);
    console.log(`  Source: ${article.source.domain}`);
    console.log(`  URL: ${article.href}\n`);
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
    "category.id": "medtop:13000000",
    "source.country.code": "us",
    "language.code": "en",
    per_page: "20",
    "sort.by": "published_at",
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
    title: query,
    "language.code": "en",
    "published_at.start": from.toISOString(),
    "published_at.end": now.toISOString(),
    "sort.by": "published_at",
    per_page: "25",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  console.log(
    `Found ${data.results.length} articles about "${query}" in the last ${days} days\n`
  );

  data.results.forEach((article, i) => {
    console.log(`${i + 1}. ${article.title}`);
    console.log(`   ${article.description?.slice(0, 120)}...`);
    console.log(`   ${article.href}\n`);
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
      "language.code": "en",
      per_page: String(limit),
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

const data = await fetchNews({ "language.code": "en", per_page: "10" });
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
    "api_key"       => $apiKey,
    "language.code" => "en",
    "per_page"      => 10,
]);

$response = file_get_contents("{$baseUrl}?{$query}");
$data     = json_decode($response, true);

echo "Total results: " . count($data["results"]) . "\n\n";

foreach ($data["results"] as $article) {
    echo "- {$article['title']}\n";
    echo "  Source: {$article['source']['domain']}\n";
    echo "  URL: {$article['href']}\n\n";
}
```

### Filter by Category and Country

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"       => $apiKey,
    "category.id"   => "medtop:13000000",
    "source.country.code" => "us",
    "language.code" => "en",
    "per_page"      => 20,
    "sort.by"       => "published_at",
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
    "api_key"            => $apiKey,
    "title"              => "artificial intelligence",
    "language.code"      => "en",
    "published_at.start" => $weekAgo->format("c"),
    "published_at.end"   => $now->format("c"),
    "sort.by"            => "relevance",
    "per_page"           => 25,
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
    echo "   {$article['href']}\n\n";
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
        "api_key"       => $apiKey,
        "language.code" => "en",
        "per_page"      => $limit,
        "page"          => $page,
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

$data = fetchNews(["language.code" => "en", "per_page" => 10]);

foreach ($data["results"] as $article) {
    echo $article["title"] . "\n";
}
```
