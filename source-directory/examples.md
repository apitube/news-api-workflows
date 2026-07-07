# Source Directory — Code Examples

Detailed examples for resolving publisher names into source IDs with `/v1/suggest/sources` and pivoting to their articles with `/v1/news/everything`.

> `/v1/suggest/sources` is a **prefix search** and returns a **flat array** (not wrapped in `results`). There is no publisher-listing or per-source profile/coverage endpoint — to measure volume, pivot with `source.id` to `/v1/news/everything` or `/v1/news/count`.

## Python

### Search Sources by Name or Domain Prefix

```python
import requests

SUGGEST_URL = "https://api.apitube.io/v1/suggest/sources"


def find_sources(prefix, api_key):
    response = requests.get(SUGGEST_URL, params={"api_key": api_key, "prefix": prefix})
    response.raise_for_status()
    return response.json()  # flat array of source objects


for source in find_sources("guardian", "YOUR_API_KEY"):
    print(f"{source['id']:>8}  {source['name']:<30} {source['domain']:<25} bias={source.get('bias')}")
```

### Pivot from a Source to Its Latest Articles

```python
import requests


def latest_articles_for_source(source_id, api_key, per_page=10):
    response = requests.get(
        "https://api.apitube.io/v1/news/everything",
        params={"api_key": api_key, "source.id": source_id, "per_page": per_page},
    )
    response.raise_for_status()
    return response.json()["results"]


# Resolve a name to a source, then load its articles
sources = find_sources("guardian", "YOUR_API_KEY")
if sources:
    source_id = sources[0]["id"]
    for article in latest_articles_for_source(source_id, "YOUR_API_KEY"):
        print(f"  {article['published_at'][:10]}  {article['title']}")
        print(f"    {article['href']}")
```

### Count a Publisher's Articles (volume)

```python
import requests


def source_volume(source_id, api_key, start="NOW-30DAYS"):
    response = requests.get(
        "https://api.apitube.io/v1/news/count",
        params={"api_key": api_key, "source.id": source_id, "published_at.start": start},
    )
    response.raise_for_status()
    return response.json()["count"]


print(source_volume(4232, "YOUR_API_KEY"))
```

## JavaScript (Node.js)

### Search Sources by Name or Domain Prefix

```javascript
async function findSources(prefix, apiKey) {
  const params = new URLSearchParams({ api_key: apiKey, prefix });
  const response = await fetch(`https://api.apitube.io/v1/suggest/sources?${params}`);
  return response.json(); // flat array of source objects
}

const sources = await findSources("guardian", "YOUR_API_KEY");
sources.forEach((source) => {
  console.log(`${source.id}  ${source.name} (${source.domain}) bias=${source.bias}`);
});
```

### Pivot from a Source to Its Latest Articles

```javascript
async function latestArticlesForSource(sourceId, apiKey, perPage = 10) {
  const params = new URLSearchParams({
    api_key: apiKey,
    "source.id": sourceId,
    per_page: String(perPage),
  });
  const response = await fetch(`https://api.apitube.io/v1/news/everything?${params}`);
  const data = await response.json();
  return data.results;
}

const sources = await findSources("guardian", "YOUR_API_KEY");
if (sources.length) {
  const articles = await latestArticlesForSource(sources[0].id, "YOUR_API_KEY");
  articles.forEach((a) => console.log(`  ${a.published_at.slice(0, 10)}  ${a.title}\n    ${a.href}`));
}
```

## PHP

### Search Sources by Name or Domain Prefix

```php
function findSources(string $prefix, string $apiKey): array {
    $query = http_build_query(["api_key" => $apiKey, "prefix" => $prefix]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/suggest/sources?{$query}"
    ), true); // flat array of source objects
}

foreach (findSources("guardian", "YOUR_API_KEY") as $source) {
    printf("%8d  %s (%s) bias=%s\n",
        $source["id"], $source["name"], $source["domain"], $source["bias"] ?? "");
}
```

### Pivot from a Source to Its Latest Articles

```php
function latestArticlesForSource(int $sourceId, string $apiKey, int $perPage = 10): array {
    $query = http_build_query([
        "api_key"   => $apiKey,
        "source.id" => $sourceId,
        "per_page"  => $perPage,
    ]);
    $data = json_decode(file_get_contents(
        "https://api.apitube.io/v1/news/everything?{$query}"
    ), true);
    return $data["results"];
}

$sources = findSources("guardian", "YOUR_API_KEY");
if ($sources) {
    foreach (latestArticlesForSource($sources[0]["id"], "YOUR_API_KEY") as $article) {
        printf("  %s  %s\n    %s\n",
            substr($article["published_at"], 0, 10), $article["title"], $article["href"]);
    }
}
```

## Notes

- `prefix` is **required** on `/v1/suggest/sources`; omitting it returns `400 ER0346`.
- The suggest response is a flat array; each item has `id`, `name`, `domain`, `type`, `bias`, and `links.self` (a `/v1/news/everything?source.id=…` URL).
- Use `source.id` (exact) or `source.domain` on `/v1/news/everything` and `/v1/news/count` to scope by publisher.
