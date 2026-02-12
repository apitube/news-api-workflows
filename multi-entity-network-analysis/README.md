# Multi-Entity Network Analysis

Workflow for mapping entity relationships through co-mention analysis, building influence networks, detecting sentiment propagation patterns, and identifying hidden connections using the [APITube News API](https://apitube.io).

## Overview

The **Multi-Entity Network Analysis** workflow constructs relationship graphs by analyzing entity co-occurrences in news articles, calculates influence scores using network centrality metrics, tracks how sentiment propagates through connected entities, and discovers non-obvious relationships. Combines graph theory, co-occurrence statistics, and temporal analysis to reveal entity ecosystems. Ideal for due diligence, competitive mapping, supply chain analysis, and political network analysis.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by organization name.                                         |
| `person.name`                 | string  | Filter by person name.                                               |
| `location.name`               | string  | Filter by location name.                                             |
| `brand.name`                  | string  | Filter by brand name.                                                |
| `title`                       | string  | Filter by keywords.                                                  |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Find co-mentioned entities with primary entity
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Tesla&source.rank.opr.min=4&language.code=en&per_page=50" | jq '.results[].entities[].name' | sort | uniq -c | sort -rn

# Analyze relationship between two entities
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=Tesla,SpaceX&language.code=en&per_page=30"

# Track sentiment in co-mentions
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Apple&sentiment.overall.polarity=negative&language.code=en&per_page=30"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class EntityNetworkAnalyzer:
    """Build and analyze entity relationship networks."""

    def __init__(self):
        self.nodes = {}  # entity -> attributes
        self.edges = defaultdict(lambda: {"weight": 0, "articles": [], "sentiment_sum": 0})
        self.articles_processed = 0

    def fetch_articles(self, entity, days=30, max_articles=200):
        """Fetch articles mentioning an entity."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        articles = []
        page = 1

        while len(articles) < max_articles:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity,
                "published_at.start": start,
                "source.rank.opr.min": 4,
                "language.code": "en",
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 50,
                "page": page,
            })
            data = resp.json()
            batch = data.get("results", [])

            if not batch:
                break

            articles.extend(batch)
            page += 1

            if len(batch) < 50:
                break

        return articles[:max_articles]

    def extract_co_occurrences(self, articles):
        """Extract entity co-occurrences from articles."""
        for article in articles:
            entities = article.get("entities", [])
            entity_names = list(set(e["name"] for e in entities if e.get("name")))
            sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)

            # Add nodes
            for entity in entities:
                name = entity.get("name")
                if name:
                    if name not in self.nodes:
                        self.nodes[name] = {
                            "type": entity.get("type", "unknown"),
                            "mention_count": 0,
                            "sentiment_sum": 0,
                        }
                    self.nodes[name]["mention_count"] += 1
                    self.nodes[name]["sentiment_sum"] += sentiment

            # Add edges (co-occurrences)
            for i, e1 in enumerate(entity_names):
                for e2 in entity_names[i+1:]:
                    edge_key = tuple(sorted([e1, e2]))
                    self.edges[edge_key]["weight"] += 1
                    self.edges[edge_key]["sentiment_sum"] += sentiment
                    self.edges[edge_key]["articles"].append(article.get("url"))

            self.articles_processed += 1

    def calculate_centrality(self):
        """Calculate degree centrality for each node."""
        centrality = defaultdict(float)

        for (e1, e2), data in self.edges.items():
            centrality[e1] += data["weight"]
            centrality[e2] += data["weight"]

        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1
        for entity in centrality:
            centrality[entity] /= max_centrality

        return dict(centrality)

    def find_bridges(self, min_weight=2):
        """Find entities that bridge different clusters."""
        # Simple bridge detection: entities connected to many other entities
        connection_counts = defaultdict(set)

        for (e1, e2), data in self.edges.items():
            if data["weight"] >= min_weight:
                connection_counts[e1].add(e2)
                connection_counts[e2].add(e1)

        bridges = []
        for entity, connections in connection_counts.items():
            if len(connections) >= 5:
                bridges.append({
                    "entity": entity,
                    "connections": len(connections),
                    "connected_to": list(connections)[:10]
                })

        return sorted(bridges, key=lambda x: x["connections"], reverse=True)

    def get_strongest_relationships(self, entity, top_n=10):
        """Get strongest relationships for an entity."""
        relationships = []

        for (e1, e2), data in self.edges.items():
            if entity in (e1, e2):
                other = e2 if e1 == entity else e1
                avg_sentiment = data["sentiment_sum"] / data["weight"] if data["weight"] else 0
                relationships.append({
                    "entity": other,
                    "co_mentions": data["weight"],
                    "avg_sentiment": avg_sentiment,
                    "relationship_type": "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
                })

        return sorted(relationships, key=lambda x: x["co_mentions"], reverse=True)[:top_n]

    def build_network(self, seed_entities, days=30, depth=1):
        """Build network starting from seed entities."""
        processed = set()
        current_level = set(seed_entities)

        for level in range(depth + 1):
            print(f"Processing level {level}: {len(current_level)} entities")

            next_level = set()
            for entity in current_level:
                if entity in processed:
                    continue

                articles = self.fetch_articles(entity, days=days, max_articles=100)
                self.extract_co_occurrences(articles)
                processed.add(entity)

                # Find new entities for next level
                for (e1, e2), data in self.edges.items():
                    if data["weight"] >= 3:
                        if e1 == entity and e2 not in processed:
                            next_level.add(e2)
                        elif e2 == entity and e1 not in processed:
                            next_level.add(e1)

            current_level = next_level

        return self

    def get_network_stats(self):
        """Get network statistics."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "articles_processed": self.articles_processed,
            "density": (2 * len(self.edges)) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0
        }

# Build network
analyzer = EntityNetworkAnalyzer()
seed_entities = ["Tesla", "SpaceX", "Elon Musk"]

print("MULTI-ENTITY NETWORK ANALYSIS")
print("=" * 60)
print(f"Seed Entities: {seed_entities}")

analyzer.build_network(seed_entities, days=30, depth=1)

stats = analyzer.get_network_stats()
print(f"\nNETWORK STATISTICS:")
print(f"  Nodes: {stats['nodes']}")
print(f"  Edges: {stats['edges']}")
print(f"  Density: {stats['density']:.4f}")

# Centrality
centrality = analyzer.calculate_centrality()
print(f"\nTOP ENTITIES BY CENTRALITY:")
for entity, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {entity}: {score:.3f}")

# Bridges
bridges = analyzer.find_bridges()
print(f"\nBRIDGE ENTITIES:")
for b in bridges[:5]:
    print(f"  {b['entity']}: connects {b['connections']} entities")

# Relationships for seed entity
print(f"\nSTRONGEST RELATIONSHIPS FOR 'Tesla':")
relationships = analyzer.get_strongest_relationships("Tesla")
for r in relationships:
    print(f"  {r['entity']}: {r['co_mentions']} co-mentions ({r['relationship_type']})")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class EntityNetworkAnalyzer {
  constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.articlesProcessed = 0;
  }

  async fetchArticles(entity, days = 30, maxArticles = 100) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];
    const articles = [];
    let page = 1;

    while (articles.length < maxArticles) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": entity,
        "published_at.start": start,
        "source.rank.opr.min": "4",
        "language.code": "en",
        per_page: "50",
        page: String(page),
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      const batch = data.results || [];

      if (batch.length === 0) break;
      articles.push(...batch);
      page++;

      if (batch.length < 50) break;
    }

    return articles.slice(0, maxArticles);
  }

  extractCoOccurrences(articles) {
    for (const article of articles) {
      const entities = article.entities || [];
      const entityNames = [...new Set(entities.map((e) => e.name).filter(Boolean))];
      const sentiment = article.sentiment?.overall?.score || 0;

      // Add nodes
      for (const entity of entities) {
        if (entity.name) {
          if (!this.nodes.has(entity.name)) {
            this.nodes.set(entity.name, {
              type: entity.type || "unknown",
              mentionCount: 0,
              sentimentSum: 0,
            });
          }
          const node = this.nodes.get(entity.name);
          node.mentionCount++;
          node.sentimentSum += sentiment;
        }
      }

      // Add edges
      for (let i = 0; i < entityNames.length; i++) {
        for (let j = i + 1; j < entityNames.length; j++) {
          const edgeKey = [entityNames[i], entityNames[j]].sort().join("|||");

          if (!this.edges.has(edgeKey)) {
            this.edges.set(edgeKey, { weight: 0, sentimentSum: 0 });
          }
          const edge = this.edges.get(edgeKey);
          edge.weight++;
          edge.sentimentSum += sentiment;
        }
      }

      this.articlesProcessed++;
    }
  }

  calculateCentrality() {
    const centrality = new Map();

    for (const [edgeKey, data] of this.edges) {
      const [e1, e2] = edgeKey.split("|||");
      centrality.set(e1, (centrality.get(e1) || 0) + data.weight);
      centrality.set(e2, (centrality.get(e2) || 0) + data.weight);
    }

    const maxCentrality = Math.max(...centrality.values()) || 1;
    for (const [entity, value] of centrality) {
      centrality.set(entity, value / maxCentrality);
    }

    return centrality;
  }

  getStrongestRelationships(entity, topN = 10) {
    const relationships = [];

    for (const [edgeKey, data] of this.edges) {
      const [e1, e2] = edgeKey.split("|||");
      if (e1 === entity || e2 === entity) {
        const other = e1 === entity ? e2 : e1;
        const avgSentiment = data.weight > 0 ? data.sentimentSum / data.weight : 0;
        relationships.push({
          entity: other,
          coMentions: data.weight,
          avgSentiment,
          relationshipType:
            avgSentiment > 0.1 ? "positive" : avgSentiment < -0.1 ? "negative" : "neutral",
        });
      }
    }

    return relationships.sort((a, b) => b.coMentions - a.coMentions).slice(0, topN);
  }

  async buildNetwork(seedEntities, days = 30) {
    for (const entity of seedEntities) {
      console.log(`Fetching articles for ${entity}...`);
      const articles = await this.fetchArticles(entity, days, 100);
      this.extractCoOccurrences(articles);
    }
    return this;
  }

  getStats() {
    return {
      nodes: this.nodes.size,
      edges: this.edges.size,
      articlesProcessed: this.articlesProcessed,
    };
  }
}

async function runAnalysis() {
  const analyzer = new EntityNetworkAnalyzer();
  const seeds = ["Tesla", "SpaceX"];

  console.log("MULTI-ENTITY NETWORK ANALYSIS");
  console.log("=".repeat(50));

  await analyzer.buildNetwork(seeds, 30);

  const stats = analyzer.getStats();
  console.log(`\nNodes: ${stats.nodes}`);
  console.log(`Edges: ${stats.edges}`);
  console.log(`Articles: ${stats.articlesProcessed}`);

  const centrality = analyzer.calculateCentrality();
  console.log("\nTop entities by centrality:");
  [...centrality.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([entity, score]) => {
      console.log(`  ${entity}: ${score.toFixed(3)}`);
    });

  console.log("\nStrongest relationships for Tesla:");
  analyzer.getStrongestRelationships("Tesla").forEach((r) => {
    console.log(`  ${r.entity}: ${r.coMentions} (${r.relationshipType})`);
  });
}

runAnalysis();
```

### PHP

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class EntityNetworkAnalyzer
{
    private string $apiKey;
    private string $baseUrl;
    private array $nodes = [];
    private array $edges = [];
    private int $articlesProcessed = 0;

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function fetchArticles(string $entity, int $days = 30, int $maxArticles = 100): array
    {
        $start = (new DateTime("-{$days} days"))->format("Y-m-d");
        $articles = [];
        $page = 1;

        while (count($articles) < $maxArticles) {
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "organization.name" => $entity,
                "published_at.start" => $start,
                "source.rank.opr.min" => 4,
                "language.code" => "en",
                "per_page" => 50,
                "page" => $page,
            ]);

            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $batch = $data["results"] ?? [];

            if (empty($batch)) break;
            $articles = array_merge($articles, $batch);
            $page++;

            if (count($batch) < 50) break;
        }

        return array_slice($articles, 0, $maxArticles);
    }

    public function extractCoOccurrences(array $articles): void
    {
        foreach ($articles as $article) {
            $entities = $article["entities"] ?? [];
            $entityNames = array_unique(array_filter(array_column($entities, "name")));
            $sentiment = $article["sentiment"]["overall"]["score"] ?? 0;

            foreach ($entities as $entity) {
                $name = $entity["name"] ?? null;
                if ($name) {
                    if (!isset($this->nodes[$name])) {
                        $this->nodes[$name] = [
                            "type" => $entity["type"] ?? "unknown",
                            "mention_count" => 0,
                            "sentiment_sum" => 0,
                        ];
                    }
                    $this->nodes[$name]["mention_count"]++;
                    $this->nodes[$name]["sentiment_sum"] += $sentiment;
                }
            }

            $entityNames = array_values($entityNames);
            for ($i = 0; $i < count($entityNames); $i++) {
                for ($j = $i + 1; $j < count($entityNames); $j++) {
                    $pair = [$entityNames[$i], $entityNames[$j]];
                    sort($pair);
                    $edgeKey = implode("|||", $pair);

                    if (!isset($this->edges[$edgeKey])) {
                        $this->edges[$edgeKey] = ["weight" => 0, "sentiment_sum" => 0];
                    }
                    $this->edges[$edgeKey]["weight"]++;
                    $this->edges[$edgeKey]["sentiment_sum"] += $sentiment;
                }
            }

            $this->articlesProcessed++;
        }
    }

    public function calculateCentrality(): array
    {
        $centrality = [];

        foreach ($this->edges as $edgeKey => $data) {
            [$e1, $e2] = explode("|||", $edgeKey);
            $centrality[$e1] = ($centrality[$e1] ?? 0) + $data["weight"];
            $centrality[$e2] = ($centrality[$e2] ?? 0) + $data["weight"];
        }

        $max = max($centrality) ?: 1;
        return array_map(fn($v) => $v / $max, $centrality);
    }

    public function getStrongestRelationships(string $entity, int $topN = 10): array
    {
        $relationships = [];

        foreach ($this->edges as $edgeKey => $data) {
            [$e1, $e2] = explode("|||", $edgeKey);
            if ($e1 === $entity || $e2 === $entity) {
                $other = $e1 === $entity ? $e2 : $e1;
                $avgSentiment = $data["weight"] > 0 ? $data["sentiment_sum"] / $data["weight"] : 0;
                $relationships[] = [
                    "entity" => $other,
                    "co_mentions" => $data["weight"],
                    "avg_sentiment" => $avgSentiment,
                ];
            }
        }

        usort($relationships, fn($a, $b) => $b["co_mentions"] <=> $a["co_mentions"]);
        return array_slice($relationships, 0, $topN);
    }

    public function buildNetwork(array $seedEntities, int $days = 30): self
    {
        foreach ($seedEntities as $entity) {
            echo "Fetching articles for {$entity}...\n";
            $articles = $this->fetchArticles($entity, $days, 100);
            $this->extractCoOccurrences($articles);
        }
        return $this;
    }

    public function getStats(): array
    {
        return [
            "nodes" => count($this->nodes),
            "edges" => count($this->edges),
            "articles_processed" => $this->articlesProcessed,
        ];
    }
}

$analyzer = new EntityNetworkAnalyzer();
$seeds = ["Tesla", "SpaceX"];

echo "MULTI-ENTITY NETWORK ANALYSIS\n";
echo str_repeat("=", 50) . "\n";

$analyzer->buildNetwork($seeds, 30);

$stats = $analyzer->getStats();
echo "\nNodes: {$stats['nodes']}\n";
echo "Edges: {$stats['edges']}\n";
echo "Articles: {$stats['articles_processed']}\n";

$centrality = $analyzer->calculateCentrality();
arsort($centrality);
echo "\nTop entities by centrality:\n";
foreach (array_slice($centrality, 0, 10, true) as $entity => $score) {
    printf("  %s: %.3f\n", $entity, $score);
}

echo "\nStrongest relationships for Tesla:\n";
foreach ($analyzer->getStrongestRelationships("Tesla") as $r) {
    printf("  %s: %d co-mentions\n", $r["entity"], $r["co_mentions"]);
}
```

## Common Use Cases

- **Due diligence** — map corporate relationships before M&A.
- **Supply chain mapping** — discover supplier and partner networks.
- **Competitive intelligence** — understand competitor ecosystems.
- **Political analysis** — map relationships between political figures.
- **Influence tracking** — identify key influencers in topics.
- **Risk propagation** — predict how issues spread through networks.
- **Partnership discovery** — find potential partners through shared connections.
- **Investor relations** — understand shareholder and analyst networks.

## See Also

- [examples.md](./examples.md) — detailed code examples for multi-entity network analysis.
