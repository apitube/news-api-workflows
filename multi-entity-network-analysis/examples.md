# Multi-Entity Network Analysis — Examples

Advanced code examples for entity relationship mapping, influence networks, sentiment propagation, and hidden connection discovery.

---

## Python — Advanced Network Graph Analysis

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json
import math

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class AdvancedNetworkAnalyzer:
    """
    Comprehensive entity network analysis with graph algorithms,
    community detection, and influence propagation modeling.
    """

    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(lambda: {
            "weight": 0,
            "sentiment_sum": 0,
            "articles": [],
            "first_seen": None,
            "last_seen": None
        })
        self.article_cache = []

    def fetch_articles_batch(self, entities, days=30, max_per_entity=150):
        """Fetch articles for multiple entities."""
        all_articles = []

        for entity in entities:
            start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            page = 1
            entity_articles = []

            while len(entity_articles) < max_per_entity:
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
                batch = resp.json().get("results", [])

                if not batch:
                    break

                entity_articles.extend(batch)
                page += 1

                if len(batch) < 50:
                    break

            all_articles.extend(entity_articles[:max_per_entity])

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        self.article_cache = unique_articles
        return unique_articles

    def build_graph(self, articles):
        """Build graph from articles."""
        for article in articles:
            entities = article.get("entities", [])
            entity_names = list(set(e["name"] for e in entities if e.get("name")))
            sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)
            pub_date = article.get("published_at", "")

            # Add nodes
            for entity in entities:
                name = entity.get("name")
                if not name:
                    continue

                if name not in self.nodes:
                    self.nodes[name] = {
                        "type": entity.get("type", "unknown"),
                        "mention_count": 0,
                        "sentiment_sum": 0,
                        "first_seen": pub_date,
                        "last_seen": pub_date,
                        "articles": []
                    }

                node = self.nodes[name]
                node["mention_count"] += 1
                node["sentiment_sum"] += sentiment
                node["articles"].append(article.get("url"))
                if pub_date < node["first_seen"]:
                    node["first_seen"] = pub_date
                if pub_date > node["last_seen"]:
                    node["last_seen"] = pub_date

            # Add edges
            for i, e1 in enumerate(entity_names):
                for e2 in entity_names[i+1:]:
                    edge_key = tuple(sorted([e1, e2]))
                    edge = self.edges[edge_key]
                    edge["weight"] += 1
                    edge["sentiment_sum"] += sentiment
                    edge["articles"].append(article.get("url"))

                    if edge["first_seen"] is None or pub_date < edge["first_seen"]:
                        edge["first_seen"] = pub_date
                    if edge["last_seen"] is None or pub_date > edge["last_seen"]:
                        edge["last_seen"] = pub_date

    def calculate_degree_centrality(self):
        """Calculate degree centrality."""
        degree = defaultdict(int)

        for (e1, e2), data in self.edges.items():
            degree[e1] += data["weight"]
            degree[e2] += data["weight"]

        max_degree = max(degree.values()) if degree else 1
        return {e: d / max_degree for e, d in degree.items()}

    def calculate_betweenness_centrality(self, sample_size=50):
        """
        Approximate betweenness centrality using sampled shortest paths.
        """
        # Build adjacency list
        adj = defaultdict(set)
        for (e1, e2), data in self.edges.items():
            if data["weight"] >= 2:  # Filter weak edges
                adj[e1].add(e2)
                adj[e2].add(e1)

        nodes = list(self.nodes.keys())
        betweenness = defaultdict(float)

        # Sample node pairs
        import random
        sample_pairs = []
        for _ in range(min(sample_size, len(nodes) * (len(nodes) - 1) // 2)):
            pair = tuple(random.sample(nodes, 2))
            sample_pairs.append(pair)

        # BFS for shortest paths
        def bfs_paths(start, end):
            if start == end:
                return [[start]]

            visited = {start}
            queue = [[start]]
            paths = []

            while queue:
                path = queue.pop(0)
                node = path[-1]

                if node == end:
                    paths.append(path)
                    continue

                for neighbor in adj.get(node, []):
                    if neighbor not in visited or neighbor == end:
                        new_path = path + [neighbor]
                        if neighbor == end:
                            paths.append(new_path)
                        else:
                            visited.add(neighbor)
                            queue.append(new_path)

            return paths

        # Count path traversals
        for start, end in sample_pairs:
            paths = bfs_paths(start, end)
            if paths:
                for path in paths:
                    for node in path[1:-1]:  # Exclude start and end
                        betweenness[node] += 1 / len(paths)

        # Normalize
        max_betweenness = max(betweenness.values()) if betweenness else 1
        return {e: b / max_betweenness for e, b in betweenness.items()}

    def calculate_pagerank(self, damping=0.85, iterations=50):
        """Calculate PageRank centrality."""
        nodes = list(self.nodes.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Build weighted adjacency
        out_weights = defaultdict(float)
        for (e1, e2), data in self.edges.items():
            out_weights[e1] += data["weight"]
            out_weights[e2] += data["weight"]

        # Initialize PageRank
        pagerank = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_pr = {}
            for node in nodes:
                rank_sum = 0
                for (e1, e2), data in self.edges.items():
                    if e1 == node:
                        rank_sum += pagerank[e2] * data["weight"] / out_weights[e2]
                    elif e2 == node:
                        rank_sum += pagerank[e1] * data["weight"] / out_weights[e1]

                new_pr[node] = (1 - damping) / n + damping * rank_sum

            pagerank = new_pr

        return pagerank

    def detect_communities(self, resolution=1.0):
        """
        Simple community detection using label propagation.
        """
        # Initialize each node with its own community
        communities = {node: i for i, node in enumerate(self.nodes.keys())}

        # Build adjacency with weights
        adj = defaultdict(dict)
        for (e1, e2), data in self.edges.items():
            adj[e1][e2] = data["weight"]
            adj[e2][e1] = data["weight"]

        # Iterate until convergence
        changed = True
        max_iterations = 20
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            for node in self.nodes.keys():
                if node not in adj:
                    continue

                # Count community weights
                community_weights = defaultdict(float)
                for neighbor, weight in adj[node].items():
                    community_weights[communities[neighbor]] += weight

                if community_weights:
                    best_community = max(community_weights.items(), key=lambda x: x[1])[0]
                    if communities[node] != best_community:
                        communities[node] = best_community
                        changed = True

            iteration += 1

        # Group by community
        community_members = defaultdict(list)
        for node, comm_id in communities.items():
            community_members[comm_id].append(node)

        return dict(community_members)

    def find_sentiment_propagation(self, source_entity, days=14):
        """Track how sentiment propagates from source entity."""
        # Filter recent articles mentioning source
        source_articles = [
            a for a in self.article_cache
            if any(e.get("name") == source_entity for e in a.get("entities", []))
        ]

        # Track sentiment by connected entities over time
        propagation = defaultdict(lambda: {"dates": [], "sentiments": []})

        for article in source_articles:
            pub_date = article.get("published_at", "")[:10]
            sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)

            for entity in article.get("entities", []):
                name = entity.get("name")
                if name and name != source_entity:
                    propagation[name]["dates"].append(pub_date)
                    propagation[name]["sentiments"].append(sentiment)

        # Calculate propagation metrics
        results = []
        for entity, data in propagation.items():
            if len(data["sentiments"]) < 2:
                continue

            avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
            results.append({
                "entity": entity,
                "co_mentions": len(data["sentiments"]),
                "avg_sentiment": avg_sentiment,
                "first_date": min(data["dates"]),
                "last_date": max(data["dates"])
            })

        return sorted(results, key=lambda x: x["co_mentions"], reverse=True)

    def find_hidden_connections(self, entity1, entity2, max_hops=3):
        """Find indirect connections between two entities."""
        # Build adjacency
        adj = defaultdict(set)
        for (e1, e2), data in self.edges.items():
            if data["weight"] >= 2:
                adj[e1].add(e2)
                adj[e2].add(e1)

        # BFS for paths
        def find_paths(start, end, max_depth):
            if start == end:
                return [[start]]

            paths = []
            queue = [(start, [start])]

            while queue:
                node, path = queue.pop(0)

                if len(path) > max_depth:
                    continue

                for neighbor in adj.get(node, []):
                    if neighbor == end:
                        paths.append(path + [neighbor])
                    elif neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

            return paths

        paths = find_paths(entity1, entity2, max_hops)

        # Analyze paths
        analyzed_paths = []
        for path in paths:
            path_strength = 1.0
            for i in range(len(path) - 1):
                edge_key = tuple(sorted([path[i], path[i+1]]))
                weight = self.edges[edge_key]["weight"]
                path_strength *= weight

            analyzed_paths.append({
                "path": path,
                "hops": len(path) - 1,
                "strength": path_strength
            })

        return sorted(analyzed_paths, key=lambda x: x["strength"], reverse=True)

    def export_for_visualization(self):
        """Export network in format suitable for visualization."""
        nodes_export = []
        for name, data in self.nodes.items():
            nodes_export.append({
                "id": name,
                "type": data["type"],
                "size": data["mention_count"],
                "sentiment": data["sentiment_sum"] / max(data["mention_count"], 1)
            })

        edges_export = []
        for (e1, e2), data in self.edges.items():
            if data["weight"] >= 2:
                edges_export.append({
                    "source": e1,
                    "target": e2,
                    "weight": data["weight"],
                    "sentiment": data["sentiment_sum"] / max(data["weight"], 1)
                })

        return {"nodes": nodes_export, "edges": edges_export}


class InfluenceScorer:
    """Calculate multi-dimensional influence scores."""

    def __init__(self, network):
        self.network = network

    def calculate_influence_score(self, entity):
        """Calculate comprehensive influence score."""
        if entity not in self.network.nodes:
            return None

        node = self.network.nodes[entity]

        # Component scores
        degree = self.network.calculate_degree_centrality().get(entity, 0)
        pagerank = self.network.calculate_pagerank().get(entity, 0)
        betweenness = self.network.calculate_betweenness_centrality().get(entity, 0)

        # Volume component
        total_mentions = sum(n["mention_count"] for n in self.network.nodes.values())
        volume_score = node["mention_count"] / total_mentions if total_mentions else 0

        # Sentiment component (normalized to 0-1)
        avg_sentiment = node["sentiment_sum"] / max(node["mention_count"], 1)
        sentiment_score = (avg_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]

        # Composite score (weighted)
        influence_score = (
            degree * 0.25 +
            pagerank * 0.30 +
            betweenness * 0.20 +
            volume_score * 0.15 +
            sentiment_score * 0.10
        )

        return {
            "entity": entity,
            "influence_score": influence_score,
            "components": {
                "degree_centrality": degree,
                "pagerank": pagerank,
                "betweenness": betweenness,
                "volume_share": volume_score,
                "sentiment": sentiment_score
            }
        }

    def rank_all_entities(self):
        """Rank all entities by influence."""
        scores = []
        for entity in self.network.nodes.keys():
            score = self.calculate_influence_score(entity)
            if score:
                scores.append(score)

        return sorted(scores, key=lambda x: x["influence_score"], reverse=True)


# Run comprehensive analysis
print("ADVANCED NETWORK GRAPH ANALYSIS")
print("=" * 70)

network = AdvancedNetworkAnalyzer()

# Seed entities
seeds = ["Tesla", "SpaceX", "Elon Musk", "NVIDIA", "OpenAI"]
print(f"Seed entities: {seeds}")

print("\nFetching articles...")
articles = network.fetch_articles_batch(seeds, days=30, max_per_entity=100)
print(f"Total unique articles: {len(articles)}")

print("\nBuilding graph...")
network.build_graph(articles)
print(f"Nodes: {len(network.nodes)}")
print(f"Edges: {len(network.edges)}")

# Centrality analysis
print("\n" + "=" * 70)
print("CENTRALITY ANALYSIS")
print("-" * 40)

degree = network.calculate_degree_centrality()
pagerank = network.calculate_pagerank()
betweenness = network.calculate_betweenness_centrality()

print("\nTop by Degree Centrality:")
for e, s in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {e}: {s:.3f}")

print("\nTop by PageRank:")
for e, s in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {e}: {s:.4f}")

print("\nTop by Betweenness:")
for e, s in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {e}: {s:.3f}")

# Community detection
print("\n" + "=" * 70)
print("COMMUNITY DETECTION")
print("-" * 40)

communities = network.detect_communities()
print(f"Communities found: {len(communities)}")

for comm_id, members in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
    print(f"\nCommunity {comm_id} ({len(members)} members):")
    print(f"  Members: {', '.join(members[:10])}")

# Influence scoring
print("\n" + "=" * 70)
print("INFLUENCE RANKING")
print("-" * 40)

scorer = InfluenceScorer(network)
rankings = scorer.rank_all_entities()

print("\nTop 15 Most Influential Entities:")
for rank in rankings[:15]:
    print(f"  {rank['entity']}: {rank['influence_score']:.4f}")
    print(f"    Degree: {rank['components']['degree_centrality']:.3f} | "
          f"PR: {rank['components']['pagerank']:.4f} | "
          f"Between: {rank['components']['betweenness']:.3f}")

# Hidden connections
print("\n" + "=" * 70)
print("HIDDEN CONNECTION ANALYSIS")
print("-" * 40)

if "Tesla" in network.nodes and "Microsoft" in network.nodes:
    paths = network.find_hidden_connections("Tesla", "Microsoft", max_hops=3)
    print("\nPaths between Tesla and Microsoft:")
    for p in paths[:5]:
        print(f"  {' → '.join(p['path'])} (strength: {p['strength']:.0f})")

# Sentiment propagation
print("\n" + "=" * 70)
print("SENTIMENT PROPAGATION FROM 'Tesla'")
print("-" * 40)

propagation = network.find_sentiment_propagation("Tesla")
for p in propagation[:10]:
    sentiment_label = "positive" if p["avg_sentiment"] > 0.1 else "negative" if p["avg_sentiment"] < -0.1 else "neutral"
    print(f"  {p['entity']}: {p['co_mentions']} co-mentions ({sentiment_label})")

# Export for visualization
print("\n" + "=" * 70)
export_data = network.export_for_visualization()
print(f"Export ready: {len(export_data['nodes'])} nodes, {len(export_data['edges'])} edges")
```

---

## JavaScript — Real-Time Network Monitoring

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class NetworkMonitor {
  constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.snapshots = [];
  }

  async fetchRecentArticles(entities, hours = 24) {
    const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
    const allArticles = [];

    for (const entity of entities) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": entity,
        "published_at.start": start,
        "source.rank.opr.min": "4",
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        per_page: "50",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      allArticles.push(...(data.results || []));
    }

    // Deduplicate
    const seen = new Set();
    return allArticles.filter((a) => {
      if (seen.has(a.url)) return false;
      seen.add(a.url);
      return true;
    });
  }

  updateGraph(articles) {
    for (const article of articles) {
      const entities = article.entities || [];
      const names = [...new Set(entities.map((e) => e.name).filter(Boolean))];
      const sentiment = article.sentiment?.overall?.score || 0;
      const pubDate = article.published_at || new Date().toISOString();

      // Update nodes
      for (const entity of entities) {
        if (!entity.name) continue;

        if (!this.nodes.has(entity.name)) {
          this.nodes.set(entity.name, {
            type: entity.type || "unknown",
            mentions: 0,
            sentimentSum: 0,
            recentMentions: [],
          });
        }

        const node = this.nodes.get(entity.name);
        node.mentions++;
        node.sentimentSum += sentiment;
        node.recentMentions.push({ date: pubDate, sentiment });

        // Keep only last 100 mentions
        if (node.recentMentions.length > 100) {
          node.recentMentions = node.recentMentions.slice(-100);
        }
      }

      // Update edges
      for (let i = 0; i < names.length; i++) {
        for (let j = i + 1; j < names.length; j++) {
          const key = [names[i], names[j]].sort().join("|||");

          if (!this.edges.has(key)) {
            this.edges.set(key, {
              weight: 0,
              sentimentSum: 0,
              recentArticles: [],
            });
          }

          const edge = this.edges.get(key);
          edge.weight++;
          edge.sentimentSum += sentiment;
          edge.recentArticles.push({ date: pubDate, url: article.url });

          if (edge.recentArticles.length > 50) {
            edge.recentArticles = edge.recentArticles.slice(-50);
          }
        }
      }
    }
  }

  takeSnapshot() {
    const snapshot = {
      timestamp: new Date().toISOString(),
      nodeCount: this.nodes.size,
      edgeCount: this.edges.size,
      topNodes: this.getTopNodes(10),
      strongestEdges: this.getStrongestEdges(10),
    };

    this.snapshots.push(snapshot);

    // Keep last 24 snapshots
    if (this.snapshots.length > 24) {
      this.snapshots = this.snapshots.slice(-24);
    }

    return snapshot;
  }

  getTopNodes(n = 10) {
    return [...this.nodes.entries()]
      .map(([name, data]) => ({
        name,
        mentions: data.mentions,
        avgSentiment: data.sentimentSum / Math.max(data.mentions, 1),
      }))
      .sort((a, b) => b.mentions - a.mentions)
      .slice(0, n);
  }

  getStrongestEdges(n = 10) {
    return [...this.edges.entries()]
      .map(([key, data]) => {
        const [e1, e2] = key.split("|||");
        return {
          entities: [e1, e2],
          weight: data.weight,
          avgSentiment: data.sentimentSum / Math.max(data.weight, 1),
        };
      })
      .sort((a, b) => b.weight - a.weight)
      .slice(0, n);
  }

  detectNetworkChanges() {
    if (this.snapshots.length < 2) return null;

    const current = this.snapshots[this.snapshots.length - 1];
    const previous = this.snapshots[this.snapshots.length - 2];

    const changes = {
      nodeGrowth: current.nodeCount - previous.nodeCount,
      edgeGrowth: current.edgeCount - previous.edgeCount,
      newTopEntities: [],
      emergingRelationships: [],
    };

    // Find new top entities
    const prevTopNames = new Set(previous.topNodes.map((n) => n.name));
    changes.newTopEntities = current.topNodes
      .filter((n) => !prevTopNames.has(n.name))
      .map((n) => n.name);

    // Find emerging relationships
    const prevEdgeSet = new Set(
      previous.strongestEdges.map((e) => e.entities.sort().join("|||"))
    );
    changes.emergingRelationships = current.strongestEdges
      .filter((e) => !prevEdgeSet.has(e.entities.sort().join("|||")))
      .map((e) => e.entities);

    return changes;
  }

  calculateInfluenceScore(entity) {
    if (!this.nodes.has(entity)) return null;

    const node = this.nodes.get(entity);
    const totalMentions = [...this.nodes.values()].reduce(
      (sum, n) => sum + n.mentions,
      0
    );

    // Degree (number of connections)
    let degree = 0;
    for (const [key] of this.edges) {
      if (key.includes(entity)) degree++;
    }

    const maxDegree = Math.max(
      ...[...this.nodes.keys()].map((e) => {
        let d = 0;
        for (const [key] of this.edges) {
          if (key.includes(e)) d++;
        }
        return d;
      })
    );

    const volumeScore = node.mentions / Math.max(totalMentions, 1);
    const degreeScore = degree / Math.max(maxDegree, 1);
    const sentimentScore = (node.sentimentSum / Math.max(node.mentions, 1) + 1) / 2;

    return {
      entity,
      score: volumeScore * 0.4 + degreeScore * 0.4 + sentimentScore * 0.2,
      components: { volumeScore, degreeScore, sentimentScore },
    };
  }

  async startMonitoring(entities, intervalMinutes = 60) {
    console.log(`Starting network monitoring for: ${entities.join(", ")}`);
    console.log(`Update interval: ${intervalMinutes} minutes\n`);

    const update = async () => {
      console.log(`[${new Date().toISOString()}] Fetching updates...`);

      const articles = await this.fetchRecentArticles(entities, intervalMinutes / 60 * 2);
      console.log(`Found ${articles.length} articles`);

      this.updateGraph(articles);
      const snapshot = this.takeSnapshot();

      console.log(`Network: ${snapshot.nodeCount} nodes, ${snapshot.edgeCount} edges`);

      const changes = this.detectNetworkChanges();
      if (changes) {
        console.log(`Changes: +${changes.nodeGrowth} nodes, +${changes.edgeGrowth} edges`);
        if (changes.newTopEntities.length > 0) {
          console.log(`New top entities: ${changes.newTopEntities.join(", ")}`);
        }
        if (changes.emergingRelationships.length > 0) {
          console.log(`Emerging relationships:`);
          changes.emergingRelationships.forEach((r) => {
            console.log(`  ${r[0]} <-> ${r[1]}`);
          });
        }
      }

      console.log("\nTop Entities:");
      snapshot.topNodes.slice(0, 5).forEach((n) => {
        console.log(`  ${n.name}: ${n.mentions} mentions`);
      });

      console.log("\n" + "-".repeat(50) + "\n");
    };

    // Initial update
    await update();

    // Schedule periodic updates (in production)
    // setInterval(update, intervalMinutes * 60 * 1000);
  }
}

// Community detector
class CommunityDetector {
  constructor(nodes, edges) {
    this.nodes = nodes;
    this.edges = edges;
  }

  buildAdjacency() {
    const adj = new Map();

    for (const [key, data] of this.edges) {
      if (data.weight < 2) continue;

      const [e1, e2] = key.split("|||");

      if (!adj.has(e1)) adj.set(e1, new Map());
      if (!adj.has(e2)) adj.set(e2, new Map());

      adj.get(e1).set(e2, data.weight);
      adj.get(e2).set(e1, data.weight);
    }

    return adj;
  }

  detectCommunities() {
    const adj = this.buildAdjacency();
    const communities = new Map();

    // Initialize each node in its own community
    let communityId = 0;
    for (const node of this.nodes.keys()) {
      communities.set(node, communityId++);
    }

    // Label propagation
    let changed = true;
    let iterations = 0;
    const maxIterations = 20;

    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;

      for (const node of this.nodes.keys()) {
        const neighbors = adj.get(node);
        if (!neighbors || neighbors.size === 0) continue;

        // Count community weights
        const communityWeights = new Map();
        for (const [neighbor, weight] of neighbors) {
          const comm = communities.get(neighbor);
          communityWeights.set(
            comm,
            (communityWeights.get(comm) || 0) + weight
          );
        }

        // Find best community
        let bestComm = communities.get(node);
        let bestWeight = 0;

        for (const [comm, weight] of communityWeights) {
          if (weight > bestWeight) {
            bestWeight = weight;
            bestComm = comm;
          }
        }

        if (communities.get(node) !== bestComm) {
          communities.set(node, bestComm);
          changed = true;
        }
      }
    }

    // Group by community
    const result = new Map();
    for (const [node, comm] of communities) {
      if (!result.has(comm)) result.set(comm, []);
      result.get(comm).push(node);
    }

    return result;
  }
}

// Run monitoring
async function main() {
  const monitor = new NetworkMonitor();
  const watchEntities = ["Tesla", "SpaceX", "NVIDIA", "OpenAI", "Apple"];

  await monitor.startMonitoring(watchEntities, 60);

  // Community detection
  console.log("\nCOMMUNITY DETECTION");
  console.log("=".repeat(50));

  const detector = new CommunityDetector(monitor.nodes, monitor.edges);
  const communities = detector.detectCommunities();

  console.log(`Found ${communities.size} communities\n`);

  for (const [commId, members] of [...communities.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 5)) {
    console.log(`Community ${commId} (${members.length} members):`);
    console.log(`  ${members.slice(0, 8).join(", ")}`);
  }

  // Influence scores
  console.log("\nINFLUENCE SCORES");
  console.log("=".repeat(50));

  const scores = watchEntities
    .map((e) => monitor.calculateInfluenceScore(e))
    .filter(Boolean)
    .sort((a, b) => b.score - a.score);

  scores.forEach((s) => {
    console.log(`${s.entity}: ${s.score.toFixed(4)}`);
    console.log(
      `  Volume: ${s.components.volumeScore.toFixed(3)} | ` +
        `Degree: ${s.components.degreeScore.toFixed(3)} | ` +
        `Sentiment: ${s.components.sentimentScore.toFixed(3)}`
    );
  });
}

main();
```

---

## PHP — Enterprise Network Analysis Service

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class EnterpriseNetworkAnalyzer
{
    private string $apiKey;
    private string $baseUrl;
    private array $nodes = [];
    private array $edges = [];
    private array $articleCache = [];

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function fetchArticles(array $entities, int $days = 30, int $maxPerEntity = 100): array
    {
        $allArticles = [];
        $start = (new DateTime("-{$days} days"))->format("Y-m-d");

        foreach ($entities as $entity) {
            $page = 1;
            $entityArticles = [];

            while (count($entityArticles) < $maxPerEntity) {
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
                $entityArticles = array_merge($entityArticles, $batch);
                $page++;

                if (count($batch) < 50) break;
            }

            $allArticles = array_merge($allArticles, array_slice($entityArticles, 0, $maxPerEntity));
        }

        // Deduplicate
        $seen = [];
        $unique = [];
        foreach ($allArticles as $article) {
            $url = $article["url"] ?? "";
            if ($url && !isset($seen[$url])) {
                $seen[$url] = true;
                $unique[] = $article;
            }
        }

        $this->articleCache = $unique;
        return $unique;
    }

    public function buildGraph(array $articles): void
    {
        foreach ($articles as $article) {
            $entities = $article["entities"] ?? [];
            $names = array_unique(array_filter(array_column($entities, "name")));
            $sentiment = $article["sentiment"]["overall"]["score"] ?? 0;

            // Nodes
            foreach ($entities as $entity) {
                $name = $entity["name"] ?? null;
                if (!$name) continue;

                if (!isset($this->nodes[$name])) {
                    $this->nodes[$name] = [
                        "type" => $entity["type"] ?? "unknown",
                        "mentions" => 0,
                        "sentiment_sum" => 0,
                    ];
                }

                $this->nodes[$name]["mentions"]++;
                $this->nodes[$name]["sentiment_sum"] += $sentiment;
            }

            // Edges
            $names = array_values($names);
            for ($i = 0; $i < count($names); $i++) {
                for ($j = $i + 1; $j < count($names); $j++) {
                    $pair = [$names[$i], $names[$j]];
                    sort($pair);
                    $key = implode("|||", $pair);

                    if (!isset($this->edges[$key])) {
                        $this->edges[$key] = ["weight" => 0, "sentiment_sum" => 0];
                    }

                    $this->edges[$key]["weight"]++;
                    $this->edges[$key]["sentiment_sum"] += $sentiment;
                }
            }
        }
    }

    public function calculateDegreeCentrality(): array
    {
        $degree = [];

        foreach ($this->edges as $key => $data) {
            [$e1, $e2] = explode("|||", $key);
            $degree[$e1] = ($degree[$e1] ?? 0) + $data["weight"];
            $degree[$e2] = ($degree[$e2] ?? 0) + $data["weight"];
        }

        $max = max($degree) ?: 1;
        return array_map(fn($d) => $d / $max, $degree);
    }

    public function calculatePageRank(float $damping = 0.85, int $iterations = 50): array
    {
        $nodes = array_keys($this->nodes);
        $n = count($nodes);
        if ($n === 0) return [];

        // Out weights
        $outWeights = [];
        foreach ($this->edges as $key => $data) {
            [$e1, $e2] = explode("|||", $key);
            $outWeights[$e1] = ($outWeights[$e1] ?? 0) + $data["weight"];
            $outWeights[$e2] = ($outWeights[$e2] ?? 0) + $data["weight"];
        }

        // Initialize
        $pr = array_fill_keys($nodes, 1.0 / $n);

        for ($iter = 0; $iter < $iterations; $iter++) {
            $newPr = [];

            foreach ($nodes as $node) {
                $rankSum = 0;

                foreach ($this->edges as $key => $data) {
                    [$e1, $e2] = explode("|||", $key);

                    if ($e1 === $node && isset($outWeights[$e2]) && $outWeights[$e2] > 0) {
                        $rankSum += $pr[$e2] * $data["weight"] / $outWeights[$e2];
                    } elseif ($e2 === $node && isset($outWeights[$e1]) && $outWeights[$e1] > 0) {
                        $rankSum += $pr[$e1] * $data["weight"] / $outWeights[$e1];
                    }
                }

                $newPr[$node] = (1 - $damping) / $n + $damping * $rankSum;
            }

            $pr = $newPr;
        }

        return $pr;
    }

    public function detectCommunities(): array
    {
        // Build adjacency
        $adj = [];
        foreach ($this->edges as $key => $data) {
            if ($data["weight"] < 2) continue;

            [$e1, $e2] = explode("|||", $key);

            if (!isset($adj[$e1])) $adj[$e1] = [];
            if (!isset($adj[$e2])) $adj[$e2] = [];

            $adj[$e1][$e2] = $data["weight"];
            $adj[$e2][$e1] = $data["weight"];
        }

        // Initialize communities
        $communities = [];
        $id = 0;
        foreach (array_keys($this->nodes) as $node) {
            $communities[$node] = $id++;
        }

        // Label propagation
        $changed = true;
        $maxIter = 20;
        $iter = 0;

        while ($changed && $iter < $maxIter) {
            $changed = false;
            $iter++;

            foreach (array_keys($this->nodes) as $node) {
                if (!isset($adj[$node])) continue;

                $commWeights = [];
                foreach ($adj[$node] as $neighbor => $weight) {
                    $comm = $communities[$neighbor];
                    $commWeights[$comm] = ($commWeights[$comm] ?? 0) + $weight;
                }

                if (!empty($commWeights)) {
                    arsort($commWeights);
                    $bestComm = array_key_first($commWeights);

                    if ($communities[$node] !== $bestComm) {
                        $communities[$node] = $bestComm;
                        $changed = true;
                    }
                }
            }
        }

        // Group
        $result = [];
        foreach ($communities as $node => $comm) {
            if (!isset($result[$comm])) $result[$comm] = [];
            $result[$comm][] = $node;
        }

        return $result;
    }

    public function calculateInfluenceScore(string $entity): ?array
    {
        if (!isset($this->nodes[$entity])) return null;

        $node = $this->nodes[$entity];
        $totalMentions = array_sum(array_column($this->nodes, "mentions"));

        $degree = $this->calculateDegreeCentrality()[$entity] ?? 0;
        $pagerank = $this->calculatePageRank()[$entity] ?? 0;

        $volumeScore = $node["mentions"] / max($totalMentions, 1);
        $sentimentScore = ($node["sentiment_sum"] / max($node["mentions"], 1) + 1) / 2;

        $score = $degree * 0.3 + $pagerank * 0.35 + $volumeScore * 0.2 + $sentimentScore * 0.15;

        return [
            "entity" => $entity,
            "score" => $score,
            "components" => [
                "degree" => $degree,
                "pagerank" => $pagerank,
                "volume" => $volumeScore,
                "sentiment" => $sentimentScore,
            ],
        ];
    }

    public function findHiddenConnections(string $e1, string $e2, int $maxHops = 3): array
    {
        $adj = [];
        foreach ($this->edges as $key => $data) {
            if ($data["weight"] < 2) continue;
            [$a, $b] = explode("|||", $key);
            if (!isset($adj[$a])) $adj[$a] = [];
            if (!isset($adj[$b])) $adj[$b] = [];
            $adj[$a][] = $b;
            $adj[$b][] = $a;
        }

        $paths = [];
        $queue = [[$e1]];

        while (!empty($queue)) {
            $path = array_shift($queue);
            $node = end($path);

            if (count($path) > $maxHops + 1) continue;

            if ($node === $e2 && count($path) > 1) {
                $paths[] = $path;
                continue;
            }

            foreach ($adj[$node] ?? [] as $neighbor) {
                if (!in_array($neighbor, $path) || $neighbor === $e2) {
                    $queue[] = array_merge($path, [$neighbor]);
                }
            }
        }

        // Calculate path strengths
        $analyzed = [];
        foreach ($paths as $path) {
            $strength = 1;
            for ($i = 0; $i < count($path) - 1; $i++) {
                $pair = [$path[$i], $path[$i + 1]];
                sort($pair);
                $key = implode("|||", $pair);
                $strength *= $this->edges[$key]["weight"] ?? 1;
            }

            $analyzed[] = [
                "path" => $path,
                "hops" => count($path) - 1,
                "strength" => $strength,
            ];
        }

        usort($analyzed, fn($a, $b) => $b["strength"] <=> $a["strength"]);
        return $analyzed;
    }

    public function getStats(): array
    {
        return [
            "nodes" => count($this->nodes),
            "edges" => count($this->edges),
            "articles" => count($this->articleCache),
        ];
    }

    public function generateReport(array $entities): array
    {
        $this->fetchArticles($entities, 30, 100);
        $this->buildGraph($this->articleCache);

        $degree = $this->calculateDegreeCentrality();
        $pagerank = $this->calculatePageRank();
        $communities = $this->detectCommunities();

        $influenceRanking = [];
        foreach (array_keys($this->nodes) as $entity) {
            $score = $this->calculateInfluenceScore($entity);
            if ($score) $influenceRanking[] = $score;
        }
        usort($influenceRanking, fn($a, $b) => $b["score"] <=> $a["score"]);

        return [
            "generated_at" => (new DateTime())->format("c"),
            "stats" => $this->getStats(),
            "top_by_degree" => array_slice(
                array_map(fn($e, $s) => ["entity" => $e, "score" => $s],
                    array_keys($degree), array_values($degree)),
                0, 15
            ),
            "top_by_pagerank" => array_slice(
                array_map(fn($e, $s) => ["entity" => $e, "score" => $s],
                    array_keys($pagerank), array_values($pagerank)),
                0, 15
            ),
            "communities" => count($communities),
            "influence_ranking" => array_slice($influenceRanking, 0, 15),
        ];
    }
}

// Run analysis
$analyzer = new EnterpriseNetworkAnalyzer();
$entities = ["Tesla", "SpaceX", "NVIDIA", "OpenAI"];

echo "ENTERPRISE NETWORK ANALYSIS\n";
echo str_repeat("=", 60) . "\n";

$report = $analyzer->generateReport($entities);

echo "\nSTATISTICS:\n";
printf("  Nodes: %d\n", $report["stats"]["nodes"]);
printf("  Edges: %d\n", $report["stats"]["edges"]);
printf("  Articles: %d\n", $report["stats"]["articles"]);

echo "\nTOP BY DEGREE CENTRALITY:\n";
foreach (array_slice($report["top_by_degree"], 0, 10) as $item) {
    arsort($report["top_by_degree"]);
}
usort($report["top_by_degree"], fn($a, $b) => $b["score"] <=> $a["score"]);
foreach (array_slice($report["top_by_degree"], 0, 10) as $item) {
    printf("  %s: %.3f\n", $item["entity"], $item["score"]);
}

echo "\nTOP BY PAGERANK:\n";
usort($report["top_by_pagerank"], fn($a, $b) => $b["score"] <=> $a["score"]);
foreach (array_slice($report["top_by_pagerank"], 0, 10) as $item) {
    printf("  %s: %.4f\n", $item["entity"], $item["score"]);
}

echo "\nINFLUENCE RANKING:\n";
foreach (array_slice($report["influence_ranking"], 0, 10) as $item) {
    printf("  %s: %.4f\n", $item["entity"], $item["score"]);
}

echo "\nCOMMUNITIES DETECTED: {$report['communities']}\n";

// Hidden connections
echo "\nHIDDEN CONNECTIONS (Tesla <-> Microsoft):\n";
$paths = $analyzer->findHiddenConnections("Tesla", "Microsoft", 3);
foreach (array_slice($paths, 0, 5) as $p) {
    echo "  " . implode(" → ", $p["path"]) . " (strength: {$p['strength']})\n";
}
```

---

## See Also

- [README.md](./README.md) — Multi-Entity Network Analysis workflow overview and quick start.
