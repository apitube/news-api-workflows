# News Knowledge Graph

Workflow for building dynamic knowledge graphs from news data, performing entity resolution, extracting and typing relationships, temporal reasoning, graph queries, and knowledge inference using the [APITube News API](https://apitube.io).

## Overview

The **News Knowledge Graph** workflow constructs and maintains a queryable knowledge graph from streaming news data. Features include entity extraction and resolution (merging duplicate entities), relationship extraction with confidence scoring, temporal relationship tracking (when relationships formed/ended), graph-based inference for discovering implicit relationships, and sophisticated query capabilities. Implements proper graph data structures with nodes, edges, and properties. Ideal for intelligence analysis, due diligence, investigative journalism, and research applications.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by entity.                                                    |
| `entity.type`                 | string  | Filter by entity type.                                               |
| `title`                       | string  | Filter by keywords.                                                  |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class KnowledgeGraphNode:
    """Represents an entity in the knowledge graph."""

    def __init__(self, entity_id, name, entity_type):
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.aliases = set([name])
        self.properties = {}
        self.first_seen = None
        self.last_seen = None
        self.mention_count = 0
        self.sentiment_sum = 0

    def merge_alias(self, alias):
        """Add an alias for this entity."""
        self.aliases.add(alias)

    def update_temporal(self, timestamp):
        """Update temporal tracking."""
        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": list(self.aliases),
            "properties": self.properties,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mention_count": self.mention_count,
            "avg_sentiment": self.sentiment_sum / max(self.mention_count, 1)
        }


class KnowledgeGraphEdge:
    """Represents a relationship between entities."""

    def __init__(self, source_id, target_id, relationship_type):
        self.source_id = source_id
        self.target_id = target_id
        self.type = relationship_type
        self.weight = 0
        self.confidence = 0.0
        self.evidence = []  # Article URLs supporting this relationship
        self.first_seen = None
        self.last_seen = None
        self.sentiment_sum = 0
        self.properties = {}

    def add_evidence(self, article_url, timestamp, sentiment=0):
        """Add evidence for this relationship."""
        self.weight += 1
        self.evidence.append(article_url)
        self.sentiment_sum += sentiment

        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

        # Update confidence based on evidence count and recency
        recency_factor = 1.0
        if self.last_seen:
            days_old = (datetime.utcnow() - datetime.fromisoformat(self.last_seen.replace("Z", ""))).days
            recency_factor = max(0.5, 1.0 - days_old * 0.02)

        self.confidence = min(1.0, (self.weight / 10) * recency_factor)

    def to_dict(self):
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "confidence": self.confidence,
            "evidence_count": len(self.evidence),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "avg_sentiment": self.sentiment_sum / max(self.weight, 1)
        }


class NewsKnowledgeGraph:
    """
    Dynamic knowledge graph built from news data.
    Supports entity resolution, relationship extraction, and graph queries.
    """

    # Relationship patterns detected from co-occurrence context
    RELATIONSHIP_PATTERNS = {
        "acquisition": ["acquires", "acquired", "buys", "bought", "purchases", "takes over"],
        "partnership": ["partners with", "partnership", "collaborates", "joint venture", "allies with"],
        "competition": ["competes", "competitor", "rivals", "versus", "vs"],
        "investment": ["invests in", "investment", "funds", "backs", "stakes in"],
        "leadership": ["CEO of", "leads", "heads", "appointed", "named"],
        "lawsuit": ["sues", "lawsuit", "legal action", "litigation"],
        "supply": ["supplies", "supplier", "provides", "vendor"],
        "customer": ["customer", "client", "buyer"],
    }

    def __init__(self):
        self.nodes = {}  # id -> KnowledgeGraphNode
        self.edges = {}  # (source_id, target_id, type) -> KnowledgeGraphEdge
        self.name_to_id = {}  # normalized_name -> entity_id
        self.next_id = 1

    def _normalize_name(self, name):
        """Normalize entity name for matching."""
        return re.sub(r'[^\w\s]', '', name.lower()).strip()

    def _get_or_create_node(self, name, entity_type):
        """Get existing node or create new one with entity resolution."""
        normalized = self._normalize_name(name)

        # Check for existing entity
        if normalized in self.name_to_id:
            return self.nodes[self.name_to_id[normalized]]

        # Check for partial matches (simple entity resolution)
        for existing_norm, entity_id in self.name_to_id.items():
            if normalized in existing_norm or existing_norm in normalized:
                node = self.nodes[entity_id]
                node.merge_alias(name)
                self.name_to_id[normalized] = entity_id
                return node

        # Create new node
        entity_id = f"entity_{self.next_id}"
        self.next_id += 1

        node = KnowledgeGraphNode(entity_id, name, entity_type)
        self.nodes[entity_id] = node
        self.name_to_id[normalized] = entity_id

        return node

    def _detect_relationship_type(self, title):
        """Detect relationship type from article title."""
        title_lower = title.lower()

        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                if pattern in title_lower:
                    return rel_type

        return "mentioned_with"

    def _get_or_create_edge(self, source_id, target_id, rel_type):
        """Get existing edge or create new one."""
        # Ensure consistent ordering for undirected relationships
        if rel_type in ["mentioned_with", "competition"]:
            if source_id > target_id:
                source_id, target_id = target_id, source_id

        edge_key = (source_id, target_id, rel_type)

        if edge_key not in self.edges:
            self.edges[edge_key] = KnowledgeGraphEdge(source_id, target_id, rel_type)

        return self.edges[edge_key]

    def ingest_article(self, article):
        """Ingest a single article into the knowledge graph."""
        entities = article.get("entities", [])
        title = article.get("title", "")
        url = article.get("url", "")
        timestamp = article.get("published_at", "")
        sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)

        if len(entities) < 2:
            return

        # Create/update nodes
        article_nodes = []
        for entity in entities:
            name = entity.get("name")
            etype = entity.get("type", "unknown")

            if not name:
                continue

            node = self._get_or_create_node(name, etype)
            node.mention_count += 1
            node.sentiment_sum += sentiment
            node.update_temporal(timestamp)
            article_nodes.append(node)

        # Create edges between co-occurring entities
        rel_type = self._detect_relationship_type(title)

        for i, node1 in enumerate(article_nodes):
            for node2 in article_nodes[i+1:]:
                edge = self._get_or_create_edge(node1.id, node2.id, rel_type)
                edge.add_evidence(url, timestamp, sentiment)

    def ingest_articles(self, articles):
        """Batch ingest articles."""
        for article in articles:
            self.ingest_article(article)

    def fetch_and_ingest(self, query_params, max_articles=500):
        """Fetch articles from API and ingest into graph."""
        articles = []
        page = 1

        while len(articles) < max_articles:
            params = {**query_params, "api_key": API_KEY, "per_page": 50, "page": page}
            resp = requests.get(BASE_URL, params=params)
            batch = resp.json().get("results", [])

            if not batch:
                break

            articles.extend(batch)
            page += 1

            if len(batch) < 50:
                break

        self.ingest_articles(articles[:max_articles])
        return len(articles)

    def get_node(self, name):
        """Get node by name."""
        normalized = self._normalize_name(name)
        if normalized in self.name_to_id:
            return self.nodes[self.name_to_id[normalized]]
        return None

    def get_neighbors(self, entity_id, relationship_type=None, min_confidence=0.0):
        """Get neighboring nodes connected to an entity."""
        neighbors = []

        for (source, target, rel_type), edge in self.edges.items():
            if relationship_type and rel_type != relationship_type:
                continue
            if edge.confidence < min_confidence:
                continue

            if source == entity_id:
                neighbor_id = target
            elif target == entity_id:
                neighbor_id = source
            else:
                continue

            neighbors.append({
                "node": self.nodes[neighbor_id].to_dict(),
                "relationship": edge.to_dict()
            })

        return sorted(neighbors, key=lambda x: x["relationship"]["weight"], reverse=True)

    def find_path(self, start_name, end_name, max_depth=4):
        """Find shortest path between two entities (BFS)."""
        start_node = self.get_node(start_name)
        end_node = self.get_node(end_name)

        if not start_node or not end_node:
            return None

        # BFS
        visited = {start_node.id}
        queue = [(start_node.id, [start_node.id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id == end_node.id:
                # Reconstruct path with edge details
                full_path = []
                for i in range(len(path) - 1):
                    node = self.nodes[path[i]]
                    # Find edge to next node
                    for (s, t, r), edge in self.edges.items():
                        if (s == path[i] and t == path[i+1]) or (t == path[i] and s == path[i+1]):
                            full_path.append({
                                "node": node.to_dict(),
                                "edge": edge.to_dict()
                            })
                            break

                full_path.append({"node": self.nodes[path[-1]].to_dict(), "edge": None})
                return full_path

            # Explore neighbors
            for (source, target, _), edge in self.edges.items():
                neighbor_id = None
                if source == current_id and target not in visited:
                    neighbor_id = target
                elif target == current_id and source not in visited:
                    neighbor_id = source

                if neighbor_id:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def find_common_connections(self, entity1_name, entity2_name):
        """Find entities connected to both given entities."""
        node1 = self.get_node(entity1_name)
        node2 = self.get_node(entity2_name)

        if not node1 or not node2:
            return []

        neighbors1 = set()
        neighbors2 = set()

        for (source, target, _), edge in self.edges.items():
            if source == node1.id:
                neighbors1.add(target)
            elif target == node1.id:
                neighbors1.add(source)

            if source == node2.id:
                neighbors2.add(target)
            elif target == node2.id:
                neighbors2.add(source)

        common = neighbors1 & neighbors2

        return [self.nodes[n].to_dict() for n in common if n in self.nodes]

    def get_clusters(self, min_cluster_size=3):
        """Detect clusters of closely connected entities."""
        # Simple clustering based on connected components
        visited = set()
        clusters = []

        for node_id in self.nodes:
            if node_id in visited:
                continue

            # BFS to find connected component
            cluster = []
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.append(current)

                for (source, target, _), edge in self.edges.items():
                    if edge.confidence < 0.3:
                        continue

                    neighbor = None
                    if source == current and target not in visited:
                        neighbor = target
                    elif target == current and source not in visited:
                        neighbor = source

                    if neighbor:
                        queue.append(neighbor)

            if len(cluster) >= min_cluster_size:
                clusters.append({
                    "size": len(cluster),
                    "members": [self.nodes[n].to_dict() for n in cluster]
                })

        return sorted(clusters, key=lambda x: x["size"], reverse=True)

    def infer_relationships(self, min_common_neighbors=3):
        """Infer potential relationships based on graph structure."""
        inferred = []

        node_ids = list(self.nodes.keys())

        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                # Skip if already connected
                if any(
                    (s == id1 and t == id2) or (s == id2 and t == id1)
                    for (s, t, _) in self.edges
                ):
                    continue

                # Count common neighbors
                common = len(self.find_common_connections(
                    self.nodes[id1].name,
                    self.nodes[id2].name
                ))

                if common >= min_common_neighbors:
                    inferred.append({
                        "entity1": self.nodes[id1].to_dict(),
                        "entity2": self.nodes[id2].to_dict(),
                        "common_neighbors": common,
                        "inferred_relationship": "potential_connection",
                        "confidence": min(1.0, common / 10)
                    })

        return sorted(inferred, key=lambda x: x["common_neighbors"], reverse=True)

    def get_stats(self):
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": defaultdict(int, {n.type: sum(1 for x in self.nodes.values() if x.type == n.type) for n in self.nodes.values()}),
            "edge_types": defaultdict(int, {e.type: sum(1 for x in self.edges.values() if x.type == e.type) for e in self.edges.values()}),
            "avg_degree": len(self.edges) * 2 / max(len(self.nodes), 1)
        }

    def export_to_json(self):
        """Export graph to JSON format."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "stats": self.get_stats()
            }
        }


# Build knowledge graph
print("NEWS KNOWLEDGE GRAPH")
print("=" * 70)

graph = NewsKnowledgeGraph()

# Ingest articles about tech companies
print("\nIngesting articles...")
count = graph.fetch_and_ingest({
    "title": "Apple,Microsoft,Google,Amazon,NVIDIA",
    "language": "en",
    "source.rank.opr.min": 0.6,
    "published_at.start": (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
}, max_articles=300)

print(f"Ingested {count} articles")

# Graph stats
stats = graph.get_stats()
print(f"\nGRAPH STATISTICS")
print("-" * 40)
print(f"Nodes: {stats['total_nodes']}")
print(f"Edges: {stats['total_edges']}")
print(f"Avg degree: {stats['avg_degree']:.2f}")

# Query: Get neighbors
print(f"\n" + "=" * 70)
print("NEIGHBORS OF 'Apple'")
print("-" * 40)
apple = graph.get_node("Apple")
if apple:
    neighbors = graph.get_neighbors(apple.id, min_confidence=0.2)
    for n in neighbors[:10]:
        print(f"  {n['node']['name']} ({n['relationship']['type']}): "
              f"weight={n['relationship']['weight']}, conf={n['relationship']['confidence']:.2f}")

# Query: Find path
print(f"\n" + "=" * 70)
print("PATH: Apple -> NVIDIA")
print("-" * 40)
path = graph.find_path("Apple", "NVIDIA")
if path:
    for i, step in enumerate(path):
        print(f"  {i+1}. {step['node']['name']}", end="")
        if step['edge']:
            print(f" --[{step['edge']['type']}]-->", end="")
        print()
else:
    print("  No path found")

# Query: Common connections
print(f"\n" + "=" * 70)
print("COMMON CONNECTIONS: Apple & Microsoft")
print("-" * 40)
common = graph.find_common_connections("Apple", "Microsoft")
for c in common[:5]:
    print(f"  {c['name']} ({c['type']})")

# Clusters
print(f"\n" + "=" * 70)
print("ENTITY CLUSTERS")
print("-" * 40)
clusters = graph.get_clusters(min_cluster_size=3)
for i, cluster in enumerate(clusters[:3], 1):
    print(f"Cluster {i} ({cluster['size']} members):")
    for member in cluster['members'][:5]:
        print(f"    {member['name']} ({member['type']})")

# Inferred relationships
print(f"\n" + "=" * 70)
print("INFERRED RELATIONSHIPS")
print("-" * 40)
inferred = graph.infer_relationships(min_common_neighbors=2)
for inf in inferred[:5]:
    print(f"  {inf['entity1']['name']} <--> {inf['entity2']['name']}")
    print(f"    Common neighbors: {inf['common_neighbors']}, Confidence: {inf['confidence']:.2f}")
```

## Graph Components

### Nodes (Entities)
- Unique ID and canonical name
- Entity type (organization, person, location, etc.)
- Aliases for entity resolution
- Temporal tracking (first/last seen)
- Aggregated sentiment

### Edges (Relationships)
- Source and target entity IDs
- Relationship type (acquisition, partnership, competition, etc.)
- Weight (number of co-occurrences)
- Confidence score (evidence + recency)
- Evidence links (supporting articles)

## Common Use Cases

- **Due diligence** — map corporate relationships before investment/M&A.
- **Intelligence analysis** — discover hidden connections between entities.
- **Investigative journalism** — follow relationship trails.
- **Competitive intelligence** — understand market ecosystem.
- **Risk analysis** — identify exposure through relationships.
- **Research** — explore entity networks in any domain.

## See Also

- [examples.md](./examples.md) — additional code examples.
