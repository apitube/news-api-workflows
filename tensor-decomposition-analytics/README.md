# Tensor Decomposition Analytics

Workflow for multi-dimensional news data analysis using tensor decomposition methods (CP, Tucker, Non-negative) to discover latent patterns across entities, topics, time, and sentiment with the [APITube News API](https://apitube.io).

## Overview

The **Tensor Decomposition Analytics** workflow applies tensor factorization techniques to extract latent structures from high-dimensional news data. Features include CP (CANDECOMP/PARAFAC) decomposition for identifying co-occurring patterns, Tucker decomposition for multi-modal analysis, non-negative tensor factorization for interpretable components, temporal pattern discovery, and anomaly detection via reconstruction error. Enables discovery of hidden relationships between entities, topics, sources, and time. Ideal for competitive intelligence, market structure analysis, thematic research, and complex pattern discovery.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by entity name.                                               |
| `topic.id`                    | string  | Filter by topic ID.                                                  |
| `category.id`                 | string  | Filter by category ID.                                               |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### Python

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


@dataclass
class TensorComponent:
    """Represents a discovered latent component."""
    component_id: int
    entity_weights: Dict[str, float]
    topic_weights: Dict[str, float]
    temporal_pattern: List[float]
    strength: float
    interpretation: str


class NewsTensorAnalyzer:
    """
    Tensor decomposition for multi-dimensional news analysis.
    Discovers latent patterns across entities, topics, and time.
    """

    def __init__(self, entities: List[str], topics: List[str], days: int = 14):
        self.entities = entities
        self.topics = topics
        self.days = days
        self.tensor = None  # 3D: entities x topics x time
        self.decomposition = None

    def fetch_tensor_data(self) -> np.ndarray:
        """
        Build 3D tensor: entities x topics x time
        Values represent article counts.
        """
        n_entities = len(self.entities)
        n_topics = len(self.topics)
        n_time = self.days

        tensor = np.zeros((n_entities, n_topics, n_time))

        print(f"Building tensor: {n_entities} entities x {n_topics} topics x {n_time} days")

        for e_idx, entity in enumerate(self.entities):
            print(f"  Fetching data for {entity}...")

            for t_idx, topic in enumerate(self.topics):
                for d in range(n_time):
                    start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
                    end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

                    params = {
                        "api_key": API_KEY,
                        "entity.name": entity,
                        "topic.id": topic,
                        "published_at.start": start,
                        "published_at.end": end,
                        "language": "en",
                        "per_page": 1,
                    }

                    try:
                        response = requests.get(BASE_URL, params=params)
                        count = response.json().get("total_results", 0)
                        tensor[e_idx, t_idx, d] = count
                    except:
                        tensor[e_idx, t_idx, d] = 0

        self.tensor = tensor
        return tensor

    def cp_decomposition(self, rank: int = 5, max_iter: int = 100) -> Dict:
        """
        CP (CANDECOMP/PARAFAC) tensor decomposition.
        Decomposes tensor into sum of rank-1 tensors.

        X ≈ Σ λ_r * a_r ⊗ b_r ⊗ c_r

        where:
        - a_r: entity factors
        - b_r: topic factors
        - c_r: temporal factors
        - λ_r: component strength
        """
        if self.tensor is None:
            self.fetch_tensor_data()

        tensor = self.tensor
        shape = tensor.shape

        # Initialize factor matrices randomly
        A = np.random.rand(shape[0], rank)  # Entities
        B = np.random.rand(shape[1], rank)  # Topics
        C = np.random.rand(shape[2], rank)  # Time

        # Alternating Least Squares (ALS)
        for iteration in range(max_iter):
            # Update A
            khatri_rao_BC = self._khatri_rao(B, C)
            unfold_0 = self._unfold(tensor, 0)
            A = unfold_0 @ khatri_rao_BC @ np.linalg.pinv(khatri_rao_BC.T @ khatri_rao_BC)

            # Update B
            khatri_rao_AC = self._khatri_rao(A, C)
            unfold_1 = self._unfold(tensor, 1)
            B = unfold_1 @ khatri_rao_AC @ np.linalg.pinv(khatri_rao_AC.T @ khatri_rao_AC)

            # Update C
            khatri_rao_AB = self._khatri_rao(A, B)
            unfold_2 = self._unfold(tensor, 2)
            C = unfold_2 @ khatri_rao_AB @ np.linalg.pinv(khatri_rao_AB.T @ khatri_rao_AB)

            # Normalize and extract lambdas
            lambdas = np.linalg.norm(A, axis=0)
            A = A / (lambdas + 1e-10)
            lambdas *= np.linalg.norm(B, axis=0)
            B = B / (np.linalg.norm(B, axis=0) + 1e-10)
            lambdas *= np.linalg.norm(C, axis=0)
            C = C / (np.linalg.norm(C, axis=0) + 1e-10)

        # Sort by strength
        order = np.argsort(-lambdas)
        lambdas = lambdas[order]
        A = A[:, order]
        B = B[:, order]
        C = C[:, order]

        # Compute reconstruction error
        reconstructed = self._reconstruct_cp(A, B, C, lambdas)
        error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

        self.decomposition = {
            "method": "CP",
            "rank": rank,
            "entity_factors": A,
            "topic_factors": B,
            "temporal_factors": C,
            "lambdas": lambdas,
            "reconstruction_error": error
        }

        return self.decomposition

    def _khatri_rao(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Khatri-Rao product (column-wise Kronecker)."""
        n_cols = A.shape[1]
        result = np.zeros((A.shape[0] * B.shape[0], n_cols))

        for i in range(n_cols):
            result[:, i] = np.kron(A[:, i], B[:, i])

        return result

    def _unfold(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold tensor along specified mode."""
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def _reconstruct_cp(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                        lambdas: np.ndarray) -> np.ndarray:
        """Reconstruct tensor from CP factors."""
        shape = (A.shape[0], B.shape[0], C.shape[0])
        result = np.zeros(shape)

        for r in range(len(lambdas)):
            result += lambdas[r] * np.outer(np.outer(A[:, r], B[:, r]).flatten(),
                                            C[:, r]).reshape(shape)

        return result

    def non_negative_decomposition(self, rank: int = 5, max_iter: int = 100) -> Dict:
        """
        Non-negative tensor factorization using multiplicative updates.
        Produces interpretable non-negative factors.
        """
        if self.tensor is None:
            self.fetch_tensor_data()

        tensor = np.maximum(self.tensor, 0)  # Ensure non-negative
        shape = tensor.shape

        # Initialize with non-negative random values
        A = np.random.rand(shape[0], rank) + 0.1
        B = np.random.rand(shape[1], rank) + 0.1
        C = np.random.rand(shape[2], rank) + 0.1

        eps = 1e-10

        for iteration in range(max_iter):
            # Update A (multiplicative update)
            unfold_0 = self._unfold(tensor, 0)
            khatri_rao_BC = self._khatri_rao(B, C)
            numerator = unfold_0 @ khatri_rao_BC
            denominator = A @ (khatri_rao_BC.T @ khatri_rao_BC) + eps
            A = A * numerator / denominator

            # Update B
            unfold_1 = self._unfold(tensor, 1)
            khatri_rao_AC = self._khatri_rao(A, C)
            numerator = unfold_1 @ khatri_rao_AC
            denominator = B @ (khatri_rao_AC.T @ khatri_rao_AC) + eps
            B = B * numerator / denominator

            # Update C
            unfold_2 = self._unfold(tensor, 2)
            khatri_rao_AB = self._khatri_rao(A, B)
            numerator = unfold_2 @ khatri_rao_AB
            denominator = C @ (khatri_rao_AB.T @ khatri_rao_AB) + eps
            C = C * numerator / denominator

        # Normalize
        lambdas = np.linalg.norm(A, axis=0)
        A = A / (lambdas + eps)
        lambdas *= np.linalg.norm(B, axis=0)
        B = B / (np.linalg.norm(B, axis=0) + eps)
        lambdas *= np.linalg.norm(C, axis=0)
        C = C / (np.linalg.norm(C, axis=0) + eps)

        # Sort by strength
        order = np.argsort(-lambdas)
        lambdas = lambdas[order]
        A = A[:, order]
        B = B[:, order]
        C = C[:, order]

        # Reconstruction error
        reconstructed = self._reconstruct_cp(A, B, C, lambdas)
        error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

        self.decomposition = {
            "method": "NTF",
            "rank": rank,
            "entity_factors": A,
            "topic_factors": B,
            "temporal_factors": C,
            "lambdas": lambdas,
            "reconstruction_error": error
        }

        return self.decomposition

    def extract_components(self) -> List[TensorComponent]:
        """Extract and interpret discovered components."""
        if self.decomposition is None:
            raise ValueError("Run decomposition first")

        components = []
        A = self.decomposition["entity_factors"]
        B = self.decomposition["topic_factors"]
        C = self.decomposition["temporal_factors"]
        lambdas = self.decomposition["lambdas"]

        for r in range(len(lambdas)):
            # Top entities for this component
            entity_weights = {
                self.entities[i]: round(float(A[i, r]), 3)
                for i in np.argsort(-A[:, r])[:5]
                if A[i, r] > 0.1
            }

            # Top topics for this component
            topic_weights = {
                self.topics[i]: round(float(B[i, r]), 3)
                for i in np.argsort(-B[:, r])[:5]
                if B[i, r] > 0.1
            }

            # Temporal pattern
            temporal_pattern = C[:, r].tolist()

            # Interpretation
            interpretation = self._interpret_component(entity_weights, topic_weights, temporal_pattern)

            components.append(TensorComponent(
                component_id=r,
                entity_weights=entity_weights,
                topic_weights=topic_weights,
                temporal_pattern=[round(v, 3) for v in temporal_pattern],
                strength=round(float(lambdas[r]), 2),
                interpretation=interpretation
            ))

        return components

    def _interpret_component(self, entities: Dict, topics: Dict,
                             temporal: List[float]) -> str:
        """Generate interpretation of a component."""
        top_entity = max(entities, key=entities.get) if entities else "unknown"
        top_topic = max(topics, key=topics.get) if topics else "unknown"

        # Temporal trend
        if len(temporal) >= 3:
            recent = np.mean(temporal[-3:])
            earlier = np.mean(temporal[:-3]) if len(temporal) > 3 else temporal[0]
            if recent > earlier * 1.2:
                trend = "increasing"
            elif recent < earlier * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return f"{top_entity}-{top_topic} association ({trend} trend)"

    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalies via reconstruction error.
        High reconstruction error indicates unusual patterns.
        """
        if self.decomposition is None or self.tensor is None:
            raise ValueError("Run decomposition first")

        A = self.decomposition["entity_factors"]
        B = self.decomposition["topic_factors"]
        C = self.decomposition["temporal_factors"]
        lambdas = self.decomposition["lambdas"]

        reconstructed = self._reconstruct_cp(A, B, C, lambdas)
        residual = np.abs(self.tensor - reconstructed)

        # Z-score normalization
        mean_residual = np.mean(residual)
        std_residual = np.std(residual)
        z_scores = (residual - mean_residual) / (std_residual + 1e-10)

        anomalies = []
        for e_idx in range(len(self.entities)):
            for t_idx in range(len(self.topics)):
                for d in range(self.days):
                    z = z_scores[e_idx, t_idx, d]
                    if z > threshold:
                        anomalies.append({
                            "entity": self.entities[e_idx],
                            "topic": self.topics[t_idx],
                            "day_offset": d,
                            "observed": float(self.tensor[e_idx, t_idx, d]),
                            "expected": float(reconstructed[e_idx, t_idx, d]),
                            "z_score": round(float(z), 2)
                        })

        return sorted(anomalies, key=lambda x: -x["z_score"])

    def find_entity_clusters(self) -> Dict[int, List[str]]:
        """Cluster entities based on their factor loadings."""
        if self.decomposition is None:
            raise ValueError("Run decomposition first")

        A = self.decomposition["entity_factors"]

        # Simple clustering: assign each entity to its dominant component
        clusters = defaultdict(list)

        for i, entity in enumerate(self.entities):
            dominant = int(np.argmax(A[i, :]))
            clusters[dominant].append(entity)

        return dict(clusters)

    def analyze(self, rank: int = 5, method: str = "ntf") -> Dict:
        """Run complete tensor analysis."""
        print(f"Running tensor decomposition analysis")
        print(f"  Entities: {self.entities}")
        print(f"  Topics: {self.topics}")
        print(f"  Time window: {self.days} days")

        # Fetch data
        self.fetch_tensor_data()
        print(f"  Tensor shape: {self.tensor.shape}")
        print(f"  Total observations: {np.sum(self.tensor):.0f}")

        # Run decomposition
        print(f"\nRunning {method.upper()} decomposition (rank={rank})...")
        if method == "cp":
            self.cp_decomposition(rank=rank)
        else:
            self.non_negative_decomposition(rank=rank)

        print(f"  Reconstruction error: {self.decomposition['reconstruction_error']:.3f}")

        # Extract components
        print("\nExtracting components...")
        components = self.extract_components()

        # Detect anomalies
        print("Detecting anomalies...")
        anomalies = self.detect_anomalies()

        # Find clusters
        clusters = self.find_entity_clusters()

        return {
            "tensor_shape": list(self.tensor.shape),
            "method": method.upper(),
            "rank": rank,
            "reconstruction_error": round(self.decomposition["reconstruction_error"], 4),
            "components": [
                {
                    "id": c.component_id,
                    "strength": c.strength,
                    "entities": c.entity_weights,
                    "topics": c.topic_weights,
                    "temporal_pattern": c.temporal_pattern,
                    "interpretation": c.interpretation
                }
                for c in components
            ],
            "anomalies": anomalies[:10],
            "entity_clusters": {f"cluster_{k}": v for k, v in clusters.items()},
            "analysis_time": datetime.utcnow().isoformat()
        }


# Run analysis
print("TENSOR DECOMPOSITION ANALYTICS")
print("=" * 70)

# Define analysis scope
entities = ["Apple", "Google", "Microsoft", "Amazon", "Meta"]
topics = ["technology", "artificial_intelligence", "business", "finance", "innovation"]

analyzer = NewsTensorAnalyzer(
    entities=entities,
    topics=topics,
    days=14
)

results = analyzer.analyze(rank=3, method="ntf")

print("\n" + "=" * 70)
print("ANALYSIS RESULTS")
print("-" * 50)

print(f"\nTensor: {results['tensor_shape']}")
print(f"Method: {results['method']}")
print(f"Reconstruction Error: {results['reconstruction_error']:.4f}")

print("\nDISCOVERED COMPONENTS:")
print("-" * 50)
for comp in results["components"]:
    print(f"\nComponent {comp['id']} (strength: {comp['strength']})")
    print(f"  Interpretation: {comp['interpretation']}")
    print(f"  Top Entities: {comp['entities']}")
    print(f"  Top Topics: {comp['topics']}")

if results["anomalies"]:
    print("\nANOMALIES DETECTED:")
    print("-" * 50)
    for a in results["anomalies"][:5]:
        print(f"  {a['entity']} + {a['topic']} (day -{a['day_offset']}): "
              f"observed={a['observed']:.0f}, expected={a['expected']:.1f}, z={a['z_score']:.1f}")

print("\nENTITY CLUSTERS:")
for cluster, members in results["entity_clusters"].items():
    print(f"  {cluster}: {members}")
```

## Decomposition Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **CP** | CANDECOMP/PARAFAC decomposition | General pattern discovery |
| **NTF** | Non-negative Tensor Factorization | Interpretable additive patterns |
| **Tucker** | Multi-linear decomposition | Heterogeneous mode sizes |

## Output Components

| Output | Description |
|--------|-------------|
| **Components** | Latent factors with entity, topic, and temporal weights |
| **Anomalies** | Unusual patterns detected via reconstruction error |
| **Clusters** | Entity groupings based on factor loadings |
| **Reconstruction Error** | Model fit quality (lower is better) |

## Common Use Cases

- **Competitive intelligence** — discover entity-topic associations and competitor clusters.
- **Market structure analysis** — identify hidden relationships in industry coverage.
- **Thematic research** — extract latent themes from multi-dimensional news data.
- **Anomaly detection** — find unusual patterns that deviate from normal structure.
- **Trend discovery** — identify temporal patterns across entity-topic combinations.
- **Portfolio analysis** — understand coverage dynamics for multiple assets.

## See Also

- [examples.md](./examples.md) — detailed code examples.
