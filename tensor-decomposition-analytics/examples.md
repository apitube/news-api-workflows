# Tensor Decomposition Analytics - Advanced Examples

## Dynamic Tensor Analysis with Temporal Slicing

Analyze how entity-topic relationships evolve over time using dynamic tensor factorization.

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
class TemporalComponent:
    """Component that evolves over time."""
    component_id: int
    entity_trajectory: Dict[str, List[float]]  # entity -> time series of weights
    topic_trajectory: Dict[str, List[float]]
    strength_trajectory: List[float]
    trend: str  # 'growing', 'stable', 'declining'
    interpretation: str

class DynamicTensorAnalyzer:
    """
    Dynamic tensor decomposition for tracking evolving relationships.
    Uses windowed tensor factorization with temporal regularization.
    """

    def __init__(self, entities: List[str], topics: List[str]):
        self.entities = entities
        self.topics = topics
        self.time_slices = []
        self.decompositions = []

    def fetch_time_slice(self, start_date: str, end_date: str) -> np.ndarray:
        """Fetch data for a single time slice (2D: entities x topics)."""
        matrix = np.zeros((len(self.entities), len(self.topics)))

        for e_idx, entity in enumerate(self.entities):
            for t_idx, topic in enumerate(self.topics):
                params = {
                    "api_key": API_KEY,
                    "entity.name": entity,
                    "topic.id": topic,
                    "published_at.start": start_date,
                    "published_at.end": end_date,
                    "language": "en",
                    "per_page": 1
                }

                try:
                    response = requests.get(BASE_URL, params=params)
                    count = response.json().get("total_results", 0)
                    matrix[e_idx, t_idx] = count
                except:
                    matrix[e_idx, t_idx] = 0

        return matrix

    def build_dynamic_tensor(self, weeks: int = 8) -> List[np.ndarray]:
        """Build sequence of time-sliced matrices."""
        self.time_slices = []

        print(f"Building dynamic tensor: {weeks} weekly slices")

        for w in range(weeks, 0, -1):
            start = (datetime.utcnow() - timedelta(weeks=w)).strftime("%Y-%m-%d")
            end = (datetime.utcnow() - timedelta(weeks=w-1)).strftime("%Y-%m-%d")

            print(f"  Fetching week {weeks - w + 1}: {start} to {end}")
            matrix = self.fetch_time_slice(start, end)
            self.time_slices.append({
                "start": start,
                "end": end,
                "matrix": matrix
            })

        return [t["matrix"] for t in self.time_slices]

    def temporal_nmf(self, matrices: List[np.ndarray], rank: int = 5,
                     temporal_reg: float = 0.1, max_iter: int = 100) -> Dict:
        """
        Non-negative Matrix Factorization with temporal regularization.
        Encourages smooth factor evolution across time slices.
        """
        n_time = len(matrices)
        n_entities = matrices[0].shape[0]
        n_topics = matrices[0].shape[1]

        # Initialize factors for each time slice
        W = [np.random.rand(n_entities, rank) + 0.1 for _ in range(n_time)]
        H = [np.random.rand(rank, n_topics) + 0.1 for _ in range(n_time)]

        eps = 1e-10

        for iteration in range(max_iter):
            for t in range(n_time):
                X = matrices[t]

                # Standard NMF update for H
                numerator = W[t].T @ X
                denominator = W[t].T @ W[t] @ H[t] + eps
                H[t] = H[t] * numerator / denominator

                # Standard NMF update for W
                numerator = X @ H[t].T
                denominator = W[t] @ H[t] @ H[t].T + eps

                # Add temporal regularization
                if t > 0:
                    denominator += temporal_reg * (W[t] - W[t-1])
                if t < n_time - 1:
                    denominator += temporal_reg * (W[t] - W[t+1])

                denominator = np.maximum(denominator, eps)
                W[t] = W[t] * numerator / denominator

        # Extract component strengths over time
        strengths = []
        for t in range(n_time):
            s = np.linalg.norm(W[t], axis=0) * np.linalg.norm(H[t], axis=1)
            strengths.append(s)

        return {
            "W": W,  # Entity factors per time
            "H": H,  # Topic factors per time
            "strengths": np.array(strengths),
            "n_time": n_time,
            "rank": rank
        }

    def extract_temporal_components(self, decomposition: Dict) -> List[TemporalComponent]:
        """Extract evolving components from decomposition."""
        W = decomposition["W"]
        H = decomposition["H"]
        strengths = decomposition["strengths"]
        n_time = decomposition["n_time"]
        rank = decomposition["rank"]

        components = []

        for r in range(rank):
            # Entity trajectory
            entity_traj = {}
            for e_idx, entity in enumerate(self.entities):
                entity_traj[entity] = [float(W[t][e_idx, r]) for t in range(n_time)]

            # Topic trajectory
            topic_traj = {}
            for t_idx, topic in enumerate(self.topics):
                topic_traj[topic] = [float(H[t][r, t_idx]) for t in range(n_time)]

            # Strength trajectory
            strength_traj = [float(strengths[t, r]) for t in range(n_time)]

            # Determine trend
            if len(strength_traj) >= 3:
                recent = np.mean(strength_traj[-3:])
                earlier = np.mean(strength_traj[:-3])
                if recent > earlier * 1.2:
                    trend = "growing"
                elif recent < earlier * 0.8:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "unknown"

            # Interpretation
            top_entity = max(entity_traj, key=lambda e: np.mean(entity_traj[e]))
            top_topic = max(topic_traj, key=lambda t: np.mean(topic_traj[t]))

            components.append(TemporalComponent(
                component_id=r,
                entity_trajectory=entity_traj,
                topic_trajectory=topic_traj,
                strength_trajectory=strength_traj,
                trend=trend,
                interpretation=f"{top_entity}-{top_topic} ({trend})"
            ))

        return components

    def detect_emerging_patterns(self, components: List[TemporalComponent]) -> List[Dict]:
        """Detect newly emerging entity-topic patterns."""
        emerging = []

        for comp in components:
            # Check for recent emergence
            strength = comp.strength_trajectory

            if len(strength) >= 4:
                first_half = np.mean(strength[:len(strength)//2])
                second_half = np.mean(strength[len(strength)//2:])

                if first_half < 0.5 and second_half > first_half * 2:
                    # Find which entities/topics are driving emergence
                    driving_entities = [
                        e for e, traj in comp.entity_trajectory.items()
                        if np.mean(traj[len(traj)//2:]) > np.mean(traj[:len(traj)//2]) * 1.5
                    ]

                    driving_topics = [
                        t for t, traj in comp.topic_trajectory.items()
                        if np.mean(traj[len(traj)//2:]) > np.mean(traj[:len(traj)//2]) * 1.5
                    ]

                    emerging.append({
                        "component_id": comp.component_id,
                        "emergence_ratio": round(second_half / max(first_half, 0.01), 2),
                        "driving_entities": driving_entities[:3],
                        "driving_topics": driving_topics[:3],
                        "current_strength": round(strength[-1], 2)
                    })

        return sorted(emerging, key=lambda x: -x["emergence_ratio"])

    def detect_declining_patterns(self, components: List[TemporalComponent]) -> List[Dict]:
        """Detect declining patterns."""
        declining = []

        for comp in components:
            strength = comp.strength_trajectory

            if len(strength) >= 4:
                first_half = np.mean(strength[:len(strength)//2])
                second_half = np.mean(strength[len(strength)//2:])

                if first_half > 1.0 and second_half < first_half * 0.5:
                    declining.append({
                        "component_id": comp.component_id,
                        "decline_ratio": round(first_half / max(second_half, 0.01), 2),
                        "interpretation": comp.interpretation,
                        "peak_strength": round(max(strength), 2)
                    })

        return sorted(declining, key=lambda x: -x["decline_ratio"])

    def analyze_dynamics(self, weeks: int = 8, rank: int = 5) -> Dict:
        """Run complete dynamic analysis."""
        print("Dynamic Tensor Analysis")
        print(f"  Entities: {self.entities}")
        print(f"  Topics: {self.topics}")

        # Build tensor
        matrices = self.build_dynamic_tensor(weeks=weeks)

        # Run decomposition
        print("\nRunning temporal NMF...")
        decomposition = self.temporal_nmf(matrices, rank=rank)

        # Extract components
        print("Extracting temporal components...")
        components = self.extract_temporal_components(decomposition)

        # Detect patterns
        emerging = self.detect_emerging_patterns(components)
        declining = self.detect_declining_patterns(components)

        return {
            "analysis_time": datetime.utcnow().isoformat(),
            "n_time_slices": weeks,
            "rank": rank,
            "time_periods": [
                {"start": t["start"], "end": t["end"]}
                for t in self.time_slices
            ],
            "components": [
                {
                    "id": c.component_id,
                    "trend": c.trend,
                    "interpretation": c.interpretation,
                    "strength_trajectory": [round(s, 2) for s in c.strength_trajectory],
                    "top_entities": sorted(
                        c.entity_trajectory.items(),
                        key=lambda x: np.mean(x[1]),
                        reverse=True
                    )[:3],
                    "top_topics": sorted(
                        c.topic_trajectory.items(),
                        key=lambda x: np.mean(x[1]),
                        reverse=True
                    )[:3]
                }
                for c in components
            ],
            "emerging_patterns": emerging,
            "declining_patterns": declining
        }


# Usage
analyzer = DynamicTensorAnalyzer(
    entities=["Apple", "Google", "Microsoft", "Amazon", "Meta"],
    topics=["artificial_intelligence", "cloud_computing", "cybersecurity", "hardware", "software"]
)

results = analyzer.analyze_dynamics(weeks=6, rank=4)

print("\n" + "=" * 60)
print("DYNAMIC TENSOR ANALYSIS RESULTS")
print("=" * 60)

print("\nTEMPORAL COMPONENTS:")
for comp in results["components"]:
    print(f"\nComponent {comp['id']}: {comp['interpretation']}")
    print(f"  Trend: {comp['trend']}")
    print(f"  Strength over time: {comp['strength_trajectory']}")

if results["emerging_patterns"]:
    print("\nEMERGING PATTERNS:")
    for p in results["emerging_patterns"]:
        print(f"  Component {p['component_id']}: {p['emergence_ratio']}x growth")
        print(f"    Driven by: {p['driving_entities']} + {p['driving_topics']}")

if results["declining_patterns"]:
    print("\nDECLINING PATTERNS:")
    for p in results["declining_patterns"]:
        print(f"  {p['interpretation']}: {p['decline_ratio']}x decline")
```

## Sparse Tensor Factorization for Large-Scale Analysis

Handle large, sparse news tensors efficiently using sparse tensor methods.

```python
import numpy as np
from scipy import sparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class SparseTensorFactorizer:
    """
    Sparse tensor factorization for large-scale analysis.
    Efficient handling of high-dimensional, sparse news data.
    """

    def __init__(self, entities: List[str], topics: List[str],
                 sources: List[str], days: int = 30):
        self.entities = entities
        self.topics = topics
        self.sources = sources  # 3rd dimension: source types
        self.days = days

        # Store as coordinate list
        self.coords = []  # List of (entity_idx, topic_idx, source_idx, time_idx, value)
        self.shape = (len(entities), len(topics), len(sources), days)

    def fetch_sparse_data(self) -> List[Tuple]:
        """Fetch data and store only non-zero entries."""
        self.coords = []

        print(f"Building sparse tensor: {self.shape}")

        for e_idx, entity in enumerate(self.entities):
            print(f"  Processing entity: {entity}")

            for t_idx, topic in enumerate(self.topics):
                for d in range(self.days):
                    start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
                    end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

                    params = {
                        "api_key": API_KEY,
                        "entity.name": entity,
                        "topic.id": topic,
                        "published_at.start": start,
                        "published_at.end": end,
                        "language": "en",
                        "per_page": 100
                    }

                    try:
                        response = requests.get(BASE_URL, params=params)
                        articles = response.json().get("results", [])

                        # Group by source type
                        source_counts = defaultdict(int)
                        for article in articles:
                            opr = article.get("source", {}).get("rankings", {}).get("opr", 0)
                            if opr >= 0.7:
                                source_counts["tier1"] += 1
                            elif opr >= 0.4:
                                source_counts["tier2"] += 1
                            else:
                                source_counts["tier3"] += 1

                        for source, count in source_counts.items():
                            if count > 0 and source in self.sources:
                                s_idx = self.sources.index(source)
                                self.coords.append((e_idx, t_idx, s_idx, d, count))
                    except:
                        continue

        print(f"  Total non-zero entries: {len(self.coords)}")
        sparsity = 1 - len(self.coords) / np.prod(self.shape)
        print(f"  Sparsity: {sparsity:.2%}")

        return self.coords

    def sparse_cp_als(self, rank: int = 5, max_iter: int = 50) -> Dict:
        """
        CP-ALS for sparse tensors.
        Only processes non-zero entries for efficiency.
        """
        # Initialize factor matrices
        A = np.random.rand(self.shape[0], rank)  # Entities
        B = np.random.rand(self.shape[1], rank)  # Topics
        C = np.random.rand(self.shape[2], rank)  # Sources
        D = np.random.rand(self.shape[3], rank)  # Time

        # Convert coords to arrays for vectorization
        if not self.coords:
            self.fetch_sparse_data()

        indices = np.array([(c[0], c[1], c[2], c[3]) for c in self.coords])
        values = np.array([c[4] for c in self.coords])

        for iteration in range(max_iter):
            # Update A (entities)
            A = self._update_factor_sparse(A, [B, C, D], indices, values, 0)

            # Update B (topics)
            B = self._update_factor_sparse(B, [A, C, D], indices, values, 1)

            # Update C (sources)
            C = self._update_factor_sparse(C, [A, B, D], indices, values, 2)

            # Update D (time)
            D = self._update_factor_sparse(D, [A, B, C], indices, values, 3)

            if (iteration + 1) % 10 == 0:
                error = self._compute_reconstruction_error(A, B, C, D, indices, values)
                print(f"  Iteration {iteration + 1}: error = {error:.4f}")

        # Extract component strengths
        lambdas = np.linalg.norm(A, axis=0)
        A = A / (lambdas + 1e-10)
        lambdas *= np.linalg.norm(B, axis=0)
        B = B / (np.linalg.norm(B, axis=0) + 1e-10)
        lambdas *= np.linalg.norm(C, axis=0)
        C = C / (np.linalg.norm(C, axis=0) + 1e-10)
        lambdas *= np.linalg.norm(D, axis=0)
        D = D / (np.linalg.norm(D, axis=0) + 1e-10)

        # Sort by strength
        order = np.argsort(-lambdas)

        return {
            "entity_factors": A[:, order],
            "topic_factors": B[:, order],
            "source_factors": C[:, order],
            "time_factors": D[:, order],
            "lambdas": lambdas[order],
            "rank": rank
        }

    def _update_factor_sparse(self, factor: np.ndarray,
                               other_factors: List[np.ndarray],
                               indices: np.ndarray, values: np.ndarray,
                               mode: int) -> np.ndarray:
        """Update one factor matrix using only sparse entries."""
        rank = factor.shape[1]
        new_factor = np.zeros_like(factor)

        # Compute Khatri-Rao product for other modes
        other_modes = [i for i in range(4) if i != mode]

        for i in range(factor.shape[0]):
            mask = indices[:, mode] == i
            if not np.any(mask):
                continue

            relevant_indices = indices[mask]
            relevant_values = values[mask]

            # Build MTTKRP (Matricized Tensor Times Khatri-Rao Product)
            rhs = np.zeros(rank)
            gram = np.zeros((rank, rank))

            for j, (idx, val) in enumerate(zip(relevant_indices, relevant_values)):
                # Product of other factors at this index
                prod = np.ones(rank)
                for k, m in enumerate(other_modes):
                    prod *= other_factors[k][idx[m], :]

                rhs += val * prod
                gram += np.outer(prod, prod)

            # Solve for new factor row
            try:
                new_factor[i, :] = np.linalg.solve(gram + 1e-6 * np.eye(rank), rhs)
            except:
                new_factor[i, :] = factor[i, :]

        return np.maximum(new_factor, 0)  # Non-negative

    def _compute_reconstruction_error(self, A, B, C, D, indices, values) -> float:
        """Compute reconstruction error on observed entries only."""
        error = 0
        for idx, val in zip(indices, values):
            reconstructed = 0
            for r in range(A.shape[1]):
                reconstructed += A[idx[0], r] * B[idx[1], r] * C[idx[2], r] * D[idx[3], r]
            error += (val - reconstructed) ** 2

        return np.sqrt(error) / len(values)

    def extract_insights(self, decomposition: Dict) -> Dict:
        """Extract insights from sparse decomposition."""
        A = decomposition["entity_factors"]
        B = decomposition["topic_factors"]
        C = decomposition["source_factors"]
        D = decomposition["time_factors"]
        lambdas = decomposition["lambdas"]

        insights = []

        for r in range(len(lambdas)):
            # Top entities
            top_entities = [(self.entities[i], round(float(A[i, r]), 3))
                          for i in np.argsort(-A[:, r])[:3] if A[i, r] > 0.1]

            # Top topics
            top_topics = [(self.topics[i], round(float(B[i, r]), 3))
                        for i in np.argsort(-B[:, r])[:3] if B[i, r] > 0.1]

            # Source distribution
            source_dist = {self.sources[i]: round(float(C[i, r]), 3)
                         for i in range(len(self.sources))}

            # Temporal pattern
            time_pattern = D[:, r].tolist()

            # Trend
            if len(time_pattern) >= 5:
                recent = np.mean(time_pattern[-5:])
                earlier = np.mean(time_pattern[:-5])
                trend = "increasing" if recent > earlier * 1.2 else \
                       "decreasing" if recent < earlier * 0.8 else "stable"
            else:
                trend = "unknown"

            insights.append({
                "component": r,
                "strength": round(float(lambdas[r]), 2),
                "entities": top_entities,
                "topics": top_topics,
                "source_distribution": source_dist,
                "temporal_trend": trend
            })

        return {
            "tensor_shape": self.shape,
            "sparsity": 1 - len(self.coords) / np.prod(self.shape),
            "components": insights
        }


# Usage
factorizer = SparseTensorFactorizer(
    entities=["Apple", "Google", "Microsoft", "Amazon", "Meta", "Netflix", "Tesla"],
    topics=["technology", "artificial_intelligence", "finance", "innovation", "business"],
    sources=["tier1", "tier2", "tier3"],
    days=14
)

# Fetch sparse data
factorizer.fetch_sparse_data()

# Run factorization
print("\nRunning sparse CP decomposition...")
decomposition = factorizer.sparse_cp_als(rank=4, max_iter=30)

# Extract insights
results = factorizer.extract_insights(decomposition)

print("\n" + "=" * 60)
print("SPARSE TENSOR ANALYSIS")
print("=" * 60)
print(f"Tensor shape: {results['tensor_shape']}")
print(f"Sparsity: {results['sparsity']:.2%}")

for comp in results["components"]:
    print(f"\nComponent {comp['component']} (strength: {comp['strength']})")
    print(f"  Entities: {comp['entities']}")
    print(f"  Topics: {comp['topics']}")
    print(f"  Source dist: {comp['source_distribution']}")
    print(f"  Trend: {comp['temporal_trend']}")
```

## Multi-Modal Tensor Analysis with Sentiment

Incorporate sentiment as an additional tensor mode for richer analysis.

```python
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class MultiModalTensorAnalyzer:
    """
    4D tensor analysis: Entities x Topics x Sentiment x Time
    Reveals sentiment-aware patterns in entity-topic relationships.
    """

    SENTIMENT_BINS = ["negative", "neutral", "positive"]

    def __init__(self, entities: List[str], topics: List[str], days: int = 14):
        self.entities = entities
        self.topics = topics
        self.days = days
        self.tensor = None

    def fetch_4d_tensor(self) -> np.ndarray:
        """Build 4D tensor with sentiment dimension."""
        shape = (len(self.entities), len(self.topics),
                len(self.SENTIMENT_BINS), self.days)
        self.tensor = np.zeros(shape)

        print(f"Building 4D tensor: {shape}")

        for e_idx, entity in enumerate(self.entities):
            print(f"  Processing: {entity}")

            for t_idx, topic in enumerate(self.topics):
                for s_idx, sentiment in enumerate(self.SENTIMENT_BINS):
                    for d in range(self.days):
                        start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
                        end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

                        params = {
                            "api_key": API_KEY,
                            "entity.name": entity,
                            "topic.id": topic,
                            "sentiment.overall.polarity": sentiment,
                            "published_at.start": start,
                            "published_at.end": end,
                            "language": "en",
                            "per_page": 1
                        }

                        try:
                            response = requests.get(BASE_URL, params=params)
                            count = response.json().get("total_results", 0)
                            self.tensor[e_idx, t_idx, s_idx, d] = count
                        except:
                            pass

        return self.tensor

    def tucker_decomposition(self, ranks: Tuple[int, int, int, int] = (3, 3, 3, 5),
                             max_iter: int = 50) -> Dict:
        """
        Tucker decomposition for 4D tensor.
        X ≈ G ×₁ A ×₂ B ×₃ C ×₄ D
        """
        if self.tensor is None:
            self.fetch_4d_tensor()

        tensor = self.tensor
        shape = tensor.shape

        # Initialize factor matrices
        A = np.random.rand(shape[0], ranks[0])  # Entities
        B = np.random.rand(shape[1], ranks[1])  # Topics
        C = np.random.rand(shape[2], ranks[2])  # Sentiment
        D = np.random.rand(shape[3], ranks[3])  # Time

        # Core tensor
        G = np.random.rand(*ranks)

        for iteration in range(max_iter):
            # Update each factor using HOSVD-like updates

            # Mode-1 (entities)
            Y1 = self._unfold_4d(tensor, 0)
            kron = np.kron(np.kron(D, C), B)
            G1 = self._unfold_4d(G, 0)
            target = Y1 @ kron @ G1.T
            A, _ = np.linalg.qr(target)
            A = A[:, :ranks[0]]

            # Mode-2 (topics)
            Y2 = self._unfold_4d(tensor, 1)
            kron = np.kron(np.kron(D, C), A)
            G2 = self._unfold_4d(G, 1)
            target = Y2 @ kron @ G2.T
            B, _ = np.linalg.qr(target)
            B = B[:, :ranks[1]]

            # Mode-3 (sentiment)
            Y3 = self._unfold_4d(tensor, 2)
            kron = np.kron(np.kron(D, B), A)
            G3 = self._unfold_4d(G, 2)
            target = Y3 @ kron @ G3.T
            C, _ = np.linalg.qr(target)
            C = C[:, :ranks[2]]

            # Mode-4 (time)
            Y4 = self._unfold_4d(tensor, 3)
            kron = np.kron(np.kron(C, B), A)
            G4 = self._unfold_4d(G, 3)
            target = Y4 @ kron @ G4.T
            D, _ = np.linalg.qr(target)
            D = D[:, :ranks[3]]

            # Update core tensor
            G = self._compute_core(tensor, A, B, C, D)

        return {
            "core": G,
            "entity_factors": A,
            "topic_factors": B,
            "sentiment_factors": C,
            "time_factors": D,
            "ranks": ranks
        }

    def _unfold_4d(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold 4D tensor along specified mode."""
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    def _compute_core(self, tensor, A, B, C, D) -> np.ndarray:
        """Compute core tensor G = X ×₁ Aᵀ ×₂ Bᵀ ×₃ Cᵀ ×₄ Dᵀ"""
        # Apply each factor transpose
        result = tensor

        # Mode-1: multiply by A.T
        result = np.tensordot(A.T, result, axes=([1], [0]))

        # Mode-2: multiply by B.T
        result = np.tensordot(B.T, result, axes=([1], [1]))
        result = np.moveaxis(result, 0, 1)

        # Mode-3: multiply by C.T
        result = np.tensordot(C.T, result, axes=([1], [2]))
        result = np.moveaxis(result, 0, 2)

        # Mode-4: multiply by D.T
        result = np.tensordot(D.T, result, axes=([1], [3]))
        result = np.moveaxis(result, 0, 3)

        return result

    def analyze_sentiment_patterns(self, decomposition: Dict) -> Dict:
        """Analyze sentiment-related patterns from decomposition."""
        C = decomposition["sentiment_factors"]
        A = decomposition["entity_factors"]
        B = decomposition["topic_factors"]
        D = decomposition["time_factors"]
        G = decomposition["core"]

        # Analyze sentiment factor loadings
        sentiment_patterns = []

        for s_idx, sentiment in enumerate(self.SENTIMENT_BINS):
            # Which factors have high sentiment loading?
            sentiment_loading = C[s_idx, :]

            for r in range(C.shape[1]):
                if abs(sentiment_loading[r]) > 0.3:
                    # Find associated entities and topics
                    entity_weights = A[:, r]
                    topic_weights = B[:, r]

                    top_entities = [
                        (self.entities[i], round(float(entity_weights[i]), 3))
                        for i in np.argsort(-np.abs(entity_weights))[:3]
                    ]

                    top_topics = [
                        (self.topics[i], round(float(topic_weights[i]), 3))
                        for i in np.argsort(-np.abs(topic_weights))[:3]
                    ]

                    sentiment_patterns.append({
                        "sentiment": sentiment,
                        "factor": r,
                        "loading": round(float(sentiment_loading[r]), 3),
                        "associated_entities": top_entities,
                        "associated_topics": top_topics
                    })

        # Entity sentiment profiles
        entity_profiles = {}
        for e_idx, entity in enumerate(self.entities):
            # Aggregate across topics and time
            entity_sentiment = self.tensor[e_idx, :, :, :].sum(axis=(0, 2))
            total = entity_sentiment.sum()
            if total > 0:
                entity_profiles[entity] = {
                    "negative": round(float(self.tensor[e_idx, :, 0, :].sum() / total), 3),
                    "neutral": round(float(self.tensor[e_idx, :, 1, :].sum() / total), 3),
                    "positive": round(float(self.tensor[e_idx, :, 2, :].sum() / total), 3)
                }

        return {
            "sentiment_patterns": sentiment_patterns,
            "entity_sentiment_profiles": entity_profiles,
            "decomposition_ranks": decomposition["ranks"]
        }


# Usage
analyzer = MultiModalTensorAnalyzer(
    entities=["Apple", "Google", "Microsoft", "Amazon"],
    topics=["artificial_intelligence", "privacy", "antitrust", "innovation"],
    days=14
)

# Run Tucker decomposition
print("Running Tucker decomposition...")
decomposition = analyzer.tucker_decomposition(ranks=(3, 3, 3, 5), max_iter=30)

# Analyze sentiment patterns
results = analyzer.analyze_sentiment_patterns(decomposition)

print("\n" + "=" * 60)
print("MULTI-MODAL TENSOR ANALYSIS")
print("=" * 60)

print("\nSENTIMENT PATTERNS:")
for p in results["sentiment_patterns"]:
    print(f"\n  {p['sentiment'].upper()} (factor {p['factor']}, loading: {p['loading']})")
    print(f"    Entities: {p['associated_entities']}")
    print(f"    Topics: {p['associated_topics']}")

print("\nENTITY SENTIMENT PROFILES:")
for entity, profile in results["entity_sentiment_profiles"].items():
    print(f"  {entity}: pos={profile['positive']:.0%}, neu={profile['neutral']:.0%}, neg={profile['negative']:.0%}")
```
