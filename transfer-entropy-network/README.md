# Transfer Entropy Network

> **Information-theoretic causality analysis with transfer entropy, conditional mutual information, effective information flow, and directed information networks** — built for the [APITube News API](https://apitube.io).

This research-grade workflow implements information-theoretic measures for detecting directed information flow between news sentiment series. Unlike correlation or Granger causality, transfer entropy captures nonlinear dependencies and provides a model-free measure of predictive information transfer.

## Overview

The Transfer Entropy Network provides:

- **Transfer Entropy Estimation** — Measure directed information flow using Kraskov-Stögbauer-Grassberger (KSG) estimator
- **Conditional Mutual Information** — Quantify shared information accounting for confounders
- **Effective Information** — Measure causal efficacy using intervention distributions
- **Information Flow Networks** — Build weighted digraphs of information transfer
- **Multivariate Extension** — Conditional transfer entropy with multiple sources
- **Significance Testing** — Surrogate data methods for statistical validation
- **Time-Lagged Analysis** — Optimal lag detection for information transfer

## Theoretical Background

### Transfer Entropy

Transfer entropy from X to Y measures the reduction in uncertainty about Y's future given Y's past, when also knowing X's past:

```
T_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})
        = I(Y_t ; X_{t-1:t-l} | Y_{t-1:t-k})
```

Where:
- `H(·)` is Shannon entropy
- `I(·;·|·)` is conditional mutual information
- `k, l` are embedding dimensions

### Key Properties

1. **Asymmetric**: T_{X→Y} ≠ T_{Y→X} (unlike correlation)
2. **Model-free**: No assumptions about functional relationships
3. **Nonlinear**: Captures any statistical dependency
4. **Conditional**: Can account for confounding variables

## Parameters

### Data Collection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity.surface_form.in` | array | Entities for information flow analysis |
| `category.in` | array | Categories: `business`, `finance`, `economy` |
| `published_at.gte` / `.lte` | datetime | Analysis time window |
| `source.rank.lte` | number | Source quality filter |
| `language.code.eq` | string | Language filter (default: `en`) |

### Estimation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding_dim` | integer | Embedding dimension k (default: 3) |
| `lag` | integer | Time lag for source (default: 1) |
| `k_neighbors` | integer | KSG k-nearest neighbors (default: 4) |
| `estimator` | string | Estimator type: `ksg`, `kernel`, `binned` |
| `n_surrogates` | integer | Surrogates for significance (default: 100) |
| `alpha` | float | Significance level (default: 0.05) |

## Quick Start

### cURL
```bash
curl -G "https://api.apitube.io/v1/news/everything" \
  --data-urlencode "entity.surface_form.in=Apple,Microsoft,Google,Amazon,Meta" \
  --data-urlencode "category.eq=technology" \
  --data-urlencode "published_at.gte=2024-01-01" \
  --data-urlencode "language.code.eq=en" \
  --data-urlencode "limit=50" \
  --data-urlencode "api_key=YOUR_API_KEY"
```

### Python
```python
import requests
import numpy as np
from scipy.special import digamma
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class TransferEntropyResult:
    """Result of transfer entropy estimation."""
    source: str
    target: str
    transfer_entropy: float
    p_value: float
    is_significant: bool
    optimal_lag: int
    effective_info_flow: float

@dataclass
class InformationFlowEdge:
    """Edge in information flow network."""
    source: str
    target: str
    weight: float
    lag: int
    significance: float

class KSGEstimator:
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.

    Uses k-nearest neighbor distances for continuous entropy estimation.
    """

    def __init__(self, k: int = 4):
        self.k = k

    def entropy(self, X: np.ndarray) -> float:
        """
        Estimate entropy H(X) using KSG method.

        H(X) ≈ ψ(N) - ψ(k) + d·log(2) + d·⟨log(ε)⟩
        """
        N, d = X.shape

        # Build k-d tree
        tree = KDTree(X)

        # Find k-th nearest neighbor distances
        distances, _ = tree.query(X, k=self.k + 1)
        epsilon = distances[:, -1]  # k-th neighbor distance

        # Avoid log(0)
        epsilon = np.maximum(epsilon, 1e-10)

        # KSG entropy estimator
        entropy = (
            digamma(N) -
            digamma(self.k) +
            d * np.log(2) +
            d * np.mean(np.log(epsilon))
        )

        return entropy

    def mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Estimate mutual information I(X;Y) using KSG method.

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        XY = np.hstack([X, Y])

        h_x = self.entropy(X)
        h_y = self.entropy(Y)
        h_xy = self.entropy(XY)

        mi = h_x + h_y - h_xy

        return max(0, mi)  # MI should be non-negative

    def conditional_mutual_information(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> float:
        """
        Estimate conditional mutual information I(X;Y|Z).

        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        XZ = np.hstack([X, Z])
        YZ = np.hstack([Y, Z])
        XYZ = np.hstack([X, Y, Z])

        h_xz = self.entropy(XZ)
        h_yz = self.entropy(YZ)
        h_z = self.entropy(Z)
        h_xyz = self.entropy(XYZ)

        cmi = h_xz + h_yz - h_z - h_xyz

        return max(0, cmi)

class TransferEntropyEstimator:
    """
    Transfer entropy estimation with multiple methods.

    Implements KSG-based and binned estimators.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        lag: int = 1,
        k_neighbors: int = 4
    ):
        self.embedding_dim = embedding_dim
        self.lag = lag
        self.ksg = KSGEstimator(k=k_neighbors)

    def embed_time_series(
        self,
        X: np.ndarray,
        dim: int,
        delay: int = 1
    ) -> np.ndarray:
        """
        Create time-delay embedding of time series.

        Returns matrix where each row is [x_t, x_{t-τ}, x_{t-2τ}, ...]
        """
        N = len(X)
        max_delay = (dim - 1) * delay

        if N <= max_delay:
            raise ValueError("Time series too short for embedding")

        embedded = np.zeros((N - max_delay, dim))

        for i in range(dim):
            start_idx = max_delay - i * delay
            end_idx = N - i * delay
            embedded[:, i] = X[start_idx:end_idx]

        return embedded

    def transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: Optional[int] = None
    ) -> float:
        """
        Estimate transfer entropy from source to target.

        T_{X→Y} = I(Y_t ; X_{t-lag:t-lag-k} | Y_{t-1:t-k})
        """
        if lag is None:
            lag = self.lag

        k = self.embedding_dim
        N = len(target)

        # Embed target history
        target_past = self.embed_time_series(target[:-1], k)

        # Embed source history with lag
        source_past = self.embed_time_series(source[lag:], k)

        # Align lengths
        min_len = min(len(target_past), len(source_past))
        target_past = target_past[-min_len:]
        source_past = source_past[-min_len:]

        # Target future (what we're predicting)
        target_future = target[-(min_len):].reshape(-1, 1)

        # Transfer entropy as conditional mutual information
        te = self.ksg.conditional_mutual_information(
            target_future,
            source_past,
            target_past
        )

        return te

    def find_optimal_lag(
        self,
        source: np.ndarray,
        target: np.ndarray,
        max_lag: int = 10
    ) -> Tuple[int, float]:
        """Find lag that maximizes transfer entropy."""
        best_lag = 1
        best_te = 0

        for lag in range(1, max_lag + 1):
            try:
                te = self.transfer_entropy(source, target, lag)
                if te > best_te:
                    best_te = te
                    best_lag = lag
            except ValueError:
                break

        return best_lag, best_te

    def significance_test(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_surrogates: int = 100,
        alpha: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Test significance using surrogate data.

        Generates surrogates by shuffling source time series.
        """
        # Observed transfer entropy
        te_observed = self.transfer_entropy(source, target)

        # Generate surrogate distribution
        te_surrogates = []

        for _ in range(n_surrogates):
            # Shuffle source (destroys temporal structure)
            source_shuffled = np.random.permutation(source)
            try:
                te_surrogate = self.transfer_entropy(source_shuffled, target)
                te_surrogates.append(te_surrogate)
            except ValueError:
                pass

        if len(te_surrogates) == 0:
            return 1.0, False

        # p-value: fraction of surrogates >= observed
        te_surrogates = np.array(te_surrogates)
        p_value = np.mean(te_surrogates >= te_observed)

        is_significant = p_value < alpha

        return p_value, is_significant

class EffectiveInformation:
    """
    Effective information (EI) for measuring causal efficacy.

    Based on Integrated Information Theory concepts.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def discretize(self, X: np.ndarray) -> np.ndarray:
        """Discretize continuous time series into bins."""
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(X, percentiles)
        return np.digitize(X, bins[1:-1])

    def transition_probability_matrix(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1
    ) -> np.ndarray:
        """
        Estimate transition probability matrix P(Y_t | X_{t-lag}).
        """
        source_disc = self.discretize(source)
        target_disc = self.discretize(target)

        # Align with lag
        source_lagged = source_disc[:-lag]
        target_current = target_disc[lag:]

        # Count transitions
        n_states = self.n_bins
        counts = np.zeros((n_states, n_states))

        for s, t in zip(source_lagged, target_current):
            counts[s, t] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        tpm = np.divide(
            counts, row_sums,
            where=row_sums > 0,
            out=np.zeros_like(counts)
        )

        return tpm

    def effective_information(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Calculate effective information.

        EI = MI(uniform intervention on X, observed Y)
        """
        tpm = self.transition_probability_matrix(source, target, lag)

        # Uniform intervention distribution
        p_x_uniform = np.ones(self.n_bins) / self.n_bins

        # Marginal distribution of Y under uniform intervention
        p_y = tpm.T @ p_x_uniform

        # Effective information
        ei = 0
        for x in range(self.n_bins):
            for y in range(self.n_bins):
                if tpm[x, y] > 0 and p_y[y] > 0:
                    ei += p_x_uniform[x] * tpm[x, y] * np.log2(tpm[x, y] / p_y[y])

        return max(0, ei)

class TransferEntropyNetwork:
    """
    Build and analyze information flow networks.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        k_neighbors: int = 4,
        significance_level: float = 0.05
    ):
        self.te_estimator = TransferEntropyEstimator(
            embedding_dim=embedding_dim,
            k_neighbors=k_neighbors
        )
        self.ei_estimator = EffectiveInformation()
        self.significance_level = significance_level
        self.edges: List[InformationFlowEdge] = []

    def build_network(
        self,
        data: Dict[str, np.ndarray],
        n_surrogates: int = 100
    ) -> List[InformationFlowEdge]:
        """
        Build information flow network from time series data.

        Tests all pairwise transfer entropies.
        """
        self.edges = []
        entities = list(data.keys())

        for source_name in entities:
            for target_name in entities:
                if source_name == target_name:
                    continue

                source = data[source_name]
                target = data[target_name]

                # Find optimal lag
                optimal_lag, te = self.te_estimator.find_optimal_lag(
                    source, target, max_lag=10
                )

                # Significance test
                p_value, is_significant = self.te_estimator.significance_test(
                    source, target, n_surrogates, self.significance_level
                )

                # Effective information
                ei = self.ei_estimator.effective_information(
                    source, target, optimal_lag
                )

                if is_significant:
                    edge = InformationFlowEdge(
                        source=source_name,
                        target=target_name,
                        weight=te,
                        lag=optimal_lag,
                        significance=1 - p_value
                    )
                    self.edges.append(edge)

        return self.edges

    def get_adjacency_matrix(self, entities: List[str]) -> np.ndarray:
        """Get weighted adjacency matrix of information flow."""
        n = len(entities)
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        adj = np.zeros((n, n))

        for edge in self.edges:
            i = entity_to_idx.get(edge.source)
            j = entity_to_idx.get(edge.target)
            if i is not None and j is not None:
                adj[i, j] = edge.weight

        return adj

    def compute_network_metrics(self) -> Dict:
        """Compute network-level information flow metrics."""
        if not self.edges:
            return {}

        # Total information flow
        total_flow = sum(e.weight for e in self.edges)

        # Average transfer entropy
        avg_te = total_flow / len(self.edges)

        # Information flow asymmetry
        flow_in = defaultdict(float)
        flow_out = defaultdict(float)

        for edge in self.edges:
            flow_out[edge.source] += edge.weight
            flow_in[edge.target] += edge.weight

        # Net flow (positive = source, negative = sink)
        all_entities = set(flow_in.keys()) | set(flow_out.keys())
        net_flow = {
            e: flow_out[e] - flow_in[e]
            for e in all_entities
        }

        # Information sources and sinks
        sources = sorted(net_flow.items(), key=lambda x: x[1], reverse=True)[:3]
        sinks = sorted(net_flow.items(), key=lambda x: x[1])[:3]

        return {
            "total_information_flow": total_flow,
            "average_transfer_entropy": avg_te,
            "n_significant_edges": len(self.edges),
            "information_sources": sources,
            "information_sinks": sinks,
            "net_flow": net_flow
        }

class TransferEntropyAnalyzer:
    """
    Complete transfer entropy analysis system with news data integration.
    """

    def __init__(
        self,
        api_key: str,
        entities: List[str],
        lookback_days: int = 180,
        embedding_dim: int = 3
    ):
        self.api_key = api_key
        self.entities = entities
        self.lookback_days = lookback_days
        self.network = TransferEntropyNetwork(embedding_dim=embedding_dim)
        self.data: Dict[str, np.ndarray] = {}

    def fetch_entity_data(self) -> Dict[str, np.ndarray]:
        """Fetch sentiment time series for all entities."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for entity in self.entities:
            series = []
            current_date = start_date

            while current_date <= end_date:
                next_date = current_date + timedelta(days=1)

                response = requests.get(BASE_URL, params={
                    "api_key": self.api_key,
                    "entity.surface_form.eq": entity,
                    "published_at.gte": current_date.strftime("%Y-%m-%d"),
                    "published_at.lt": next_date.strftime("%Y-%m-%d"),
                    "category.in": "business,technology,finance",
                    "language.code.eq": "en",
                    "limit": 50
                })

                articles = response.json().get("results", [])
                if articles:
                    sentiments = [
                        a.get("sentiment", {}).get("overall", 0)
                        for a in articles
                    ]
                    series.append(np.mean(sentiments))
                else:
                    series.append(0)

                current_date = next_date

            self.data[entity] = np.array(series)

        return self.data

    def analyze_information_flow(
        self,
        n_surrogates: int = 50
    ) -> Dict:
        """Run complete information flow analysis."""
        if not self.data:
            self.fetch_entity_data()

        # Build network
        self.network.build_network(self.data, n_surrogates)

        # Compute metrics
        metrics = self.network.compute_network_metrics()

        # Get adjacency matrix
        adj_matrix = self.network.get_adjacency_matrix(self.entities)

        # Identify key flows
        top_flows = sorted(
            self.network.edges,
            key=lambda e: e.weight,
            reverse=True
        )[:10]

        return {
            "network_metrics": metrics,
            "adjacency_matrix": adj_matrix.tolist(),
            "entities": self.entities,
            "top_information_flows": [
                {
                    "source": e.source,
                    "target": e.target,
                    "transfer_entropy": e.weight,
                    "optimal_lag": e.lag,
                    "confidence": e.significance
                }
                for e in top_flows
            ],
            "timestamp": datetime.now().isoformat()
        }

    def compute_pairwise_flow(
        self,
        source: str,
        target: str
    ) -> TransferEntropyResult:
        """Compute detailed transfer entropy between specific pair."""
        if not self.data:
            self.fetch_entity_data()

        source_data = self.data.get(source)
        target_data = self.data.get(target)

        if source_data is None or target_data is None:
            raise ValueError(f"Data not found for {source} or {target}")

        # Find optimal lag
        optimal_lag, te = self.network.te_estimator.find_optimal_lag(
            source_data, target_data
        )

        # Significance test
        p_value, is_significant = self.network.te_estimator.significance_test(
            source_data, target_data
        )

        # Effective information
        ei = self.network.ei_estimator.effective_information(
            source_data, target_data, optimal_lag
        )

        return TransferEntropyResult(
            source=source,
            target=target,
            transfer_entropy=te,
            p_value=p_value,
            is_significant=is_significant,
            optimal_lag=optimal_lag,
            effective_info_flow=ei
        )

    def generate_flow_report(self) -> Dict:
        """Generate comprehensive information flow report."""
        analysis = self.analyze_information_flow()

        # Compute bidirectional flows
        bidirectional = []
        edges_by_pair = {}

        for edge in self.network.edges:
            pair = tuple(sorted([edge.source, edge.target]))
            if pair not in edges_by_pair:
                edges_by_pair[pair] = []
            edges_by_pair[pair].append(edge)

        for pair, edges in edges_by_pair.items():
            if len(edges) == 2:
                # Bidirectional flow
                net_flow = edges[0].weight - edges[1].weight
                dominant_direction = edges[0] if edges[0].weight > edges[1].weight else edges[1]
                bidirectional.append({
                    "entities": list(pair),
                    "dominant_flow": f"{dominant_direction.source} → {dominant_direction.target}",
                    "net_transfer_entropy": abs(net_flow),
                    "flow_asymmetry": abs(net_flow) / (edges[0].weight + edges[1].weight)
                })

        return {
            **analysis,
            "bidirectional_flows": bidirectional,
            "summary": {
                "most_influential": analysis["network_metrics"].get("information_sources", []),
                "most_reactive": analysis["network_metrics"].get("information_sinks", []),
                "network_density": len(self.network.edges) / (
                    len(self.entities) * (len(self.entities) - 1)
                )
            }
        }

# Usage
entities = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]

analyzer = TransferEntropyAnalyzer(
    API_KEY,
    entities,
    lookback_days=180,
    embedding_dim=3
)

report = analyzer.generate_flow_report()
print(report)

# Specific pair analysis
result = analyzer.compute_pairwise_flow("Apple", "Microsoft")
print(f"TE(Apple → Microsoft) = {result.transfer_entropy:.4f}")
print(f"p-value = {result.p_value:.4f}")
print(f"Optimal lag = {result.optimal_lag} days")
```

### JavaScript
```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

// Digamma function approximation
function digamma(x) {
  if (x < 6) {
    return digamma(x + 1) - 1 / x;
  }
  return Math.log(x) - 1 / (2 * x) - 1 / (12 * x * x) +
         1 / (120 * x * x * x * x);
}

class KSGEstimator {
  constructor(k = 4) {
    this.k = k;
  }

  euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  findKNearestDistances(X, k) {
    const N = X.length;
    const distances = [];

    for (let i = 0; i < N; i++) {
      const dists = [];
      for (let j = 0; j < N; j++) {
        if (i !== j) {
          dists.push(this.euclideanDistance(X[i], X[j]));
        }
      }
      dists.sort((a, b) => a - b);
      distances.push(dists[k - 1] || 1e-10);
    }

    return distances;
  }

  entropy(X) {
    const N = X.length;
    const d = X[0].length;

    const epsilons = this.findKNearestDistances(X, this.k);

    let sumLogEps = 0;
    for (const eps of epsilons) {
      sumLogEps += Math.log(Math.max(eps, 1e-10));
    }

    return digamma(N) - digamma(this.k) + d * Math.log(2) +
           d * sumLogEps / N;
  }

  mutualInformation(X, Y) {
    // Ensure 2D
    const X2d = X[0].length ? X : X.map(x => [x]);
    const Y2d = Y[0].length ? Y : Y.map(y => [y]);

    const XY = X2d.map((x, i) => [...x, ...Y2d[i]]);

    const hX = this.entropy(X2d);
    const hY = this.entropy(Y2d);
    const hXY = this.entropy(XY);

    return Math.max(0, hX + hY - hXY);
  }

  conditionalMutualInformation(X, Y, Z) {
    const X2d = X[0]?.length ? X : X.map(x => [x]);
    const Y2d = Y[0]?.length ? Y : Y.map(y => [y]);
    const Z2d = Z[0]?.length ? Z : Z.map(z => [z]);

    const XZ = X2d.map((x, i) => [...x, ...Z2d[i]]);
    const YZ = Y2d.map((y, i) => [...y, ...Z2d[i]]);
    const XYZ = X2d.map((x, i) => [...x, ...Y2d[i], ...Z2d[i]]);

    const hXZ = this.entropy(XZ);
    const hYZ = this.entropy(YZ);
    const hZ = this.entropy(Z2d);
    const hXYZ = this.entropy(XYZ);

    return Math.max(0, hXZ + hYZ - hZ - hXYZ);
  }
}

class TransferEntropyEstimator {
  constructor(embeddingDim = 3, lag = 1, kNeighbors = 4) {
    this.embeddingDim = embeddingDim;
    this.lag = lag;
    this.ksg = new KSGEstimator(kNeighbors);
  }

  embedTimeSeries(X, dim, delay = 1) {
    const N = X.length;
    const maxDelay = (dim - 1) * delay;

    if (N <= maxDelay) {
      throw new Error("Time series too short");
    }

    const embedded = [];
    for (let t = maxDelay; t < N; t++) {
      const row = [];
      for (let d = 0; d < dim; d++) {
        row.push(X[t - d * delay]);
      }
      embedded.push(row);
    }

    return embedded;
  }

  transferEntropy(source, target, lag = null) {
    lag = lag ?? this.lag;
    const k = this.embeddingDim;

    // Embed target history
    const targetPast = this.embedTimeSeries(target.slice(0, -1), k);

    // Embed source with lag
    const sourcePast = this.embedTimeSeries(source.slice(lag), k);

    // Align lengths
    const minLen = Math.min(targetPast.length, sourcePast.length);
    const targetPastAligned = targetPast.slice(-minLen);
    const sourcePastAligned = sourcePast.slice(-minLen);

    // Target future
    const targetFuture = target.slice(-minLen).map(t => [t]);

    // Transfer entropy as CMI
    return this.ksg.conditionalMutualInformation(
      targetFuture,
      sourcePastAligned,
      targetPastAligned
    );
  }

  findOptimalLag(source, target, maxLag = 10) {
    let bestLag = 1;
    let bestTE = 0;

    for (let lag = 1; lag <= maxLag; lag++) {
      try {
        const te = this.transferEntropy(source, target, lag);
        if (te > bestTE) {
          bestTE = te;
          bestLag = lag;
        }
      } catch (e) {
        break;
      }
    }

    return { optimalLag: bestLag, transferEntropy: bestTE };
  }

  significanceTest(source, target, nSurrogates = 50, alpha = 0.05) {
    const teObserved = this.transferEntropy(source, target);

    let countGreater = 0;

    for (let s = 0; s < nSurrogates; s++) {
      const sourceShuffled = [...source].sort(() => Math.random() - 0.5);
      try {
        const teSurrogate = this.transferEntropy(sourceShuffled, target);
        if (teSurrogate >= teObserved) countGreater++;
      } catch (e) {}
    }

    const pValue = countGreater / nSurrogates;
    return { pValue, isSignificant: pValue < alpha };
  }
}

class TransferEntropyAnalyzer {
  constructor(apiKey, entities, lookbackDays = 180) {
    this.apiKey = apiKey;
    this.entities = entities;
    this.lookbackDays = lookbackDays;
    this.teEstimator = new TransferEntropyEstimator(3, 1, 4);
    this.data = {};
  }

  async fetchEntityData() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - this.lookbackDays);

    for (const entity of this.entities) {
      const series = [];
      const current = new Date(startDate);

      while (current <= endDate) {
        const next = new Date(current);
        next.setDate(next.getDate() + 1);

        const params = new URLSearchParams({
          api_key: this.apiKey,
          "entity.surface_form.eq": entity,
          "published_at.gte": current.toISOString().split("T")[0],
          "published_at.lt": next.toISOString().split("T")[0],
          "category.in": "business,technology,finance",
          "language.code.eq": "en",
          limit: "50"
        });

        try {
          const response = await fetch(`${BASE_URL}?${params}`);
          const data = await response.json();
          const articles = data.results || [];

          if (articles.length > 0) {
            const sentiments = articles.map(a => a.sentiment?.overall ?? 0);
            series.push(sentiments.reduce((a, b) => a + b, 0) / sentiments.length);
          } else {
            series.push(0);
          }
        } catch (e) {
          series.push(0);
        }

        current.setDate(current.getDate() + 1);
      }

      this.data[entity] = series;
    }

    return this.data;
  }

  async analyzeInformationFlow(nSurrogates = 30) {
    if (Object.keys(this.data).length === 0) {
      await this.fetchEntityData();
    }

    const edges = [];

    for (const source of this.entities) {
      for (const target of this.entities) {
        if (source === target) continue;

        const sourceData = this.data[source];
        const targetData = this.data[target];

        const { optimalLag, transferEntropy } =
          this.teEstimator.findOptimalLag(sourceData, targetData);

        const { pValue, isSignificant } =
          this.teEstimator.significanceTest(sourceData, targetData, nSurrogates);

        if (isSignificant) {
          edges.push({
            source,
            target,
            transferEntropy,
            optimalLag,
            pValue,
            significance: 1 - pValue
          });
        }
      }
    }

    // Sort by transfer entropy
    edges.sort((a, b) => b.transferEntropy - a.transferEntropy);

    // Compute network metrics
    const flowOut = {}, flowIn = {};
    for (const edge of edges) {
      flowOut[edge.source] = (flowOut[edge.source] || 0) + edge.transferEntropy;
      flowIn[edge.target] = (flowIn[edge.target] || 0) + edge.transferEntropy;
    }

    const netFlow = {};
    for (const entity of this.entities) {
      netFlow[entity] = (flowOut[entity] || 0) - (flowIn[entity] || 0);
    }

    return {
      edges,
      networkMetrics: {
        totalFlow: edges.reduce((s, e) => s + e.transferEntropy, 0),
        nSignificantEdges: edges.length,
        netFlow
      },
      timestamp: new Date().toISOString()
    };
  }
}

// Usage
const entities = ["Apple", "Microsoft", "Google", "Amazon", "Meta"];
const analyzer = new TransferEntropyAnalyzer(API_KEY, entities, 180);

analyzer.analyzeInformationFlow().then(report => {
  console.log("Top information flows:");
  report.edges.slice(0, 5).forEach(e => {
    console.log(`${e.source} → ${e.target}: TE=${e.transferEntropy.toFixed(4)}`);
  });
});
```

### PHP
```php
<?php

class KSGEstimator {
    private int $k;

    public function __construct(int $k = 4) {
        $this->k = $k;
    }

    private function digamma(float $x): float {
        if ($x < 6) {
            return $this->digamma($x + 1) - 1 / $x;
        }
        return log($x) - 1 / (2 * $x) - 1 / (12 * $x * $x);
    }

    private function euclideanDistance(array $a, array $b): float {
        $sum = 0;
        for ($i = 0; $i < count($a); $i++) {
            $sum += ($a[$i] - $b[$i]) ** 2;
        }
        return sqrt($sum);
    }

    private function findKNearestDistances(array $X, int $k): array {
        $N = count($X);
        $distances = [];

        for ($i = 0; $i < $N; $i++) {
            $dists = [];
            for ($j = 0; $j < $N; $j++) {
                if ($i !== $j) {
                    $dists[] = $this->euclideanDistance($X[$i], $X[$j]);
                }
            }
            sort($dists);
            $distances[] = $dists[$k - 1] ?? 1e-10;
        }

        return $distances;
    }

    public function entropy(array $X): float {
        $N = count($X);
        $d = count($X[0]);

        $epsilons = $this->findKNearestDistances($X, $this->k);

        $sumLogEps = 0;
        foreach ($epsilons as $eps) {
            $sumLogEps += log(max($eps, 1e-10));
        }

        return $this->digamma($N) - $this->digamma($this->k) +
               $d * log(2) + $d * $sumLogEps / $N;
    }

    public function mutualInformation(array $X, array $Y): float {
        $X2d = is_array($X[0]) ? $X : array_map(fn($x) => [$x], $X);
        $Y2d = is_array($Y[0]) ? $Y : array_map(fn($y) => [$y], $Y);

        $XY = array_map(
            fn($x, $y) => array_merge($x, $y),
            $X2d, $Y2d
        );

        $hX = $this->entropy($X2d);
        $hY = $this->entropy($Y2d);
        $hXY = $this->entropy($XY);

        return max(0, $hX + $hY - $hXY);
    }

    public function conditionalMutualInformation(array $X, array $Y, array $Z): float {
        $X2d = is_array($X[0] ?? null) ? $X : array_map(fn($x) => [$x], $X);
        $Y2d = is_array($Y[0] ?? null) ? $Y : array_map(fn($y) => [$y], $Y);
        $Z2d = is_array($Z[0] ?? null) ? $Z : array_map(fn($z) => [$z], $Z);

        $XZ = array_map(fn($x, $z) => array_merge($x, $z), $X2d, $Z2d);
        $YZ = array_map(fn($y, $z) => array_merge($y, $z), $Y2d, $Z2d);
        $XYZ = array_map(
            fn($x, $y, $z) => array_merge($x, $y, $z),
            $X2d, $Y2d, $Z2d
        );

        $hXZ = $this->entropy($XZ);
        $hYZ = $this->entropy($YZ);
        $hZ = $this->entropy($Z2d);
        $hXYZ = $this->entropy($XYZ);

        return max(0, $hXZ + $hYZ - $hZ - $hXYZ);
    }
}

class TransferEntropyEstimator {
    private int $embeddingDim;
    private int $lag;
    private KSGEstimator $ksg;

    public function __construct(int $embeddingDim = 3, int $lag = 1, int $kNeighbors = 4) {
        $this->embeddingDim = $embeddingDim;
        $this->lag = $lag;
        $this->ksg = new KSGEstimator($kNeighbors);
    }

    private function embedTimeSeries(array $X, int $dim, int $delay = 1): array {
        $N = count($X);
        $maxDelay = ($dim - 1) * $delay;

        if ($N <= $maxDelay) {
            throw new Exception("Time series too short");
        }

        $embedded = [];
        for ($t = $maxDelay; $t < $N; $t++) {
            $row = [];
            for ($d = 0; $d < $dim; $d++) {
                $row[] = $X[$t - $d * $delay];
            }
            $embedded[] = $row;
        }

        return $embedded;
    }

    public function transferEntropy(array $source, array $target, ?int $lag = null): float {
        $lag = $lag ?? $this->lag;
        $k = $this->embeddingDim;

        $targetPast = $this->embedTimeSeries(array_slice($target, 0, -1), $k);
        $sourcePast = $this->embedTimeSeries(array_slice($source, $lag), $k);

        $minLen = min(count($targetPast), count($sourcePast));
        $targetPast = array_slice($targetPast, -$minLen);
        $sourcePast = array_slice($sourcePast, -$minLen);

        $targetFuture = array_map(
            fn($t) => [$t],
            array_slice($target, -$minLen)
        );

        return $this->ksg->conditionalMutualInformation(
            $targetFuture,
            $sourcePast,
            $targetPast
        );
    }

    public function findOptimalLag(array $source, array $target, int $maxLag = 10): array {
        $bestLag = 1;
        $bestTE = 0;

        for ($lag = 1; $lag <= $maxLag; $lag++) {
            try {
                $te = $this->transferEntropy($source, $target, $lag);
                if ($te > $bestTE) {
                    $bestTE = $te;
                    $bestLag = $lag;
                }
            } catch (Exception $e) {
                break;
            }
        }

        return ['optimalLag' => $bestLag, 'transferEntropy' => $bestTE];
    }
}

class TransferEntropyAnalyzer {
    private string $apiKey;
    private array $entities;
    private int $lookbackDays;
    private TransferEntropyEstimator $teEstimator;
    private array $data = [];

    public function __construct(
        string $apiKey,
        array $entities,
        int $lookbackDays = 180
    ) {
        $this->apiKey = $apiKey;
        $this->entities = $entities;
        $this->lookbackDays = $lookbackDays;
        $this->teEstimator = new TransferEntropyEstimator(3, 1, 4);
    }

    public function fetchEntityData(): array {
        $endDate = new DateTime();
        $startDate = (clone $endDate)->modify("-{$this->lookbackDays} days");

        foreach ($this->entities as $entity) {
            $series = [];
            $current = clone $startDate;

            while ($current <= $endDate) {
                $next = clone $current;
                $next->modify('+1 day');

                $params = http_build_query([
                    'api_key' => $this->apiKey,
                    'entity.surface_form.eq' => $entity,
                    'published_at.gte' => $current->format('Y-m-d'),
                    'published_at.lt' => $next->format('Y-m-d'),
                    'category.in' => 'business,technology,finance',
                    'language.code.eq' => 'en',
                    'limit' => 50
                ]);

                $response = @file_get_contents(
                    "https://api.apitube.io/v1/news/everything?{$params}"
                );

                if ($response) {
                    $data = json_decode($response, true);
                    $articles = $data['results'] ?? [];

                    if (count($articles) > 0) {
                        $sentiments = array_map(
                            fn($a) => $a['sentiment']['overall'] ?? 0,
                            $articles
                        );
                        $series[] = array_sum($sentiments) / count($sentiments);
                    } else {
                        $series[] = 0;
                    }
                } else {
                    $series[] = 0;
                }

                $current->modify('+1 day');
            }

            $this->data[$entity] = $series;
        }

        return $this->data;
    }

    public function analyzeInformationFlow(): array {
        if (empty($this->data)) {
            $this->fetchEntityData();
        }

        $edges = [];

        foreach ($this->entities as $source) {
            foreach ($this->entities as $target) {
                if ($source === $target) continue;

                $result = $this->teEstimator->findOptimalLag(
                    $this->data[$source],
                    $this->data[$target]
                );

                if ($result['transferEntropy'] > 0.01) {
                    $edges[] = [
                        'source' => $source,
                        'target' => $target,
                        'transfer_entropy' => $result['transferEntropy'],
                        'optimal_lag' => $result['optimalLag']
                    ];
                }
            }
        }

        usort($edges, fn($a, $b) =>
            $b['transfer_entropy'] <=> $a['transfer_entropy']
        );

        return [
            'edges' => $edges,
            'timestamp' => (new DateTime())->format('c')
        ];
    }
}

// Usage
$entities = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Meta'];
$analyzer = new TransferEntropyAnalyzer('YOUR_API_KEY', $entities, 180);
print_r($analyzer->analyzeInformationFlow());
```

## Use Cases

### Market Microstructure
- Information leadership detection
- Price discovery analysis
- Market efficiency assessment

### Corporate Intelligence
- Cross-company influence mapping
- Supply chain information flow
- Competitive dynamics analysis

### Media Analysis
- News propagation patterns
- Source influence ranking
- Echo chamber detection
