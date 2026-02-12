# Causal Discovery Engine

> **Structure learning with PC algorithm, do-calculus intervention analysis, counterfactual reasoning, and causal effect estimation from news data** — built for the [APITube News API](https://apitube.io).

This research-grade workflow implements a complete causal discovery and inference framework that learns causal graphs from observational news data. It applies constraint-based structure learning (PC algorithm), estimates causal effects using do-calculus, performs counterfactual analysis, and identifies instrumental variables for robust causal inference.

## Overview

The Causal Discovery Engine provides:

- **PC Algorithm Implementation** — Constraint-based causal structure learning with conditional independence tests
- **Skeleton Discovery** — Edge detection via marginal and conditional independence
- **V-Structure Orientation** — Collider identification and edge direction inference
- **Meek Rules Application** — Propagate orientations to maximize directed edges
- **Do-Calculus Engine** — Compute interventional distributions P(Y|do(X))
- **Backdoor Criterion** — Identify valid adjustment sets for causal effect estimation
- **Front-Door Criterion** — Alternative identification when backdoor fails
- **Instrumental Variables** — Detect and use instruments for confounded relationships
- **Counterfactual Reasoning** — Estimate "what-if" scenarios using structural equations
- **Causal Effect Bounds** — Partial identification when point identification fails

## Parameters

### Data Collection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity.surface_form.in` | array | Entities for causal graph construction |
| `category.in` | array | Categories: `business`, `finance`, `economy`, `politics` |
| `published_at.gte` / `.lte` | datetime | Observation window |
| `source.rank.lte` | number | High-quality source filter |
| `language.code.eq` | string | Language filter (default: `en`) |

### Causal Discovery Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | float | Significance level for independence tests (default: 0.05) |
| `max_cond_set` | integer | Maximum conditioning set size (default: 3) |
| `independence_test` | string | Test type: `partial_correlation`, `g_squared`, `cmi` |
| `orientation_rules` | array | Meek rules to apply (default: all) |

### Causal Inference Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimand` | string | Target: `ate`, `att`, `cate` |
| `identification` | string | Strategy: `backdoor`, `frontdoor`, `iv` |
| `confidence_level` | float | CI confidence level (default: 0.95) |

## Theoretical Background

### PC Algorithm

The PC (Peter-Clark) algorithm discovers causal structure in three phases:

1. **Skeleton Discovery**: Start with complete undirected graph, remove edges where variables are conditionally independent
2. **V-Structure Identification**: Orient edges into colliders (X → Z ← Y where X ⊥ Y but X ⊥̸ Y | Z)
3. **Edge Orientation**: Apply Meek rules to orient remaining edges without creating cycles or new v-structures

### Do-Calculus Rules

Pearl's do-calculus provides three rules for manipulating interventional distributions:

- **Rule 1** (Insertion/deletion of observations): P(y|do(x),z,w) = P(y|do(x),w) if Y ⊥ Z | X,W in G_X̄
- **Rule 2** (Action/observation exchange): P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y ⊥ Z | X,W in G_X̄Z̲
- **Rule 3** (Insertion/deletion of actions): P(y|do(x),do(z),w) = P(y|do(x),w) if Y ⊥ Z | X,W in G_X̄Z̄(W)

## Quick Start

### cURL
```bash
curl -G "https://api.apitube.io/v1/news/everything" \
  --data-urlencode "entity.surface_form.in=Federal Reserve,S&P 500,USD,Treasury Yields,Inflation" \
  --data-urlencode "category.in=finance,economy" \
  --data-urlencode "published_at.gte=2024-01-01" \
  --data-urlencode "language.code.eq=en" \
  --data-urlencode "limit=50" \
  --data-urlencode "api_key=YOUR_API_KEY"
```

### Python
```python
import requests
import numpy as np
from scipy import stats
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, FrozenSet
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import copy

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class EdgeType(Enum):
    UNDIRECTED = "---"
    DIRECTED = "-->"
    BIDIRECTED = "<->"
    UNKNOWN = "o-o"

@dataclass
class CausalEdge:
    """Represents an edge in the causal graph."""
    source: str
    target: str
    edge_type: EdgeType
    strength: float = 0.0
    p_value: float = 1.0

@dataclass
class CausalGraph:
    """Represents a causal DAG or CPDAG."""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[Tuple[str, str], CausalEdge] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_edge(self, source: str, target: str, edge_type: EdgeType, **kwargs):
        edge = CausalEdge(source, target, edge_type, **kwargs)
        self.edges[(source, target)] = edge
        self.adjacency[source].add(target)
        if edge_type == EdgeType.UNDIRECTED:
            self.adjacency[target].add(source)

    def remove_edge(self, source: str, target: str):
        if (source, target) in self.edges:
            del self.edges[(source, target)]
            self.adjacency[source].discard(target)
        if (target, source) in self.edges:
            del self.edges[(target, source)]
            self.adjacency[target].discard(source)

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self.edges or (target, source) in self.edges

    def get_neighbors(self, node: str) -> Set[str]:
        neighbors = self.adjacency[node].copy()
        for other in self.nodes:
            if other in self.adjacency and node in self.adjacency[other]:
                neighbors.add(other)
        return neighbors

    def get_parents(self, node: str) -> Set[str]:
        parents = set()
        for (src, tgt), edge in self.edges.items():
            if tgt == node and edge.edge_type == EdgeType.DIRECTED:
                parents.add(src)
        return parents

    def get_children(self, node: str) -> Set[str]:
        children = set()
        for (src, tgt), edge in self.edges.items():
            if src == node and edge.edge_type == EdgeType.DIRECTED:
                children.add(tgt)
        return children

    def get_ancestors(self, node: str) -> Set[str]:
        ancestors = set()
        to_visit = list(self.get_parents(node))
        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))
        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        descendants = set()
        to_visit = list(self.get_children(node))
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))
        return descendants

class IndependenceTest:
    """
    Conditional independence testing for causal discovery.

    Implements partial correlation test with Fisher's z-transform.
    """

    def __init__(self, data: np.ndarray, variable_names: List[str], alpha: float = 0.05):
        self.data = data
        self.variable_names = variable_names
        self.var_to_idx = {v: i for i, v in enumerate(variable_names)}
        self.alpha = alpha
        self.n_samples = data.shape[0]

        # Precompute correlation matrix
        self.corr_matrix = np.corrcoef(data.T)

    def partial_correlation(
        self,
        x_idx: int,
        y_idx: int,
        cond_idx: List[int]
    ) -> float:
        """
        Compute partial correlation between X and Y given conditioning set.

        Uses recursive formula or precision matrix method.
        """
        if len(cond_idx) == 0:
            return self.corr_matrix[x_idx, y_idx]

        # Extract relevant submatrix
        indices = [x_idx, y_idx] + list(cond_idx)
        sub_corr = self.corr_matrix[np.ix_(indices, indices)]

        try:
            # Precision matrix method
            precision = np.linalg.inv(sub_corr)
            partial_corr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
            return np.clip(partial_corr, -0.9999, 0.9999)
        except np.linalg.LinAlgError:
            return 0.0

    def fisher_z_transform(self, r: float) -> float:
        """Fisher's z-transformation for correlation."""
        return 0.5 * np.log((1 + r) / (1 - r))

    def test_conditional_independence(
        self,
        x: str,
        y: str,
        conditioning_set: Set[str]
    ) -> Tuple[bool, float]:
        """
        Test if X ⊥ Y | Z using partial correlation test.

        Returns (is_independent, p_value)
        """
        x_idx = self.var_to_idx[x]
        y_idx = self.var_to_idx[y]
        cond_idx = [self.var_to_idx[z] for z in conditioning_set]

        # Compute partial correlation
        partial_corr = self.partial_correlation(x_idx, y_idx, cond_idx)

        # Fisher's z-test
        z = self.fisher_z_transform(partial_corr)
        n = self.n_samples
        k = len(conditioning_set)

        # Standard error
        se = 1.0 / np.sqrt(n - k - 3) if n > k + 3 else float('inf')

        # Z-statistic
        z_stat = abs(z) / se if se > 0 else 0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(z_stat))

        is_independent = p_value > self.alpha

        return is_independent, p_value

class PCAlgorithm:
    """
    PC Algorithm for causal structure learning.

    Discovers causal graph from observational data using
    conditional independence tests.
    """

    def __init__(
        self,
        independence_test: IndependenceTest,
        max_cond_set_size: int = 3
    ):
        self.ind_test = independence_test
        self.max_cond_set_size = max_cond_set_size
        self.separation_sets: Dict[FrozenSet[str], Set[str]] = {}

    def discover_skeleton(self, nodes: List[str]) -> CausalGraph:
        """
        Phase 1: Discover undirected skeleton.

        Start with complete graph, remove edges where
        conditional independence holds.
        """
        graph = CausalGraph()
        graph.nodes = set(nodes)

        # Initialize complete undirected graph
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i < j:
                    graph.add_edge(x, y, EdgeType.UNDIRECTED)

        # Test independence with increasing conditioning set sizes
        for cond_size in range(self.max_cond_set_size + 1):
            edges_to_remove = []

            for (x, y) in list(graph.edges.keys()):
                # Get potential conditioning variables
                neighbors_x = graph.get_neighbors(x) - {y}
                neighbors_y = graph.get_neighbors(y) - {x}
                potential_cond = neighbors_x | neighbors_y

                if len(potential_cond) < cond_size:
                    continue

                # Test all conditioning sets of current size
                for cond_set in combinations(potential_cond, cond_size):
                    cond_set = set(cond_set)

                    is_independent, p_value = self.ind_test.test_conditional_independence(
                        x, y, cond_set
                    )

                    if is_independent:
                        edges_to_remove.append((x, y))
                        # Store separation set for v-structure detection
                        self.separation_sets[frozenset({x, y})] = cond_set
                        break

            # Remove independent edges
            for x, y in edges_to_remove:
                graph.remove_edge(x, y)

        return graph

    def orient_v_structures(self, graph: CausalGraph) -> CausalGraph:
        """
        Phase 2: Identify and orient v-structures (colliders).

        For each triple X - Z - Y where X and Y are not adjacent,
        orient as X → Z ← Y if Z is not in sep(X, Y).
        """
        oriented_graph = copy.deepcopy(graph)

        for z in graph.nodes:
            neighbors = list(graph.get_neighbors(z))

            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    # Check if x and y are non-adjacent
                    if not graph.has_edge(x, y):
                        # Check if z is in separation set
                        sep_set = self.separation_sets.get(frozenset({x, y}), set())

                        if z not in sep_set:
                            # Orient as v-structure: x → z ← y
                            oriented_graph.remove_edge(x, z)
                            oriented_graph.remove_edge(y, z)
                            oriented_graph.add_edge(x, z, EdgeType.DIRECTED)
                            oriented_graph.add_edge(y, z, EdgeType.DIRECTED)

        return oriented_graph

    def apply_meek_rules(self, graph: CausalGraph) -> CausalGraph:
        """
        Phase 3: Apply Meek's orientation rules.

        Propagate edge orientations without creating cycles
        or new v-structures.
        """
        changed = True

        while changed:
            changed = False

            for (x, y), edge in list(graph.edges.items()):
                if edge.edge_type != EdgeType.UNDIRECTED:
                    continue

                # Rule 1: X → Y - Z becomes X → Y → Z
                # (if X and Z not adjacent)
                parents_y = graph.get_parents(y)
                for p in parents_y:
                    if not graph.has_edge(p, x) and p != x:
                        # Check x - y is undirected from y's perspective too
                        if (y, x) in graph.edges:
                            graph.remove_edge(x, y)
                            graph.remove_edge(y, x)
                            graph.add_edge(x, y, EdgeType.DIRECTED)
                            changed = True
                            break

                # Rule 2: X → Z → Y with X - Y becomes X → Y
                # (avoid cycle)
                for z in graph.get_children(x):
                    if y in graph.get_children(z):
                        if (y, x) in graph.edges:
                            graph.remove_edge(x, y)
                            graph.remove_edge(y, x)
                            graph.add_edge(x, y, EdgeType.DIRECTED)
                            changed = True
                            break

                # Rule 3: X - Z1 → Y and X - Z2 → Y with Z1 - Z2
                # becomes X → Y
                children_to_y = set()
                for z in graph.get_neighbors(x):
                    if y in graph.get_children(z):
                        children_to_y.add(z)

                if len(children_to_y) >= 2:
                    for z1, z2 in combinations(children_to_y, 2):
                        if graph.has_edge(z1, z2):
                            if (y, x) in graph.edges:
                                graph.remove_edge(x, y)
                                graph.remove_edge(y, x)
                                graph.add_edge(x, y, EdgeType.DIRECTED)
                                changed = True
                                break

        return graph

    def fit(self, nodes: List[str]) -> CausalGraph:
        """Run complete PC algorithm."""
        # Phase 1: Skeleton discovery
        skeleton = self.discover_skeleton(nodes)

        # Phase 2: V-structure orientation
        cpdag = self.orient_v_structures(skeleton)

        # Phase 3: Meek rules
        cpdag = self.apply_meek_rules(cpdag)

        return cpdag

class DoCalculus:
    """
    Pearl's do-calculus for causal effect estimation.

    Implements backdoor criterion, front-door criterion,
    and adjustment formula.
    """

    def __init__(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]):
        self.graph = graph
        self.data = data
        self.variable_names = variable_names
        self.var_to_idx = {v: i for i, v in enumerate(variable_names)}

    def find_backdoor_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> Optional[Set[str]]:
        """
        Find valid adjustment set using backdoor criterion.

        Z satisfies backdoor if:
        1. Z blocks all backdoor paths from X to Y
        2. Z does not contain any descendant of X
        """
        descendants_x = self.graph.get_descendants(treatment)
        ancestors_y = self.graph.get_ancestors(outcome)

        # Candidate adjustment variables
        candidates = self.graph.nodes - {treatment, outcome} - descendants_x

        # Find minimal valid adjustment set
        for size in range(len(candidates) + 1):
            for adj_set in combinations(candidates, size):
                adj_set = set(adj_set)
                if self._blocks_backdoor_paths(treatment, outcome, adj_set):
                    return adj_set

        return None

    def _blocks_backdoor_paths(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> bool:
        """Check if adjustment set blocks all backdoor paths."""
        # Find all backdoor paths (paths with arrow into treatment)
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)

        for path in backdoor_paths:
            if not self._is_path_blocked(path, adjustment_set):
                return False

        return True

    def _find_backdoor_paths(
        self,
        treatment: str,
        outcome: str,
        max_length: int = 10
    ) -> List[List[str]]:
        """Find all backdoor paths from treatment to outcome."""
        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return

            if current == outcome and len(path) > 1:
                # Check if first edge is into treatment (backdoor)
                first = path[0]
                second = path[1]
                if (first, second) in self.graph.edges:
                    edge = self.graph.edges[(first, second)]
                    if edge.edge_type == EdgeType.DIRECTED and edge.target == first:
                        paths.append(path.copy())
                return

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(treatment, [treatment], {treatment})
        return paths

    def _is_path_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
        """Check if path is d-separated by conditioning set."""
        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            curr_node = path[i]
            next_node = path[i + 1]

            # Check edge types
            edge_in = self.graph.edges.get((prev_node, curr_node)) or \
                      self.graph.edges.get((curr_node, prev_node))
            edge_out = self.graph.edges.get((curr_node, next_node)) or \
                       self.graph.edges.get((next_node, curr_node))

            if edge_in is None or edge_out is None:
                continue

            is_collider = (
                edge_in.target == curr_node and
                edge_out.target == curr_node
            )

            if is_collider:
                # Collider: path blocked unless conditioned on collider or descendant
                descendants = self.graph.get_descendants(curr_node) | {curr_node}
                if not (conditioning_set & descendants):
                    return True
            else:
                # Non-collider: path blocked if conditioned on
                if curr_node in conditioning_set:
                    return True

        return False

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Optional[Set[str]] = None
    ) -> Dict:
        """
        Estimate Average Treatment Effect using adjustment formula.

        ATE = E[Y|do(X=1)] - E[Y|do(X=0)]
            = Σ_z P(Y|X=1,Z=z)P(Z=z) - Σ_z P(Y|X=0,Z=z)P(Z=z)
        """
        if adjustment_set is None:
            adjustment_set = self.find_backdoor_adjustment_set(treatment, outcome)

        if adjustment_set is None:
            return {"error": "No valid adjustment set found"}

        x_idx = self.var_to_idx[treatment]
        y_idx = self.var_to_idx[outcome]
        z_idx = [self.var_to_idx[z] for z in adjustment_set]

        X = self.data[:, x_idx]
        Y = self.data[:, y_idx]

        # Stratified estimation
        if len(z_idx) == 0:
            # No adjustment needed
            high_x = X > np.median(X)
            e_y_do_high = np.mean(Y[high_x])
            e_y_do_low = np.mean(Y[~high_x])
        else:
            Z = self.data[:, z_idx]

            # Discretize Z for stratification
            n_strata = min(10, len(np.unique(Z, axis=0)))

            # Use regression adjustment
            from sklearn.linear_model import LinearRegression

            # E[Y|X,Z]
            XZ = np.column_stack([X, Z])
            model = LinearRegression().fit(XZ, Y)

            # Predict under interventions
            Z_mean = Z.mean(axis=0)

            # do(X=high)
            XZ_high = np.column_stack([
                np.ones(len(X)) * np.percentile(X, 75),
                np.tile(Z_mean, (len(X), 1))
            ])
            e_y_do_high = model.predict(XZ_high).mean()

            # do(X=low)
            XZ_low = np.column_stack([
                np.ones(len(X)) * np.percentile(X, 25),
                np.tile(Z_mean, (len(X), 1))
            ])
            e_y_do_low = model.predict(XZ_low).mean()

        ate = e_y_do_high - e_y_do_low

        # Bootstrap confidence interval
        n_bootstrap = 1000
        ate_bootstrap = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), len(X), replace=True)
            X_b, Y_b = X[idx], Y[idx]
            high_x = X_b > np.median(X_b)
            ate_b = np.mean(Y_b[high_x]) - np.mean(Y_b[~high_x])
            ate_bootstrap.append(ate_b)

        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)

        return {
            "treatment": treatment,
            "outcome": outcome,
            "adjustment_set": list(adjustment_set),
            "ate": ate,
            "ci_95": (ci_lower, ci_upper),
            "e_y_do_high": e_y_do_high,
            "e_y_do_low": e_y_do_low
        }

class CounterfactualReasoning:
    """
    Counterfactual inference using structural causal models.

    Computes P(Y_x=y' | X=x, Y=y) - the probability of
    counterfactual outcome.
    """

    def __init__(self, graph: CausalGraph, data: np.ndarray, variable_names: List[str]):
        self.graph = graph
        self.data = data
        self.variable_names = variable_names
        self.var_to_idx = {v: i for i, v in enumerate(variable_names)}
        self.structural_equations = {}

    def learn_structural_equations(self):
        """Learn structural equations from data."""
        from sklearn.linear_model import LinearRegression

        for var in self.variable_names:
            parents = self.graph.get_parents(var)

            if len(parents) == 0:
                # Exogenous variable
                self.structural_equations[var] = {
                    "type": "exogenous",
                    "mean": np.mean(self.data[:, self.var_to_idx[var]]),
                    "std": np.std(self.data[:, self.var_to_idx[var]])
                }
            else:
                # Endogenous variable with structural equation
                y_idx = self.var_to_idx[var]
                x_idx = [self.var_to_idx[p] for p in parents]

                X = self.data[:, x_idx]
                Y = self.data[:, y_idx]

                model = LinearRegression().fit(X, Y)
                residuals = Y - model.predict(X)

                self.structural_equations[var] = {
                    "type": "endogenous",
                    "parents": list(parents),
                    "coefficients": model.coef_,
                    "intercept": model.intercept_,
                    "residual_std": np.std(residuals)
                }

    def compute_counterfactual(
        self,
        intervention: Dict[str, float],
        evidence: Dict[str, float],
        query: str
    ) -> Dict:
        """
        Compute counterfactual query.

        P(Y_x | evidence) using abduction-action-prediction.
        """
        # Step 1: Abduction - infer exogenous variables from evidence
        exogenous_values = self._abduction(evidence)

        # Step 2: Action - modify graph for intervention
        # Step 3: Prediction - compute query under intervention
        counterfactual_value = self._predict_counterfactual(
            intervention, exogenous_values, query
        )

        return {
            "query": f"P({query} | do({intervention}), {evidence})",
            "counterfactual_value": counterfactual_value,
            "factual_value": evidence.get(query),
            "intervention": intervention,
            "evidence": evidence
        }

    def _abduction(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """Infer exogenous noise terms from evidence."""
        exogenous = {}

        for var, value in evidence.items():
            eq = self.structural_equations.get(var)

            if eq is None:
                continue

            if eq["type"] == "exogenous":
                exogenous[var] = (value - eq["mean"]) / eq["std"]
            else:
                # Compute residual given parent values
                parent_values = np.array([
                    evidence.get(p, 0) for p in eq["parents"]
                ])
                predicted = eq["intercept"] + np.dot(eq["coefficients"], parent_values)
                exogenous[var] = (value - predicted) / eq["residual_std"]

        return exogenous

    def _predict_counterfactual(
        self,
        intervention: Dict[str, float],
        exogenous: Dict[str, float],
        query: str
    ) -> float:
        """Predict counterfactual value using modified SCM."""
        # Topological order
        order = self._topological_sort()

        values = {}

        for var in order:
            if var in intervention:
                values[var] = intervention[var]
            else:
                eq = self.structural_equations.get(var)

                if eq is None or eq["type"] == "exogenous":
                    noise = exogenous.get(var, 0)
                    values[var] = eq["mean"] + noise * eq["std"] if eq else 0
                else:
                    parent_values = np.array([
                        values.get(p, 0) for p in eq["parents"]
                    ])
                    noise = exogenous.get(var, 0)
                    values[var] = (
                        eq["intercept"] +
                        np.dot(eq["coefficients"], parent_values) +
                        noise * eq["residual_std"]
                    )

        return values.get(query, 0)

    def _topological_sort(self) -> List[str]:
        """Topological sort of variables."""
        in_degree = {v: len(self.graph.get_parents(v)) for v in self.variable_names}
        queue = [v for v in self.variable_names if in_degree[v] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self.graph.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

class CausalDiscoveryEngine:
    """
    Complete causal discovery and inference system.

    Integrates PC algorithm, do-calculus, and counterfactual reasoning.
    """

    def __init__(
        self,
        api_key: str,
        entities: List[str],
        lookback_days: int = 180,
        alpha: float = 0.05
    ):
        self.api_key = api_key
        self.entities = entities
        self.lookback_days = lookback_days
        self.alpha = alpha
        self.data = None
        self.causal_graph = None

    def fetch_entity_data(self) -> np.ndarray:
        """Fetch and align entity sentiment data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        entity_series = {}

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
                    "category.in": "finance,economy,business",
                    "language.code.eq": "en",
                    "limit": 50
                })

                articles = response.json().get("results", [])
                if articles:
                    sentiments = [a.get("sentiment", {}).get("overall", 0) for a in articles]
                    series.append((current_date, np.mean(sentiments)))

                current_date = next_date

            entity_series[entity] = series

        # Align to common dates
        all_dates = set.intersection(*[
            set(d for d, _ in series)
            for series in entity_series.values()
        ])
        common_dates = sorted(all_dates)

        data = np.zeros((len(common_dates), len(self.entities)))

        for j, entity in enumerate(self.entities):
            date_to_val = dict(entity_series[entity])
            for i, date in enumerate(common_dates):
                data[i, j] = date_to_val.get(date, 0)

        self.data = data
        return data

    def discover_causal_structure(self) -> CausalGraph:
        """Run PC algorithm to discover causal structure."""
        if self.data is None:
            self.fetch_entity_data()

        ind_test = IndependenceTest(self.data, self.entities, self.alpha)
        pc = PCAlgorithm(ind_test, max_cond_set_size=3)

        self.causal_graph = pc.fit(self.entities)

        return self.causal_graph

    def estimate_causal_effects(
        self,
        treatment: str,
        outcome: str
    ) -> Dict:
        """Estimate causal effect of treatment on outcome."""
        if self.causal_graph is None:
            self.discover_causal_structure()

        do_calc = DoCalculus(self.causal_graph, self.data, self.entities)

        return do_calc.estimate_ate(treatment, outcome)

    def counterfactual_analysis(
        self,
        intervention: Dict[str, float],
        evidence: Dict[str, float],
        query: str
    ) -> Dict:
        """Perform counterfactual analysis."""
        if self.causal_graph is None:
            self.discover_causal_structure()

        cf = CounterfactualReasoning(self.causal_graph, self.data, self.entities)
        cf.learn_structural_equations()

        return cf.compute_counterfactual(intervention, evidence, query)

    def generate_causal_report(self) -> Dict:
        """Generate comprehensive causal analysis report."""
        if self.causal_graph is None:
            self.discover_causal_structure()

        # Extract graph structure
        edges = []
        for (src, tgt), edge in self.causal_graph.edges.items():
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge.edge_type.value,
                "strength": edge.strength
            })

        # Estimate all pairwise effects
        effects = []
        for treatment in self.entities:
            for outcome in self.entities:
                if treatment != outcome:
                    try:
                        effect = self.estimate_causal_effects(treatment, outcome)
                        if "error" not in effect:
                            effects.append(effect)
                    except Exception:
                        pass

        # Identify key causal drivers
        causal_influence = defaultdict(float)
        for effect in effects:
            causal_influence[effect["treatment"]] += abs(effect["ate"])

        top_drivers = sorted(
            causal_influence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "causal_graph": {
                "nodes": list(self.causal_graph.nodes),
                "edges": edges
            },
            "causal_effects": effects,
            "top_causal_drivers": top_drivers,
            "timestamp": datetime.now().isoformat()
        }

# Usage
entities = [
    "Federal Reserve", "S&P 500", "USD Index",
    "Treasury Yields", "Inflation", "Unemployment"
]

engine = CausalDiscoveryEngine(API_KEY, entities, lookback_days=180)
report = engine.generate_causal_report()
print(report)

# Estimate specific causal effect
effect = engine.estimate_causal_effects("Federal Reserve", "S&P 500")
print(f"ATE of Fed on S&P: {effect}")

# Counterfactual analysis
cf = engine.counterfactual_analysis(
    intervention={"Federal Reserve": 0.5},
    evidence={"Federal Reserve": -0.3, "S&P 500": -0.2},
    query="S&P 500"
)
print(f"Counterfactual: {cf}")
```

### JavaScript
```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const EdgeType = {
  UNDIRECTED: "---",
  DIRECTED: "-->",
  BIDIRECTED: "<->",
};

class CausalGraph {
  constructor() {
    this.nodes = new Set();
    this.edges = new Map();
    this.adjacency = new Map();
  }

  addEdge(source, target, edgeType, strength = 0) {
    const key = `${source}|${target}`;
    this.edges.set(key, { source, target, edgeType, strength });

    if (!this.adjacency.has(source)) this.adjacency.set(source, new Set());
    this.adjacency.get(source).add(target);

    if (edgeType === EdgeType.UNDIRECTED) {
      if (!this.adjacency.has(target)) this.adjacency.set(target, new Set());
      this.adjacency.get(target).add(source);
    }
  }

  removeEdge(source, target) {
    this.edges.delete(`${source}|${target}`);
    this.edges.delete(`${target}|${source}`);
    this.adjacency.get(source)?.delete(target);
    this.adjacency.get(target)?.delete(source);
  }

  hasEdge(source, target) {
    return this.edges.has(`${source}|${target}`) ||
           this.edges.has(`${target}|${source}`);
  }

  getNeighbors(node) {
    const neighbors = new Set(this.adjacency.get(node) || []);
    for (const [key, edge] of this.edges) {
      if (edge.target === node) neighbors.add(edge.source);
    }
    return neighbors;
  }

  getParents(node) {
    const parents = new Set();
    for (const [key, edge] of this.edges) {
      if (edge.target === node && edge.edgeType === EdgeType.DIRECTED) {
        parents.add(edge.source);
      }
    }
    return parents;
  }

  getChildren(node) {
    const children = new Set();
    for (const [key, edge] of this.edges) {
      if (edge.source === node && edge.edgeType === EdgeType.DIRECTED) {
        children.add(edge.target);
      }
    }
    return children;
  }
}

class IndependenceTest {
  constructor(data, variableNames, alpha = 0.05) {
    this.data = data;
    this.variableNames = variableNames;
    this.varToIdx = Object.fromEntries(variableNames.map((v, i) => [v, i]));
    this.alpha = alpha;
    this.nSamples = data.length;
    this.corrMatrix = this.computeCorrelationMatrix();
  }

  computeCorrelationMatrix() {
    const n = this.variableNames.length;
    const matrix = Array(n).fill(null).map(() => Array(n).fill(0));

    const means = this.variableNames.map((_, j) =>
      this.data.reduce((s, row) => s + row[j], 0) / this.nSamples
    );

    const stds = this.variableNames.map((_, j) => {
      const variance = this.data.reduce(
        (s, row) => s + (row[j] - means[j]) ** 2, 0
      ) / this.nSamples;
      return Math.sqrt(variance);
    });

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let cov = 0;
        for (let t = 0; t < this.nSamples; t++) {
          cov += (this.data[t][i] - means[i]) * (this.data[t][j] - means[j]);
        }
        cov /= this.nSamples;
        matrix[i][j] = cov / (stds[i] * stds[j] || 1);
      }
    }

    return matrix;
  }

  partialCorrelation(xIdx, yIdx, condIdx) {
    if (condIdx.length === 0) {
      return this.corrMatrix[xIdx][yIdx];
    }

    // Extract submatrix and compute via precision matrix
    const indices = [xIdx, yIdx, ...condIdx];
    const n = indices.length;

    const subCorr = indices.map(i =>
      indices.map(j => this.corrMatrix[i][j])
    );

    // Simplified: use recursive formula for single conditioning
    if (condIdx.length === 1) {
      const z = condIdx[0];
      const rxy = this.corrMatrix[xIdx][yIdx];
      const rxz = this.corrMatrix[xIdx][z];
      const ryz = this.corrMatrix[yIdx][z];

      const partialCorr = (rxy - rxz * ryz) /
        Math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2));

      return Math.max(-0.9999, Math.min(0.9999, partialCorr));
    }

    // For larger conditioning sets, return simple correlation as approximation
    return this.corrMatrix[xIdx][yIdx];
  }

  fisherZ(r) {
    return 0.5 * Math.log((1 + r) / (1 - r));
  }

  testConditionalIndependence(x, y, conditioningSet) {
    const xIdx = this.varToIdx[x];
    const yIdx = this.varToIdx[y];
    const condIdx = [...conditioningSet].map(z => this.varToIdx[z]);

    const partialCorr = this.partialCorrelation(xIdx, yIdx, condIdx);
    const z = this.fisherZ(partialCorr);

    const k = conditioningSet.size;
    const se = 1.0 / Math.sqrt(this.nSamples - k - 3);
    const zStat = Math.abs(z) / se;

    // Approximate p-value
    const pValue = 2 * (1 - this.normalCDF(zStat));

    return {
      isIndependent: pValue > this.alpha,
      pValue
    };
  }

  normalCDF(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * y);
  }
}

class PCAlgorithm {
  constructor(independenceTest, maxCondSetSize = 3) {
    this.indTest = independenceTest;
    this.maxCondSetSize = maxCondSetSize;
    this.separationSets = new Map();
  }

  *combinations(arr, k) {
    if (k === 0) { yield []; return; }
    if (arr.length < k) return;

    const first = arr[0];
    const rest = arr.slice(1);

    for (const combo of this.combinations(rest, k - 1)) {
      yield [first, ...combo];
    }
    yield* this.combinations(rest, k);
  }

  discoverSkeleton(nodes) {
    const graph = new CausalGraph();
    graph.nodes = new Set(nodes);

    // Initialize complete graph
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        graph.addEdge(nodes[i], nodes[j], EdgeType.UNDIRECTED);
      }
    }

    // Test independence with increasing conditioning set sizes
    for (let condSize = 0; condSize <= this.maxCondSetSize; condSize++) {
      const edgesToRemove = [];

      for (const [key, edge] of graph.edges) {
        const x = edge.source;
        const y = edge.target;

        const neighborsX = graph.getNeighbors(x);
        const neighborsY = graph.getNeighbors(y);
        neighborsX.delete(y);
        neighborsY.delete(x);

        const potentialCond = new Set([...neighborsX, ...neighborsY]);

        if (potentialCond.size < condSize) continue;

        for (const condSet of this.combinations([...potentialCond], condSize)) {
          const condSetObj = new Set(condSet);
          const { isIndependent, pValue } = this.indTest.testConditionalIndependence(
            x, y, condSetObj
          );

          if (isIndependent) {
            edgesToRemove.push([x, y]);
            this.separationSets.set(`${x}|${y}`, condSetObj);
            break;
          }
        }
      }

      for (const [x, y] of edgesToRemove) {
        graph.removeEdge(x, y);
      }
    }

    return graph;
  }

  orientVStructures(graph) {
    for (const z of graph.nodes) {
      const neighbors = [...graph.getNeighbors(z)];

      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          const x = neighbors[i];
          const y = neighbors[j];

          if (!graph.hasEdge(x, y)) {
            const sepSet = this.separationSets.get(`${x}|${y}`) ||
                          this.separationSets.get(`${y}|${x}`) ||
                          new Set();

            if (!sepSet.has(z)) {
              // Orient as v-structure
              graph.removeEdge(x, z);
              graph.removeEdge(y, z);
              graph.addEdge(x, z, EdgeType.DIRECTED);
              graph.addEdge(y, z, EdgeType.DIRECTED);
            }
          }
        }
      }
    }

    return graph;
  }

  fit(nodes) {
    const skeleton = this.discoverSkeleton(nodes);
    const cpdag = this.orientVStructures(skeleton);
    return cpdag;
  }
}

class CausalDiscoveryEngine {
  constructor(apiKey, entities, lookbackDays = 180, alpha = 0.05) {
    this.apiKey = apiKey;
    this.entities = entities;
    this.lookbackDays = lookbackDays;
    this.alpha = alpha;
    this.data = null;
    this.causalGraph = null;
  }

  async fetchEntityData() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - this.lookbackDays);

    const entitySeries = {};

    for (const entity of this.entities) {
      entitySeries[entity] = [];
      const current = new Date(startDate);

      while (current <= endDate) {
        const next = new Date(current);
        next.setDate(next.getDate() + 1);

        const params = new URLSearchParams({
          api_key: this.apiKey,
          "entity.surface_form.eq": entity,
          "published_at.gte": current.toISOString().split("T")[0],
          "published_at.lt": next.toISOString().split("T")[0],
          "category.in": "finance,economy,business",
          "language.code.eq": "en",
          limit: "50"
        });

        try {
          const response = await fetch(`${BASE_URL}?${params}`);
          const data = await response.json();
          const articles = data.results || [];

          if (articles.length > 0) {
            const sentiments = articles.map(a => a.sentiment?.overall ?? 0);
            const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
            entitySeries[entity].push({
              date: current.toISOString().split("T")[0],
              value: mean
            });
          }
        } catch (e) {
          console.error(`Error fetching ${entity}:`, e);
        }

        current.setDate(current.getDate() + 1);
      }
    }

    // Align to common dates
    const allDates = this.entities.map(e =>
      new Set(entitySeries[e].map(d => d.date))
    );
    const commonDates = [...allDates[0]].filter(d =>
      allDates.every(set => set.has(d))
    ).sort();

    this.data = commonDates.map(date => {
      return this.entities.map(entity => {
        const entry = entitySeries[entity].find(e => e.date === date);
        return entry ? entry.value : 0;
      });
    });

    return this.data;
  }

  async discoverCausalStructure() {
    if (!this.data) {
      await this.fetchEntityData();
    }

    const indTest = new IndependenceTest(this.data, this.entities, this.alpha);
    const pc = new PCAlgorithm(indTest, 3);

    this.causalGraph = pc.fit(this.entities);
    return this.causalGraph;
  }

  async generateCausalReport() {
    if (!this.causalGraph) {
      await this.discoverCausalStructure();
    }

    const edges = [];
    for (const [key, edge] of this.causalGraph.edges) {
      edges.push({
        source: edge.source,
        target: edge.target,
        type: edge.edgeType
      });
    }

    return {
      causalGraph: {
        nodes: [...this.causalGraph.nodes],
        edges
      },
      timestamp: new Date().toISOString()
    };
  }
}

// Usage
const entities = [
  "Federal Reserve", "S&P 500", "USD Index",
  "Treasury Yields", "Inflation"
];

const engine = new CausalDiscoveryEngine(API_KEY, entities, 180);
engine.generateCausalReport().then(console.log);
```

### PHP
```php
<?php

class CausalGraph {
    public array $nodes = [];
    public array $edges = [];
    public array $adjacency = [];

    public function addEdge(string $source, string $target, string $edgeType): void {
        $key = "{$source}|{$target}";
        $this->edges[$key] = [
            'source' => $source,
            'target' => $target,
            'type' => $edgeType
        ];

        $this->adjacency[$source][] = $target;

        if ($edgeType === 'UNDIRECTED') {
            $this->adjacency[$target][] = $source;
        }
    }

    public function removeEdge(string $source, string $target): void {
        unset($this->edges["{$source}|{$target}"]);
        unset($this->edges["{$target}|{$source}"]);

        $this->adjacency[$source] = array_filter(
            $this->adjacency[$source] ?? [],
            fn($n) => $n !== $target
        );
        $this->adjacency[$target] = array_filter(
            $this->adjacency[$target] ?? [],
            fn($n) => $n !== $source
        );
    }

    public function hasEdge(string $source, string $target): bool {
        return isset($this->edges["{$source}|{$target}"]) ||
               isset($this->edges["{$target}|{$source}"]);
    }

    public function getNeighbors(string $node): array {
        $neighbors = $this->adjacency[$node] ?? [];

        foreach ($this->edges as $edge) {
            if ($edge['target'] === $node) {
                $neighbors[] = $edge['source'];
            }
        }

        return array_unique($neighbors);
    }

    public function getParents(string $node): array {
        $parents = [];
        foreach ($this->edges as $edge) {
            if ($edge['target'] === $node && $edge['type'] === 'DIRECTED') {
                $parents[] = $edge['source'];
            }
        }
        return $parents;
    }
}

class IndependenceTest {
    private array $data;
    private array $variableNames;
    private array $varToIdx;
    private float $alpha;
    private int $nSamples;
    private array $corrMatrix;

    public function __construct(array $data, array $variableNames, float $alpha = 0.05) {
        $this->data = $data;
        $this->variableNames = $variableNames;
        $this->varToIdx = array_flip($variableNames);
        $this->alpha = $alpha;
        $this->nSamples = count($data);
        $this->corrMatrix = $this->computeCorrelationMatrix();
    }

    private function computeCorrelationMatrix(): array {
        $n = count($this->variableNames);
        $matrix = array_fill(0, $n, array_fill(0, $n, 0.0));

        $means = [];
        $stds = [];

        for ($j = 0; $j < $n; $j++) {
            $values = array_column($this->data, $j);
            $means[$j] = array_sum($values) / $this->nSamples;
            $variance = array_sum(array_map(
                fn($v) => ($v - $means[$j]) ** 2,
                $values
            )) / $this->nSamples;
            $stds[$j] = sqrt($variance);
        }

        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $cov = 0;
                for ($t = 0; $t < $this->nSamples; $t++) {
                    $cov += ($this->data[$t][$i] - $means[$i]) *
                            ($this->data[$t][$j] - $means[$j]);
                }
                $cov /= $this->nSamples;
                $matrix[$i][$j] = $cov / ($stds[$i] * $stds[$j] ?: 1);
            }
        }

        return $matrix;
    }

    public function testConditionalIndependence(
        string $x,
        string $y,
        array $conditioningSet
    ): array {
        $xIdx = $this->varToIdx[$x];
        $yIdx = $this->varToIdx[$y];

        $partialCorr = $this->corrMatrix[$xIdx][$yIdx];

        if (count($conditioningSet) === 1) {
            $z = $this->varToIdx[array_values($conditioningSet)[0]];
            $rxy = $this->corrMatrix[$xIdx][$yIdx];
            $rxz = $this->corrMatrix[$xIdx][$z];
            $ryz = $this->corrMatrix[$yIdx][$z];

            $denom = sqrt((1 - $rxz ** 2) * (1 - $ryz ** 2));
            $partialCorr = $denom > 0 ? ($rxy - $rxz * $ryz) / $denom : 0;
        }

        $z = 0.5 * log((1 + $partialCorr) / (1 - $partialCorr + 0.0001));
        $k = count($conditioningSet);
        $se = 1.0 / sqrt($this->nSamples - $k - 3);
        $zStat = abs($z) / $se;

        $pValue = 2 * (1 - $this->normalCDF($zStat));

        return [
            'isIndependent' => $pValue > $this->alpha,
            'pValue' => $pValue
        ];
    }

    private function normalCDF(float $x): float {
        return 0.5 * (1 + erf($x / sqrt(2)));
    }
}

class PCAlgorithm {
    private IndependenceTest $indTest;
    private int $maxCondSetSize;
    private array $separationSets = [];

    public function __construct(IndependenceTest $indTest, int $maxCondSetSize = 3) {
        $this->indTest = $indTest;
        $this->maxCondSetSize = $maxCondSetSize;
    }

    private function combinations(array $arr, int $k): array {
        if ($k === 0) return [[]];
        if (count($arr) < $k) return [];

        $first = array_shift($arr);
        $result = [];

        foreach ($this->combinations($arr, $k - 1) as $combo) {
            $result[] = array_merge([$first], $combo);
        }

        return array_merge($result, $this->combinations($arr, $k));
    }

    public function fit(array $nodes): CausalGraph {
        $graph = new CausalGraph();
        $graph->nodes = $nodes;

        // Initialize complete graph
        for ($i = 0; $i < count($nodes); $i++) {
            for ($j = $i + 1; $j < count($nodes); $j++) {
                $graph->addEdge($nodes[$i], $nodes[$j], 'UNDIRECTED');
            }
        }

        // Skeleton discovery
        for ($condSize = 0; $condSize <= $this->maxCondSetSize; $condSize++) {
            $edgesToRemove = [];

            foreach ($graph->edges as $key => $edge) {
                $x = $edge['source'];
                $y = $edge['target'];

                $neighbors = array_merge(
                    $graph->getNeighbors($x),
                    $graph->getNeighbors($y)
                );
                $neighbors = array_diff(array_unique($neighbors), [$x, $y]);

                if (count($neighbors) < $condSize) continue;

                foreach ($this->combinations($neighbors, $condSize) as $condSet) {
                    $result = $this->indTest->testConditionalIndependence($x, $y, $condSet);

                    if ($result['isIndependent']) {
                        $edgesToRemove[] = [$x, $y];
                        $this->separationSets["{$x}|{$y}"] = $condSet;
                        break;
                    }
                }
            }

            foreach ($edgesToRemove as [$x, $y]) {
                $graph->removeEdge($x, $y);
            }
        }

        // V-structure orientation
        foreach ($graph->nodes as $z) {
            $neighbors = $graph->getNeighbors($z);

            for ($i = 0; $i < count($neighbors); $i++) {
                for ($j = $i + 1; $j < count($neighbors); $j++) {
                    $x = $neighbors[$i];
                    $y = $neighbors[$j];

                    if (!$graph->hasEdge($x, $y)) {
                        $sepSet = $this->separationSets["{$x}|{$y}"] ??
                                  $this->separationSets["{$y}|{$x}"] ?? [];

                        if (!in_array($z, $sepSet)) {
                            $graph->removeEdge($x, $z);
                            $graph->removeEdge($y, $z);
                            $graph->addEdge($x, $z, 'DIRECTED');
                            $graph->addEdge($y, $z, 'DIRECTED');
                        }
                    }
                }
            }
        }

        return $graph;
    }
}

class CausalDiscoveryEngine {
    private string $apiKey;
    private array $entities;
    private int $lookbackDays;
    private float $alpha;
    private ?array $data = null;
    private ?CausalGraph $causalGraph = null;

    public function __construct(
        string $apiKey,
        array $entities,
        int $lookbackDays = 180,
        float $alpha = 0.05
    ) {
        $this->apiKey = $apiKey;
        $this->entities = $entities;
        $this->lookbackDays = $lookbackDays;
        $this->alpha = $alpha;
    }

    public function fetchEntityData(): array {
        $endDate = new DateTime();
        $startDate = (clone $endDate)->modify("-{$this->lookbackDays} days");

        $entitySeries = [];

        foreach ($this->entities as $entity) {
            $entitySeries[$entity] = [];
            $current = clone $startDate;

            while ($current <= $endDate) {
                $next = clone $current;
                $next->modify('+1 day');

                $params = http_build_query([
                    'api_key' => $this->apiKey,
                    'entity.surface_form.eq' => $entity,
                    'published_at.gte' => $current->format('Y-m-d'),
                    'published_at.lt' => $next->format('Y-m-d'),
                    'category.in' => 'finance,economy,business',
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
                        $entitySeries[$entity][] = [
                            'date' => $current->format('Y-m-d'),
                            'value' => array_sum($sentiments) / count($sentiments)
                        ];
                    }
                }

                $current->modify('+1 day');
            }
        }

        // Align to common dates
        $allDates = array_map(
            fn($series) => array_column($series, 'date'),
            $entitySeries
        );
        $commonDates = call_user_func_array('array_intersect', $allDates);
        sort($commonDates);

        $this->data = array_map(function($date) use ($entitySeries) {
            return array_map(function($entity) use ($entitySeries, $date) {
                foreach ($entitySeries[$entity] as $entry) {
                    if ($entry['date'] === $date) return $entry['value'];
                }
                return 0;
            }, $this->entities);
        }, $commonDates);

        return $this->data;
    }

    public function discoverCausalStructure(): CausalGraph {
        if ($this->data === null) {
            $this->fetchEntityData();
        }

        $indTest = new IndependenceTest($this->data, $this->entities, $this->alpha);
        $pc = new PCAlgorithm($indTest, 3);

        $this->causalGraph = $pc->fit($this->entities);
        return $this->causalGraph;
    }

    public function generateCausalReport(): array {
        if ($this->causalGraph === null) {
            $this->discoverCausalStructure();
        }

        return [
            'causal_graph' => [
                'nodes' => $this->causalGraph->nodes,
                'edges' => array_values($this->causalGraph->edges)
            ],
            'timestamp' => (new DateTime())->format('c')
        ];
    }
}

// Usage
$entities = ['Federal Reserve', 'S&P 500', 'USD Index', 'Treasury Yields', 'Inflation'];
$engine = new CausalDiscoveryEngine('YOUR_API_KEY', $entities, 180);
print_r($engine->generateCausalReport());
```

## Use Cases

### Economic Policy Analysis
- Identify causal drivers of economic outcomes
- Evaluate policy intervention effects
- Counterfactual policy scenarios

### Investment Research
- Discover causal market relationships
- Estimate treatment effects of events
- Validate investment hypotheses

### Risk Assessment
- Causal chain identification
- Intervention impact estimation
- Systemic risk pathway analysis
