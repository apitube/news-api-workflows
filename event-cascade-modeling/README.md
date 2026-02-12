# Predictive Event Cascade Modeling

Workflow for modeling how news events cascade through markets and entities, predicting secondary impacts using graph propagation, Monte Carlo simulation for scenario analysis, causal inference, and multi-step impact forecasting using the [APITube News API](https://apitube.io).

## Overview

The **Predictive Event Cascade Modeling** workflow implements sophisticated event propagation analysis by modeling how initial events trigger secondary effects across connected entities, using graph-based cascade simulation, Monte Carlo methods for probability estimation, historical pattern matching for impact prediction, and temporal lag analysis. Features include impact probability matrices, cascade visualization, scenario trees, and automated impact forecasting. Ideal for risk management, scenario planning, trading desks, and strategic analysis.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Quick Start

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import random
import math

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class EntityRelationshipGraph:
    """
    Graph representing relationships between entities for cascade modeling.
    """

    def __init__(self):
        self.nodes = {}  # entity -> attributes
        self.edges = defaultdict(lambda: {"weight": 0, "lag_days": []})
        self.historical_cascades = []

    def add_entity(self, name, entity_type="organization", sector=None):
        """Add entity node."""
        self.nodes[name] = {
            "type": entity_type,
            "sector": sector,
            "baseline_coverage": 0,
            "volatility": 0
        }

    def add_relationship(self, source, target, relationship_type, weight=1.0):
        """Add directed relationship edge."""
        key = (source, target)
        self.edges[key]["weight"] = weight
        self.edges[key]["type"] = relationship_type

    def learn_from_articles(self, articles):
        """Learn relationships from article co-occurrences."""
        for article in articles:
            entities = [e.get("name") for e in article.get("entities", []) if e.get("name")]

            # Add nodes
            for entity in article.get("entities", []):
                if entity.get("name") and entity.get("name") not in self.nodes:
                    self.add_entity(entity.get("name"), entity.get("type", "organization"))

            # Add/update edges
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    self.edges[(e1, e2)]["weight"] += 1
                    self.edges[(e2, e1)]["weight"] += 1

    def get_connected_entities(self, entity, min_weight=2):
        """Get entities connected to given entity."""
        connected = []
        for (source, target), data in self.edges.items():
            if source == entity and data["weight"] >= min_weight:
                connected.append({"entity": target, "weight": data["weight"]})
        return sorted(connected, key=lambda x: x["weight"], reverse=True)


class ImpactPropagationModel:
    """
    Models how impact propagates through entity network.
    """

    def __init__(self, graph):
        self.graph = graph
        self.propagation_matrix = {}
        self.historical_impacts = defaultdict(list)

    def calculate_transmission_probability(self, source, target):
        """Calculate probability of impact transmission between entities."""
        edge_key = (source, target)
        if edge_key not in self.graph.edges:
            return 0

        edge = self.graph.edges[edge_key]
        weight = edge["weight"]

        # Normalize by source's total connections
        total_weight = sum(
            d["weight"] for (s, _), d in self.graph.edges.items()
            if s == source
        )

        if total_weight == 0:
            return 0

        # Base probability from edge weight
        base_prob = weight / total_weight

        # Adjust for relationship type
        type_multipliers = {
            "supplier": 0.8,
            "customer": 0.7,
            "competitor": 0.5,
            "partner": 0.6,
            "subsidiary": 0.9,
            "investor": 0.4
        }
        multiplier = type_multipliers.get(edge.get("type", ""), 0.5)

        return min(1.0, base_prob * multiplier * 2)  # Scale up

    def build_propagation_matrix(self):
        """Build probability matrix for all entity pairs."""
        entities = list(self.graph.nodes.keys())

        for source in entities:
            for target in entities:
                if source != target:
                    prob = self.calculate_transmission_probability(source, target)
                    if prob > 0:
                        self.propagation_matrix[(source, target)] = prob

        return self.propagation_matrix

    def simulate_cascade(self, initial_entity, initial_impact, max_steps=5, decay=0.7):
        """Simulate single cascade from initial event."""
        cascade = [{
            "step": 0,
            "entity": initial_entity,
            "impact": initial_impact,
            "cumulative_prob": 1.0
        }]

        affected = {initial_entity: initial_impact}
        frontier = [(initial_entity, initial_impact, 1.0)]

        for step in range(1, max_steps + 1):
            new_frontier = []

            for source, source_impact, source_prob in frontier:
                # Get potential targets
                for (s, target), prob in self.propagation_matrix.items():
                    if s == source and target not in affected:
                        # Calculate transmitted impact
                        transmitted_impact = source_impact * decay * prob

                        if transmitted_impact > 0.01:  # Threshold
                            cumulative_prob = source_prob * prob

                            cascade.append({
                                "step": step,
                                "entity": target,
                                "impact": transmitted_impact,
                                "source": source,
                                "transmission_prob": prob,
                                "cumulative_prob": cumulative_prob
                            })

                            affected[target] = transmitted_impact
                            new_frontier.append((target, transmitted_impact, cumulative_prob))

            frontier = new_frontier
            if not frontier:
                break

        return cascade

    def learn_historical_lag(self, source_entity, target_entity, articles_source, articles_target):
        """Learn typical lag between source and target entity impacts."""
        # Find coverage spikes in source
        source_spikes = self._find_coverage_spikes(articles_source)

        # Find corresponding spikes in target
        target_spikes = self._find_coverage_spikes(articles_target)

        # Calculate lags
        lags = []
        for source_date in source_spikes:
            for target_date in target_spikes:
                lag = (target_date - source_date).days
                if 0 < lag <= 14:  # Within 2 weeks
                    lags.append(lag)

        if lags:
            self.historical_impacts[(source_entity, target_entity)] = {
                "avg_lag": statistics.mean(lags),
                "median_lag": statistics.median(lags),
                "observations": len(lags)
            }

        return lags

    def _find_coverage_spikes(self, articles, threshold=2.0):
        """Find dates with coverage spikes."""
        daily_counts = defaultdict(int)

        for article in articles:
            date = article.get("published_at", "")[:10]
            daily_counts[date] += 1

        if not daily_counts:
            return []

        counts = list(daily_counts.values())
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 1

        spikes = []
        for date, count in daily_counts.items():
            if count > mean + threshold * std:
                spikes.append(datetime.fromisoformat(date))

        return spikes


class MonteCarloSimulator:
    """
    Monte Carlo simulation for cascade probability estimation.
    """

    def __init__(self, propagation_model, num_simulations=1000):
        self.model = propagation_model
        self.num_simulations = num_simulations

    def run_simulation(self, initial_entity, initial_impact, max_steps=5):
        """Run Monte Carlo simulation of cascades."""
        results = defaultdict(lambda: {"impact_sum": 0, "hit_count": 0, "impacts": []})

        for _ in range(self.num_simulations):
            cascade = self._simulate_stochastic_cascade(
                initial_entity, initial_impact, max_steps
            )

            for event in cascade:
                entity = event["entity"]
                impact = event["impact"]
                results[entity]["impact_sum"] += impact
                results[entity]["hit_count"] += 1
                results[entity]["impacts"].append(impact)

        # Calculate statistics
        simulation_results = {}
        for entity, data in results.items():
            hit_prob = data["hit_count"] / self.num_simulations
            avg_impact = data["impact_sum"] / max(data["hit_count"], 1)

            if data["impacts"]:
                impact_std = statistics.stdev(data["impacts"]) if len(data["impacts"]) > 1 else 0
                percentile_95 = sorted(data["impacts"])[int(len(data["impacts"]) * 0.95)] if data["impacts"] else 0
            else:
                impact_std = 0
                percentile_95 = 0

            simulation_results[entity] = {
                "hit_probability": hit_prob,
                "expected_impact": avg_impact * hit_prob,
                "conditional_impact": avg_impact,
                "impact_volatility": impact_std,
                "var_95": percentile_95
            }

        return simulation_results

    def _simulate_stochastic_cascade(self, initial_entity, initial_impact, max_steps, decay=0.7):
        """Single stochastic cascade simulation."""
        cascade = [{"entity": initial_entity, "impact": initial_impact, "step": 0}]
        affected = {initial_entity: initial_impact}
        frontier = [(initial_entity, initial_impact)]

        for step in range(1, max_steps + 1):
            new_frontier = []

            for source, source_impact in frontier:
                for (s, target), prob in self.model.propagation_matrix.items():
                    if s == source and target not in affected:
                        # Stochastic transmission
                        if random.random() < prob:
                            # Random impact with noise
                            noise = random.gauss(1, 0.2)
                            transmitted = source_impact * decay * noise

                            if transmitted > 0.01:
                                cascade.append({
                                    "entity": target,
                                    "impact": transmitted,
                                    "step": step
                                })
                                affected[target] = transmitted
                                new_frontier.append((target, transmitted))

            frontier = new_frontier
            if not frontier:
                break

        return cascade

    def calculate_var(self, initial_entity, initial_impact, confidence=0.95):
        """Calculate Value at Risk for cascade impact."""
        results = self.run_simulation(initial_entity, initial_impact)

        total_impacts = []
        for _ in range(self.num_simulations):
            total = initial_impact
            for entity, data in results.items():
                if entity != initial_entity:
                    if random.random() < data["hit_probability"]:
                        total += data["conditional_impact"] * random.gauss(1, 0.2)
            total_impacts.append(total)

        sorted_impacts = sorted(total_impacts)
        var_index = int(len(sorted_impacts) * confidence)
        var = sorted_impacts[var_index] if var_index < len(sorted_impacts) else sorted_impacts[-1]

        return {
            "var": var,
            "confidence": confidence,
            "mean_total_impact": statistics.mean(total_impacts),
            "max_total_impact": max(total_impacts)
        }


class ScenarioAnalyzer:
    """
    Scenario analysis for event impacts.
    """

    SCENARIO_TEMPLATES = {
        "earnings_miss": {
            "initial_impact": 0.8,
            "affected_relationships": ["investor", "analyst", "competitor"],
            "decay": 0.6
        },
        "product_recall": {
            "initial_impact": 0.9,
            "affected_relationships": ["customer", "supplier", "competitor"],
            "decay": 0.7
        },
        "ceo_departure": {
            "initial_impact": 0.7,
            "affected_relationships": ["investor", "partner", "subsidiary"],
            "decay": 0.5
        },
        "regulatory_action": {
            "initial_impact": 0.85,
            "affected_relationships": ["competitor", "partner", "industry"],
            "decay": 0.65
        },
        "acquisition_announcement": {
            "initial_impact": 0.75,
            "affected_relationships": ["competitor", "customer", "supplier"],
            "decay": 0.55
        }
    }

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    def analyze_scenario(self, entity, scenario_type):
        """Analyze impact of scenario on entity and network."""
        template = self.SCENARIO_TEMPLATES.get(scenario_type, {
            "initial_impact": 0.5,
            "affected_relationships": [],
            "decay": 0.5
        })

        # Run simulation
        results = self.monte_carlo.run_simulation(
            entity,
            template["initial_impact"],
            max_steps=4
        )

        # Calculate VaR
        var_result = self.monte_carlo.calculate_var(
            entity,
            template["initial_impact"],
            confidence=0.95
        )

        # Identify most at-risk entities
        at_risk = sorted(
            [(e, d) for e, d in results.items() if e != entity],
            key=lambda x: x[1]["expected_impact"],
            reverse=True
        )[:10]

        return {
            "scenario": scenario_type,
            "entity": entity,
            "initial_impact": template["initial_impact"],
            "cascade_results": results,
            "var_95": var_result["var"],
            "mean_total_impact": var_result["mean_total_impact"],
            "most_at_risk": [
                {
                    "entity": e,
                    "probability": d["hit_probability"],
                    "expected_impact": d["expected_impact"]
                }
                for e, d in at_risk
            ]
        }

    def compare_scenarios(self, entity, scenarios):
        """Compare multiple scenarios for an entity."""
        comparisons = []

        for scenario in scenarios:
            analysis = self.analyze_scenario(entity, scenario)
            comparisons.append({
                "scenario": scenario,
                "var_95": analysis["var_95"],
                "mean_impact": analysis["mean_total_impact"],
                "entities_affected": len([r for r in analysis["cascade_results"].values() if r["hit_probability"] > 0.1])
            })

        return sorted(comparisons, key=lambda x: x["var_95"], reverse=True)


class EventCascadeSystem:
    """
    Complete event cascade modeling system.
    """

    def __init__(self):
        self.graph = EntityRelationshipGraph()
        self.propagation_model = ImpactPropagationModel(self.graph)
        self.monte_carlo = None
        self.scenario_analyzer = None

    def build_network(self, seed_entities, days=30):
        """Build entity network from news data."""
        print("Building entity network...")

        all_articles = []
        for entity in seed_entities:
            print(f"  Fetching articles for {entity}...")

            start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity,
                "published_at.start": start,
                "source.rank.opr.min": 4,
                "language.code": "en",
                "per_page": 100,
            })

            articles = resp.json().get("results", [])
            all_articles.extend(articles)

        # Learn relationships
        self.graph.learn_from_articles(all_articles)

        # Build propagation matrix
        self.propagation_model.build_propagation_matrix()

        # Initialize Monte Carlo
        self.monte_carlo = MonteCarloSimulator(self.propagation_model, num_simulations=500)
        self.scenario_analyzer = ScenarioAnalyzer(self.monte_carlo)

        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "articles_processed": len(all_articles)
        }

    def predict_cascade(self, entity, impact_magnitude=0.8):
        """Predict cascade from event at entity."""
        if not self.monte_carlo:
            raise ValueError("Network not built. Call build_network first.")

        return self.monte_carlo.run_simulation(entity, impact_magnitude)

    def run_scenario_analysis(self, entity, scenario):
        """Run scenario analysis."""
        if not self.scenario_analyzer:
            raise ValueError("Network not built. Call build_network first.")

        return self.scenario_analyzer.analyze_scenario(entity, scenario)

    def get_risk_report(self, entity):
        """Generate risk report for entity."""
        scenarios = ["earnings_miss", "product_recall", "ceo_departure", "regulatory_action"]
        comparisons = self.scenario_analyzer.compare_scenarios(entity, scenarios)

        # Get connected entities at risk
        connected = self.graph.get_connected_entities(entity, min_weight=3)

        return {
            "entity": entity,
            "generated_at": datetime.utcnow().isoformat(),
            "scenario_comparison": comparisons,
            "connected_entities": connected[:10],
            "network_position": {
                "direct_connections": len(connected),
                "propagation_reach": len(self.propagation_model.propagation_matrix)
            }
        }


# Run cascade modeling
print("PREDICTIVE EVENT CASCADE MODELING")
print("=" * 70)

system = EventCascadeSystem()

# Build network
seeds = ["Apple", "Microsoft", "Google", "Amazon", "NVIDIA", "Tesla"]
network_stats = system.build_network(seeds, days=30)

print(f"\nNetwork built:")
print(f"  Nodes: {network_stats['nodes']}")
print(f"  Edges: {network_stats['edges']}")
print(f"  Articles: {network_stats['articles_processed']}")

# Predict cascade from Apple event
print("\n" + "=" * 70)
print("CASCADE PREDICTION: Apple Event")
print("-" * 50)

cascade_results = system.predict_cascade("Apple", impact_magnitude=0.8)

print(f"{'Entity':<20} {'Hit Prob':>10} {'Exp Impact':>12} {'VaR 95':>10}")
print("-" * 55)

for entity, data in sorted(cascade_results.items(), key=lambda x: x[1]["expected_impact"], reverse=True)[:10]:
    print(f"{entity:<20} {data['hit_probability']:>10.1%} {data['expected_impact']:>12.3f} {data['var_95']:>10.3f}")

# Scenario analysis
print("\n" + "=" * 70)
print("SCENARIO ANALYSIS: Apple")
print("-" * 50)

for scenario in ["earnings_miss", "product_recall", "regulatory_action"]:
    analysis = system.run_scenario_analysis("Apple", scenario)
    print(f"\n{scenario.upper()}:")
    print(f"  Initial Impact: {analysis['initial_impact']}")
    print(f"  VaR 95%: {analysis['var_95']:.3f}")
    print(f"  Mean Total Impact: {analysis['mean_total_impact']:.3f}")
    print(f"  Most at risk:")
    for risk in analysis['most_at_risk'][:3]:
        print(f"    {risk['entity']}: prob={risk['probability']:.1%}, impact={risk['expected_impact']:.3f}")

# Risk report
print("\n" + "=" * 70)
print("RISK REPORT: Apple")
print("-" * 50)

report = system.get_risk_report("Apple")

print("Scenario Risk Ranking (by VaR 95%):")
for comp in report["scenario_comparison"]:
    print(f"  {comp['scenario']}: VaR={comp['var_95']:.3f}, affected={comp['entities_affected']}")

print(f"\nDirect connections: {report['network_position']['direct_connections']}")
```

## Cascade Model Components

| Component | Function |
|-----------|----------|
| EntityRelationshipGraph | Network of entity relationships |
| ImpactPropagationModel | Transmission probability calculation |
| MonteCarloSimulator | Stochastic cascade simulation |
| ScenarioAnalyzer | Predefined scenario analysis |

## Scenario Templates

| Scenario | Initial Impact | Key Relationships |
|----------|----------------|-------------------|
| earnings_miss | 0.8 | investor, analyst, competitor |
| product_recall | 0.9 | customer, supplier, competitor |
| ceo_departure | 0.7 | investor, partner, subsidiary |
| regulatory_action | 0.85 | competitor, partner, industry |

## Common Use Cases

- **Risk management** — predict cascade risks from events.
- **Scenario planning** — model what-if scenarios.
- **Portfolio risk** — assess cross-holding contagion.
- **Supply chain** — model disruption propagation.
- **Crisis preparation** — understand potential cascade paths.

## See Also

- [examples.md](./examples.md) — additional code examples.
