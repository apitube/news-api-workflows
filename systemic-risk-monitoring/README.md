# Systemic Risk Monitoring

> **Comprehensive systemic risk assessment with Value-at-Risk (VaR), Expected Shortfall (ES), network-based risk indicators, stress testing, and early warning systems** — built for the [APITube News API](https://apitube.io).

This enterprise-grade workflow implements a multi-dimensional systemic risk monitoring framework that combines market-wide sentiment analysis with advanced risk metrics. It provides real-time risk assessment, tail risk quantification, stress scenario analysis, and early warning signals for systemic events.

## Overview

The Systemic Risk Monitoring system provides:

- **Value-at-Risk (VaR) Estimation** — Parametric, historical, and Monte Carlo VaR from news sentiment
- **Expected Shortfall (ES)** — Conditional VaR for tail risk quantification
- **Network Risk Metrics** — DebtRank-inspired interconnectedness measures
- **Stress Testing Framework** — Scenario-based impact assessment
- **Absorption Ratio** — PCA-based systemic risk indicator
- **Turbulence Index** — Mahalanobis distance for regime detection
- **SRISK-like Metrics** — Capital shortfall estimation under stress

## Parameters

### Data Collection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `organization.name` | string | Financial institutions to monitor (comma-separated) |
| `category.id` | string | Category ID: `medtop:04000000` (economy/business/finance) |
| `published_at.start` / `.end` | datetime | Analysis window |
| `source.rank.opr.min` | number | High-authority source filter (0-7) |
| `sentiment.overall.score.max` | number | Negative sentiment threshold for stress events |
| `language.code` | string | Language filter (default: `en`) |

### Risk Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `var_confidence` | float | VaR confidence level (default: 0.99) |
| `var_horizon` | integer | Risk horizon in days (default: 10) |
| `monte_carlo_paths` | integer | MC simulation paths (default: 10000) |
| `stress_severity` | float | Stress scenario severity multiplier |
| `lookback_window` | integer | Historical data window (default: 252) |

### Risk Indicators

| Indicator | Description |
|-----------|-------------|
| **VaR** | Maximum expected loss at confidence level |
| **ES/CVaR** | Expected loss beyond VaR threshold |
| **Absorption Ratio** | Fraction of variance explained by top PCs |
| **Turbulence Index** | Statistical distance from normal regime |
| **Interconnectedness** | Network-based contagion risk measure |
| **Capital Buffer** | Estimated capital adequacy under stress |

## Quick Start

### cURL
```bash
curl -G "https://api.apitube.io/v1/news/everything" \
  --data-urlencode "organization.name=JPMorgan,Goldman Sachs,Bank of America,Citigroup,Wells Fargo" \
  --data-urlencode "category.id=medtop:04000000" \
  --data-urlencode "published_at.start=2024-01-01" \
  --data-urlencode "sentiment.overall.score.max=0" \
  --data-urlencode "language.code=en" \
  --data-urlencode "per_page=50" \
  --data-urlencode "api_key=YOUR_API_KEY"
```

### Python
```python
import requests
import numpy as np
from scipy import stats
from scipy.linalg import eigh
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class RiskObservation:
    """Daily risk observation for an entity."""
    entity: str
    date: datetime
    sentiment_mean: float
    sentiment_vol: float
    coverage_volume: int
    negative_ratio: float  # Fraction of negative articles

@dataclass
class VaRResult:
    """Value-at-Risk calculation result."""
    var_parametric: float
    var_historical: float
    var_monte_carlo: float
    expected_shortfall: float
    confidence_level: float
    horizon_days: int

@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    severity: float
    sentiment_shock: float
    coverage_multiplier: float
    duration_days: int

@dataclass
class SystemicRiskReport:
    """Comprehensive systemic risk assessment."""
    timestamp: datetime
    var_system: VaRResult
    absorption_ratio: float
    turbulence_index: float
    interconnectedness_score: float
    capital_buffer_ratio: float
    stress_test_results: Dict[str, float]
    risk_level: str
    early_warning_signals: List[str]

class ValueAtRiskCalculator:
    """
    Multi-method Value-at-Risk calculator.

    Implements parametric, historical simulation, and Monte Carlo VaR.
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        horizon_days: int = 10,
        monte_carlo_paths: int = 10000
    ):
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        self.monte_carlo_paths = monte_carlo_paths

    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Parametric VaR assuming normal distribution.

        VaR = -μ + σ * z_α * √T
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_alpha = stats.norm.ppf(self.confidence_level)

        var = portfolio_value * (
            -mu * self.horizon_days +
            z_alpha * sigma * np.sqrt(self.horizon_days)
        )

        return max(0, var)

    def calculate_historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Historical simulation VaR.

        Uses empirical distribution of returns.
        """
        # Aggregate returns over horizon
        if len(returns) < self.horizon_days:
            horizon_returns = returns
        else:
            horizon_returns = np.array([
                np.sum(returns[i:i+self.horizon_days])
                for i in range(len(returns) - self.horizon_days + 1)
            ])

        # VaR is the negative of the (1-α) percentile
        var_percentile = np.percentile(
            horizon_returns,
            (1 - self.confidence_level) * 100
        )

        return max(0, -var_percentile * portfolio_value)

    def calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Monte Carlo simulation VaR.

        Simulates future paths using fitted distribution.
        """
        mu = np.mean(returns)
        sigma = np.std(returns)

        # Simulate paths
        simulated_returns = np.random.normal(
            mu * self.horizon_days,
            sigma * np.sqrt(self.horizon_days),
            self.monte_carlo_paths
        )

        # VaR from simulated distribution
        var_percentile = np.percentile(
            simulated_returns,
            (1 - self.confidence_level) * 100
        )

        return max(0, -var_percentile * portfolio_value)

    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Expected Shortfall (Conditional VaR).

        Average loss beyond VaR threshold.
        """
        # Aggregate returns over horizon
        if len(returns) < self.horizon_days:
            horizon_returns = returns
        else:
            horizon_returns = np.array([
                np.sum(returns[i:i+self.horizon_days])
                for i in range(len(returns) - self.horizon_days + 1)
            ])

        # Find VaR threshold
        var_threshold = np.percentile(
            horizon_returns,
            (1 - self.confidence_level) * 100
        )

        # Average of returns below VaR
        tail_returns = horizon_returns[horizon_returns <= var_threshold]

        if len(tail_returns) == 0:
            return self.calculate_historical_var(returns, portfolio_value)

        expected_shortfall = -np.mean(tail_returns) * portfolio_value

        return max(0, expected_shortfall)

    def compute_full_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> VaRResult:
        """Compute VaR using all methods."""
        return VaRResult(
            var_parametric=self.calculate_parametric_var(returns, portfolio_value),
            var_historical=self.calculate_historical_var(returns, portfolio_value),
            var_monte_carlo=self.calculate_monte_carlo_var(returns, portfolio_value),
            expected_shortfall=self.calculate_expected_shortfall(returns, portfolio_value),
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days
        )

class AbsorptionRatioCalculator:
    """
    Absorption Ratio for systemic risk measurement.

    Measures fraction of total variance explained by a fixed number
    of eigenvectors. High AR indicates high systemic risk.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def calculate(self, returns_matrix: np.ndarray) -> float:
        """
        Calculate absorption ratio.

        AR = sum(λ_1...λ_n) / sum(all λ)
        """
        if returns_matrix.shape[1] < self.n_components:
            n_components = returns_matrix.shape[1]
        else:
            n_components = self.n_components

        # Covariance matrix
        cov_matrix = np.cov(returns_matrix.T)

        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        # Absorption ratio
        total_variance = np.sum(eigenvalues)
        explained_variance = np.sum(eigenvalues[:n_components])

        if total_variance == 0:
            return 0.0

        return explained_variance / total_variance

class TurbulenceIndex:
    """
    Mahalanobis distance-based turbulence measure.

    Detects when current market state is statistically unusual.
    """

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.historical_mean = None
        self.historical_cov_inv = None

    def fit(self, returns_matrix: np.ndarray) -> "TurbulenceIndex":
        """Fit on historical data."""
        self.historical_mean = np.mean(returns_matrix, axis=0)

        cov_matrix = np.cov(returns_matrix.T)
        # Regularize covariance matrix
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

        try:
            self.historical_cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            self.historical_cov_inv = np.linalg.pinv(cov_matrix)

        return self

    def calculate(self, current_returns: np.ndarray) -> float:
        """
        Calculate turbulence index for current period.

        T = (r - μ)' Σ^(-1) (r - μ)
        """
        if self.historical_mean is None:
            raise ValueError("Must fit before calculating")

        diff = current_returns - self.historical_mean

        turbulence = diff @ self.historical_cov_inv @ diff

        return float(turbulence)

    def is_turbulent(
        self,
        current_returns: np.ndarray,
        threshold_percentile: float = 0.95
    ) -> Tuple[bool, float]:
        """Check if current state is turbulent."""
        turbulence = self.calculate(current_returns)

        # Chi-square distribution threshold
        n_assets = len(current_returns)
        threshold = stats.chi2.ppf(threshold_percentile, n_assets)

        return turbulence > threshold, turbulence

class NetworkRiskAnalyzer:
    """
    Network-based systemic risk analysis.

    Measures interconnectedness and contagion potential.
    """

    def __init__(self):
        self.adjacency_matrix = None
        self.node_names = []

    def build_network(
        self,
        returns_matrix: np.ndarray,
        entity_names: List[str],
        correlation_threshold: float = 0.5
    ) -> None:
        """Build network from correlation structure."""
        self.node_names = entity_names
        n_entities = len(entity_names)

        # Correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)

        # Adjacency matrix (edges where correlation > threshold)
        self.adjacency_matrix = np.zeros((n_entities, n_entities))

        for i in range(n_entities):
            for j in range(n_entities):
                if i != j and abs(corr_matrix[i, j]) > correlation_threshold:
                    self.adjacency_matrix[i, j] = abs(corr_matrix[i, j])

    def calculate_interconnectedness(self) -> float:
        """
        Calculate network interconnectedness score.

        Based on network density and average connection strength.
        """
        if self.adjacency_matrix is None:
            return 0.0

        n = len(self.node_names)
        max_edges = n * (n - 1)

        # Density: fraction of possible edges that exist
        actual_edges = np.sum(self.adjacency_matrix > 0)
        density = actual_edges / max_edges if max_edges > 0 else 0

        # Average strength of existing connections
        nonzero = self.adjacency_matrix[self.adjacency_matrix > 0]
        avg_strength = np.mean(nonzero) if len(nonzero) > 0 else 0

        # Combined score
        return density * avg_strength

    def identify_systemically_important(
        self,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Identify systemically important entities.

        Uses eigenvector centrality (PageRank-like).
        """
        if self.adjacency_matrix is None:
            return []

        # Eigenvector centrality
        eigenvalues, eigenvectors = np.linalg.eigh(self.adjacency_matrix)
        centrality = np.abs(eigenvectors[:, -1])  # Largest eigenvalue's vector

        # Normalize
        centrality = centrality / centrality.sum()

        # Rank entities
        ranked = sorted(
            zip(self.node_names, centrality),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_n]

    def simulate_contagion(
        self,
        initial_shock_entity: str,
        shock_magnitude: float = 0.5,
        propagation_rate: float = 0.3,
        max_rounds: int = 10
    ) -> Dict[str, float]:
        """
        Simulate contagion spread through network.

        Returns final stress levels for each entity.
        """
        if self.adjacency_matrix is None:
            return {}

        n = len(self.node_names)
        stress_levels = np.zeros(n)

        # Initial shock
        try:
            shock_idx = self.node_names.index(initial_shock_entity)
            stress_levels[shock_idx] = shock_magnitude
        except ValueError:
            return {}

        # Propagation rounds
        for round_num in range(max_rounds):
            new_stress = stress_levels.copy()

            for i in range(n):
                # Receive stress from connected nodes
                for j in range(n):
                    if i != j and self.adjacency_matrix[j, i] > 0:
                        contagion = (
                            stress_levels[j] *
                            self.adjacency_matrix[j, i] *
                            propagation_rate
                        )
                        new_stress[i] = min(1.0, new_stress[i] + contagion)

            # Check convergence
            if np.allclose(stress_levels, new_stress, atol=1e-6):
                break

            stress_levels = new_stress

        return dict(zip(self.node_names, stress_levels))

class StressTestingFramework:
    """
    Stress testing framework for scenario analysis.
    """

    PREDEFINED_SCENARIOS = [
        StressScenario("Mild Downturn", 1.0, -0.3, 1.5, 5),
        StressScenario("Moderate Stress", 1.5, -0.5, 2.0, 10),
        StressScenario("Severe Crisis", 2.0, -0.8, 3.0, 20),
        StressScenario("2008-Like Event", 3.0, -0.95, 4.0, 30),
        StressScenario("Black Swan", 4.0, -1.0, 5.0, 45)
    ]

    def __init__(self, base_sentiment_vol: float = 0.3):
        self.base_sentiment_vol = base_sentiment_vol

    def run_stress_test(
        self,
        current_sentiment: float,
        current_coverage: int,
        scenario: StressScenario
    ) -> Dict:
        """Run stress test for a single scenario."""
        # Stressed sentiment
        stressed_sentiment = current_sentiment + scenario.sentiment_shock
        stressed_sentiment = max(-1.0, stressed_sentiment)

        # Stressed coverage
        stressed_coverage = int(current_coverage * scenario.coverage_multiplier)

        # Impact estimation (simplified capital impact model)
        sentiment_impact = abs(scenario.sentiment_shock) * scenario.severity
        coverage_impact = np.log1p(stressed_coverage - current_coverage) / 10

        total_impact = sentiment_impact + coverage_impact

        # Estimated capital shortfall (as percentage)
        capital_shortfall_pct = min(100, total_impact * 20)

        return {
            "scenario": scenario.name,
            "stressed_sentiment": stressed_sentiment,
            "stressed_coverage": stressed_coverage,
            "total_impact": total_impact,
            "capital_shortfall_pct": capital_shortfall_pct,
            "recovery_time_days": scenario.duration_days * scenario.severity
        }

    def run_all_scenarios(
        self,
        current_sentiment: float,
        current_coverage: int
    ) -> List[Dict]:
        """Run all predefined stress scenarios."""
        return [
            self.run_stress_test(current_sentiment, current_coverage, scenario)
            for scenario in self.PREDEFINED_SCENARIOS
        ]

class SystemicRiskMonitor:
    """
    Complete systemic risk monitoring system.

    Integrates all risk metrics with news data collection.
    """

    def __init__(
        self,
        api_key: str,
        entities: List[str],
        lookback_days: int = 252,
        var_confidence: float = 0.99,
        var_horizon: int = 10
    ):
        self.api_key = api_key
        self.entities = entities
        self.lookback_days = lookback_days

        # Initialize components
        self.var_calculator = ValueAtRiskCalculator(
            confidence_level=var_confidence,
            horizon_days=var_horizon
        )
        self.absorption_calculator = AbsorptionRatioCalculator(n_components=3)
        self.turbulence_index = TurbulenceIndex(lookback_window=60)
        self.network_analyzer = NetworkRiskAnalyzer()
        self.stress_tester = StressTestingFramework()

        # Data storage
        self.observations: Dict[str, List[RiskObservation]] = defaultdict(list)

    def fetch_entity_data(
        self,
        entity: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[RiskObservation]:
        """Fetch daily risk observations for an entity."""
        observations = []
        current_date = start_date

        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)

            response = requests.get(BASE_URL, params={
                "api_key": self.api_key,
                "organization.name": entity,
                "published_at.start": current_date.strftime("%Y-%m-%d"),
                "published_at.end": next_date.strftime("%Y-%m-%d"),
                "category.id": "medtop:04000000",
                "language.code": "en",
                "per_page": 50
            })

            articles = response.json().get("results", [])

            if articles:
                sentiments = [
                    a.get("sentiment", {}).get("overall", 0)
                    for a in articles
                ]
                negative_count = sum(1 for s in sentiments if s < 0)

                obs = RiskObservation(
                    entity=entity,
                    date=current_date,
                    sentiment_mean=np.mean(sentiments),
                    sentiment_vol=np.std(sentiments) if len(sentiments) > 1 else 0,
                    coverage_volume=len(articles),
                    negative_ratio=negative_count / len(articles)
                )
                observations.append(obs)

            current_date = next_date

        return observations

    def collect_all_data(self) -> None:
        """Collect data for all entities."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for entity in self.entities:
            self.observations[entity] = self.fetch_entity_data(
                entity, start_date, end_date
            )

    def build_returns_matrix(self) -> Tuple[np.ndarray, List[datetime]]:
        """Build aligned returns matrix from observations."""
        # Find common dates
        all_dates = [
            set(obs.date for obs in self.observations[entity])
            for entity in self.entities
        ]
        common_dates = sorted(set.intersection(*all_dates))

        if len(common_dates) < 30:
            raise ValueError("Insufficient overlapping data")

        # Build matrix (sentiment changes as "returns")
        n_dates = len(common_dates)
        n_entities = len(self.entities)
        returns = np.zeros((n_dates - 1, n_entities))

        for j, entity in enumerate(self.entities):
            obs_dict = {obs.date: obs for obs in self.observations[entity]}

            for i, date in enumerate(common_dates[1:]):
                prev_date = common_dates[i]
                if date in obs_dict and prev_date in obs_dict:
                    returns[i, j] = (
                        obs_dict[date].sentiment_mean -
                        obs_dict[prev_date].sentiment_mean
                    )

        return returns, common_dates

    def calculate_system_var(self) -> VaRResult:
        """Calculate system-wide VaR."""
        returns, _ = self.build_returns_matrix()

        # Equal-weighted portfolio
        portfolio_returns = returns.mean(axis=1)

        return self.var_calculator.compute_full_var(portfolio_returns)

    def calculate_absorption_ratio(self) -> float:
        """Calculate absorption ratio."""
        returns, _ = self.build_returns_matrix()
        return self.absorption_calculator.calculate(returns)

    def calculate_turbulence(self) -> Tuple[bool, float]:
        """Calculate turbulence index."""
        returns, _ = self.build_returns_matrix()

        # Fit on historical data
        self.turbulence_index.fit(returns[:-10])

        # Check recent period
        recent_returns = returns[-5:].mean(axis=0)

        return self.turbulence_index.is_turbulent(recent_returns)

    def calculate_interconnectedness(self) -> float:
        """Calculate network interconnectedness."""
        returns, _ = self.build_returns_matrix()

        self.network_analyzer.build_network(
            returns,
            self.entities,
            correlation_threshold=0.5
        )

        return self.network_analyzer.calculate_interconnectedness()

    def run_stress_tests(self) -> List[Dict]:
        """Run stress test scenarios."""
        # Get current system state
        current_sentiment = np.mean([
            self.observations[e][-1].sentiment_mean
            for e in self.entities
            if self.observations[e]
        ])

        current_coverage = sum(
            self.observations[e][-1].coverage_volume
            for e in self.entities
            if self.observations[e]
        )

        return self.stress_tester.run_all_scenarios(
            current_sentiment, current_coverage
        )

    def generate_early_warnings(
        self,
        var_result: VaRResult,
        absorption_ratio: float,
        is_turbulent: bool,
        turbulence_value: float,
        interconnectedness: float
    ) -> List[str]:
        """Generate early warning signals."""
        warnings = []

        # VaR warning
        if var_result.var_parametric > 0.5:
            warnings.append(
                f"HIGH VAR ALERT: System VaR at {var_result.var_parametric:.1%}"
            )

        # Expected shortfall warning
        if var_result.expected_shortfall > 0.7:
            warnings.append(
                f"TAIL RISK ALERT: Expected Shortfall at {var_result.expected_shortfall:.1%}"
            )

        # Absorption ratio warning
        if absorption_ratio > 0.8:
            warnings.append(
                f"CONCENTRATION ALERT: Absorption Ratio at {absorption_ratio:.1%}"
            )

        # Turbulence warning
        if is_turbulent:
            warnings.append(
                f"TURBULENCE ALERT: Market state abnormal (T={turbulence_value:.2f})"
            )

        # Interconnectedness warning
        if interconnectedness > 0.6:
            warnings.append(
                f"CONTAGION RISK: High interconnectedness ({interconnectedness:.1%})"
            )

        return warnings

    def calculate_capital_buffer_ratio(
        self,
        var_result: VaRResult,
        stress_results: List[Dict]
    ) -> float:
        """
        Estimate capital buffer adequacy ratio.

        > 1.0 means adequate buffer, < 1.0 means insufficient.
        """
        # Worst case stress impact
        worst_case = max(r["capital_shortfall_pct"] for r in stress_results)

        # Assume 8% base capital requirement
        base_capital = 8.0

        # Additional buffer from VaR (simplified)
        var_buffer = var_result.expected_shortfall * 100 * 2.5

        total_capital = base_capital + var_buffer
        required_capital = worst_case

        if required_capital == 0:
            return float("inf")

        return total_capital / required_capital

    def generate_report(self) -> SystemicRiskReport:
        """Generate comprehensive systemic risk report."""
        self.collect_all_data()

        # Calculate all metrics
        var_result = self.calculate_system_var()
        absorption_ratio = self.calculate_absorption_ratio()
        is_turbulent, turbulence_value = self.calculate_turbulence()
        interconnectedness = self.calculate_interconnectedness()
        stress_results = self.run_stress_tests()
        capital_buffer = self.calculate_capital_buffer_ratio(var_result, stress_results)

        # Early warnings
        warnings = self.generate_early_warnings(
            var_result,
            absorption_ratio,
            is_turbulent,
            turbulence_value,
            interconnectedness
        )

        # Overall risk level
        risk_score = (
            var_result.expected_shortfall * 0.25 +
            absorption_ratio * 0.2 +
            (1 if is_turbulent else 0) * 0.2 +
            interconnectedness * 0.2 +
            (1 - min(1, capital_buffer)) * 0.15
        )

        risk_level = (
            "CRITICAL" if risk_score > 0.7 else
            "HIGH" if risk_score > 0.5 else
            "ELEVATED" if risk_score > 0.3 else
            "MODERATE" if risk_score > 0.15 else
            "LOW"
        )

        return SystemicRiskReport(
            timestamp=datetime.now(),
            var_system=var_result,
            absorption_ratio=absorption_ratio,
            turbulence_index=turbulence_value,
            interconnectedness_score=interconnectedness,
            capital_buffer_ratio=capital_buffer,
            stress_test_results={
                r["scenario"]: r["capital_shortfall_pct"]
                for r in stress_results
            },
            risk_level=risk_level,
            early_warning_signals=warnings
        )

# Usage
entities = [
    "JPMorgan Chase", "Bank of America", "Citigroup",
    "Goldman Sachs", "Morgan Stanley", "Wells Fargo"
]

monitor = SystemicRiskMonitor(
    API_KEY,
    entities,
    lookback_days=180,
    var_confidence=0.99,
    var_horizon=10
)

report = monitor.generate_report()
print(f"Risk Level: {report.risk_level}")
print(f"System VaR (99%, 10-day): {report.var_system.var_parametric:.2%}")
print(f"Expected Shortfall: {report.var_system.expected_shortfall:.2%}")
print(f"Absorption Ratio: {report.absorption_ratio:.2%}")
print(f"Turbulence Index: {report.turbulence_index:.2f}")
print(f"Interconnectedness: {report.interconnectedness_score:.2%}")
print(f"Capital Buffer Ratio: {report.capital_buffer_ratio:.2f}x")
print(f"Early Warnings: {report.early_warning_signals}")
```

### JavaScript
```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class ValueAtRiskCalculator {
  constructor(confidenceLevel = 0.99, horizonDays = 10, mcPaths = 10000) {
    this.confidenceLevel = confidenceLevel;
    this.horizonDays = horizonDays;
    this.mcPaths = mcPaths;
  }

  normalPPF(p) {
    // Approximation of inverse normal CDF
    const a = [
      -3.969683028665376e1, 2.209460984245205e2,
      -2.759285104469687e2, 1.383577518672690e2,
      -3.066479806614716e1, 2.506628277459239e0
    ];
    const b = [
      -5.447609879822406e1, 1.615858368580409e2,
      -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1
    ];
    const c = [
      -7.784894002430293e-3, -3.223964580411365e-1,
      -2.400758277161838e0, -2.549732539343734e0,
      4.374664141464968e0, 2.938163982698783e0
    ];
    const d = [
      7.784695709041462e-3, 3.224671290700398e-1,
      2.445134137142996e0, 3.754408661907416e0
    ];

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q, r;

    if (p < pLow) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p <= pHigh) {
      q = p - 0.5;
      r = q * q;
      return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
             (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
  }

  mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  std(arr) {
    const m = this.mean(arr);
    return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length);
  }

  percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    if (lower === upper) return sorted[lower];
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
  }

  calculateParametricVaR(returns, portfolioValue = 1.0) {
    const mu = this.mean(returns);
    const sigma = this.std(returns);
    const zAlpha = this.normalPPF(this.confidenceLevel);

    const var_ = portfolioValue * (
      -mu * this.horizonDays +
      zAlpha * sigma * Math.sqrt(this.horizonDays)
    );

    return Math.max(0, var_);
  }

  calculateHistoricalVaR(returns, portfolioValue = 1.0) {
    let horizonReturns;
    if (returns.length < this.horizonDays) {
      horizonReturns = returns;
    } else {
      horizonReturns = [];
      for (let i = 0; i <= returns.length - this.horizonDays; i++) {
        let sum = 0;
        for (let j = 0; j < this.horizonDays; j++) {
          sum += returns[i + j];
        }
        horizonReturns.push(sum);
      }
    }

    const varPercentile = this.percentile(
      horizonReturns,
      (1 - this.confidenceLevel) * 100
    );

    return Math.max(0, -varPercentile * portfolioValue);
  }

  calculateExpectedShortfall(returns, portfolioValue = 1.0) {
    let horizonReturns;
    if (returns.length < this.horizonDays) {
      horizonReturns = returns;
    } else {
      horizonReturns = [];
      for (let i = 0; i <= returns.length - this.horizonDays; i++) {
        let sum = 0;
        for (let j = 0; j < this.horizonDays; j++) {
          sum += returns[i + j];
        }
        horizonReturns.push(sum);
      }
    }

    const varThreshold = this.percentile(
      horizonReturns,
      (1 - this.confidenceLevel) * 100
    );

    const tailReturns = horizonReturns.filter(r => r <= varThreshold);

    if (tailReturns.length === 0) {
      return this.calculateHistoricalVaR(returns, portfolioValue);
    }

    return Math.max(0, -this.mean(tailReturns) * portfolioValue);
  }

  computeFullVaR(returns, portfolioValue = 1.0) {
    return {
      varParametric: this.calculateParametricVaR(returns, portfolioValue),
      varHistorical: this.calculateHistoricalVaR(returns, portfolioValue),
      expectedShortfall: this.calculateExpectedShortfall(returns, portfolioValue),
      confidenceLevel: this.confidenceLevel,
      horizonDays: this.horizonDays
    };
  }
}

class AbsorptionRatioCalculator {
  constructor(nComponents = 5) {
    this.nComponents = nComponents;
  }

  calculate(returnsMatrix) {
    const nEntities = returnsMatrix[0].length;
    const nComponents = Math.min(this.nComponents, nEntities);

    // Compute covariance matrix
    const means = [];
    for (let j = 0; j < nEntities; j++) {
      means.push(returnsMatrix.reduce((s, row) => s + row[j], 0) / returnsMatrix.length);
    }

    const cov = Array(nEntities).fill(null).map(() => Array(nEntities).fill(0));
    for (let i = 0; i < nEntities; i++) {
      for (let j = 0; j < nEntities; j++) {
        let sum = 0;
        for (let t = 0; t < returnsMatrix.length; t++) {
          sum += (returnsMatrix[t][i] - means[i]) * (returnsMatrix[t][j] - means[j]);
        }
        cov[i][j] = sum / (returnsMatrix.length - 1);
      }
    }

    // Power iteration for eigenvalues (simplified)
    const eigenvalues = [];
    for (let i = 0; i < nEntities; i++) {
      eigenvalues.push(cov[i][i]); // Diagonal approximation
    }
    eigenvalues.sort((a, b) => b - a);

    const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
    const explainedVariance = eigenvalues.slice(0, nComponents).reduce((a, b) => a + b, 0);

    return totalVariance > 0 ? explainedVariance / totalVariance : 0;
  }
}

class SystemicRiskMonitor {
  constructor(apiKey, entities, lookbackDays = 180) {
    this.apiKey = apiKey;
    this.entities = entities;
    this.lookbackDays = lookbackDays;
    this.observations = {};
    this.varCalculator = new ValueAtRiskCalculator(0.99, 10);
    this.absorptionCalculator = new AbsorptionRatioCalculator(3);
  }

  async fetchEntityData(entity, startDate, endDate) {
    const observations = [];
    const current = new Date(startDate);

    while (current <= endDate) {
      const next = new Date(current);
      next.setDate(next.getDate() + 1);

      const params = new URLSearchParams({
        api_key: this.apiKey,
        "organization.name": entity,
        "published_at.start": current.toISOString().split("T")[0],
        "published_at.end": next.toISOString().split("T")[0],
        "category.id": "medtop:04000000",
        "language.code": "en",
        per_page: "50"
      });

      try {
        const response = await fetch(`${BASE_URL}?${params}`);
        const data = await response.json();
        const articles = data.results || [];

        if (articles.length > 0) {
          const sentiments = articles.map(a => a.sentiment?.overall ?? 0);
          const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
          const negCount = sentiments.filter(s => s < 0).length;

          observations.push({
            entity,
            date: new Date(current),
            sentimentMean: mean,
            coverageVolume: articles.length,
            negativeRatio: negCount / articles.length
          });
        }
      } catch (e) {
        console.error(`Error fetching ${entity}:`, e);
      }

      current.setDate(current.getDate() + 1);
    }

    return observations;
  }

  async collectAllData() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - this.lookbackDays);

    for (const entity of this.entities) {
      this.observations[entity] = await this.fetchEntityData(entity, startDate, endDate);
    }
  }

  buildReturnsMatrix() {
    const allDates = Object.values(this.observations).map(
      obs => new Set(obs.map(o => o.date.toISOString().split("T")[0]))
    );

    const commonDates = [...allDates[0]].filter(
      d => allDates.every(set => set.has(d))
    ).sort();

    const returns = [];

    for (let i = 1; i < commonDates.length; i++) {
      const row = [];
      for (const entity of this.entities) {
        const obs = this.observations[entity];
        const curr = obs.find(o => o.date.toISOString().split("T")[0] === commonDates[i]);
        const prev = obs.find(o => o.date.toISOString().split("T")[0] === commonDates[i-1]);

        if (curr && prev) {
          row.push(curr.sentimentMean - prev.sentimentMean);
        } else {
          row.push(0);
        }
      }
      returns.push(row);
    }

    return returns;
  }

  calculateSystemVaR() {
    const returns = this.buildReturnsMatrix();
    const portfolioReturns = returns.map(
      row => row.reduce((a, b) => a + b, 0) / row.length
    );
    return this.varCalculator.computeFullVaR(portfolioReturns);
  }

  calculateAbsorptionRatio() {
    const returns = this.buildReturnsMatrix();
    return this.absorptionCalculator.calculate(returns);
  }

  calculateInterconnectedness() {
    const returns = this.buildReturnsMatrix();
    const n = this.entities.length;

    // Calculate correlations
    let highCorrCount = 0;
    let totalCorr = 0;
    let pairs = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const xi = returns.map(row => row[i]);
        const xj = returns.map(row => row[j]);
        const corr = this.correlation(xi, xj);

        if (Math.abs(corr) > 0.5) highCorrCount++;
        totalCorr += Math.abs(corr);
        pairs++;
      }
    }

    return (highCorrCount / pairs) * (totalCorr / pairs);
  }

  correlation(x, y) {
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;

    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    return num / Math.sqrt(denX * denY);
  }

  determineRiskLevel(varResult, absorptionRatio, interconnectedness) {
    const riskScore =
      varResult.expectedShortfall * 0.3 +
      absorptionRatio * 0.35 +
      interconnectedness * 0.35;

    if (riskScore > 0.7) return "CRITICAL";
    if (riskScore > 0.5) return "HIGH";
    if (riskScore > 0.3) return "ELEVATED";
    if (riskScore > 0.15) return "MODERATE";
    return "LOW";
  }

  async generateReport() {
    await this.collectAllData();

    const varResult = this.calculateSystemVaR();
    const absorptionRatio = this.calculateAbsorptionRatio();
    const interconnectedness = this.calculateInterconnectedness();
    const riskLevel = this.determineRiskLevel(varResult, absorptionRatio, interconnectedness);

    const warnings = [];
    if (varResult.expectedShortfall > 0.5) {
      warnings.push(`TAIL RISK: ES at ${(varResult.expectedShortfall * 100).toFixed(1)}%`);
    }
    if (absorptionRatio > 0.8) {
      warnings.push(`CONCENTRATION: AR at ${(absorptionRatio * 100).toFixed(1)}%`);
    }
    if (interconnectedness > 0.5) {
      warnings.push(`CONTAGION: High interconnectedness`);
    }

    return {
      timestamp: new Date().toISOString(),
      riskLevel,
      systemVaR: varResult,
      absorptionRatio,
      interconnectedness,
      earlyWarnings: warnings
    };
  }
}

// Usage
const entities = [
  "JPMorgan Chase", "Bank of America", "Citigroup",
  "Goldman Sachs", "Morgan Stanley"
];

const monitor = new SystemicRiskMonitor(API_KEY, entities, 180);
monitor.generateReport().then(console.log);
```

### PHP
```php
<?php

class ValueAtRiskCalculator {
    private float $confidenceLevel;
    private int $horizonDays;

    public function __construct(
        float $confidenceLevel = 0.99,
        int $horizonDays = 10
    ) {
        $this->confidenceLevel = $confidenceLevel;
        $this->horizonDays = $horizonDays;
    }

    private function mean(array $arr): float {
        return array_sum($arr) / count($arr);
    }

    private function std(array $arr): float {
        $mean = $this->mean($arr);
        $variance = array_sum(array_map(
            fn($x) => ($x - $mean) ** 2,
            $arr
        )) / count($arr);
        return sqrt($variance);
    }

    private function percentile(array $arr, float $p): float {
        sort($arr);
        $idx = ($p / 100) * (count($arr) - 1);
        $lower = (int) floor($idx);
        $upper = (int) ceil($idx);

        if ($lower === $upper) return $arr[$lower];

        return $arr[$lower] + ($arr[$upper] - $arr[$lower]) * ($idx - $lower);
    }

    public function calculateParametricVaR(
        array $returns,
        float $portfolioValue = 1.0
    ): float {
        $mu = $this->mean($returns);
        $sigma = $this->std($returns);

        // Z-score for confidence level (approximation)
        $zAlpha = 2.326; // 99% confidence

        $var = $portfolioValue * (
            -$mu * $this->horizonDays +
            $zAlpha * $sigma * sqrt($this->horizonDays)
        );

        return max(0, $var);
    }

    public function calculateHistoricalVaR(
        array $returns,
        float $portfolioValue = 1.0
    ): float {
        if (count($returns) < $this->horizonDays) {
            $horizonReturns = $returns;
        } else {
            $horizonReturns = [];
            for ($i = 0; $i <= count($returns) - $this->horizonDays; $i++) {
                $sum = 0;
                for ($j = 0; $j < $this->horizonDays; $j++) {
                    $sum += $returns[$i + $j];
                }
                $horizonReturns[] = $sum;
            }
        }

        $varPercentile = $this->percentile(
            $horizonReturns,
            (1 - $this->confidenceLevel) * 100
        );

        return max(0, -$varPercentile * $portfolioValue);
    }

    public function calculateExpectedShortfall(
        array $returns,
        float $portfolioValue = 1.0
    ): float {
        if (count($returns) < $this->horizonDays) {
            $horizonReturns = $returns;
        } else {
            $horizonReturns = [];
            for ($i = 0; $i <= count($returns) - $this->horizonDays; $i++) {
                $sum = 0;
                for ($j = 0; $j < $this->horizonDays; $j++) {
                    $sum += $returns[$i + $j];
                }
                $horizonReturns[] = $sum;
            }
        }

        $varThreshold = $this->percentile(
            $horizonReturns,
            (1 - $this->confidenceLevel) * 100
        );

        $tailReturns = array_filter(
            $horizonReturns,
            fn($r) => $r <= $varThreshold
        );

        if (empty($tailReturns)) {
            return $this->calculateHistoricalVaR($returns, $portfolioValue);
        }

        return max(0, -$this->mean(array_values($tailReturns)) * $portfolioValue);
    }

    public function computeFullVaR(
        array $returns,
        float $portfolioValue = 1.0
    ): array {
        return [
            'var_parametric' => $this->calculateParametricVaR($returns, $portfolioValue),
            'var_historical' => $this->calculateHistoricalVaR($returns, $portfolioValue),
            'expected_shortfall' => $this->calculateExpectedShortfall($returns, $portfolioValue),
            'confidence_level' => $this->confidenceLevel,
            'horizon_days' => $this->horizonDays
        ];
    }
}

class SystemicRiskMonitor {
    private string $apiKey;
    private array $entities;
    private int $lookbackDays;
    private array $observations = [];
    private ValueAtRiskCalculator $varCalculator;

    public function __construct(
        string $apiKey,
        array $entities,
        int $lookbackDays = 180
    ) {
        $this->apiKey = $apiKey;
        $this->entities = $entities;
        $this->lookbackDays = $lookbackDays;
        $this->varCalculator = new ValueAtRiskCalculator(0.99, 10);
    }

    public function fetchEntityData(
        string $entity,
        DateTime $startDate,
        DateTime $endDate
    ): array {
        $observations = [];
        $current = clone $startDate;

        while ($current <= $endDate) {
            $next = clone $current;
            $next->modify('+1 day');

            $params = http_build_query([
                'api_key' => $this->apiKey,
                'organization.name' => $entity,
                'published_at.start' => $current->format('Y-m-d'),
                'published_at.end' => $next->format('Y-m-d'),
                'category.id' => 'medtop:04000000',
                'language.code' => 'en',
                'per_page' => 50
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
                    $negCount = count(array_filter($sentiments, fn($s) => $s < 0));

                    $observations[] = [
                        'entity' => $entity,
                        'date' => $current->format('Y-m-d'),
                        'sentiment_mean' => array_sum($sentiments) / count($sentiments),
                        'coverage_volume' => count($articles),
                        'negative_ratio' => $negCount / count($articles)
                    ];
                }
            }

            $current->modify('+1 day');
        }

        return $observations;
    }

    public function collectAllData(): void {
        $endDate = new DateTime();
        $startDate = (clone $endDate)->modify("-{$this->lookbackDays} days");

        foreach ($this->entities as $entity) {
            $this->observations[$entity] = $this->fetchEntityData(
                $entity, $startDate, $endDate
            );
        }
    }

    public function buildReturnsMatrix(): array {
        $allDates = array_map(
            fn($obs) => array_column($obs, 'date'),
            $this->observations
        );

        $commonDates = call_user_func_array('array_intersect', $allDates);
        sort($commonDates);

        $returns = [];

        for ($i = 1; $i < count($commonDates); $i++) {
            $row = [];
            foreach ($this->entities as $entity) {
                $obs = $this->observations[$entity];
                $currIdx = array_search($commonDates[$i], array_column($obs, 'date'));
                $prevIdx = array_search($commonDates[$i-1], array_column($obs, 'date'));

                if ($currIdx !== false && $prevIdx !== false) {
                    $row[] = $obs[$currIdx]['sentiment_mean'] - $obs[$prevIdx]['sentiment_mean'];
                } else {
                    $row[] = 0;
                }
            }
            $returns[] = $row;
        }

        return $returns;
    }

    private function correlation(array $x, array $y): float {
        $n = count($x);
        $meanX = array_sum($x) / $n;
        $meanY = array_sum($y) / $n;

        $num = 0;
        $denX = 0;
        $denY = 0;

        for ($i = 0; $i < $n; $i++) {
            $dx = $x[$i] - $meanX;
            $dy = $y[$i] - $meanY;
            $num += $dx * $dy;
            $denX += $dx * $dx;
            $denY += $dy * $dy;
        }

        return $num / sqrt($denX * $denY);
    }

    public function calculateInterconnectedness(): float {
        $returns = $this->buildReturnsMatrix();
        $n = count($this->entities);

        $highCorrCount = 0;
        $totalCorr = 0;
        $pairs = 0;

        for ($i = 0; $i < $n; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $xi = array_column($returns, $i);
                $xj = array_column($returns, $j);
                $corr = $this->correlation($xi, $xj);

                if (abs($corr) > 0.5) $highCorrCount++;
                $totalCorr += abs($corr);
                $pairs++;
            }
        }

        return ($highCorrCount / $pairs) * ($totalCorr / $pairs);
    }

    public function generateReport(): array {
        $this->collectAllData();

        $returns = $this->buildReturnsMatrix();
        $portfolioReturns = array_map(
            fn($row) => array_sum($row) / count($row),
            $returns
        );

        $varResult = $this->varCalculator->computeFullVaR($portfolioReturns);
        $interconnectedness = $this->calculateInterconnectedness();

        // Risk level determination
        $riskScore =
            $varResult['expected_shortfall'] * 0.4 +
            $interconnectedness * 0.6;

        $riskLevel = match(true) {
            $riskScore > 0.7 => 'CRITICAL',
            $riskScore > 0.5 => 'HIGH',
            $riskScore > 0.3 => 'ELEVATED',
            $riskScore > 0.15 => 'MODERATE',
            default => 'LOW'
        };

        $warnings = [];
        if ($varResult['expected_shortfall'] > 0.5) {
            $warnings[] = sprintf(
                'TAIL RISK: ES at %.1f%%',
                $varResult['expected_shortfall'] * 100
            );
        }
        if ($interconnectedness > 0.5) {
            $warnings[] = 'CONTAGION: High interconnectedness';
        }

        return [
            'timestamp' => (new DateTime())->format('c'),
            'risk_level' => $riskLevel,
            'system_var' => $varResult,
            'interconnectedness' => $interconnectedness,
            'early_warnings' => $warnings
        ];
    }
}

// Usage
$entities = [
    'JPMorgan Chase', 'Bank of America', 'Citigroup',
    'Goldman Sachs', 'Morgan Stanley'
];

$monitor = new SystemicRiskMonitor('YOUR_API_KEY', $entities, 180);
print_r($monitor->generateReport());
```

## Use Cases

### Financial Stability Monitoring
- Central bank systemic risk assessment
- Macro-prudential supervision
- Financial stability reports

### Risk Management
- Enterprise-wide risk aggregation
- Tail risk quantification
- Stress testing frameworks

### Regulatory Compliance
- Basel III/IV capital requirements
- SRISK and CoVaR reporting
- Systemically important institution identification

### Investment Management
- Risk-adjusted portfolio construction
- Drawdown protection strategies
- Crisis alpha generation
