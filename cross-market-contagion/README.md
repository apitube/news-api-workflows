# Cross-Market Contagion Analysis

> **Financial contagion modeling with Granger causality testing, spillover effect quantification, dynamic conditional correlation, and cross-market shock propagation analysis** — built for the [APITube News API](https://apitube.io).

This enterprise-grade workflow implements sophisticated econometric methods to detect and quantify how news sentiment shocks propagate across markets, sectors, and geographic regions. It identifies contagion pathways, measures spillover intensity, and provides early warning for systemic risk events.

## Overview

The Cross-Market Contagion Analysis system provides:

- **Granger Causality Testing** — Determine if sentiment in one market predicts another with statistical significance
- **Dynamic Conditional Correlation (DCC)** — Track time-varying correlation between market sentiment series
- **Spillover Index** — Diebold-Yilmaz methodology for directional spillover measurement
- **Impulse Response Functions** — Quantify shock propagation magnitude and duration
- **Contagion Network Mapping** — Visualize cross-market influence paths
- **Systemic Risk Indicators** — Aggregate contagion metrics into early warning signals

## Parameters

### Data Collection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity.surface_form.in` | array | Multiple markets/indices to monitor |
| `category.in` | array | Categories: `business`, `finance`, `economy` |
| `published_at.gte` / `.lte` | datetime | Analysis time window |
| `source.rank.lte` | number | Filter high-authority sources |
| `language.code.in` | array | Languages for multi-regional analysis |
| `country.code.in` | array | Countries for geographic contagion |

### Econometric Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `var_lags` | integer | VAR model lag order (default: 5) |
| `granger_max_lags` | integer | Maximum lags for Granger causality (default: 10) |
| `dcc_params` | object | DCC-GARCH parameters (alpha, beta) |
| `spillover_horizon` | integer | Forecast horizon for spillover decomposition |
| `significance_level` | float | P-value threshold for Granger tests (default: 0.05) |

### Market Definitions

| Market Type | Examples |
|-------------|----------|
| **Indices** | S&P 500, FTSE 100, DAX, Nikkei 225, Shanghai Composite |
| **Sectors** | Technology, Financials, Energy, Healthcare, Real Estate |
| **Regions** | US, Europe, Asia-Pacific, Emerging Markets |
| **Assets** | Equities, Bonds, Commodities, Currencies, Crypto |

## Quick Start

### cURL
```bash
curl -G "https://api.apitube.io/v1/news/everything" \
  --data-urlencode "entity.surface_form.in=S&P 500,FTSE 100,DAX,Nikkei 225" \
  --data-urlencode "category.in=finance,business" \
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
from scipy.linalg import cholesky, solve_triangular
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class MarketSentimentSeries:
    """Time series of market sentiment observations."""
    market: str
    timestamps: List[datetime] = field(default_factory=list)
    sentiment_mean: List[float] = field(default_factory=list)
    sentiment_std: List[float] = field(default_factory=list)
    volume: List[int] = field(default_factory=list)

    def to_array(self) -> np.ndarray:
        return np.array(self.sentiment_mean)

@dataclass
class GrangerResult:
    """Result of Granger causality test."""
    cause: str
    effect: str
    f_statistic: float
    p_value: float
    optimal_lag: int
    is_significant: bool

@dataclass
class SpilloverDecomposition:
    """Spillover index decomposition."""
    from_market: str
    to_market: str
    spillover_pct: float
    direction: str  # "to" or "from"

class VectorAutoregression:
    """
    Vector Autoregression (VAR) model implementation.

    Used for Granger causality testing and impulse response analysis.
    """

    def __init__(self, lags: int = 5):
        self.lags = lags
        self.coef = None
        self.intercept = None
        self.residuals = None
        self.sigma = None
        self.variable_names = []

    def _create_lagged_matrix(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged regressor matrix for VAR estimation."""
        n_obs, n_vars = Y.shape
        n_effective = n_obs - self.lags

        # Dependent variable
        Y_dep = Y[self.lags:]

        # Lagged regressors
        X = np.ones((n_effective, 1 + n_vars * self.lags))  # Intercept + lags

        for lag in range(1, self.lags + 1):
            start_col = 1 + (lag - 1) * n_vars
            end_col = 1 + lag * n_vars
            X[:, start_col:end_col] = Y[self.lags - lag:n_obs - lag]

        return Y_dep, X

    def fit(self, Y: np.ndarray, variable_names: Optional[List[str]] = None) -> "VectorAutoregression":
        """Fit VAR model using OLS."""
        n_obs, n_vars = Y.shape

        if variable_names:
            self.variable_names = variable_names
        else:
            self.variable_names = [f"var_{i}" for i in range(n_vars)]

        Y_dep, X = self._create_lagged_matrix(Y)

        # OLS estimation: beta = (X'X)^{-1} X'Y
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.coef = XtX_inv @ X.T @ Y_dep

        # Residuals and covariance
        Y_fitted = X @ self.coef
        self.residuals = Y_dep - Y_fitted
        self.sigma = (self.residuals.T @ self.residuals) / (len(Y_dep) - X.shape[1])

        # Extract intercept
        self.intercept = self.coef[0]

        return self

    def granger_causality_test(
        self,
        Y: np.ndarray,
        cause_idx: int,
        effect_idx: int,
        max_lags: int = 10
    ) -> GrangerResult:
        """
        Test Granger causality from cause to effect.

        H0: Lags of cause variable do not help predict effect variable
        """
        best_lag = 1
        best_p_value = 1.0
        best_f_stat = 0.0

        for lag in range(1, max_lags + 1):
            # Unrestricted model: effect ~ lags(effect) + lags(cause)
            self.lags = lag
            Y_dep, X_unrestricted = self._create_lagged_matrix(Y)

            # Restricted model: effect ~ lags(effect) only
            # Remove cause variable lags from X
            n_vars = Y.shape[1]
            restricted_cols = [0]  # Intercept
            for l in range(1, lag + 1):
                for v in range(n_vars):
                    if v != cause_idx:
                        col_idx = 1 + (l - 1) * n_vars + v
                        restricted_cols.append(col_idx)

            X_restricted = X_unrestricted[:, restricted_cols]

            # Fit both models for effect variable
            y_effect = Y_dep[:, effect_idx]

            # Unrestricted RSS
            beta_u = np.linalg.lstsq(X_unrestricted, y_effect, rcond=None)[0]
            rss_u = np.sum((y_effect - X_unrestricted @ beta_u) ** 2)

            # Restricted RSS
            beta_r = np.linalg.lstsq(X_restricted, y_effect, rcond=None)[0]
            rss_r = np.sum((y_effect - X_restricted @ beta_r) ** 2)

            # F-test
            n = len(y_effect)
            df_r = n - len(restricted_cols)
            df_u = n - X_unrestricted.shape[1]
            df_diff = lag  # Number of restrictions

            if df_u > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_u)
                p_value = 1 - stats.f.cdf(f_stat, df_diff, df_u)

                if p_value < best_p_value:
                    best_p_value = p_value
                    best_f_stat = f_stat
                    best_lag = lag

        return GrangerResult(
            cause=self.variable_names[cause_idx] if self.variable_names else f"var_{cause_idx}",
            effect=self.variable_names[effect_idx] if self.variable_names else f"var_{effect_idx}",
            f_statistic=best_f_stat,
            p_value=best_p_value,
            optimal_lag=best_lag,
            is_significant=best_p_value < 0.05
        )

    def impulse_response(
        self,
        periods: int = 20,
        shock_idx: int = 0,
        shock_size: float = 1.0
    ) -> np.ndarray:
        """
        Compute impulse response function.

        Returns response of all variables to a unit shock in shock_idx.
        """
        n_vars = len(self.variable_names)

        # Companion form coefficients
        irf = np.zeros((periods, n_vars))

        # Cholesky decomposition for orthogonalized shocks
        try:
            P = cholesky(self.sigma, lower=True)
        except np.linalg.LinAlgError:
            P = np.eye(n_vars)

        # Initial shock
        shock = np.zeros(n_vars)
        shock[shock_idx] = shock_size
        orthog_shock = P @ shock

        # Simulate responses
        Y_sim = np.zeros((periods + self.lags, n_vars))
        Y_sim[self.lags] = orthog_shock

        for t in range(self.lags + 1, periods + self.lags):
            # Y_t = intercept + sum(A_l * Y_{t-l})
            Y_sim[t] = self.intercept.copy()
            for l in range(1, self.lags + 1):
                start_idx = 1 + (l - 1) * n_vars
                end_idx = 1 + l * n_vars
                A_l = self.coef[start_idx:end_idx].T
                Y_sim[t] += A_l @ Y_sim[t - l]

        irf = Y_sim[self.lags:]

        return irf

class DynamicConditionalCorrelation:
    """
    DCC-GARCH model for time-varying correlation.

    Tracks how correlation between markets evolves over time.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.93):
        self.alpha = alpha
        self.beta = beta
        self.unconditional_corr = None
        self.conditional_corr_series = []

    def fit(self, returns: np.ndarray) -> "DynamicConditionalCorrelation":
        """
        Fit DCC model to standardized returns.

        Args:
            returns: (T, N) array of standardized returns
        """
        T, N = returns.shape

        # Unconditional correlation
        self.unconditional_corr = np.corrcoef(returns.T)

        # Initialize Q_t with unconditional correlation
        Q_bar = self.unconditional_corr.copy()
        Q_t = Q_bar.copy()

        self.conditional_corr_series = []

        for t in range(T):
            # Update Q_t
            epsilon_t = returns[t:t+1].T @ returns[t:t+1]  # Outer product

            Q_t = (1 - self.alpha - self.beta) * Q_bar + \
                  self.alpha * epsilon_t + \
                  self.beta * Q_t

            # Normalize to correlation matrix
            Q_diag = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(Q_diag, Q_diag)

            self.conditional_corr_series.append(R_t.copy())

        return self

    def get_pairwise_correlation(
        self,
        idx1: int,
        idx2: int
    ) -> np.ndarray:
        """Extract time-varying correlation between two markets."""
        return np.array([
            R[idx1, idx2] for R in self.conditional_corr_series
        ])

    def detect_correlation_breakdown(
        self,
        idx1: int,
        idx2: int,
        threshold: float = 0.3
    ) -> List[int]:
        """Detect sudden correlation changes (contagion indicators)."""
        corr_series = self.get_pairwise_correlation(idx1, idx2)

        breakpoints = []
        for t in range(1, len(corr_series)):
            change = abs(corr_series[t] - corr_series[t-1])
            if change > threshold:
                breakpoints.append(t)

        return breakpoints

class SpilloverIndex:
    """
    Diebold-Yilmaz Spillover Index.

    Measures directional spillovers between markets using
    forecast error variance decomposition.
    """

    def __init__(self, horizon: int = 10, var_lags: int = 5):
        self.horizon = horizon
        self.var_lags = var_lags
        self.spillover_matrix = None
        self.total_spillover = None

    def compute(
        self,
        Y: np.ndarray,
        variable_names: List[str]
    ) -> Dict:
        """
        Compute spillover index and decomposition.

        Returns:
            Dictionary with spillover metrics
        """
        n_vars = Y.shape[1]

        # Fit VAR model
        var = VectorAutoregression(lags=self.var_lags)
        var.fit(Y, variable_names)

        # Forecast error variance decomposition
        # Using generalized variance decomposition (order-invariant)
        fevd = self._generalized_fevd(var, self.horizon)

        # Normalize rows to sum to 100
        fevd_normalized = fevd / fevd.sum(axis=1, keepdims=True) * 100

        # Spillover matrix
        self.spillover_matrix = fevd_normalized

        # Total spillover index
        # Sum of off-diagonal elements / n * 100
        off_diag_sum = fevd_normalized.sum() - np.trace(fevd_normalized)
        self.total_spillover = off_diag_sum / n_vars

        # Directional spillovers
        spillover_to = fevd_normalized.sum(axis=0) - np.diag(fevd_normalized)
        spillover_from = fevd_normalized.sum(axis=1) - np.diag(fevd_normalized)
        spillover_net = spillover_to - spillover_from

        return {
            "total_spillover_index": self.total_spillover,
            "spillover_matrix": self.spillover_matrix,
            "directional_to": dict(zip(variable_names, spillover_to)),
            "directional_from": dict(zip(variable_names, spillover_from)),
            "net_spillover": dict(zip(variable_names, spillover_net)),
            "variable_names": variable_names
        }

    def _generalized_fevd(
        self,
        var: VectorAutoregression,
        horizon: int
    ) -> np.ndarray:
        """Compute generalized forecast error variance decomposition."""
        n_vars = len(var.variable_names)
        sigma = var.sigma

        # Compute impulse responses
        irf_matrix = np.zeros((horizon, n_vars, n_vars))

        for shock_idx in range(n_vars):
            irf = var.impulse_response(horizon, shock_idx, 1.0)
            irf_matrix[:, :, shock_idx] = irf

        # FEVD computation
        fevd = np.zeros((n_vars, n_vars))
        sigma_diag = np.diag(sigma)

        for i in range(n_vars):
            for j in range(n_vars):
                # Numerator: cumulative squared response
                numerator = np.sum(irf_matrix[:, i, j] ** 2)

                # Denominator: total forecast error variance
                denominator = np.sum([
                    np.sum(irf_matrix[:, i, k] ** 2) * sigma[k, k]
                    for k in range(n_vars)
                ])

                if denominator > 0:
                    fevd[i, j] = (numerator * sigma[j, j]) / denominator

        return fevd

class CrossMarketContagionAnalyzer:
    """
    Complete cross-market contagion analysis system.

    Integrates multiple econometric methods with news data collection.
    """

    def __init__(
        self,
        api_key: str,
        markets: List[str],
        lookback_days: int = 180
    ):
        self.api_key = api_key
        self.markets = markets
        self.lookback_days = lookback_days
        self.sentiment_series: Dict[str, MarketSentimentSeries] = {}

    def fetch_market_sentiment(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime
    ) -> MarketSentimentSeries:
        """Fetch daily sentiment series for a market."""
        series = MarketSentimentSeries(market=market)
        current_date = start_date

        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)

            response = requests.get(BASE_URL, params={
                "api_key": self.api_key,
                "entity.surface_form.eq": market,
                "published_at.gte": current_date.strftime("%Y-%m-%d"),
                "published_at.lt": next_date.strftime("%Y-%m-%d"),
                "category.in": "finance,business",
                "language.code.eq": "en",
                "limit": 50
            })

            articles = response.json().get("results", [])

            if articles:
                sentiments = [
                    a.get("sentiment", {}).get("overall", 0)
                    for a in articles
                ]

                series.timestamps.append(current_date)
                series.sentiment_mean.append(np.mean(sentiments))
                series.sentiment_std.append(
                    np.std(sentiments) if len(sentiments) > 1 else 0
                )
                series.volume.append(len(articles))

            current_date = next_date

        return series

    def collect_all_markets(self) -> None:
        """Collect sentiment data for all markets."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for market in self.markets:
            self.sentiment_series[market] = self.fetch_market_sentiment(
                market, start_date, end_date
            )

    def align_series(self) -> Tuple[np.ndarray, List[datetime]]:
        """Align all market series to common timestamps."""
        # Find common timestamps
        all_timestamps = [
            set(s.timestamps) for s in self.sentiment_series.values()
        ]
        common_timestamps = sorted(set.intersection(*all_timestamps))

        if len(common_timestamps) < 30:
            raise ValueError("Insufficient overlapping data points")

        # Create aligned matrix
        n_obs = len(common_timestamps)
        n_markets = len(self.markets)
        Y = np.zeros((n_obs, n_markets))

        for j, market in enumerate(self.markets):
            series = self.sentiment_series[market]
            timestamp_to_idx = {t: i for i, t in enumerate(series.timestamps)}

            for i, ts in enumerate(common_timestamps):
                if ts in timestamp_to_idx:
                    Y[i, j] = series.sentiment_mean[timestamp_to_idx[ts]]

        return Y, common_timestamps

    def run_granger_analysis(self) -> List[GrangerResult]:
        """Run pairwise Granger causality tests."""
        Y, _ = self.align_series()

        var = VectorAutoregression(lags=5)
        var.variable_names = self.markets

        results = []

        for i, cause in enumerate(self.markets):
            for j, effect in enumerate(self.markets):
                if i != j:
                    result = var.granger_causality_test(Y, i, j, max_lags=10)
                    results.append(result)

        return results

    def compute_spillover_index(self) -> Dict:
        """Compute Diebold-Yilmaz spillover index."""
        Y, _ = self.align_series()

        spillover = SpilloverIndex(horizon=10, var_lags=5)
        return spillover.compute(Y, self.markets)

    def analyze_dynamic_correlation(self) -> Dict:
        """Analyze time-varying correlation structure."""
        Y, timestamps = self.align_series()

        # Standardize returns
        Y_std = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-6)

        dcc = DynamicConditionalCorrelation(alpha=0.05, beta=0.93)
        dcc.fit(Y_std)

        # Extract pairwise correlations
        correlations = {}
        for i, m1 in enumerate(self.markets):
            for j, m2 in enumerate(self.markets):
                if i < j:
                    pair = f"{m1}__{m2}"
                    correlations[pair] = {
                        "correlation_series": dcc.get_pairwise_correlation(i, j).tolist(),
                        "breakpoints": dcc.detect_correlation_breakdown(i, j),
                        "current_correlation": dcc.conditional_corr_series[-1][i, j]
                    }

        return {
            "pairwise_correlations": correlations,
            "timestamps": [t.isoformat() for t in timestamps]
        }

    def compute_impulse_responses(self) -> Dict:
        """Compute impulse response functions for each market."""
        Y, _ = self.align_series()

        var = VectorAutoregression(lags=5)
        var.fit(Y, self.markets)

        responses = {}

        for i, shocked_market in enumerate(self.markets):
            irf = var.impulse_response(periods=20, shock_idx=i, shock_size=1.0)

            responses[shocked_market] = {
                target: irf[:, j].tolist()
                for j, target in enumerate(self.markets)
            }

        return responses

    def identify_contagion_pathways(self) -> Dict:
        """Identify significant contagion pathways."""
        granger_results = self.run_granger_analysis()
        spillover = self.compute_spillover_index()

        pathways = []

        for result in granger_results:
            if result.is_significant:
                # Get spillover magnitude
                cause_idx = self.markets.index(result.cause)
                effect_idx = self.markets.index(result.effect)
                spillover_magnitude = spillover["spillover_matrix"][effect_idx, cause_idx]

                pathways.append({
                    "from": result.cause,
                    "to": result.effect,
                    "granger_p_value": result.p_value,
                    "optimal_lag": result.optimal_lag,
                    "spillover_magnitude": spillover_magnitude
                })

        # Sort by spillover magnitude
        pathways.sort(key=lambda x: x["spillover_magnitude"], reverse=True)

        return {
            "significant_pathways": pathways,
            "total_spillover_index": spillover["total_spillover_index"],
            "net_transmitters": [
                m for m in self.markets
                if spillover["net_spillover"][m] > 0
            ],
            "net_receivers": [
                m for m in self.markets
                if spillover["net_spillover"][m] < 0
            ]
        }

    def generate_systemic_risk_report(self) -> Dict:
        """Generate comprehensive systemic risk assessment."""
        self.collect_all_markets()

        spillover = self.compute_spillover_index()
        pathways = self.identify_contagion_pathways()
        correlations = self.analyze_dynamic_correlation()

        # Compute average correlation level
        avg_correlation = np.mean([
            v["current_correlation"]
            for v in correlations["pairwise_correlations"].values()
        ])

        # Systemic risk score (0-100)
        systemic_risk_score = min(100, (
            spillover["total_spillover_index"] * 0.4 +
            avg_correlation * 30 +
            len(pathways["significant_pathways"]) * 5
        ))

        return {
            "systemic_risk_score": systemic_risk_score,
            "total_spillover_index": spillover["total_spillover_index"],
            "average_correlation": avg_correlation,
            "contagion_pathways": pathways["significant_pathways"][:10],
            "net_transmitters": pathways["net_transmitters"],
            "net_receivers": pathways["net_receivers"],
            "risk_level": (
                "CRITICAL" if systemic_risk_score > 75 else
                "HIGH" if systemic_risk_score > 50 else
                "MODERATE" if systemic_risk_score > 25 else
                "LOW"
            ),
            "timestamp": datetime.now().isoformat()
        }

# Usage
markets = ["S&P 500", "FTSE 100", "DAX", "Nikkei 225", "Shanghai Composite"]
analyzer = CrossMarketContagionAnalyzer(API_KEY, markets, lookback_days=90)
report = analyzer.generate_systemic_risk_report()
print(report)
```

### JavaScript
```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class VectorAutoregression {
  constructor(lags = 5) {
    this.lags = lags;
    this.coef = null;
    this.variableNames = [];
  }

  createLaggedMatrix(Y) {
    const [nObs, nVars] = [Y.length, Y[0].length];
    const nEffective = nObs - this.lags;

    const YDep = Y.slice(this.lags);

    const X = [];
    for (let t = 0; t < nEffective; t++) {
      const row = [1]; // Intercept
      for (let lag = 1; lag <= this.lags; lag++) {
        for (let v = 0; v < nVars; v++) {
          row.push(Y[this.lags + t - lag][v]);
        }
      }
      X.push(row);
    }

    return { YDep, X };
  }

  fit(Y, variableNames = null) {
    this.variableNames = variableNames ||
      Y[0].map((_, i) => `var_${i}`);

    const { YDep, X } = this.createLaggedMatrix(Y);

    // Simple OLS (pseudo-inverse approximation)
    this.coef = this.solveOLS(X, YDep);

    return this;
  }

  solveOLS(X, Y) {
    const n = X.length;
    const p = X[0].length;
    const k = Y[0].length;

    // X'X
    const XtX = Array(p).fill(null).map(() => Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        for (let t = 0; t < n; t++) {
          XtX[i][j] += X[t][i] * X[t][j];
        }
      }
    }

    // X'Y
    const XtY = Array(p).fill(null).map(() => Array(k).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < k; j++) {
        for (let t = 0; t < n; t++) {
          XtY[i][j] += X[t][i] * Y[t][j];
        }
      }
    }

    // Solve using simple inverse approximation
    const XtXInv = this.pseudoInverse(XtX);

    const beta = Array(p).fill(null).map(() => Array(k).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < k; j++) {
        for (let l = 0; l < p; l++) {
          beta[i][j] += XtXInv[i][l] * XtY[l][j];
        }
      }
    }

    return beta;
  }

  pseudoInverse(A) {
    const n = A.length;
    const result = Array(n).fill(null).map((_, i) =>
      Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
    );

    // Gauss-Jordan elimination
    const augmented = A.map((row, i) => [...row, ...result[i]]);

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = k;
        }
      }
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

      const pivot = augmented[i][i];
      if (Math.abs(pivot) < 1e-10) continue;

      for (let j = 0; j < 2 * n; j++) {
        augmented[i][j] /= pivot;
      }

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = augmented[k][i];
          for (let j = 0; j < 2 * n; j++) {
            augmented[k][j] -= factor * augmented[i][j];
          }
        }
      }
    }

    return augmented.map(row => row.slice(n));
  }

  grangerCausalityTest(Y, causeIdx, effectIdx, maxLags = 10) {
    let bestPValue = 1.0;
    let bestFStat = 0;
    let bestLag = 1;

    const nVars = Y[0].length;

    for (let lag = 1; lag <= maxLags; lag++) {
      this.lags = lag;
      const { YDep, X: XUnrestricted } = this.createLaggedMatrix(Y);

      // Restricted model columns (exclude cause variable lags)
      const restrictedCols = [0]; // Intercept
      for (let l = 1; l <= lag; l++) {
        for (let v = 0; v < nVars; v++) {
          if (v !== causeIdx) {
            restrictedCols.push(1 + (l - 1) * nVars + v);
          }
        }
      }

      const XRestricted = XUnrestricted.map(row =>
        restrictedCols.map(c => row[c])
      );
      const yEffect = YDep.map(row => row[effectIdx]);

      // Unrestricted RSS
      const betaU = this.solveSingleOLS(XUnrestricted, yEffect);
      let rssU = 0;
      for (let t = 0; t < yEffect.length; t++) {
        let pred = 0;
        for (let j = 0; j < XUnrestricted[t].length; j++) {
          pred += XUnrestricted[t][j] * betaU[j];
        }
        rssU += (yEffect[t] - pred) ** 2;
      }

      // Restricted RSS
      const betaR = this.solveSingleOLS(XRestricted, yEffect);
      let rssR = 0;
      for (let t = 0; t < yEffect.length; t++) {
        let pred = 0;
        for (let j = 0; j < XRestricted[t].length; j++) {
          pred += XRestricted[t][j] * betaR[j];
        }
        rssR += (yEffect[t] - pred) ** 2;
      }

      // F-test (approximate p-value)
      const n = yEffect.length;
      const dfDiff = lag;
      const dfU = n - XUnrestricted[0].length;

      if (dfU > 0 && rssU > 0) {
        const fStat = ((rssR - rssU) / dfDiff) / (rssU / dfU);
        // Approximate p-value using chi-square approximation
        const pValue = Math.exp(-0.5 * fStat * dfDiff / dfU);

        if (pValue < bestPValue) {
          bestPValue = pValue;
          bestFStat = fStat;
          bestLag = lag;
        }
      }
    }

    return {
      cause: this.variableNames[causeIdx],
      effect: this.variableNames[effectIdx],
      fStatistic: bestFStat,
      pValue: bestPValue,
      optimalLag: bestLag,
      isSignificant: bestPValue < 0.05
    };
  }

  solveSingleOLS(X, y) {
    const n = X.length;
    const p = X[0].length;

    const XtX = Array(p).fill(null).map(() => Array(p).fill(0));
    const Xty = Array(p).fill(0);

    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        for (let t = 0; t < n; t++) {
          XtX[i][j] += X[t][i] * X[t][j];
        }
      }
      for (let t = 0; t < n; t++) {
        Xty[i] += X[t][i] * y[t];
      }
    }

    const XtXInv = this.pseudoInverse(XtX);

    return XtXInv.map(row =>
      row.reduce((sum, val, j) => sum + val * Xty[j], 0)
    );
  }
}

class CrossMarketContagionAnalyzer {
  constructor(apiKey, markets, lookbackDays = 90) {
    this.apiKey = apiKey;
    this.markets = markets;
    this.lookbackDays = lookbackDays;
    this.sentimentSeries = {};
  }

  async fetchMarketSentiment(market, startDate, endDate) {
    const series = {
      market,
      timestamps: [],
      sentimentMean: [],
      sentimentStd: [],
      volume: []
    };

    const current = new Date(startDate);
    while (current <= endDate) {
      const next = new Date(current);
      next.setDate(next.getDate() + 1);

      const params = new URLSearchParams({
        api_key: this.apiKey,
        "entity.surface_form.eq": market,
        "published_at.gte": current.toISOString().split("T")[0],
        "published_at.lt": next.toISOString().split("T")[0],
        "category.in": "finance,business",
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
          const variance = sentiments.reduce((s, x) => s + (x - mean) ** 2, 0) / sentiments.length;

          series.timestamps.push(new Date(current));
          series.sentimentMean.push(mean);
          series.sentimentStd.push(Math.sqrt(variance));
          series.volume.push(articles.length);
        }
      } catch (e) {
        console.error(`Error fetching ${market}:`, e);
      }

      current.setDate(current.getDate() + 1);
    }

    return series;
  }

  async collectAllMarkets() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - this.lookbackDays);

    for (const market of this.markets) {
      this.sentimentSeries[market] = await this.fetchMarketSentiment(
        market, startDate, endDate
      );
    }
  }

  alignSeries() {
    const allTimestamps = Object.values(this.sentimentSeries).map(
      s => new Set(s.timestamps.map(t => t.toISOString().split("T")[0]))
    );

    const commonDates = [...allTimestamps[0]].filter(
      d => allTimestamps.every(set => set.has(d))
    ).sort();

    const Y = commonDates.map(date => {
      return this.markets.map(market => {
        const series = this.sentimentSeries[market];
        const idx = series.timestamps.findIndex(
          t => t.toISOString().split("T")[0] === date
        );
        return idx >= 0 ? series.sentimentMean[idx] : 0;
      });
    });

    return { Y, timestamps: commonDates };
  }

  runGrangerAnalysis() {
    const { Y } = this.alignSeries();

    const var_ = new VectorAutoregression(5);
    var_.variableNames = this.markets;

    const results = [];

    for (let i = 0; i < this.markets.length; i++) {
      for (let j = 0; j < this.markets.length; j++) {
        if (i !== j) {
          const result = var_.grangerCausalityTest(Y, i, j, 10);
          results.push(result);
        }
      }
    }

    return results;
  }

  async generateSystemicRiskReport() {
    await this.collectAllMarkets();

    const grangerResults = this.runGrangerAnalysis();
    const significantPathways = grangerResults
      .filter(r => r.isSignificant)
      .sort((a, b) => a.pValue - b.pValue);

    // Compute average cross-correlation
    const { Y } = this.alignSeries();
    let totalCorr = 0, pairCount = 0;

    for (let i = 0; i < this.markets.length; i++) {
      for (let j = i + 1; j < this.markets.length; j++) {
        const xi = Y.map(row => row[i]);
        const xj = Y.map(row => row[j]);
        const corr = this.correlation(xi, xj);
        totalCorr += Math.abs(corr);
        pairCount++;
      }
    }

    const avgCorrelation = totalCorr / pairCount;

    const systemicRiskScore = Math.min(100,
      significantPathways.length * 8 +
      avgCorrelation * 50
    );

    return {
      systemicRiskScore,
      averageCorrelation: avgCorrelation,
      significantPathways: significantPathways.slice(0, 10),
      riskLevel:
        systemicRiskScore > 75 ? "CRITICAL" :
        systemicRiskScore > 50 ? "HIGH" :
        systemicRiskScore > 25 ? "MODERATE" : "LOW",
      timestamp: new Date().toISOString()
    };
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
}

// Usage
const markets = ["S&P 500", "FTSE 100", "DAX", "Nikkei 225"];
const analyzer = new CrossMarketContagionAnalyzer(API_KEY, markets, 90);
analyzer.generateSystemicRiskReport().then(console.log);
```

### PHP
```php
<?php

class VectorAutoregression {
    private int $lags;
    private ?array $coef = null;
    private array $variableNames = [];

    public function __construct(int $lags = 5) {
        $this->lags = $lags;
    }

    private function createLaggedMatrix(array $Y): array {
        $nObs = count($Y);
        $nVars = count($Y[0]);
        $nEffective = $nObs - $this->lags;

        $YDep = array_slice($Y, $this->lags);

        $X = [];
        for ($t = 0; $t < $nEffective; $t++) {
            $row = [1.0]; // Intercept
            for ($lag = 1; $lag <= $this->lags; $lag++) {
                for ($v = 0; $v < $nVars; $v++) {
                    $row[] = $Y[$this->lags + $t - $lag][$v];
                }
            }
            $X[] = $row;
        }

        return ['YDep' => $YDep, 'X' => $X];
    }

    public function fit(array $Y, ?array $variableNames = null): self {
        $this->variableNames = $variableNames ??
            array_map(fn($i) => "var_$i", range(0, count($Y[0]) - 1));

        $result = $this->createLaggedMatrix($Y);
        $this->coef = $this->solveOLS($result['X'], $result['YDep']);

        return $this;
    }

    private function solveOLS(array $X, array $Y): array {
        $n = count($X);
        $p = count($X[0]);
        $k = count($Y[0]);

        // X'X
        $XtX = array_fill(0, $p, array_fill(0, $p, 0.0));
        for ($i = 0; $i < $p; $i++) {
            for ($j = 0; $j < $p; $j++) {
                for ($t = 0; $t < $n; $t++) {
                    $XtX[$i][$j] += $X[$t][$i] * $X[$t][$j];
                }
            }
        }

        // X'Y
        $XtY = array_fill(0, $p, array_fill(0, $k, 0.0));
        for ($i = 0; $i < $p; $i++) {
            for ($j = 0; $j < $k; $j++) {
                for ($t = 0; $t < $n; $t++) {
                    $XtY[$i][$j] += $X[$t][$i] * $Y[$t][$j];
                }
            }
        }

        $XtXInv = $this->pseudoInverse($XtX);

        $beta = array_fill(0, $p, array_fill(0, $k, 0.0));
        for ($i = 0; $i < $p; $i++) {
            for ($j = 0; $j < $k; $j++) {
                for ($l = 0; $l < $p; $l++) {
                    $beta[$i][$j] += $XtXInv[$i][$l] * $XtY[$l][$j];
                }
            }
        }

        return $beta;
    }

    private function pseudoInverse(array $A): array {
        $n = count($A);
        $result = [];
        for ($i = 0; $i < $n; $i++) {
            $result[$i] = array_fill(0, $n, 0.0);
            $result[$i][$i] = 1.0;
        }

        $augmented = [];
        for ($i = 0; $i < $n; $i++) {
            $augmented[$i] = array_merge($A[$i], $result[$i]);
        }

        for ($i = 0; $i < $n; $i++) {
            $maxRow = $i;
            for ($k = $i + 1; $k < $n; $k++) {
                if (abs($augmented[$k][$i]) > abs($augmented[$maxRow][$i])) {
                    $maxRow = $k;
                }
            }
            [$augmented[$i], $augmented[$maxRow]] = [$augmented[$maxRow], $augmented[$i]];

            $pivot = $augmented[$i][$i];
            if (abs($pivot) < 1e-10) continue;

            for ($j = 0; $j < 2 * $n; $j++) {
                $augmented[$i][$j] /= $pivot;
            }

            for ($k = 0; $k < $n; $k++) {
                if ($k !== $i) {
                    $factor = $augmented[$k][$i];
                    for ($j = 0; $j < 2 * $n; $j++) {
                        $augmented[$k][$j] -= $factor * $augmented[$i][$j];
                    }
                }
            }
        }

        return array_map(fn($row) => array_slice($row, $n), $augmented);
    }

    public function grangerCausalityTest(
        array $Y,
        int $causeIdx,
        int $effectIdx,
        int $maxLags = 10
    ): array {
        $bestPValue = 1.0;
        $bestFStat = 0;
        $bestLag = 1;
        $nVars = count($Y[0]);

        for ($lag = 1; $lag <= $maxLags; $lag++) {
            $this->lags = $lag;
            $result = $this->createLaggedMatrix($Y);
            $XUnrestricted = $result['X'];
            $YDep = $result['YDep'];

            $restrictedCols = [0];
            for ($l = 1; $l <= $lag; $l++) {
                for ($v = 0; $v < $nVars; $v++) {
                    if ($v !== $causeIdx) {
                        $restrictedCols[] = 1 + ($l - 1) * $nVars + $v;
                    }
                }
            }

            $XRestricted = array_map(
                fn($row) => array_map(fn($c) => $row[$c], $restrictedCols),
                $XUnrestricted
            );
            $yEffect = array_column($YDep, $effectIdx);

            $betaU = $this->solveSingleOLS($XUnrestricted, $yEffect);
            $rssU = 0;
            foreach ($yEffect as $t => $y) {
                $pred = 0;
                foreach ($XUnrestricted[$t] as $j => $x) {
                    $pred += $x * $betaU[$j];
                }
                $rssU += ($y - $pred) ** 2;
            }

            $betaR = $this->solveSingleOLS($XRestricted, $yEffect);
            $rssR = 0;
            foreach ($yEffect as $t => $y) {
                $pred = 0;
                foreach ($XRestricted[$t] as $j => $x) {
                    $pred += $x * $betaR[$j];
                }
                $rssR += ($y - $pred) ** 2;
            }

            $n = count($yEffect);
            $dfDiff = $lag;
            $dfU = $n - count($XUnrestricted[0]);

            if ($dfU > 0 && $rssU > 0) {
                $fStat = (($rssR - $rssU) / $dfDiff) / ($rssU / $dfU);
                $pValue = exp(-0.5 * $fStat * $dfDiff / $dfU);

                if ($pValue < $bestPValue) {
                    $bestPValue = $pValue;
                    $bestFStat = $fStat;
                    $bestLag = $lag;
                }
            }
        }

        return [
            'cause' => $this->variableNames[$causeIdx],
            'effect' => $this->variableNames[$effectIdx],
            'f_statistic' => $bestFStat,
            'p_value' => $bestPValue,
            'optimal_lag' => $bestLag,
            'is_significant' => $bestPValue < 0.05
        ];
    }

    private function solveSingleOLS(array $X, array $y): array {
        $n = count($X);
        $p = count($X[0]);

        $XtX = array_fill(0, $p, array_fill(0, $p, 0.0));
        $Xty = array_fill(0, $p, 0.0);

        for ($i = 0; $i < $p; $i++) {
            for ($j = 0; $j < $p; $j++) {
                for ($t = 0; $t < $n; $t++) {
                    $XtX[$i][$j] += $X[$t][$i] * $X[$t][$j];
                }
            }
            for ($t = 0; $t < $n; $t++) {
                $Xty[$i] += $X[$t][$i] * $y[$t];
            }
        }

        $XtXInv = $this->pseudoInverse($XtX);

        return array_map(
            fn($row) => array_sum(array_map(
                fn($val, $j) => $val * $Xty[$j],
                $row,
                array_keys($row)
            )),
            $XtXInv
        );
    }

    public function getVariableNames(): array {
        return $this->variableNames;
    }
}

class CrossMarketContagionAnalyzer {
    private string $apiKey;
    private array $markets;
    private int $lookbackDays;
    private array $sentimentSeries = [];

    public function __construct(
        string $apiKey,
        array $markets,
        int $lookbackDays = 90
    ) {
        $this->apiKey = $apiKey;
        $this->markets = $markets;
        $this->lookbackDays = $lookbackDays;
    }

    public function fetchMarketSentiment(
        string $market,
        DateTime $startDate,
        DateTime $endDate
    ): array {
        $series = [
            'market' => $market,
            'timestamps' => [],
            'sentimentMean' => [],
            'volume' => []
        ];

        $current = clone $startDate;
        while ($current <= $endDate) {
            $next = clone $current;
            $next->modify('+1 day');

            $params = http_build_query([
                'api_key' => $this->apiKey,
                'entity.surface_form.eq' => $market,
                'published_at.gte' => $current->format('Y-m-d'),
                'published_at.lt' => $next->format('Y-m-d'),
                'category.in' => 'finance,business',
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

                    $series['timestamps'][] = $current->format('Y-m-d');
                    $series['sentimentMean'][] =
                        array_sum($sentiments) / count($sentiments);
                    $series['volume'][] = count($articles);
                }
            }

            $current->modify('+1 day');
        }

        return $series;
    }

    public function collectAllMarkets(): void {
        $endDate = new DateTime();
        $startDate = (clone $endDate)->modify("-{$this->lookbackDays} days");

        foreach ($this->markets as $market) {
            $this->sentimentSeries[$market] = $this->fetchMarketSentiment(
                $market, $startDate, $endDate
            );
        }
    }

    public function alignSeries(): array {
        $allDates = array_map(
            fn($s) => $s['timestamps'],
            $this->sentimentSeries
        );

        $commonDates = call_user_func_array('array_intersect', $allDates);
        sort($commonDates);

        $Y = [];
        foreach ($commonDates as $date) {
            $row = [];
            foreach ($this->markets as $market) {
                $series = $this->sentimentSeries[$market];
                $idx = array_search($date, $series['timestamps']);
                $row[] = $idx !== false ? $series['sentimentMean'][$idx] : 0;
            }
            $Y[] = $row;
        }

        return ['Y' => $Y, 'timestamps' => $commonDates];
    }

    public function runGrangerAnalysis(): array {
        $aligned = $this->alignSeries();
        $Y = $aligned['Y'];

        $var = new VectorAutoregression(5);
        $var->fit($Y, $this->markets);

        $results = [];

        for ($i = 0; $i < count($this->markets); $i++) {
            for ($j = 0; $j < count($this->markets); $j++) {
                if ($i !== $j) {
                    $results[] = $var->grangerCausalityTest($Y, $i, $j, 10);
                }
            }
        }

        return $results;
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

    public function generateSystemicRiskReport(): array {
        $this->collectAllMarkets();

        $grangerResults = $this->runGrangerAnalysis();
        $significantPathways = array_filter(
            $grangerResults,
            fn($r) => $r['is_significant']
        );
        usort($significantPathways, fn($a, $b) => $a['p_value'] <=> $b['p_value']);

        $aligned = $this->alignSeries();
        $Y = $aligned['Y'];

        $totalCorr = 0;
        $pairCount = 0;

        for ($i = 0; $i < count($this->markets); $i++) {
            for ($j = $i + 1; $j < count($this->markets); $j++) {
                $xi = array_column($Y, $i);
                $xj = array_column($Y, $j);
                $totalCorr += abs($this->correlation($xi, $xj));
                $pairCount++;
            }
        }

        $avgCorrelation = $totalCorr / $pairCount;

        $systemicRiskScore = min(100,
            count($significantPathways) * 8 +
            $avgCorrelation * 50
        );

        $riskLevel = match(true) {
            $systemicRiskScore > 75 => 'CRITICAL',
            $systemicRiskScore > 50 => 'HIGH',
            $systemicRiskScore > 25 => 'MODERATE',
            default => 'LOW'
        };

        return [
            'systemic_risk_score' => $systemicRiskScore,
            'average_correlation' => $avgCorrelation,
            'significant_pathways' => array_slice($significantPathways, 0, 10),
            'risk_level' => $riskLevel,
            'timestamp' => (new DateTime())->format('c')
        ];
    }
}

// Usage
$markets = ['S&P 500', 'FTSE 100', 'DAX', 'Nikkei 225'];
$analyzer = new CrossMarketContagionAnalyzer('YOUR_API_KEY', $markets, 90);
print_r($analyzer->generateSystemicRiskReport());
```

## Use Cases

### Systemic Risk Monitoring
- Cross-market contagion early warning
- Spillover intensity tracking
- Crisis propagation analysis

### Global Portfolio Management
- Geographic diversification validation
- Correlation regime monitoring
- Tail risk assessment

### Regulatory Compliance
- Systemic risk reporting
- Interconnectedness analysis
- Stress testing inputs

### Research
- Contagion pathway mapping
- Spillover dynamics studies
- Market integration analysis
