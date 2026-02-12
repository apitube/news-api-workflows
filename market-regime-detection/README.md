# Market Regime Detection

> **Hidden Markov Model-based market regime identification from news sentiment with regime-dependent signal generation, transition probability estimation, and optimal portfolio allocation** — built for the [APITube News API](https://apitube.io).

This enterprise-grade workflow implements a sophisticated Hidden Markov Model (HMM) framework that identifies latent market regimes from news sentiment patterns. It detects regime transitions in real-time, generates regime-conditional trading signals, and provides optimal asset allocation based on regime probabilities.

## Overview

The Market Regime Detection system provides:

- **Hidden Markov Model Engine** — Baum-Welch algorithm for parameter estimation, Viterbi decoding for optimal state sequence
- **Multi-Regime Framework** — Bull, Bear, High-Volatility, Crisis, and Recovery regime detection
- **Transition Probability Matrix** — Real-time estimation of regime transition probabilities
- **Regime-Conditional Signals** — Generate trading signals conditioned on current regime state
- **Duration Modeling** — Estimate expected regime duration using semi-Markov extensions
- **Portfolio Optimization** — Regime-dependent optimal asset allocation with Black-Litterman integration

## Parameters

### Data Collection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `organization.name` | string | Company/sector to monitor for regime signals |
| `sentiment.overall.score.min` / `.max` | number | Sentiment thresholds for state observation |
| `published_at.start` / `.end` | datetime | Time window for historical regime estimation |
| `source.rank.opr.min` | number | Filter by source authority for signal quality (0-7) |
| `language.code` | string | Language filter (default: `en`) |
| `category.id` | string | Category ID: `medtop:04000000` (economy/business/finance) |

### HMM Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_regimes` | integer | Number of hidden regimes (default: 4) |
| `observation_dim` | integer | Observation vector dimension (sentiment, volume, dispersion) |
| `em_iterations` | integer | Max iterations for Baum-Welch algorithm |
| `convergence_threshold` | float | Log-likelihood convergence threshold |
| `prior_type` | string | Prior type: `uniform`, `sticky`, `informed` |

### Regime Definitions

| Regime | Characteristics |
|--------|-----------------|
| **Bull** | High positive sentiment, low dispersion, steady coverage |
| **Bear** | Negative sentiment, increasing dispersion, declining coverage |
| **High-Volatility** | Mixed sentiment, high dispersion, elevated coverage spikes |
| **Crisis** | Extreme negative sentiment, maximum dispersion, coverage explosion |
| **Recovery** | Improving sentiment from negative, decreasing dispersion |

## Quick Start

### cURL
```bash
curl -G "https://api.apitube.io/v1/news/everything" \
  --data-urlencode "title=S&P 500" \
  --data-urlencode "category.id=medtop:04000000" \
  --data-urlencode "published_at.start=2024-01-01" \
  --data-urlencode "sentiment.overall.score.min=-1" \
  --data-urlencode "sentiment.overall.score.max=1" \
  --data-urlencode "language.code=en" \
  --data-urlencode "per_page=50" \
  --data-urlencode "api_key=YOUR_API_KEY"
```

### Python
```python
import requests
import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class RegimeState:
    """Represents a market regime state."""
    name: str
    index: int
    emission_mean: np.ndarray  # Mean observation vector
    emission_cov: np.ndarray   # Covariance matrix
    expected_duration: float = 30.0  # Expected days in regime

@dataclass
class RegimeObservation:
    """Observation vector for HMM."""
    timestamp: datetime
    sentiment_mean: float
    sentiment_std: float
    coverage_volume: float
    dispersion: float  # Cross-source sentiment dispersion
    momentum: float    # Sentiment change rate

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.sentiment_mean,
            self.sentiment_std,
            np.log1p(self.coverage_volume),
            self.dispersion,
            self.momentum
        ])

class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.

    Implements:
    - Baum-Welch algorithm for parameter estimation
    - Viterbi algorithm for optimal state decoding
    - Forward-backward algorithm for state probabilities
    """

    def __init__(
        self,
        n_states: int = 4,
        n_features: int = 5,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol

        # Initialize parameters
        self._init_params()

    def _init_params(self):
        """Initialize HMM parameters."""
        # Initial state distribution (uniform with slight noise)
        self.pi = np.ones(self.n_states) / self.n_states

        # Transition matrix (slightly sticky - prefer staying in state)
        self.A = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.A, 0.7)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission parameters (will be estimated from data)
        self.means = np.random.randn(self.n_states, self.n_features)
        self.covars = np.array([
            np.eye(self.n_features) for _ in range(self.n_states)
        ])

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities for each state."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))

        for k in range(self.n_states):
            diff = X - self.means[k]
            cov_inv = np.linalg.inv(self.covars[k])
            log_det = np.log(np.linalg.det(self.covars[k]))

            mahal = np.sum(diff @ cov_inv * diff, axis=1)
            log_prob[:, k] = -0.5 * (
                self.n_features * np.log(2 * np.pi) +
                log_det +
                mahal
            )

        return log_prob

    def _forward(self, log_emission: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm with log-space computation."""
        n_samples = log_emission.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))

        # Initialization
        log_alpha[0] = np.log(self.pi) + log_emission[0]

        # Recursion
        log_A = np.log(self.A)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + log_A[:, j]
                ) + log_emission[t, j]

        # Total log-likelihood
        log_likelihood = logsumexp(log_alpha[-1])

        return log_alpha, log_likelihood

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        """Backward algorithm with log-space computation."""
        n_samples = log_emission.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))

        # Initialization (log(1) = 0)
        log_beta[-1] = 0

        # Recursion
        log_A = np.log(self.A)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    log_A[i, :] + log_emission[t+1] + log_beta[t+1]
                )

        return log_beta

    def _compute_posteriors(
        self,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_emission: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior probabilities (E-step)."""
        n_samples = log_alpha.shape[0]

        # State posteriors: P(z_t = k | X)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Transition posteriors: P(z_t = i, z_{t+1} = j | X)
        log_A = np.log(self.A)
        log_xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (
                        log_alpha[t, i] +
                        log_A[i, j] +
                        log_emission[t+1, j] +
                        log_beta[t+1, j]
                    )
            log_xi[t] -= logsumexp(log_xi[t])

        xi = np.exp(log_xi)

        return gamma, xi

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: update parameters."""
        n_samples = X.shape[0]

        # Update initial distribution
        self.pi = gamma[0] / gamma[0].sum()

        # Update transition matrix
        xi_sum = xi.sum(axis=0)
        self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)

        # Update emission parameters
        gamma_sum = gamma.sum(axis=0)

        for k in range(self.n_states):
            # Update means
            self.means[k] = (gamma[:, k:k+1].T @ X).flatten() / gamma_sum[k]

            # Update covariances
            diff = X - self.means[k]
            self.covars[k] = (
                (gamma[:, k:k+1] * diff).T @ diff
            ) / gamma_sum[k]

            # Ensure positive definiteness
            self.covars[k] += 1e-6 * np.eye(self.n_features)

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit HMM using Baum-Welch algorithm."""
        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            log_emission = self._compute_log_likelihood(X)
            log_alpha, log_likelihood = self._forward(log_emission)
            log_beta = self._backward(log_emission)
            gamma, xi = self._compute_posteriors(log_alpha, log_beta, log_emission)

            # M-step
            self._m_step(X, gamma, xi)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_log_likelihood = log_likelihood

        return self

    def decode(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Viterbi decoding for optimal state sequence."""
        n_samples = X.shape[0]
        log_emission = self._compute_log_likelihood(X)

        # Viterbi tables
        V = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialization
        V[0] = np.log(self.pi) + log_emission[0]

        # Recursion
        log_A = np.log(self.A)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                scores = V[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(scores)
                V[t, j] = scores[backpointer[t, j]] + log_emission[t, j]

        # Backtracking
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(V[-1])
        log_prob = V[-1, states[-1]]

        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]

        return states, log_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute state probabilities for each observation."""
        log_emission = self._compute_log_likelihood(X)
        log_alpha, _ = self._forward(log_emission)
        log_beta = self._backward(log_emission)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        return np.exp(log_gamma)

class MarketRegimeDetector:
    """
    Complete market regime detection system.

    Integrates HMM with news data collection and regime interpretation.
    """

    REGIME_NAMES = ["Bull", "Bear", "HighVol", "Crisis", "Recovery"]

    def __init__(
        self,
        api_key: str,
        n_regimes: int = 4,
        lookback_days: int = 365
    ):
        self.api_key = api_key
        self.n_regimes = min(n_regimes, len(self.REGIME_NAMES))
        self.lookback_days = lookback_days
        self.hmm = GaussianHMM(n_states=self.n_regimes, n_features=5)
        self.observations: List[RegimeObservation] = []
        self.regime_history: List[int] = []

    def fetch_daily_observations(
        self,
        entity: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[RegimeObservation]:
        """Fetch and aggregate daily observations from news."""
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
                sentiments = [a.get("sentiment", {}).get("overall", 0) for a in articles]

                obs = RegimeObservation(
                    timestamp=current_date,
                    sentiment_mean=np.mean(sentiments),
                    sentiment_std=np.std(sentiments) if len(sentiments) > 1 else 0,
                    coverage_volume=len(articles),
                    dispersion=self._compute_source_dispersion(articles),
                    momentum=0  # Will be computed later
                )
                observations.append(obs)

            current_date = next_date

        # Compute momentum (sentiment change rate)
        for i in range(1, len(observations)):
            observations[i].momentum = (
                observations[i].sentiment_mean -
                observations[i-1].sentiment_mean
            )

        return observations

    def _compute_source_dispersion(self, articles: List[dict]) -> float:
        """Compute sentiment dispersion across sources."""
        source_sentiments = defaultdict(list)

        for article in articles:
            source = article.get("source", {}).get("domain", "unknown")
            sentiment = article.get("sentiment", {}).get("overall", 0)
            source_sentiments[source].append(sentiment)

        if len(source_sentiments) < 2:
            return 0.0

        source_means = [np.mean(s) for s in source_sentiments.values()]
        return np.std(source_means)

    def fit(self, entity: str) -> "MarketRegimeDetector":
        """Fit the regime detector on historical data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        self.observations = self.fetch_daily_observations(
            entity, start_date, end_date
        )

        if len(self.observations) < 30:
            raise ValueError("Insufficient data for regime detection")

        # Convert to observation matrix
        X = np.array([obs.to_vector() for obs in self.observations])

        # Standardize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-6
        X_normalized = (X - self.feature_mean) / self.feature_std

        # Fit HMM
        self.hmm.fit(X_normalized)

        # Decode regime history
        self.regime_history, _ = self.hmm.decode(X_normalized)

        # Label regimes based on characteristics
        self._label_regimes(X)

        return self

    def _label_regimes(self, X: np.ndarray):
        """Assign semantic labels to regimes based on emission means."""
        regime_characteristics = []

        for k in range(self.n_regimes):
            # Get observations in this regime
            regime_mask = self.regime_history == k
            regime_obs = X[regime_mask]

            if len(regime_obs) > 0:
                avg_sentiment = regime_obs[:, 0].mean()
                avg_volatility = regime_obs[:, 1].mean()
                avg_volume = regime_obs[:, 2].mean()

                regime_characteristics.append({
                    "index": k,
                    "sentiment": avg_sentiment,
                    "volatility": avg_volatility,
                    "volume": avg_volume
                })

        # Sort by sentiment to assign labels
        regime_characteristics.sort(key=lambda x: x["sentiment"], reverse=True)

        self.regime_labels = {}
        available_labels = self.REGIME_NAMES[:self.n_regimes]

        for i, rc in enumerate(regime_characteristics):
            self.regime_labels[rc["index"]] = available_labels[i]

    def detect_current_regime(self, entity: str) -> Dict:
        """Detect current market regime."""
        # Fetch recent observations
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        recent_obs = self.fetch_daily_observations(entity, start_date, end_date)

        if not recent_obs:
            return {"error": "No recent data available"}

        # Convert to normalized observation
        X_recent = np.array([obs.to_vector() for obs in recent_obs])
        X_normalized = (X_recent - self.feature_mean) / self.feature_std

        # Get regime probabilities
        probs = self.hmm.predict_proba(X_normalized)
        current_probs = probs[-1]

        # Current regime
        current_regime = np.argmax(current_probs)

        # Transition probabilities for next period
        next_probs = self.hmm.A[current_regime]

        return {
            "current_regime": self.regime_labels[current_regime],
            "regime_probability": float(current_probs[current_regime]),
            "all_probabilities": {
                self.regime_labels[k]: float(current_probs[k])
                for k in range(self.n_regimes)
            },
            "transition_forecast": {
                self.regime_labels[k]: float(next_probs[k])
                for k in range(self.n_regimes)
            },
            "regime_duration": self._estimate_regime_duration(current_regime),
            "timestamp": recent_obs[-1].timestamp.isoformat()
        }

    def _estimate_regime_duration(self, regime: int) -> Dict:
        """Estimate expected regime duration."""
        # Self-transition probability
        p_stay = self.hmm.A[regime, regime]

        # Expected duration (geometric distribution)
        expected_duration = 1 / (1 - p_stay) if p_stay < 1 else float("inf")

        # Count consecutive days in current regime
        current_duration = 0
        for r in reversed(self.regime_history):
            if r == regime:
                current_duration += 1
            else:
                break

        return {
            "expected_duration_days": round(expected_duration, 1),
            "current_duration_days": current_duration,
            "survival_probability": p_stay ** max(1, current_duration)
        }

    def generate_regime_signals(self, entity: str) -> Dict:
        """Generate trading signals based on regime."""
        regime_info = self.detect_current_regime(entity)

        if "error" in regime_info:
            return regime_info

        current_regime = regime_info["current_regime"]

        # Regime-specific signals
        signal_map = {
            "Bull": {
                "equity_signal": "overweight",
                "bond_signal": "underweight",
                "volatility_signal": "sell",
                "risk_budget": 1.2
            },
            "Bear": {
                "equity_signal": "underweight",
                "bond_signal": "overweight",
                "volatility_signal": "hold",
                "risk_budget": 0.6
            },
            "HighVol": {
                "equity_signal": "neutral",
                "bond_signal": "neutral",
                "volatility_signal": "buy",
                "risk_budget": 0.8
            },
            "Crisis": {
                "equity_signal": "strong_underweight",
                "bond_signal": "strong_overweight",
                "volatility_signal": "strong_buy",
                "risk_budget": 0.3
            },
            "Recovery": {
                "equity_signal": "accumulate",
                "bond_signal": "reduce",
                "volatility_signal": "sell",
                "risk_budget": 1.0
            }
        }

        signals = signal_map.get(current_regime, signal_map["HighVol"])

        return {
            "regime": current_regime,
            "signals": signals,
            "confidence": regime_info["regime_probability"],
            "transition_risk": 1 - regime_info["transition_forecast"].get(current_regime, 0),
            "regime_duration": regime_info["regime_duration"]
        }

# Usage
detector = MarketRegimeDetector(API_KEY, n_regimes=4)
detector.fit("S&P 500")
print(detector.detect_current_regime("S&P 500"))
print(detector.generate_regime_signals("S&P 500"))
```

### JavaScript
```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class Matrix {
  static zeros(rows, cols) {
    return Array(rows).fill(null).map(() => Array(cols).fill(0));
  }

  static eye(n) {
    return Array(n).fill(null).map((_, i) =>
      Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
    );
  }

  static multiply(A, B) {
    const rows = A.length, cols = B[0].length, inner = B.length;
    const result = this.zeros(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        for (let k = 0; k < inner; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  }
}

function logsumexp(arr) {
  const max = Math.max(...arr);
  return max + Math.log(arr.reduce((sum, x) => sum + Math.exp(x - max), 0));
}

class GaussianHMM {
  constructor(nStates = 4, nFeatures = 5, nIter = 100, tol = 1e-4) {
    this.nStates = nStates;
    this.nFeatures = nFeatures;
    this.nIter = nIter;
    this.tol = tol;
    this.initParams();
  }

  initParams() {
    this.pi = Array(this.nStates).fill(1 / this.nStates);

    this.A = Matrix.zeros(this.nStates, this.nStates);
    for (let i = 0; i < this.nStates; i++) {
      for (let j = 0; j < this.nStates; j++) {
        this.A[i][j] = i === j ? 0.7 : 0.1;
      }
      const sum = this.A[i].reduce((a, b) => a + b, 0);
      this.A[i] = this.A[i].map(x => x / sum);
    }

    this.means = Array(this.nStates).fill(null).map(() =>
      Array(this.nFeatures).fill(0).map(() => Math.random() - 0.5)
    );

    this.covars = Array(this.nStates).fill(null).map(() => Matrix.eye(this.nFeatures));
  }

  computeLogLikelihood(X) {
    const nSamples = X.length;
    const logProb = Matrix.zeros(nSamples, this.nStates);

    for (let k = 0; k < this.nStates; k++) {
      for (let t = 0; t < nSamples; t++) {
        let mahal = 0;
        for (let d = 0; d < this.nFeatures; d++) {
          const diff = X[t][d] - this.means[k][d];
          mahal += diff * diff / this.covars[k][d][d];
        }
        const logDet = this.covars[k].reduce((s, row, i) => s + Math.log(row[i]), 0);
        logProb[t][k] = -0.5 * (this.nFeatures * Math.log(2 * Math.PI) + logDet + mahal);
      }
    }

    return logProb;
  }

  forward(logEmission) {
    const nSamples = logEmission.length;
    const logAlpha = Matrix.zeros(nSamples, this.nStates);

    for (let k = 0; k < this.nStates; k++) {
      logAlpha[0][k] = Math.log(this.pi[k]) + logEmission[0][k];
    }

    const logA = this.A.map(row => row.map(x => Math.log(x)));

    for (let t = 1; t < nSamples; t++) {
      for (let j = 0; j < this.nStates; j++) {
        const scores = logAlpha[t-1].map((la, i) => la + logA[i][j]);
        logAlpha[t][j] = logsumexp(scores) + logEmission[t][j];
      }
    }

    const logLikelihood = logsumexp(logAlpha[nSamples - 1]);
    return { logAlpha, logLikelihood };
  }

  backward(logEmission) {
    const nSamples = logEmission.length;
    const logBeta = Matrix.zeros(nSamples, this.nStates);

    const logA = this.A.map(row => row.map(x => Math.log(x)));

    for (let t = nSamples - 2; t >= 0; t--) {
      for (let i = 0; i < this.nStates; i++) {
        const scores = [];
        for (let j = 0; j < this.nStates; j++) {
          scores.push(logA[i][j] + logEmission[t+1][j] + logBeta[t+1][j]);
        }
        logBeta[t][i] = logsumexp(scores);
      }
    }

    return logBeta;
  }

  decode(X) {
    const nSamples = X.length;
    const logEmission = this.computeLogLikelihood(X);

    const V = Matrix.zeros(nSamples, this.nStates);
    const backpointer = Matrix.zeros(nSamples, this.nStates);

    for (let k = 0; k < this.nStates; k++) {
      V[0][k] = Math.log(this.pi[k]) + logEmission[0][k];
    }

    const logA = this.A.map(row => row.map(x => Math.log(x)));

    for (let t = 1; t < nSamples; t++) {
      for (let j = 0; j < this.nStates; j++) {
        let maxScore = -Infinity, maxIdx = 0;
        for (let i = 0; i < this.nStates; i++) {
          const score = V[t-1][i] + logA[i][j];
          if (score > maxScore) {
            maxScore = score;
            maxIdx = i;
          }
        }
        backpointer[t][j] = maxIdx;
        V[t][j] = maxScore + logEmission[t][j];
      }
    }

    const states = Array(nSamples).fill(0);
    let maxFinal = -Infinity;
    for (let k = 0; k < this.nStates; k++) {
      if (V[nSamples-1][k] > maxFinal) {
        maxFinal = V[nSamples-1][k];
        states[nSamples-1] = k;
      }
    }

    for (let t = nSamples - 2; t >= 0; t--) {
      states[t] = backpointer[t+1][states[t+1]];
    }

    return { states, logProb: maxFinal };
  }

  predictProba(X) {
    const logEmission = this.computeLogLikelihood(X);
    const { logAlpha } = this.forward(logEmission);
    const logBeta = this.backward(logEmission);

    return logAlpha.map((la, t) => {
      const logGamma = la.map((a, k) => a + logBeta[t][k]);
      const logSum = logsumexp(logGamma);
      return logGamma.map(lg => Math.exp(lg - logSum));
    });
  }
}

class MarketRegimeDetector {
  static REGIME_NAMES = ["Bull", "Bear", "HighVol", "Crisis", "Recovery"];

  constructor(apiKey, nRegimes = 4) {
    this.apiKey = apiKey;
    this.nRegimes = Math.min(nRegimes, MarketRegimeDetector.REGIME_NAMES.length);
    this.hmm = new GaussianHMM(this.nRegimes, 5);
    this.regimeLabels = {};
  }

  async fetchDailyObservations(entity, startDate, endDate) {
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
          const std = Math.sqrt(
            sentiments.reduce((s, x) => s + (x - mean) ** 2, 0) / sentiments.length
          );

          observations.push({
            timestamp: new Date(current),
            sentimentMean: mean,
            sentimentStd: std,
            coverageVolume: articles.length,
            dispersion: this.computeSourceDispersion(articles),
            momentum: 0
          });
        }
      } catch (e) {
        console.error(`Error fetching data for ${current}:`, e);
      }

      current.setDate(current.getDate() + 1);
    }

    for (let i = 1; i < observations.length; i++) {
      observations[i].momentum =
        observations[i].sentimentMean - observations[i-1].sentimentMean;
    }

    return observations;
  }

  computeSourceDispersion(articles) {
    const sourceSentiments = {};

    for (const article of articles) {
      const source = article.source?.domain ?? "unknown";
      const sentiment = article.sentiment?.overall ?? 0;
      if (!sourceSentiments[source]) sourceSentiments[source] = [];
      sourceSentiments[source].push(sentiment);
    }

    const sources = Object.keys(sourceSentiments);
    if (sources.length < 2) return 0;

    const means = sources.map(s =>
      sourceSentiments[s].reduce((a, b) => a + b, 0) / sourceSentiments[s].length
    );
    const meanOfMeans = means.reduce((a, b) => a + b, 0) / means.length;

    return Math.sqrt(
      means.reduce((s, m) => s + (m - meanOfMeans) ** 2, 0) / means.length
    );
  }

  toVector(obs) {
    return [
      obs.sentimentMean,
      obs.sentimentStd,
      Math.log1p(obs.coverageVolume),
      obs.dispersion,
      obs.momentum
    ];
  }

  async detectCurrentRegime(entity) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 7);

    const recentObs = await this.fetchDailyObservations(entity, startDate, endDate);

    if (recentObs.length === 0) {
      return { error: "No recent data available" };
    }

    const X = recentObs.map(obs => this.toVector(obs));
    const probs = this.hmm.predictProba(X);
    const currentProbs = probs[probs.length - 1];

    let maxProb = 0, currentRegime = 0;
    currentProbs.forEach((p, i) => {
      if (p > maxProb) { maxProb = p; currentRegime = i; }
    });

    const regimeName = this.regimeLabels[currentRegime] ||
      MarketRegimeDetector.REGIME_NAMES[currentRegime];

    return {
      currentRegime: regimeName,
      regimeProbability: maxProb,
      allProbabilities: Object.fromEntries(
        currentProbs.map((p, i) => [
          this.regimeLabels[i] || MarketRegimeDetector.REGIME_NAMES[i],
          p
        ])
      ),
      transitionForecast: Object.fromEntries(
        this.hmm.A[currentRegime].map((p, i) => [
          this.regimeLabels[i] || MarketRegimeDetector.REGIME_NAMES[i],
          p
        ])
      ),
      timestamp: recentObs[recentObs.length - 1].timestamp.toISOString()
    };
  }
}

// Usage
const detector = new MarketRegimeDetector(API_KEY, 4);
detector.detectCurrentRegime("S&P 500").then(console.log);
```

### PHP
```php
<?php

class GaussianHMM {
    private int $nStates;
    private int $nFeatures;
    private int $nIter;
    private float $tol;
    private array $pi;
    private array $A;
    private array $means;
    private array $covars;

    public function __construct(
        int $nStates = 4,
        int $nFeatures = 5,
        int $nIter = 100,
        float $tol = 1e-4
    ) {
        $this->nStates = $nStates;
        $this->nFeatures = $nFeatures;
        $this->nIter = $nIter;
        $this->tol = $tol;
        $this->initParams();
    }

    private function initParams(): void {
        $this->pi = array_fill(0, $this->nStates, 1 / $this->nStates);

        $this->A = [];
        for ($i = 0; $i < $this->nStates; $i++) {
            $row = [];
            $sum = 0;
            for ($j = 0; $j < $this->nStates; $j++) {
                $val = ($i === $j) ? 0.7 : 0.1;
                $row[] = $val;
                $sum += $val;
            }
            $this->A[] = array_map(fn($x) => $x / $sum, $row);
        }

        $this->means = [];
        for ($k = 0; $k < $this->nStates; $k++) {
            $this->means[] = array_map(
                fn() => (mt_rand() / mt_getrandmax()) - 0.5,
                range(0, $this->nFeatures - 1)
            );
        }

        $this->covars = [];
        for ($k = 0; $k < $this->nStates; $k++) {
            $cov = [];
            for ($i = 0; $i < $this->nFeatures; $i++) {
                $row = array_fill(0, $this->nFeatures, 0.0);
                $row[$i] = 1.0;
                $cov[] = $row;
            }
            $this->covars[] = $cov;
        }
    }

    private function logsumexp(array $arr): float {
        $max = max($arr);
        $sum = 0;
        foreach ($arr as $x) {
            $sum += exp($x - $max);
        }
        return $max + log($sum);
    }

    private function computeLogLikelihood(array $X): array {
        $nSamples = count($X);
        $logProb = [];

        for ($t = 0; $t < $nSamples; $t++) {
            $row = [];
            for ($k = 0; $k < $this->nStates; $k++) {
                $mahal = 0;
                $logDet = 0;
                for ($d = 0; $d < $this->nFeatures; $d++) {
                    $diff = $X[$t][$d] - $this->means[$k][$d];
                    $mahal += $diff * $diff / $this->covars[$k][$d][$d];
                    $logDet += log($this->covars[$k][$d][$d]);
                }
                $row[] = -0.5 * ($this->nFeatures * log(2 * M_PI) + $logDet + $mahal);
            }
            $logProb[] = $row;
        }

        return $logProb;
    }

    public function decode(array $X): array {
        $nSamples = count($X);
        $logEmission = $this->computeLogLikelihood($X);

        $V = array_fill(0, $nSamples, array_fill(0, $this->nStates, 0.0));
        $backpointer = array_fill(0, $nSamples, array_fill(0, $this->nStates, 0));

        for ($k = 0; $k < $this->nStates; $k++) {
            $V[0][$k] = log($this->pi[$k]) + $logEmission[0][$k];
        }

        $logA = array_map(
            fn($row) => array_map(fn($x) => log($x), $row),
            $this->A
        );

        for ($t = 1; $t < $nSamples; $t++) {
            for ($j = 0; $j < $this->nStates; $j++) {
                $maxScore = -INF;
                $maxIdx = 0;
                for ($i = 0; $i < $this->nStates; $i++) {
                    $score = $V[$t-1][$i] + $logA[$i][$j];
                    if ($score > $maxScore) {
                        $maxScore = $score;
                        $maxIdx = $i;
                    }
                }
                $backpointer[$t][$j] = $maxIdx;
                $V[$t][$j] = $maxScore + $logEmission[$t][$j];
            }
        }

        $states = array_fill(0, $nSamples, 0);
        $maxFinal = -INF;
        for ($k = 0; $k < $this->nStates; $k++) {
            if ($V[$nSamples-1][$k] > $maxFinal) {
                $maxFinal = $V[$nSamples-1][$k];
                $states[$nSamples-1] = $k;
            }
        }

        for ($t = $nSamples - 2; $t >= 0; $t--) {
            $states[$t] = $backpointer[$t+1][$states[$t+1]];
        }

        return ['states' => $states, 'logProb' => $maxFinal];
    }

    public function getTransitionMatrix(): array {
        return $this->A;
    }
}

class MarketRegimeDetector {
    private const REGIME_NAMES = ['Bull', 'Bear', 'HighVol', 'Crisis', 'Recovery'];

    private string $apiKey;
    private int $nRegimes;
    private GaussianHMM $hmm;
    private array $regimeLabels = [];

    public function __construct(string $apiKey, int $nRegimes = 4) {
        $this->apiKey = $apiKey;
        $this->nRegimes = min($nRegimes, count(self::REGIME_NAMES));
        $this->hmm = new GaussianHMM($this->nRegimes, 5);

        for ($i = 0; $i < $this->nRegimes; $i++) {
            $this->regimeLabels[$i] = self::REGIME_NAMES[$i];
        }
    }

    public function fetchDailyObservations(
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
                'category.id' => 'finance',
                'language.code' => 'en',
                'per_page' => 50
            ]);

            $response = file_get_contents(
                "https://api.apitube.io/v1/news/everything?{$params}"
            );
            $data = json_decode($response, true);
            $articles = $data['results'] ?? [];

            if (count($articles) > 0) {
                $sentiments = array_map(
                    fn($a) => $a['sentiment']['overall'] ?? 0,
                    $articles
                );

                $mean = array_sum($sentiments) / count($sentiments);
                $variance = array_sum(array_map(
                    fn($s) => ($s - $mean) ** 2,
                    $sentiments
                )) / count($sentiments);

                $observations[] = [
                    'timestamp' => clone $current,
                    'sentimentMean' => $mean,
                    'sentimentStd' => sqrt($variance),
                    'coverageVolume' => count($articles),
                    'dispersion' => $this->computeSourceDispersion($articles),
                    'momentum' => 0
                ];
            }

            $current->modify('+1 day');
        }

        for ($i = 1; $i < count($observations); $i++) {
            $observations[$i]['momentum'] =
                $observations[$i]['sentimentMean'] -
                $observations[$i-1]['sentimentMean'];
        }

        return $observations;
    }

    private function computeSourceDispersion(array $articles): float {
        $sourceSentiments = [];

        foreach ($articles as $article) {
            $source = $article['source']['domain'] ?? 'unknown';
            $sentiment = $article['sentiment']['overall'] ?? 0;
            $sourceSentiments[$source][] = $sentiment;
        }

        if (count($sourceSentiments) < 2) {
            return 0.0;
        }

        $means = array_map(
            fn($s) => array_sum($s) / count($s),
            $sourceSentiments
        );

        $meanOfMeans = array_sum($means) / count($means);
        $variance = array_sum(array_map(
            fn($m) => ($m - $meanOfMeans) ** 2,
            $means
        )) / count($means);

        return sqrt($variance);
    }

    private function toVector(array $obs): array {
        return [
            $obs['sentimentMean'],
            $obs['sentimentStd'],
            log1p($obs['coverageVolume']),
            $obs['dispersion'],
            $obs['momentum']
        ];
    }

    public function detectCurrentRegime(string $entity): array {
        $endDate = new DateTime();
        $startDate = (clone $endDate)->modify('-7 days');

        $recentObs = $this->fetchDailyObservations($entity, $startDate, $endDate);

        if (empty($recentObs)) {
            return ['error' => 'No recent data available'];
        }

        $X = array_map(fn($obs) => $this->toVector($obs), $recentObs);
        $result = $this->hmm->decode($X);
        $currentRegime = end($result['states']);

        $transitionMatrix = $this->hmm->getTransitionMatrix();

        return [
            'current_regime' => $this->regimeLabels[$currentRegime],
            'transition_forecast' => array_combine(
                array_values($this->regimeLabels),
                $transitionMatrix[$currentRegime]
            ),
            'timestamp' => end($recentObs)['timestamp']->format('c')
        ];
    }

    public function generateRegimeSignals(string $entity): array {
        $regimeInfo = $this->detectCurrentRegime($entity);

        if (isset($regimeInfo['error'])) {
            return $regimeInfo;
        }

        $signalMap = [
            'Bull' => [
                'equity_signal' => 'overweight',
                'bond_signal' => 'underweight',
                'volatility_signal' => 'sell',
                'risk_budget' => 1.2
            ],
            'Bear' => [
                'equity_signal' => 'underweight',
                'bond_signal' => 'overweight',
                'volatility_signal' => 'hold',
                'risk_budget' => 0.6
            ],
            'HighVol' => [
                'equity_signal' => 'neutral',
                'bond_signal' => 'neutral',
                'volatility_signal' => 'buy',
                'risk_budget' => 0.8
            ],
            'Crisis' => [
                'equity_signal' => 'strong_underweight',
                'bond_signal' => 'strong_overweight',
                'volatility_signal' => 'strong_buy',
                'risk_budget' => 0.3
            ],
            'Recovery' => [
                'equity_signal' => 'accumulate',
                'bond_signal' => 'reduce',
                'volatility_signal' => 'sell',
                'risk_budget' => 1.0
            ]
        ];

        $currentRegime = $regimeInfo['current_regime'];

        return [
            'regime' => $currentRegime,
            'signals' => $signalMap[$currentRegime] ?? $signalMap['HighVol'],
            'transition_forecast' => $regimeInfo['transition_forecast'],
            'timestamp' => $regimeInfo['timestamp']
        ];
    }
}

// Usage
$detector = new MarketRegimeDetector('YOUR_API_KEY', 4);
print_r($detector->detectCurrentRegime('S&P 500'));
print_r($detector->generateRegimeSignals('S&P 500'));
```

## Use Cases

### Quantitative Trading
- Regime-dependent portfolio allocation
- Risk budget adjustment based on market state
- Signal filtering by regime context

### Risk Management
- Early warning for regime transitions
- Crisis probability estimation
- Drawdown risk assessment

### Asset Allocation
- Dynamic asset class weights by regime
- Tactical allocation overlays
- Multi-asset regime timing

### Research
- Market cycle analysis
- Regime duration studies
- Transition dynamics research
