# Particle Filter Nowcasting

Workflow for real-time probabilistic state estimation and nowcasting using particle filter algorithms applied to news flow dynamics with the [APITube News API](https://apitube.io).

## Overview

The **Particle Filter Nowcasting** workflow implements Sequential Monte Carlo (SMC) methods to estimate latent states of news-driven phenomena in real-time. Features include adaptive particle weighting, state space modeling for sentiment and coverage dynamics, multi-modal posterior estimation, resampling strategies, and uncertainty quantification. Uses Bayesian inference to combine prior knowledge with streaming news observations. Ideal for real-time event monitoring, sentiment nowcasting, crisis state estimation, and any application requiring probabilistic tracking of rapidly evolving situations.

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
| `title`                       | string  | Filter by keywords in title.                                         |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (-1.0 to 1.0).                              |
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
from dataclasses import dataclass
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


@dataclass
class Particle:
    """Represents a single particle in the filter."""
    state: np.ndarray  # [sentiment_level, coverage_rate, momentum]
    weight: float
    history: List[np.ndarray]


@dataclass
class NowcastResult:
    """Result of a nowcasting step."""
    timestamp: datetime
    state_estimate: Dict[str, float]
    uncertainty: Dict[str, float]
    posterior_samples: List[Dict]
    effective_sample_size: float
    observation: Dict


class NewsParticleFilter:
    """
    Particle filter for real-time news state estimation.
    Tracks latent sentiment, coverage dynamics, and momentum.
    """

    # State dimensions
    STATE_DIM = 3  # sentiment_level, coverage_rate, momentum

    # Process noise (state transition uncertainty)
    PROCESS_NOISE = np.array([0.05, 0.1, 0.08])

    # Observation noise
    OBSERVATION_NOISE = np.array([0.1, 0.15])

    def __init__(self, n_particles: int = 500, entity: str = ""):
        self.n_particles = n_particles
        self.entity = entity
        self.particles: List[Particle] = []
        self.history = []
        self.baseline = None

        # Initialize particles
        self._initialize_particles()

    def _initialize_particles(self):
        """Initialize particles from prior distribution."""
        self.particles = []

        for _ in range(self.n_particles):
            # Prior: sentiment ~ N(0, 0.3), coverage_rate ~ Gamma(2, 0.5), momentum ~ N(0, 0.2)
            state = np.array([
                np.random.normal(0, 0.3),  # sentiment_level
                np.random.gamma(2, 0.5),    # coverage_rate (articles/hour)
                np.random.normal(0, 0.2)    # momentum
            ])

            self.particles.append(Particle(
                state=state,
                weight=1.0 / self.n_particles,
                history=[]
            ))

    def fetch_baseline(self, days: int = 7) -> Dict:
        """Fetch baseline statistics for the entity."""
        daily_stats = []

        for d in range(days):
            start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
            end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

            params = {
                "api_key": API_KEY,
                "entity.name": self.entity,
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 100,
            }

            response = requests.get(BASE_URL, params=params)
            articles = response.json().get("results", [])

            if articles:
                sentiments = [
                    a.get("sentiment", {}).get("overall", {}).get("score", 0)
                    for a in articles
                ]
                daily_stats.append({
                    "volume": len(articles),
                    "sentiment_mean": np.mean(sentiments),
                    "sentiment_std": np.std(sentiments) if len(sentiments) > 1 else 0.3
                })

        if not daily_stats:
            return {"volume_mean": 10, "volume_std": 5, "sentiment_mean": 0, "sentiment_std": 0.3}

        self.baseline = {
            "volume_mean": np.mean([s["volume"] for s in daily_stats]) / 24,  # per hour
            "volume_std": np.std([s["volume"] for s in daily_stats]) / 24,
            "sentiment_mean": np.mean([s["sentiment_mean"] for s in daily_stats]),
            "sentiment_std": np.mean([s["sentiment_std"] for s in daily_stats])
        }

        return self.baseline

    def fetch_observation(self, hours: float = 1) -> Dict:
        """Fetch current observation from news stream."""
        start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "api_key": API_KEY,
            "entity.name": self.entity,
            "published_at.start": start,
            "language": "en",
            "per_page": 100,
            "sort.by": "published_at",
            "sort.order": "desc",
        }

        response = requests.get(BASE_URL, params=params)
        articles = response.json().get("results", [])

        if not articles:
            return {
                "volume": 0,
                "sentiment": 0,
                "sentiment_std": 0.3,
                "high_authority_ratio": 0,
                "timestamp": datetime.utcnow()
            }

        sentiments = [
            a.get("sentiment", {}).get("overall", {}).get("score", 0)
            for a in articles
        ]

        high_auth = sum(
            1 for a in articles
            if a.get("source", {}).get("rankings", {}).get("opr", 0) >= 0.7
        )

        return {
            "volume": len(articles) / hours,  # Normalize to per-hour
            "sentiment": np.mean(sentiments),
            "sentiment_std": np.std(sentiments) if len(sentiments) > 1 else 0.3,
            "high_authority_ratio": high_auth / len(articles),
            "timestamp": datetime.utcnow()
        }

    def state_transition(self, particle: Particle, dt: float = 1.0) -> np.ndarray:
        """
        State transition model.
        Propagates particle state forward with stochastic dynamics.
        """
        state = particle.state.copy()

        # State dynamics:
        # sentiment_level evolves slowly with momentum
        # coverage_rate mean-reverts to baseline
        # momentum decays with noise

        if self.baseline:
            baseline_rate = self.baseline["volume_mean"]
            baseline_sentiment = self.baseline["sentiment_mean"]
        else:
            baseline_rate = 1.0
            baseline_sentiment = 0

        # Sentiment evolves with momentum
        state[0] = state[0] + state[2] * dt + np.random.normal(0, self.PROCESS_NOISE[0])

        # Coverage rate mean-reverts
        mean_reversion = 0.3
        state[1] = state[1] + mean_reversion * (baseline_rate - state[1]) * dt
        state[1] += np.random.normal(0, self.PROCESS_NOISE[1])
        state[1] = max(0, state[1])  # Coverage rate must be positive

        # Momentum decays
        state[2] = state[2] * 0.9 + np.random.normal(0, self.PROCESS_NOISE[2])

        return state

    def observation_likelihood(self, state: np.ndarray, observation: Dict) -> float:
        """
        Calculate likelihood of observation given state.
        Uses Gaussian likelihood model.
        """
        obs_sentiment = observation["sentiment"]
        obs_volume = observation["volume"]

        # Sentiment likelihood
        sentiment_diff = obs_sentiment - state[0]
        sentiment_var = self.OBSERVATION_NOISE[0] ** 2 + observation.get("sentiment_std", 0.3) ** 2
        sentiment_likelihood = np.exp(-0.5 * sentiment_diff ** 2 / sentiment_var)

        # Volume likelihood (log-normal)
        if state[1] > 0 and obs_volume > 0:
            log_vol_diff = np.log(obs_volume + 1) - np.log(state[1] + 1)
            volume_var = self.OBSERVATION_NOISE[1] ** 2
            volume_likelihood = np.exp(-0.5 * log_vol_diff ** 2 / volume_var)
        else:
            volume_likelihood = 0.1

        return sentiment_likelihood * volume_likelihood

    def update(self, observation: Dict) -> NowcastResult:
        """
        Run one particle filter update step.
        1. Propagate particles
        2. Update weights based on observation
        3. Resample if needed
        4. Compute estimates
        """
        # Propagate particles
        for particle in self.particles:
            particle.state = self.state_transition(particle)
            particle.history.append(particle.state.copy())

        # Update weights
        total_weight = 0
        for particle in self.particles:
            likelihood = self.observation_likelihood(particle.state, observation)
            particle.weight *= likelihood
            total_weight += particle.weight

        # Normalize weights
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # Reset to uniform if all weights collapsed
            for particle in self.particles:
                particle.weight = 1.0 / self.n_particles

        # Calculate effective sample size
        weights = np.array([p.weight for p in self.particles])
        ess = 1.0 / np.sum(weights ** 2)

        # Resample if ESS too low
        if ess < self.n_particles / 2:
            self._resample()

        # Compute state estimates
        estimate = self._compute_estimates()

        result = NowcastResult(
            timestamp=observation.get("timestamp", datetime.utcnow()),
            state_estimate={
                "sentiment_level": estimate["mean"][0],
                "coverage_rate": estimate["mean"][1],
                "momentum": estimate["mean"][2]
            },
            uncertainty={
                "sentiment_level": estimate["std"][0],
                "coverage_rate": estimate["std"][1],
                "momentum": estimate["std"][2]
            },
            posterior_samples=[
                {
                    "sentiment_level": p.state[0],
                    "coverage_rate": p.state[1],
                    "momentum": p.state[2],
                    "weight": p.weight
                }
                for p in self.particles[:50]  # Sample
            ],
            effective_sample_size=ess,
            observation=observation
        )

        self.history.append(result)
        return result

    def _resample(self):
        """Systematic resampling of particles."""
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)

        # Systematic resampling
        u = np.random.uniform(0, 1.0 / self.n_particles)
        positions = (u + np.arange(self.n_particles) / self.n_particles)

        new_particles = []
        j = 0
        for pos in positions:
            while cumsum[j] < pos and j < len(cumsum) - 1:
                j += 1
            new_particles.append(Particle(
                state=self.particles[j].state.copy(),
                weight=1.0 / self.n_particles,
                history=self.particles[j].history.copy()
            ))

        self.particles = new_particles

    def _compute_estimates(self) -> Dict:
        """Compute weighted mean and std of state."""
        weights = np.array([p.weight for p in self.particles])
        states = np.array([p.state for p in self.particles])

        mean = np.average(states, axis=0, weights=weights)

        # Weighted variance
        variance = np.average((states - mean) ** 2, axis=0, weights=weights)
        std = np.sqrt(variance)

        return {"mean": mean, "std": std}

    def forecast(self, steps: int = 6) -> List[Dict]:
        """
        Generate probabilistic forecast by propagating particles.
        """
        forecasts = []

        # Copy current particles
        forecast_particles = [
            Particle(state=p.state.copy(), weight=p.weight, history=[])
            for p in self.particles
        ]

        for step in range(1, steps + 1):
            # Propagate all particles
            for particle in forecast_particles:
                particle.state = self.state_transition(particle)

            # Compute forecast distribution
            weights = np.array([p.weight for p in forecast_particles])
            states = np.array([p.state for p in forecast_particles])

            mean = np.average(states, axis=0, weights=weights)
            variance = np.average((states - mean) ** 2, axis=0, weights=weights)
            std = np.sqrt(variance)

            # Compute quantiles
            sentiment_values = states[:, 0]
            coverage_values = states[:, 1]

            forecasts.append({
                "step": step,
                "hours_ahead": step,
                "timestamp": (datetime.utcnow() + timedelta(hours=step)).isoformat(),
                "sentiment": {
                    "mean": round(float(mean[0]), 3),
                    "std": round(float(std[0]), 3),
                    "q10": round(float(np.percentile(sentiment_values, 10)), 3),
                    "q50": round(float(np.percentile(sentiment_values, 50)), 3),
                    "q90": round(float(np.percentile(sentiment_values, 90)), 3)
                },
                "coverage_rate": {
                    "mean": round(float(mean[1]), 2),
                    "std": round(float(std[1]), 2),
                    "q10": round(float(np.percentile(coverage_values, 10)), 2),
                    "q50": round(float(np.percentile(coverage_values, 50)), 2),
                    "q90": round(float(np.percentile(coverage_values, 90)), 2)
                },
                "momentum": {
                    "mean": round(float(mean[2]), 3),
                    "std": round(float(std[2]), 3)
                }
            })

        return forecasts

    def run_nowcasting(self, iterations: int = 12, interval_hours: float = 1) -> Dict:
        """
        Run continuous nowcasting loop.
        """
        print(f"Starting particle filter nowcasting for: {self.entity}")
        print(f"Particles: {self.n_particles}")

        # Initialize baseline
        print("Fetching baseline...")
        self.fetch_baseline()
        print(f"  Baseline volume: {self.baseline['volume_mean']:.2f}/hour")
        print(f"  Baseline sentiment: {self.baseline['sentiment_mean']:.3f}")

        results = []

        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")

            # Fetch observation
            obs = self.fetch_observation(hours=interval_hours)
            print(f"  Observed: volume={obs['volume']:.1f}, sentiment={obs['sentiment']:.3f}")

            # Update filter
            result = self.update(obs)

            print(f"  Estimated: sentiment={result.state_estimate['sentiment_level']:.3f} "
                  f"(+/- {result.uncertainty['sentiment_level']:.3f})")
            print(f"  ESS: {result.effective_sample_size:.0f}/{self.n_particles}")

            results.append({
                "iteration": i + 1,
                "timestamp": result.timestamp.isoformat(),
                "observation": {
                    "volume": obs["volume"],
                    "sentiment": obs["sentiment"]
                },
                "estimate": result.state_estimate,
                "uncertainty": result.uncertainty,
                "ess": result.effective_sample_size
            })

        # Generate forecast
        print("\nGenerating forecast...")
        forecast = self.forecast(steps=6)

        return {
            "entity": self.entity,
            "n_particles": self.n_particles,
            "baseline": self.baseline,
            "nowcast_history": results,
            "forecast": forecast,
            "final_state": {
                "sentiment_level": round(results[-1]["estimate"]["sentiment_level"], 3),
                "coverage_rate": round(results[-1]["estimate"]["coverage_rate"], 2),
                "momentum": round(results[-1]["estimate"]["momentum"], 3)
            },
            "interpretation": self._interpret_state(results[-1])
        }

    def _interpret_state(self, result: Dict) -> Dict:
        """Interpret the current state estimate."""
        sentiment = result["estimate"]["sentiment_level"]
        momentum = result["estimate"]["momentum"]
        coverage = result["estimate"]["coverage_rate"]

        if sentiment > 0.3:
            sentiment_state = "positive"
        elif sentiment < -0.3:
            sentiment_state = "negative"
        else:
            sentiment_state = "neutral"

        if momentum > 0.1:
            trend = "improving"
        elif momentum < -0.1:
            trend = "deteriorating"
        else:
            trend = "stable"

        if self.baseline:
            coverage_ratio = coverage / max(self.baseline["volume_mean"], 0.1)
            if coverage_ratio > 1.5:
                attention = "elevated"
            elif coverage_ratio < 0.5:
                attention = "low"
            else:
                attention = "normal"
        else:
            attention = "unknown"

        return {
            "sentiment_state": sentiment_state,
            "trend": trend,
            "attention": attention,
            "summary": f"Sentiment is {sentiment_state} and {trend}, with {attention} media attention"
        }


# Run nowcasting
print("PARTICLE FILTER NOWCASTING")
print("=" * 70)

pf = NewsParticleFilter(n_particles=500, entity="Tesla")

results = pf.run_nowcasting(iterations=6, interval_hours=2)

print("\n" + "=" * 70)
print("NOWCASTING RESULTS")
print("-" * 50)
print(f"Entity: {results['entity']}")
print(f"\nFinal State Estimate:")
print(f"  Sentiment: {results['final_state']['sentiment_level']:.3f}")
print(f"  Coverage Rate: {results['final_state']['coverage_rate']:.1f}/hour")
print(f"  Momentum: {results['final_state']['momentum']:.3f}")
print(f"\nInterpretation: {results['interpretation']['summary']}")

print("\n6-HOUR FORECAST:")
print("-" * 50)
for f in results["forecast"]:
    print(f"  +{f['hours_ahead']}h: sentiment={f['sentiment']['mean']:.3f} "
          f"[{f['sentiment']['q10']:.2f}, {f['sentiment']['q90']:.2f}]")
```

## State Variables

| State | Description | Unit |
|-------|-------------|------|
| `sentiment_level` | Latent sentiment state | -1 to 1 |
| `coverage_rate` | Media coverage intensity | articles/hour |
| `momentum` | Sentiment change velocity | per hour |

## Algorithm Components

| Component | Description |
|-----------|-------------|
| **State Transition** | Stochastic dynamics with mean-reversion |
| **Observation Model** | Gaussian likelihood for sentiment and log-normal for volume |
| **Resampling** | Systematic resampling when ESS < N/2 |
| **Forecasting** | Particle propagation for probabilistic forecasts |

## Common Use Cases

- **Real-time crisis monitoring** — track sentiment evolution during breaking events.
- **Market sentiment nowcasting** — probabilistic state estimation for trading signals.
- **Reputation tracking** — continuous monitoring with uncertainty quantification.
- **Event impact assessment** — real-time estimation of developing situations.
- **Risk monitoring** — early warning with posterior probability estimates.
- **Research** — Bayesian inference for media dynamics modeling.

## See Also

- [examples.md](./examples.md) — detailed code examples.
