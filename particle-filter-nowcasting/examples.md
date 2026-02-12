# Particle Filter Nowcasting - Advanced Examples

## Multi-Entity Crisis Nowcasting System

Track multiple entities simultaneously during crisis events with correlated state estimation.

```python
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class EntityState:
    """State estimate for a single entity."""
    entity: str
    sentiment: float
    sentiment_uncertainty: float
    coverage_rate: float
    coverage_uncertainty: float
    momentum: float
    crisis_probability: float

@dataclass
class SystemState:
    """Combined state of all monitored entities."""
    timestamp: datetime
    entities: Dict[str, EntityState]
    correlation_matrix: np.ndarray
    system_risk: float
    contagion_detected: bool

class MultiEntityCrisisNowcaster:
    """
    Particle filter for tracking multiple correlated entities.
    Detects crisis contagion across entity network.
    """

    # Crisis thresholds
    CRISIS_SENTIMENT_THRESHOLD = -0.4
    CRISIS_COVERAGE_MULTIPLIER = 3.0
    CONTAGION_CORRELATION_THRESHOLD = 0.7

    def __init__(self, api_key: str, entities: List[str], n_particles: int = 300):
        self.api_key = api_key
        self.entities = entities
        self.n_particles = n_particles
        self.base_url = "https://api.apitube.io/v1/news/everything"

        # State dimensions per entity: [sentiment, coverage_rate, momentum]
        self.state_dim_per_entity = 3
        self.total_dim = len(entities) * self.state_dim_per_entity

        # Particles: each particle is a full system state
        self.particles = None
        self.weights = None
        self.baselines = {}

        self._initialize_particles()

    def _initialize_particles(self):
        """Initialize particles for all entities."""
        self.particles = np.zeros((self.n_particles, self.total_dim))

        for i in range(self.n_particles):
            for j, entity in enumerate(self.entities):
                offset = j * self.state_dim_per_entity
                self.particles[i, offset] = np.random.normal(0, 0.3)      # sentiment
                self.particles[i, offset + 1] = np.random.gamma(2, 0.5)   # coverage
                self.particles[i, offset + 2] = np.random.normal(0, 0.2)  # momentum

        self.weights = np.ones(self.n_particles) / self.n_particles

    async def fetch_entity_observation(self, session: aiohttp.ClientSession,
                                        entity: str, hours: float = 2) -> Dict:
        """Fetch current observation for an entity."""
        start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        params = {
            "api_key": self.api_key,
            "organization.name": entity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100
        }

        async with session.get(self.base_url, params=params) as response:
            data = await response.json()
            articles = data.get("results", [])

        if not articles:
            return {
                "entity": entity,
                "volume": 0,
                "sentiment": 0,
                "sentiment_std": 0.3
            }

        sentiments = [
            a.get("sentiment", {}).get("overall", {}).get("score", 0)
            for a in articles
        ]

        return {
            "entity": entity,
            "volume": len(articles) / hours,
            "sentiment": np.mean(sentiments),
            "sentiment_std": np.std(sentiments) if len(sentiments) > 1 else 0.3
        }

    async def fetch_all_observations(self) -> Dict[str, Dict]:
        """Fetch observations for all entities concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_entity_observation(session, entity)
                for entity in self.entities
            ]
            results = await asyncio.gather(*tasks)

        return {r["entity"]: r for r in results}

    async def update_baselines(self):
        """Update baseline statistics for all entities."""
        async with aiohttp.ClientSession() as session:
            for entity in self.entities:
                daily_volumes = []

                for d in range(1, 8):
                    start = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")
                    end = (datetime.utcnow() - timedelta(days=d-1)).strftime("%Y-%m-%d")

                    params = {
                        "api_key": self.api_key,
                        "organization.name": entity,
                        "published_at.start": start,
                        "published_at.end": end,
                        "language.code": "en",
                        "per_page": 1
                    }

                    async with session.get(self.base_url, params=params) as response:
                        data = await response.json()
                        daily_volumes.append(len(data.get("results", [])))

                self.baselines[entity] = {
                    "volume_mean": np.mean(daily_volumes) / 24,
                    "volume_std": np.std(daily_volumes) / 24
                }

    def state_transition(self, particles: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Joint state transition with cross-entity correlation.
        """
        new_particles = particles.copy()
        n = len(self.entities)

        # Process noise
        sentiment_noise = 0.05
        coverage_noise = 0.1
        momentum_noise = 0.08

        for i in range(self.n_particles):
            for j, entity in enumerate(self.entities):
                offset = j * self.state_dim_per_entity

                sentiment = new_particles[i, offset]
                coverage = new_particles[i, offset + 1]
                momentum = new_particles[i, offset + 2]

                # Cross-entity influence (contagion)
                neighbor_sentiment = 0
                for k, other in enumerate(self.entities):
                    if k != j:
                        other_offset = k * self.state_dim_per_entity
                        other_sentiment = new_particles[i, other_offset]
                        # Contagion effect
                        neighbor_sentiment += 0.1 * other_sentiment

                neighbor_sentiment /= max(n - 1, 1)

                # Update sentiment with momentum and contagion
                new_sentiment = sentiment + momentum * dt + 0.2 * neighbor_sentiment
                new_sentiment += np.random.normal(0, sentiment_noise)
                new_particles[i, offset] = np.clip(new_sentiment, -1, 1)

                # Coverage mean-reverts
                baseline = self.baselines.get(entity, {}).get("volume_mean", 1.0)
                new_coverage = coverage + 0.3 * (baseline - coverage) * dt
                new_coverage += np.random.normal(0, coverage_noise)
                new_particles[i, offset + 1] = max(0, new_coverage)

                # Momentum decays
                new_momentum = momentum * 0.9 + np.random.normal(0, momentum_noise)
                new_particles[i, offset + 2] = new_momentum

        return new_particles

    def observation_likelihood(self, particle: np.ndarray,
                                observations: Dict[str, Dict]) -> float:
        """Calculate joint likelihood of all observations."""
        total_log_likelihood = 0

        for j, entity in enumerate(self.entities):
            if entity not in observations:
                continue

            obs = observations[entity]
            offset = j * self.state_dim_per_entity

            state_sentiment = particle[offset]
            state_coverage = particle[offset + 1]

            obs_sentiment = obs["sentiment"]
            obs_volume = obs["volume"]

            # Sentiment likelihood
            sentiment_diff = obs_sentiment - state_sentiment
            sentiment_var = 0.01 + obs.get("sentiment_std", 0.3) ** 2
            sentiment_ll = -0.5 * sentiment_diff ** 2 / sentiment_var

            # Volume likelihood (log-normal)
            if state_coverage > 0 and obs_volume > 0:
                log_vol_diff = np.log(obs_volume + 1) - np.log(state_coverage + 1)
                volume_ll = -0.5 * log_vol_diff ** 2 / 0.04
            else:
                volume_ll = -5

            total_log_likelihood += sentiment_ll + volume_ll

        return np.exp(total_log_likelihood)

    def update(self, observations: Dict[str, Dict]) -> SystemState:
        """Run particle filter update step."""
        # Propagate
        self.particles = self.state_transition(self.particles)

        # Update weights
        for i in range(self.n_particles):
            likelihood = self.observation_likelihood(self.particles[i], observations)
            self.weights[i] *= likelihood

        # Normalize
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Effective sample size
        ess = 1.0 / np.sum(self.weights ** 2)

        # Resample if needed
        if ess < self.n_particles / 2:
            self._resample()

        # Compute state estimates
        entity_states = {}
        for j, entity in enumerate(self.entities):
            offset = j * self.state_dim_per_entity

            sentiment_values = self.particles[:, offset]
            coverage_values = self.particles[:, offset + 1]
            momentum_values = self.particles[:, offset + 2]

            mean_sentiment = np.average(sentiment_values, weights=self.weights)
            mean_coverage = np.average(coverage_values, weights=self.weights)
            mean_momentum = np.average(momentum_values, weights=self.weights)

            std_sentiment = np.sqrt(np.average(
                (sentiment_values - mean_sentiment) ** 2, weights=self.weights
            ))
            std_coverage = np.sqrt(np.average(
                (coverage_values - mean_coverage) ** 2, weights=self.weights
            ))

            # Crisis probability
            crisis_prob = np.sum(self.weights[sentiment_values < self.CRISIS_SENTIMENT_THRESHOLD])

            entity_states[entity] = EntityState(
                entity=entity,
                sentiment=round(float(mean_sentiment), 3),
                sentiment_uncertainty=round(float(std_sentiment), 3),
                coverage_rate=round(float(mean_coverage), 2),
                coverage_uncertainty=round(float(std_coverage), 2),
                momentum=round(float(mean_momentum), 3),
                crisis_probability=round(float(crisis_prob), 3)
            )

        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix()

        # System-level risk
        crisis_probs = [es.crisis_probability for es in entity_states.values()]
        system_risk = np.mean(crisis_probs) + 0.5 * max(crisis_probs)

        # Contagion detection
        contagion = np.max(np.abs(correlation_matrix - np.eye(len(self.entities)))) > \
                    self.CONTAGION_CORRELATION_THRESHOLD

        return SystemState(
            timestamp=datetime.utcnow(),
            entities=entity_states,
            correlation_matrix=correlation_matrix,
            system_risk=round(float(system_risk), 3),
            contagion_detected=contagion
        )

    def _resample(self):
        """Systematic resampling."""
        cumsum = np.cumsum(self.weights)
        u = np.random.uniform(0, 1.0 / self.n_particles)
        positions = (u + np.arange(self.n_particles) / self.n_particles)

        new_particles = np.zeros_like(self.particles)
        j = 0
        for i, pos in enumerate(positions):
            while cumsum[j] < pos:
                j += 1
            new_particles[i] = self.particles[j]

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def _compute_correlation_matrix(self) -> np.ndarray:
        """Compute sentiment correlation matrix from particles."""
        n = len(self.entities)
        sentiment_samples = np.zeros((self.n_particles, n))

        for j in range(n):
            offset = j * self.state_dim_per_entity
            sentiment_samples[:, j] = self.particles[:, offset]

        # Weighted correlation
        corr = np.corrcoef(sentiment_samples.T)
        return np.round(corr, 3)

    async def run_crisis_monitoring(self, cycles: int = 10,
                                     interval_minutes: int = 15) -> List[Dict]:
        """Run continuous crisis monitoring."""
        print("Starting multi-entity crisis nowcasting...")
        print(f"Entities: {self.entities}")

        # Initialize baselines
        await self.update_baselines()
        print("Baselines initialized")

        history = []

        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle + 1}/{cycles} ---")

            # Fetch observations
            observations = await self.fetch_all_observations()

            # Update filter
            state = self.update(observations)

            # Report
            print(f"Time: {state.timestamp.strftime('%H:%M:%S')}")
            print(f"System Risk: {state.system_risk:.3f}")
            print(f"Contagion Detected: {state.contagion_detected}")

            print("\nEntity States:")
            for entity, es in state.entities.items():
                crisis_flag = " [CRISIS]" if es.crisis_probability > 0.5 else ""
                print(f"  {entity}: sentiment={es.sentiment:.3f} (+/-{es.sentiment_uncertainty:.2f}), "
                      f"coverage={es.coverage_rate:.1f}, P(crisis)={es.crisis_probability:.2f}{crisis_flag}")

            history.append({
                "cycle": cycle + 1,
                "timestamp": state.timestamp.isoformat(),
                "system_risk": state.system_risk,
                "contagion": state.contagion_detected,
                "entities": {
                    e: {
                        "sentiment": es.sentiment,
                        "sentiment_uncertainty": es.sentiment_uncertainty,
                        "crisis_probability": es.crisis_probability
                    }
                    for e, es in state.entities.items()
                }
            })

            if cycle < cycles - 1:
                await asyncio.sleep(interval_minutes * 60)

        return history


# Usage
async def main():
    nowcaster = MultiEntityCrisisNowcaster(
        api_key="YOUR_API_KEY",
        entities=["Boeing", "Airbus", "Lockheed Martin", "Northrop Grumman"],
        n_particles=400
    )

    results = await nowcaster.run_crisis_monitoring(cycles=6, interval_minutes=30)

    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print(json.dumps(results[-1], indent=2))

asyncio.run(main())
```

## Regime-Switching Particle Filter

Detect and track regime changes in news coverage dynamics.

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class CoverageRegime(Enum):
    """Possible coverage regimes."""
    QUIET = "quiet"           # Low volume, stable sentiment
    NORMAL = "normal"         # Average volume and sentiment
    ELEVATED = "elevated"     # Higher than normal attention
    CRISIS = "crisis"         # High volume, negative sentiment
    RECOVERY = "recovery"     # Improving sentiment, moderate volume

@dataclass
class RegimeState:
    """State with regime indicator."""
    sentiment: float
    coverage_rate: float
    momentum: float
    regime: CoverageRegime
    regime_probability: Dict[CoverageRegime, float]

class RegimeSwitchingFilter:
    """
    Particle filter with regime-switching dynamics.
    Different state transition models for different regimes.
    """

    # Regime transition probabilities (per hour)
    TRANSITION_PROBS = {
        CoverageRegime.QUIET: {
            CoverageRegime.QUIET: 0.90,
            CoverageRegime.NORMAL: 0.08,
            CoverageRegime.ELEVATED: 0.02,
            CoverageRegime.CRISIS: 0.00,
            CoverageRegime.RECOVERY: 0.00
        },
        CoverageRegime.NORMAL: {
            CoverageRegime.QUIET: 0.05,
            CoverageRegime.NORMAL: 0.85,
            CoverageRegime.ELEVATED: 0.08,
            CoverageRegime.CRISIS: 0.02,
            CoverageRegime.RECOVERY: 0.00
        },
        CoverageRegime.ELEVATED: {
            CoverageRegime.QUIET: 0.02,
            CoverageRegime.NORMAL: 0.10,
            CoverageRegime.ELEVATED: 0.78,
            CoverageRegime.CRISIS: 0.10,
            CoverageRegime.RECOVERY: 0.00
        },
        CoverageRegime.CRISIS: {
            CoverageRegime.QUIET: 0.00,
            CoverageRegime.NORMAL: 0.00,
            CoverageRegime.ELEVATED: 0.05,
            CoverageRegime.CRISIS: 0.85,
            CoverageRegime.RECOVERY: 0.10
        },
        CoverageRegime.RECOVERY: {
            CoverageRegime.QUIET: 0.05,
            CoverageRegime.NORMAL: 0.20,
            CoverageRegime.ELEVATED: 0.05,
            CoverageRegime.CRISIS: 0.05,
            CoverageRegime.RECOVERY: 0.65
        }
    }

    # Regime-specific dynamics
    REGIME_DYNAMICS = {
        CoverageRegime.QUIET: {
            "sentiment_mean": 0.1,
            "sentiment_vol": 0.05,
            "coverage_mean": 0.5,
            "coverage_vol": 0.1
        },
        CoverageRegime.NORMAL: {
            "sentiment_mean": 0.0,
            "sentiment_vol": 0.1,
            "coverage_mean": 1.0,
            "coverage_vol": 0.2
        },
        CoverageRegime.ELEVATED: {
            "sentiment_mean": 0.0,
            "sentiment_vol": 0.15,
            "coverage_mean": 2.0,
            "coverage_vol": 0.4
        },
        CoverageRegime.CRISIS: {
            "sentiment_mean": -0.4,
            "sentiment_vol": 0.2,
            "coverage_mean": 4.0,
            "coverage_vol": 0.6
        },
        CoverageRegime.RECOVERY: {
            "sentiment_mean": -0.1,
            "sentiment_vol": 0.15,
            "coverage_mean": 1.5,
            "coverage_vol": 0.3
        }
    }

    def __init__(self, entity: str, n_particles: int = 500):
        self.entity = entity
        self.n_particles = n_particles

        # Particle state: [sentiment, coverage, momentum, regime_index]
        self.particles = np.zeros((n_particles, 4))
        self.weights = np.ones(n_particles) / n_particles
        self.baseline_coverage = 1.0

        self._initialize()

    def _initialize(self):
        """Initialize particles with regime distribution."""
        regimes = list(CoverageRegime)

        for i in range(self.n_particles):
            # Start mostly in NORMAL
            regime_idx = np.random.choice(
                len(regimes),
                p=[0.1, 0.7, 0.1, 0.05, 0.05]
            )
            regime = regimes[regime_idx]
            dynamics = self.REGIME_DYNAMICS[regime]

            self.particles[i, 0] = np.random.normal(
                dynamics["sentiment_mean"], dynamics["sentiment_vol"]
            )
            self.particles[i, 1] = max(0.1, np.random.normal(
                dynamics["coverage_mean"], dynamics["coverage_vol"]
            ))
            self.particles[i, 2] = np.random.normal(0, 0.1)
            self.particles[i, 3] = regime_idx

    def fetch_baseline(self, days: int = 7) -> float:
        """Fetch baseline coverage rate."""
        daily_volumes = []

        for d in range(days):
            start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
            end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

            params = {
                "api_key": API_KEY,
                "organization.name": self.entity,
                "published_at.start": start,
                "published_at.end": end,
                "language.code": "en",
                "per_page": 1
            }

            response = requests.get(BASE_URL, params=params)
            daily_volumes.append(len(response.json().get("results", [])))

        self.baseline_coverage = np.mean(daily_volumes) / 24 if daily_volumes else 1.0
        return self.baseline_coverage

    def fetch_observation(self, hours: float = 2) -> Dict:
        """Fetch current observation."""
        start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "api_key": API_KEY,
            "organization.name": self.entity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100
        }

        response = requests.get(BASE_URL, params=params)
        articles = response.json().get("results", [])

        if not articles:
            return {"volume": 0, "sentiment": 0, "sentiment_std": 0.3}

        sentiments = [
            a.get("sentiment", {}).get("overall", {}).get("score", 0)
            for a in articles
        ]

        return {
            "volume": len(articles) / hours,
            "sentiment": np.mean(sentiments),
            "sentiment_std": np.std(sentiments) if len(sentiments) > 1 else 0.3
        }

    def state_transition(self):
        """Propagate particles with regime-switching."""
        regimes = list(CoverageRegime)

        for i in range(self.n_particles):
            current_regime_idx = int(self.particles[i, 3])
            current_regime = regimes[current_regime_idx]

            # Sample new regime
            trans_probs = self.TRANSITION_PROBS[current_regime]
            probs = [trans_probs[r] for r in regimes]
            new_regime_idx = np.random.choice(len(regimes), p=probs)
            new_regime = regimes[new_regime_idx]

            # Get dynamics for new regime
            dynamics = self.REGIME_DYNAMICS[new_regime]

            # Update sentiment (mean-reverting within regime)
            sentiment = self.particles[i, 0]
            momentum = self.particles[i, 2]

            new_sentiment = sentiment + momentum
            new_sentiment += 0.2 * (dynamics["sentiment_mean"] - sentiment)
            new_sentiment += np.random.normal(0, dynamics["sentiment_vol"])
            self.particles[i, 0] = np.clip(new_sentiment, -1, 1)

            # Update coverage (regime-specific level)
            coverage = self.particles[i, 1]
            target_coverage = dynamics["coverage_mean"] * self.baseline_coverage
            new_coverage = coverage + 0.3 * (target_coverage - coverage)
            new_coverage += np.random.normal(0, dynamics["coverage_vol"] * self.baseline_coverage)
            self.particles[i, 1] = max(0.1, new_coverage)

            # Update momentum
            new_momentum = momentum * 0.8 + np.random.normal(0, 0.1)
            self.particles[i, 2] = new_momentum

            # Update regime
            self.particles[i, 3] = new_regime_idx

    def observation_likelihood(self, particle: np.ndarray, obs: Dict) -> float:
        """Calculate likelihood given regime."""
        regimes = list(CoverageRegime)
        regime = regimes[int(particle[3])]
        dynamics = self.REGIME_DYNAMICS[regime]

        state_sentiment = particle[0]
        state_coverage = particle[1]

        obs_sentiment = obs["sentiment"]
        obs_volume = obs["volume"]

        # Sentiment likelihood (regime-adjusted)
        sentiment_diff = obs_sentiment - state_sentiment
        sentiment_var = dynamics["sentiment_vol"] ** 2 + obs.get("sentiment_std", 0.3) ** 2
        sentiment_ll = np.exp(-0.5 * sentiment_diff ** 2 / sentiment_var)

        # Volume likelihood (regime-adjusted)
        if state_coverage > 0 and obs_volume > 0:
            log_vol_diff = np.log(obs_volume + 1) - np.log(state_coverage + 1)
            volume_var = (dynamics["coverage_vol"] * 0.5) ** 2
            volume_ll = np.exp(-0.5 * log_vol_diff ** 2 / volume_var)
        else:
            volume_ll = 0.1

        return sentiment_ll * volume_ll

    def update(self, obs: Dict) -> RegimeState:
        """Run filter update."""
        # Propagate
        self.state_transition()

        # Update weights
        for i in range(self.n_particles):
            likelihood = self.observation_likelihood(self.particles[i], obs)
            self.weights[i] *= likelihood

        # Normalize
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Resample if needed
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n_particles / 2:
            self._resample()

        # Compute estimates
        mean_sentiment = np.average(self.particles[:, 0], weights=self.weights)
        mean_coverage = np.average(self.particles[:, 1], weights=self.weights)
        mean_momentum = np.average(self.particles[:, 2], weights=self.weights)

        # Regime probabilities
        regimes = list(CoverageRegime)
        regime_probs = {}
        for r_idx, regime in enumerate(regimes):
            mask = self.particles[:, 3] == r_idx
            regime_probs[regime] = float(np.sum(self.weights[mask]))

        # Most likely regime
        most_likely_regime = max(regime_probs, key=regime_probs.get)

        return RegimeState(
            sentiment=round(float(mean_sentiment), 3),
            coverage_rate=round(float(mean_coverage), 2),
            momentum=round(float(mean_momentum), 3),
            regime=most_likely_regime,
            regime_probability={r: round(p, 3) for r, p in regime_probs.items()}
        )

    def _resample(self):
        """Systematic resampling."""
        cumsum = np.cumsum(self.weights)
        u = np.random.uniform(0, 1.0 / self.n_particles)
        positions = u + np.arange(self.n_particles) / self.n_particles

        new_particles = np.zeros_like(self.particles)
        j = 0
        for i, pos in enumerate(positions):
            while cumsum[j] < pos:
                j += 1
            new_particles[i] = self.particles[j].copy()

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def run_monitoring(self, iterations: int = 12) -> List[Dict]:
        """Run regime monitoring."""
        print(f"Starting regime-switching filter for: {self.entity}")

        # Initialize baseline
        self.fetch_baseline()
        print(f"Baseline coverage: {self.baseline_coverage:.2f}/hour")

        history = []

        for i in range(iterations):
            obs = self.fetch_observation()
            state = self.update(obs)

            print(f"\nIteration {i+1}:")
            print(f"  Observation: volume={obs['volume']:.1f}, sentiment={obs['sentiment']:.3f}")
            print(f"  Estimated: sentiment={state.sentiment:.3f}, coverage={state.coverage_rate:.1f}")
            print(f"  REGIME: {state.regime.value.upper()}")

            # Show regime probabilities
            print(f"  Probabilities:", end="")
            for r, p in state.regime_probability.items():
                if p > 0.05:
                    print(f" {r.value}:{p:.2f}", end="")
            print()

            history.append({
                "iteration": i + 1,
                "sentiment": state.sentiment,
                "coverage": state.coverage_rate,
                "regime": state.regime.value,
                "regime_probs": {r.value: p for r, p in state.regime_probability.items()}
            })

        return history


# Usage
pf = RegimeSwitchingFilter(entity="Tesla", n_particles=500)

history = pf.run_monitoring(iterations=8)

print("\n" + "=" * 60)
print("REGIME HISTORY")
for h in history:
    print(f"  {h['iteration']}: {h['regime'].upper()} "
          f"(sentiment={h['sentiment']:.2f}, coverage={h['coverage']:.1f})")
```

## Adaptive Resampling with Kernel Smoothing

Advanced particle filter with adaptive resampling and kernel density estimation.

```python
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class AdaptiveParticleFilter:
    """
    Particle filter with adaptive resampling and kernel smoothing.
    Prevents particle degeneracy while maintaining diversity.
    """

    def __init__(self, entity: str, n_particles: int = 1000):
        self.entity = entity
        self.n_particles = n_particles

        # State: [sentiment, coverage_rate, momentum]
        self.particles = np.zeros((n_particles, 3))
        self.weights = np.ones(n_particles) / n_particles
        self.baseline = None

        # Adaptive parameters
        self.bandwidth = None
        self.ess_threshold = n_particles / 3
        self.regularization_noise = 0.01

        self._initialize()

    def _initialize(self):
        """Initialize from prior."""
        self.particles[:, 0] = np.random.normal(0, 0.3, self.n_particles)
        self.particles[:, 1] = np.random.gamma(2, 0.5, self.n_particles)
        self.particles[:, 2] = np.random.normal(0, 0.2, self.n_particles)

    def kernel_smooth_resample(self):
        """
        Resample with kernel smoothing to maintain diversity.
        Uses optimal bandwidth selection.
        """
        # Compute optimal bandwidth (Silverman's rule)
        n_eff = 1.0 / np.sum(self.weights ** 2)
        d = self.particles.shape[1]

        # Weighted standard deviations
        means = np.average(self.particles, axis=0, weights=self.weights)
        stds = np.sqrt(np.average(
            (self.particles - means) ** 2, axis=0, weights=self.weights
        ))

        # Silverman bandwidth
        h = (4 / ((d + 2) * n_eff)) ** (1 / (d + 4)) * stds
        self.bandwidth = h

        # Resample indices
        cumsum = np.cumsum(self.weights)
        u = np.random.uniform(0, 1.0 / self.n_particles)
        positions = u + np.arange(self.n_particles) / self.n_particles

        new_particles = np.zeros_like(self.particles)
        j = 0
        for i, pos in enumerate(positions):
            while cumsum[j] < pos:
                j += 1

            # Add kernel smoothing noise
            noise = np.random.normal(0, h)
            new_particles[i] = self.particles[j] + noise

        # Ensure valid ranges
        new_particles[:, 0] = np.clip(new_particles[:, 0], -1, 1)
        new_particles[:, 1] = np.maximum(new_particles[:, 1], 0.01)

        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles

    def adaptive_resample(self):
        """Resample only when ESS drops below threshold."""
        ess = 1.0 / np.sum(self.weights ** 2)

        if ess < self.ess_threshold:
            self.kernel_smooth_resample()
            return True
        return False

    def state_transition(self):
        """Propagate particles."""
        # Sentiment dynamics
        self.particles[:, 0] += self.particles[:, 2]  # momentum
        self.particles[:, 0] *= 0.98  # slight mean reversion
        self.particles[:, 0] += np.random.normal(0, 0.05, self.n_particles)
        self.particles[:, 0] = np.clip(self.particles[:, 0], -1, 1)

        # Coverage dynamics (mean-reverting)
        if self.baseline:
            target = self.baseline
            self.particles[:, 1] += 0.2 * (target - self.particles[:, 1])
        self.particles[:, 1] += np.random.normal(0, 0.1, self.n_particles)
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 0.01)

        # Momentum decay
        self.particles[:, 2] *= 0.9
        self.particles[:, 2] += np.random.normal(0, 0.08, self.n_particles)

    def observation_likelihood(self, obs: Dict) -> np.ndarray:
        """Vectorized likelihood computation."""
        obs_sentiment = obs["sentiment"]
        obs_volume = obs["volume"]
        obs_std = obs.get("sentiment_std", 0.3)

        # Sentiment likelihood
        sentiment_diff = obs_sentiment - self.particles[:, 0]
        sentiment_var = 0.01 + obs_std ** 2
        sentiment_ll = np.exp(-0.5 * sentiment_diff ** 2 / sentiment_var)

        # Volume likelihood
        log_vol_diff = np.log(obs_volume + 1) - np.log(self.particles[:, 1] + 1)
        volume_ll = np.exp(-0.5 * log_vol_diff ** 2 / 0.04)

        return sentiment_ll * volume_ll

    def update(self, obs: Dict) -> Dict:
        """Run filter update with adaptive resampling."""
        # Propagate
        self.state_transition()

        # Update weights
        likelihoods = self.observation_likelihood(obs)
        self.weights *= likelihoods

        # Normalize
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Adaptive resample
        resampled = self.adaptive_resample()

        # Compute estimates with kernel density
        estimates = self.compute_kde_estimates()
        estimates["resampled"] = resampled
        estimates["ess"] = 1.0 / np.sum(self.weights ** 2)

        return estimates

    def compute_kde_estimates(self) -> Dict:
        """Compute estimates using kernel density estimation."""
        # KDE for sentiment (1D)
        try:
            sentiment_kde = gaussian_kde(
                self.particles[:, 0],
                weights=self.weights,
                bw_method='silverman'
            )

            # Find mode
            x_grid = np.linspace(-1, 1, 200)
            density = sentiment_kde(x_grid)
            mode_sentiment = x_grid[np.argmax(density)]

            # Credible interval
            sorted_idx = np.argsort(self.particles[:, 0])
            sorted_weights = self.weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)

            lower_idx = np.searchsorted(cumsum, 0.1)
            upper_idx = np.searchsorted(cumsum, 0.9)

            lower_bound = self.particles[sorted_idx[lower_idx], 0]
            upper_bound = self.particles[sorted_idx[min(upper_idx, len(sorted_idx)-1)], 0]

        except:
            mode_sentiment = np.average(self.particles[:, 0], weights=self.weights)
            lower_bound = mode_sentiment - 0.2
            upper_bound = mode_sentiment + 0.2

        return {
            "sentiment": {
                "mean": round(float(np.average(self.particles[:, 0], weights=self.weights)), 3),
                "mode": round(float(mode_sentiment), 3),
                "ci_80": [round(float(lower_bound), 3), round(float(upper_bound), 3)]
            },
            "coverage_rate": {
                "mean": round(float(np.average(self.particles[:, 1], weights=self.weights)), 2),
                "std": round(float(np.sqrt(np.average(
                    (self.particles[:, 1] - np.average(self.particles[:, 1], weights=self.weights)) ** 2,
                    weights=self.weights
                ))), 2)
            },
            "momentum": round(float(np.average(self.particles[:, 2], weights=self.weights)), 3)
        }


# Usage
apf = AdaptiveParticleFilter(entity="Apple", n_particles=1000)

# Simulate some updates
for i in range(10):
    # Simulated observation
    obs = {
        "sentiment": np.random.normal(0.1, 0.3),
        "volume": np.random.gamma(5, 2),
        "sentiment_std": 0.25
    }

    result = apf.update(obs)

    print(f"Step {i+1}:")
    print(f"  Sentiment: {result['sentiment']['mean']:.3f} "
          f"(mode: {result['sentiment']['mode']:.3f}) "
          f"CI: {result['sentiment']['ci_80']}")
    print(f"  ESS: {result['ess']:.0f}, Resampled: {result['resampled']}")
```
