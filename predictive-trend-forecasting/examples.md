# Predictive Trend Forecasting - Advanced Examples

## Multi-Topic Portfolio Forecasting

Forecast multiple topics simultaneously and optimize content portfolio allocation.

```python
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import json

@dataclass
class TopicMetrics:
    topic: str
    current_volume: int
    trend_score: float  # -1 (declining) to 1 (viral)
    volatility: float
    forecast_growth: float
    sentiment_trend: float
    confidence: float

class PortfolioForecaster:
    """Multi-topic portfolio optimization with forecasting."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apitube.io/v1/news/everything"

    async def fetch_topic_data(self, session: aiohttp.ClientSession,
                               topic: str, days: int = 14) -> pd.DataFrame:
        """Fetch historical data for a single topic."""
        data_points = []

        for d in range(days):
            end_time = datetime.utcnow() - timedelta(days=d)
            start_time = end_time - timedelta(days=1)

            params = {
                "api_key": self.api_key,
                "topic.id": topic,
                "published_at.start": start_time.isoformat() + "Z",
                "published_at.end": end_time.isoformat() + "Z",
                "per_page": 100
            }

            try:
                async with session.get(self.base_url, params=params) as response:
                    result = await response.json()
                    articles = result.get("results", [])

                    neg = sum(1 for a in articles
                             if a.get('sentiment', {}).get('overall', {}).get('polarity') == 'negative')
                    pos = sum(1 for a in articles
                             if a.get('sentiment', {}).get('overall', {}).get('polarity') == 'positive')

                    data_points.append({
                        'date': end_time.date(),
                        'volume': len(articles),
                        'sentiment_score': (pos - neg) / max(len(articles), 1)
                    })
            except:
                continue

        return pd.DataFrame(data_points).sort_values('date')

    def calculate_topic_metrics(self, df: pd.DataFrame, topic: str) -> TopicMetrics:
        """Calculate comprehensive metrics for a topic."""
        if len(df) < 5:
            return TopicMetrics(
                topic=topic, current_volume=0, trend_score=0,
                volatility=1.0, forecast_growth=0, sentiment_trend=0, confidence=0
            )

        volumes = df['volume'].values
        sentiments = df['sentiment_score'].values

        # Current volume
        current_volume = int(volumes[-1])

        # Trend score: normalized growth rate
        growth_rates = np.diff(volumes) / np.maximum(volumes[:-1], 1)
        recent_growth = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else np.mean(growth_rates)

        # Normalize to -1 to 1 scale
        trend_score = np.tanh(recent_growth * 5)  # Scale factor

        # Volatility (coefficient of variation)
        volatility = np.std(volumes) / max(np.mean(volumes), 1)

        # Forecast growth (simple linear projection)
        x = np.arange(len(volumes))
        slope, _ = np.polyfit(x, volumes, 1)
        forecast_growth = slope / max(np.mean(volumes), 1)

        # Sentiment trend
        if len(sentiments) >= 3:
            sent_slope, _ = np.polyfit(np.arange(len(sentiments)), sentiments, 1)
            sentiment_trend = np.tanh(sent_slope * 10)
        else:
            sentiment_trend = 0

        # Confidence based on data quality
        confidence = min(0.9, 0.3 + len(df) / 50 + (1 - volatility) * 0.3)

        return TopicMetrics(
            topic=topic,
            current_volume=current_volume,
            trend_score=round(trend_score, 3),
            volatility=round(volatility, 3),
            forecast_growth=round(forecast_growth, 3),
            sentiment_trend=round(sentiment_trend, 3),
            confidence=round(confidence, 3)
        )

    async def analyze_portfolio(self, topics: List[str]) -> Dict:
        """Analyze multiple topics as a portfolio."""
        async with aiohttp.ClientSession() as session:
            # Fetch data for all topics concurrently
            tasks = [self.fetch_topic_data(session, topic) for topic in topics]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics = []
        for topic, df in zip(topics, results):
            if isinstance(df, Exception) or df.empty:
                continue
            m = self.calculate_topic_metrics(df, topic)
            metrics.append(m)

        return {
            'analyzed_at': datetime.utcnow().isoformat(),
            'topics': [
                {
                    'topic': m.topic,
                    'current_volume': m.current_volume,
                    'trend_score': m.trend_score,
                    'volatility': m.volatility,
                    'forecast_growth': m.forecast_growth,
                    'sentiment_trend': m.sentiment_trend,
                    'confidence': m.confidence,
                    'recommendation': self._get_recommendation(m)
                }
                for m in metrics
            ],
            'portfolio_summary': self._summarize_portfolio(metrics),
            'optimal_allocation': self._optimize_allocation(metrics)
        }

    def _get_recommendation(self, m: TopicMetrics) -> str:
        """Get recommendation for a single topic."""
        if m.trend_score > 0.5 and m.sentiment_trend > 0:
            return "STRONG_BUY: High growth + positive sentiment"
        elif m.trend_score > 0.3:
            return "BUY: Growing trend"
        elif m.trend_score < -0.3 and m.sentiment_trend < 0:
            return "SELL: Declining with negative sentiment"
        elif m.trend_score < -0.5:
            return "STRONG_SELL: Sharp decline"
        elif m.volatility > 0.8:
            return "WATCH: High volatility"
        else:
            return "HOLD: Stable"

    def _summarize_portfolio(self, metrics: List[TopicMetrics]) -> Dict:
        """Summarize portfolio metrics."""
        if not metrics:
            return {}

        return {
            'total_topics': len(metrics),
            'avg_trend_score': round(np.mean([m.trend_score for m in metrics]), 3),
            'avg_volatility': round(np.mean([m.volatility for m in metrics]), 3),
            'top_performers': [
                m.topic for m in sorted(metrics, key=lambda x: x.trend_score, reverse=True)[:3]
            ],
            'declining': [m.topic for m in metrics if m.trend_score < -0.2],
            'emerging': [m.topic for m in metrics if m.trend_score > 0.3]
        }

    def _optimize_allocation(self, metrics: List[TopicMetrics]) -> Dict[str, float]:
        """Optimize content allocation across topics using Modern Portfolio Theory."""
        if len(metrics) < 2:
            return {m.topic: 1.0 / len(metrics) for m in metrics} if metrics else {}

        n = len(metrics)

        # Expected returns (trend score + sentiment trend)
        returns = np.array([m.trend_score + 0.3 * m.sentiment_trend for m in metrics])

        # Risk (volatility)
        risks = np.array([m.volatility for m in metrics])

        # Simple correlation assumption (topics are partially correlated)
        correlation = 0.3

        # Covariance matrix
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i][j] = risks[i] ** 2
                else:
                    cov_matrix[i][j] = correlation * risks[i] * risks[j]

        # Optimization: maximize Sharpe-like ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / max(port_vol, 0.001)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        bounds = [(0.05, 0.5) for _ in range(n)]  # Min 5%, max 50% per topic

        # Initial guess
        x0 = np.array([1/n] * n)

        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
        else:
            # Fallback to equal weights
            weights = np.array([1/n] * n)

        return {
            metrics[i].topic: round(float(weights[i]), 3)
            for i in range(n)
        }


# Usage
async def main():
    forecaster = PortfolioForecaster(api_key="YOUR_API_KEY")

    topics = [
        "artificial_intelligence",
        "cryptocurrency",
        "climate_change",
        "electric_vehicles",
        "space_exploration",
        "cybersecurity",
        "quantum_computing"
    ]

    portfolio = await forecaster.analyze_portfolio(topics)
    print(json.dumps(portfolio, indent=2))

asyncio.run(main())
```

## Trend Breakout Detection System

Detect early signals of trend breakouts before they go mainstream.

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import json

@dataclass
class BreakoutSignal:
    topic: str
    signal_type: str  # 'emerging', 'accelerating', 'viral_potential', 'mainstream_transition'
    strength: float  # 0-1
    trigger_metrics: Dict
    detected_at: datetime
    confidence: float

class TrendBreakoutDetector:
    """Detect early trend breakouts using multiple signal analysis."""

    # Breakout signal definitions
    SIGNALS = {
        'authority_adoption': {
            'description': 'High-authority sources starting to cover',
            'weight': 0.25
        },
        'velocity_acceleration': {
            'description': 'Publication rate accelerating',
            'weight': 0.20
        },
        'geographic_expansion': {
            'description': 'Spreading to new regions/languages',
            'weight': 0.20
        },
        'sentiment_shift': {
            'description': 'Sentiment becoming more polarized',
            'weight': 0.15
        },
        'cross_topic_spillover': {
            'description': 'Appearing in related topic coverage',
            'weight': 0.20
        }
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apitube.io/v1/news/everything"
        self.baseline_cache = {}
        self.signal_history = deque(maxlen=1000)

    def fetch_current_metrics(self, topic: str, hours: int = 6) -> Dict:
        """Fetch current metrics for a topic."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        params = {
            "api_key": self.api_key,
            "topic.id": topic,
            "published_at.start": start_time.isoformat() + "Z",
            "published_at.end": end_time.isoformat() + "Z",
            "per_page": 100
        }

        response = requests.get(self.base_url, params=params)
        articles = response.json().get("results", [])

        if not articles:
            return {'volume': 0, 'high_authority': 0, 'languages': set(), 'countries': set()}

        # Extract metrics
        high_authority = sum(1 for a in articles
                           if a.get('source', {}).get('rankings', {}).get('opr', 0) >= 5)

        languages = set(a.get('language', 'unknown') for a in articles)
        countries = set(a.get('source', {}).get('location', {}).get('country_code', 'unknown')
                       for a in articles)

        sentiment_scores = [
            a.get('sentiment', {}).get('overall', {}).get('score', 0)
            for a in articles
        ]

        return {
            'volume': len(articles),
            'high_authority': high_authority,
            'high_authority_ratio': high_authority / max(len(articles), 1),
            'languages': languages,
            'language_count': len(languages),
            'countries': countries,
            'country_count': len(countries),
            'sentiment_mean': np.mean(sentiment_scores) if sentiment_scores else 0,
            'sentiment_std': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0,
            'articles': articles[:20]  # Keep sample
        }

    def fetch_baseline(self, topic: str, days: int = 7) -> Dict:
        """Fetch baseline metrics for comparison."""
        cache_key = f"{topic}_{days}"

        if cache_key in self.baseline_cache:
            cached = self.baseline_cache[cache_key]
            if (datetime.utcnow() - cached['fetched_at']).total_seconds() < 3600:
                return cached['data']

        daily_volumes = []
        daily_authority_ratios = []
        all_languages = set()
        all_countries = set()

        for d in range(1, days + 1):
            end_time = datetime.utcnow() - timedelta(days=d)
            start_time = end_time - timedelta(days=1)

            params = {
                "api_key": self.api_key,
                "topic.id": topic,
                "published_at.start": start_time.isoformat() + "Z",
                "published_at.end": end_time.isoformat() + "Z",
                "per_page": 100
            }

            response = requests.get(self.base_url, params=params)
            articles = response.json().get("results", [])

            if articles:
                daily_volumes.append(len(articles))
                high_auth = sum(1 for a in articles
                               if a.get('source', {}).get('rankings', {}).get('opr', 0) >= 5)
                daily_authority_ratios.append(high_auth / len(articles))

                for a in articles:
                    all_languages.add(a.get('language', 'unknown'))
                    all_countries.add(a.get('source', {}).get('location', {}).get('country_code', 'unknown'))

        baseline = {
            'volume_mean': np.mean(daily_volumes) / 4 if daily_volumes else 10,  # Per 6 hours
            'volume_std': np.std(daily_volumes) / 4 if daily_volumes else 5,
            'authority_ratio_mean': np.mean(daily_authority_ratios) if daily_authority_ratios else 0.1,
            'known_languages': all_languages,
            'known_countries': all_countries,
            'language_count_baseline': len(all_languages),
            'country_count_baseline': len(all_countries)
        }

        self.baseline_cache[cache_key] = {
            'data': baseline,
            'fetched_at': datetime.utcnow()
        }

        return baseline

    def calculate_authority_adoption_signal(self, current: Dict, baseline: Dict) -> float:
        """Detect when high-authority sources start covering more."""
        current_ratio = current.get('high_authority_ratio', 0)
        baseline_ratio = baseline.get('authority_ratio_mean', 0.1)

        if baseline_ratio == 0:
            return min(1.0, current_ratio * 2)

        ratio_increase = (current_ratio - baseline_ratio) / baseline_ratio

        # Significant if authority coverage increased by >50%
        return min(1.0, max(0, ratio_increase / 0.5))

    def calculate_velocity_acceleration_signal(self, current: Dict, baseline: Dict) -> float:
        """Detect accelerating publication velocity."""
        current_volume = current.get('volume', 0)
        baseline_mean = baseline.get('volume_mean', 10)
        baseline_std = baseline.get('volume_std', 5)

        if baseline_std == 0:
            baseline_std = 1

        z_score = (current_volume - baseline_mean) / baseline_std

        # Signal strength based on z-score
        if z_score > 3:
            return 1.0
        elif z_score > 2:
            return 0.8
        elif z_score > 1:
            return 0.5
        elif z_score > 0.5:
            return 0.2
        return 0

    def calculate_geographic_expansion_signal(self, current: Dict, baseline: Dict) -> float:
        """Detect expansion to new geographic regions."""
        current_languages = current.get('languages', set())
        current_countries = current.get('countries', set())
        baseline_languages = baseline.get('known_languages', set())
        baseline_countries = baseline.get('known_countries', set())

        new_languages = current_languages - baseline_languages
        new_countries = current_countries - baseline_countries

        # Weight new entries
        language_signal = min(1.0, len(new_languages) / 3)  # 3 new languages = max signal
        country_signal = min(1.0, len(new_countries) / 5)   # 5 new countries = max signal

        return (language_signal + country_signal) / 2

    def calculate_sentiment_shift_signal(self, current: Dict, baseline: Dict) -> float:
        """Detect increased sentiment polarization."""
        current_std = current.get('sentiment_std', 0)

        # Higher standard deviation = more polarized opinions = potential breakout
        # Typical std is around 0.3, breakout often shows >0.5
        if current_std > 0.6:
            return 1.0
        elif current_std > 0.5:
            return 0.7
        elif current_std > 0.4:
            return 0.4
        return 0

    def detect_breakout(self, topic: str) -> Optional[BreakoutSignal]:
        """Run full breakout detection for a topic."""
        current = self.fetch_current_metrics(topic)
        baseline = self.fetch_baseline(topic)

        # Calculate individual signals
        signals = {
            'authority_adoption': self.calculate_authority_adoption_signal(current, baseline),
            'velocity_acceleration': self.calculate_velocity_acceleration_signal(current, baseline),
            'geographic_expansion': self.calculate_geographic_expansion_signal(current, baseline),
            'sentiment_shift': self.calculate_sentiment_shift_signal(current, baseline)
        }

        # Calculate composite score
        composite_score = sum(
            signals[sig] * self.SIGNALS[sig]['weight']
            for sig in signals
        )

        # Determine signal type
        if composite_score < 0.2:
            return None  # No significant signal

        if signals['velocity_acceleration'] > 0.8 and signals['authority_adoption'] > 0.5:
            signal_type = 'viral_potential'
        elif signals['authority_adoption'] > 0.7:
            signal_type = 'mainstream_transition'
        elif signals['velocity_acceleration'] > 0.6:
            signal_type = 'accelerating'
        else:
            signal_type = 'emerging'

        # Calculate confidence
        confidence = min(0.95, 0.4 + composite_score * 0.5)

        breakout = BreakoutSignal(
            topic=topic,
            signal_type=signal_type,
            strength=round(composite_score, 3),
            trigger_metrics={
                'signals': signals,
                'current_volume': current['volume'],
                'baseline_volume': round(baseline['volume_mean'], 1),
                'authority_ratio': round(current.get('high_authority_ratio', 0), 3),
                'new_regions': len(current.get('languages', set()) - baseline.get('known_languages', set()))
            },
            detected_at=datetime.utcnow(),
            confidence=round(confidence, 3)
        )

        self.signal_history.append(breakout)
        return breakout

    def scan_topics(self, topics: List[str]) -> List[BreakoutSignal]:
        """Scan multiple topics for breakout signals."""
        breakouts = []

        for topic in topics:
            signal = self.detect_breakout(topic)
            if signal:
                breakouts.append(signal)

        # Sort by strength
        breakouts.sort(key=lambda x: x.strength, reverse=True)
        return breakouts


# Usage
detector = TrendBreakoutDetector(api_key="YOUR_API_KEY")

# Topics to monitor
topics = [
    "artificial_intelligence",
    "cryptocurrency",
    "quantum_computing",
    "electric_vehicles",
    "space_exploration",
    "biotechnology",
    "renewable_energy",
    "cybersecurity",
    "metaverse",
    "nuclear_fusion"
]

# Scan for breakouts
breakouts = detector.scan_topics(topics)

print("\n🚀 BREAKOUT SIGNALS DETECTED:\n")
for b in breakouts:
    print(f"Topic: {b.topic}")
    print(f"  Signal Type: {b.signal_type}")
    print(f"  Strength: {b.strength:.2%}")
    print(f"  Confidence: {b.confidence:.2%}")
    print(f"  Triggers:")
    for k, v in b.trigger_metrics['signals'].items():
        if v > 0.3:
            print(f"    - {k}: {v:.2%}")
    print()
```

## Cyclical Pattern Detection and Seasonal Forecasting

Identify recurring patterns (weekly, monthly) and generate seasonal forecasts.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal
from scipy.fft import fft
import requests
from typing import Dict, List, Tuple

class SeasonalForecaster:
    """Detect and forecast cyclical/seasonal patterns in news coverage."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apitube.io/v1/news/everything"

    def fetch_long_history(self, topic: str, weeks: int = 8) -> pd.DataFrame:
        """Fetch long-term historical data for pattern detection."""
        data_points = []

        for w in range(weeks):
            for d in range(7):
                for h in [0, 6, 12, 18]:  # 4 data points per day
                    target_time = datetime.utcnow() - timedelta(weeks=w, days=d, hours=h)
                    start_time = target_time - timedelta(hours=6)

                    params = {
                        "api_key": self.api_key,
                        "topic.id": topic,
                        "published_at.start": start_time.isoformat() + "Z",
                        "published_at.end": target_time.isoformat() + "Z",
                        "per_page": 100
                    }

                    try:
                        response = requests.get(self.base_url, params=params)
                        articles = response.json().get("results", [])

                        data_points.append({
                            'timestamp': target_time,
                            'volume': len(articles),
                            'hour': target_time.hour,
                            'day_of_week': target_time.weekday(),
                            'week': w
                        })
                    except:
                        continue

        df = pd.DataFrame(data_points)
        return df.sort_values('timestamp').reset_index(drop=True)

    def detect_weekly_pattern(self, df: pd.DataFrame) -> Dict:
        """Detect weekly (day-of-week) patterns."""
        if len(df) < 28:  # Need at least 4 weeks
            return {'detected': False, 'reason': 'insufficient_data'}

        # Group by day of week
        daily_pattern = df.groupby('day_of_week')['volume'].agg(['mean', 'std']).reset_index()
        daily_pattern.columns = ['day', 'mean', 'std']

        # Check if pattern is significant
        overall_mean = df['volume'].mean()
        pattern_variance = np.var(daily_pattern['mean'])
        noise_variance = np.mean(daily_pattern['std'] ** 2)

        # Signal-to-noise ratio
        snr = pattern_variance / max(noise_variance, 0.001)

        if snr < 0.1:
            return {'detected': False, 'reason': 'no_significant_pattern'}

        # Find peak and trough days
        peak_day = int(daily_pattern.loc[daily_pattern['mean'].idxmax(), 'day'])
        trough_day = int(daily_pattern.loc[daily_pattern['mean'].idxmin(), 'day'])

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        return {
            'detected': True,
            'pattern_type': 'weekly',
            'signal_to_noise': round(snr, 3),
            'peak_day': day_names[peak_day],
            'peak_volume': round(float(daily_pattern.loc[daily_pattern['day'] == peak_day, 'mean'].values[0]), 1),
            'trough_day': day_names[trough_day],
            'trough_volume': round(float(daily_pattern.loc[daily_pattern['day'] == trough_day, 'mean'].values[0]), 1),
            'daily_factors': {
                day_names[int(row['day'])]: round(row['mean'] / overall_mean, 3)
                for _, row in daily_pattern.iterrows()
            }
        }

    def detect_hourly_pattern(self, df: pd.DataFrame) -> Dict:
        """Detect intraday (hourly) patterns."""
        if len(df) < 100:
            return {'detected': False, 'reason': 'insufficient_data'}

        # Group by hour
        hourly_pattern = df.groupby('hour')['volume'].agg(['mean', 'std']).reset_index()
        hourly_pattern.columns = ['hour', 'mean', 'std']

        overall_mean = df['volume'].mean()
        pattern_variance = np.var(hourly_pattern['mean'])
        noise_variance = np.mean(hourly_pattern['std'] ** 2)

        snr = pattern_variance / max(noise_variance, 0.001)

        if snr < 0.05:
            return {'detected': False, 'reason': 'no_significant_pattern'}

        peak_hour = int(hourly_pattern.loc[hourly_pattern['mean'].idxmax(), 'hour'])
        trough_hour = int(hourly_pattern.loc[hourly_pattern['mean'].idxmin(), 'hour'])

        return {
            'detected': True,
            'pattern_type': 'intraday',
            'signal_to_noise': round(snr, 3),
            'peak_hour': f"{peak_hour:02d}:00 UTC",
            'trough_hour': f"{trough_hour:02d}:00 UTC",
            'hourly_factors': {
                f"{int(row['hour']):02d}:00": round(row['mean'] / max(overall_mean, 1), 3)
                for _, row in hourly_pattern.iterrows()
            }
        }

    def spectral_analysis(self, df: pd.DataFrame) -> Dict:
        """Use FFT to detect dominant frequencies/cycles."""
        if len(df) < 56:  # Need sufficient data
            return {'detected': False, 'reason': 'insufficient_data'}

        volumes = df['volume'].values

        # Detrend
        detrended = signal.detrend(volumes)

        # FFT
        n = len(detrended)
        fft_vals = fft(detrended)
        freqs = np.fft.fftfreq(n)

        # Get power spectrum (positive frequencies only)
        positive_freqs = freqs[:n//2]
        power = np.abs(fft_vals[:n//2]) ** 2

        # Find dominant frequencies (excluding DC component)
        power[0] = 0  # Remove DC
        peak_indices = np.argsort(power)[-5:][::-1]

        dominant_cycles = []
        for idx in peak_indices:
            if power[idx] > np.mean(power) * 3 and positive_freqs[idx] > 0:
                period = 1 / positive_freqs[idx]
                if 4 <= period <= n/2:  # Reasonable period range
                    dominant_cycles.append({
                        'period_hours': round(period * 6, 1),  # Convert to hours (6h intervals)
                        'power': round(float(power[idx]), 1),
                        'interpretation': self._interpret_period(period * 6)
                    })

        return {
            'detected': len(dominant_cycles) > 0,
            'dominant_cycles': dominant_cycles[:3],
            'analysis': 'spectral_fft'
        }

    def _interpret_period(self, hours: float) -> str:
        """Interpret the meaning of a cycle period."""
        if 20 <= hours <= 28:
            return "daily_cycle"
        elif 160 <= hours <= 176:
            return "weekly_cycle"
        elif 672 <= hours <= 744:
            return "monthly_cycle"
        elif hours < 12:
            return "intraday_cycle"
        else:
            return f"custom_{int(hours)}h_cycle"

    def generate_seasonal_forecast(self, df: pd.DataFrame,
                                    weekly_pattern: Dict,
                                    hourly_pattern: Dict,
                                    horizon_hours: int = 168) -> List[Dict]:
        """Generate forecast incorporating seasonal patterns."""
        if not weekly_pattern.get('detected') and not hourly_pattern.get('detected'):
            return []

        # Base forecast (simple average)
        base = df['volume'].mean()

        forecasts = []
        current_time = datetime.utcnow()

        for h in range(0, horizon_hours, 6):  # 6-hour intervals
            forecast_time = current_time + timedelta(hours=h)

            # Apply seasonal factors
            factor = 1.0

            if weekly_pattern.get('detected'):
                day_name = forecast_time.strftime('%A')
                day_factors = weekly_pattern.get('daily_factors', {})
                factor *= day_factors.get(day_name, 1.0)

            if hourly_pattern.get('detected'):
                hour_key = f"{forecast_time.hour:02d}:00"
                hour_factors = hourly_pattern.get('hourly_factors', {})
                # Find closest hour
                closest_hour = min(hour_factors.keys(),
                                  key=lambda x: abs(int(x.split(':')[0]) - forecast_time.hour))
                factor *= hour_factors.get(closest_hour, 1.0)

            forecasts.append({
                'timestamp': forecast_time.isoformat(),
                'forecast': round(base * factor, 1),
                'day': forecast_time.strftime('%A'),
                'hour': forecast_time.hour,
                'seasonal_factor': round(factor, 3)
            })

        return forecasts

    def full_seasonal_analysis(self, topic: str) -> Dict:
        """Complete seasonal analysis and forecasting."""
        print(f"Fetching historical data for {topic}...")
        df = self.fetch_long_history(topic, weeks=6)

        if df.empty:
            return {'error': 'no_data'}

        print("Detecting weekly patterns...")
        weekly = self.detect_weekly_pattern(df)

        print("Detecting hourly patterns...")
        hourly = self.detect_hourly_pattern(df)

        print("Running spectral analysis...")
        spectral = self.spectral_analysis(df)

        print("Generating seasonal forecast...")
        forecast = self.generate_seasonal_forecast(df, weekly, hourly)

        return {
            'topic': topic,
            'analyzed_at': datetime.utcnow().isoformat(),
            'data_points': len(df),
            'patterns': {
                'weekly': weekly,
                'hourly': hourly,
                'spectral': spectral
            },
            'forecast': forecast[:28],  # Next 7 days
            'recommendations': self._generate_recommendations(weekly, hourly)
        }

    def _generate_recommendations(self, weekly: Dict, hourly: Dict) -> List[str]:
        """Generate actionable recommendations from patterns."""
        recs = []

        if weekly.get('detected'):
            peak = weekly.get('peak_day')
            trough = weekly.get('trough_day')
            recs.append(f"📅 Best day for content: {peak} (highest coverage)")
            recs.append(f"📉 Lowest coverage day: {trough} (opportunity for differentiation)")

        if hourly.get('detected'):
            peak = hourly.get('peak_hour')
            recs.append(f"⏰ Peak news hour: {peak} (schedule releases accordingly)")

        if not recs:
            recs.append("📊 No strong seasonal patterns detected - coverage is relatively uniform")

        return recs


# Usage
forecaster = SeasonalForecaster(api_key="YOUR_API_KEY")

analysis = forecaster.full_seasonal_analysis("technology")
print(json.dumps(analysis, indent=2))
```
