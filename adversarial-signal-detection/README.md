# Adversarial Signal Detection

Workflow for detecting coordinated inauthentic behavior, manipulation campaigns, and adversarial information operations in news media using the [APITube News API](https://apitube.io).

## Overview

The **Adversarial Signal Detection** workflow implements advanced detection techniques to identify potential information manipulation, coordinated amplification networks, and synthetic content campaigns. Features include temporal burst analysis, source network clustering, narrative coordination detection, bot-like behavior patterns, and cross-platform propagation tracking. Uses statistical anomaly detection, graph analysis, and behavioral fingerprinting. Ideal for media integrity teams, fact-checkers, platform trust & safety, and researchers studying information warfare.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `title`                       | string  | Filter by keywords in title.                                         |
| `text`                        | string  | Filter by keywords in article text.                                  |
| `organization.name`           | string  | Filter by organization name.                                         |
| `source.domain`               | string  | Filter by source domain.                                             |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `source.rank.opr.max`         | number  | Maximum source authority (0–7).                                     |
| `language.code`               | string  | Filter by language code.                                             |
| `country`                     | string  | Filter by country code.                                              |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import re

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


@dataclass
class AnomalySignal:
    """Represents a detected adversarial signal."""
    signal_type: str
    severity: float  # 0-1
    confidence: float  # 0-1
    evidence: Dict
    detected_at: datetime
    affected_entities: List[str]


class AdversarialDetector:
    """
    Multi-signal adversarial detection engine.
    Identifies coordinated inauthentic behavior and manipulation campaigns.
    """

    # Detection thresholds
    BURST_THRESHOLD = 3.0  # Standard deviations above baseline
    COORDINATION_THRESHOLD = 0.7  # Similarity score for coordination
    LOW_AUTHORITY_THRESHOLD = 2  # OPR below this is suspicious
    TEMPORAL_WINDOW_MINUTES = 30  # Window for burst detection

    def __init__(self):
        self.baseline_cache = {}
        self.detected_signals = []

    def fetch_articles(self, query: str, hours: int = 24,
                       min_authority: float = None, max_authority: float = None) -> List[Dict]:
        """Fetch articles matching query within time window."""
        start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "api_key": API_KEY,
            "title": query,
            "published_at.start": start,
            "language.code.eq": "en",
            "per_page": 100,
            "sort.by": "published_at",
            "sort.order": "asc",
        }

        if min_authority:
            params["source.rank.opr.min"] = min_authority
        if max_authority:
            params["source.rank.opr.max"] = max_authority

        response = requests.get(BASE_URL, params=params)
        return response.json().get("results", [])

    def detect_temporal_burst(self, articles: List[Dict]) -> Optional[AnomalySignal]:
        """
        Detect abnormal temporal clustering of articles.
        Coordinated campaigns often show unnatural publication timing.
        """
        if len(articles) < 10:
            return None

        # Extract timestamps and bucket by 30-minute windows
        timestamps = []
        for article in articles:
            pub_time = article.get("published_at", "")
            if pub_time:
                try:
                    dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                    timestamps.append(dt)
                except:
                    continue

        if len(timestamps) < 10:
            return None

        # Bucket into 30-minute intervals
        timestamps.sort()
        start_time = timestamps[0]
        buckets = defaultdict(int)

        for ts in timestamps:
            bucket = int((ts - start_time).total_seconds() / (self.TEMPORAL_WINDOW_MINUTES * 60))
            buckets[bucket] += 1

        if len(buckets) < 3:
            return None

        # Calculate statistics
        counts = list(buckets.values())
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 1

        # Find burst windows
        bursts = []
        for bucket, count in buckets.items():
            z_score = (count - mean) / max(std, 1)
            if z_score > self.BURST_THRESHOLD:
                bursts.append({
                    "bucket": bucket,
                    "count": count,
                    "z_score": z_score,
                    "time_offset_hours": bucket * 0.5
                })

        if not bursts:
            return None

        max_burst = max(bursts, key=lambda x: x["z_score"])

        return AnomalySignal(
            signal_type="temporal_burst",
            severity=min(1.0, max_burst["z_score"] / 5),
            confidence=min(0.95, 0.5 + len(bursts) * 0.1),
            evidence={
                "burst_count": len(bursts),
                "max_z_score": round(max_burst["z_score"], 2),
                "baseline_mean": round(mean, 1),
                "baseline_std": round(std, 1),
                "bursts": bursts[:5]
            },
            detected_at=datetime.utcnow(),
            affected_entities=[]
        )

    def detect_source_network_anomaly(self, articles: List[Dict]) -> Optional[AnomalySignal]:
        """
        Detect unusual concentration in low-authority sources.
        Manipulation campaigns often use many low-quality outlets.
        """
        if len(articles) < 5:
            return None

        # Categorize sources by authority
        authority_buckets = {"high": [], "medium": [], "low": [], "unknown": []}

        for article in articles:
            source = article.get("source", {})
            opr = source.get("rankings", {}).get("opr", 0)
            domain = source.get("domain", "unknown")

            if opr >= 5:
                authority_buckets["high"].append(domain)
            elif opr >= 3:
                authority_buckets["medium"].append(domain)
            elif opr > 0:
                authority_buckets["low"].append(domain)
            else:
                authority_buckets["unknown"].append(domain)

        total = len(articles)
        low_ratio = len(authority_buckets["low"]) / total
        unknown_ratio = len(authority_buckets["unknown"]) / total
        suspicious_ratio = low_ratio + unknown_ratio

        # Check for unusual low-authority concentration
        if suspicious_ratio < 0.6:
            return None

        # Count unique domains
        low_domains = set(authority_buckets["low"])
        high_domains = set(authority_buckets["high"])

        return AnomalySignal(
            signal_type="source_network_anomaly",
            severity=min(1.0, suspicious_ratio),
            confidence=min(0.9, 0.4 + (suspicious_ratio - 0.6) * 2),
            evidence={
                "low_authority_ratio": round(low_ratio, 3),
                "unknown_ratio": round(unknown_ratio, 3),
                "suspicious_ratio": round(suspicious_ratio, 3),
                "unique_low_domains": len(low_domains),
                "unique_high_domains": len(high_domains),
                "sample_low_domains": list(low_domains)[:10]
            },
            detected_at=datetime.utcnow(),
            affected_entities=[]
        )

    def detect_narrative_coordination(self, articles: List[Dict]) -> Optional[AnomalySignal]:
        """
        Detect suspicious similarity in article titles/content.
        Coordinated campaigns often use templated messaging.
        """
        if len(articles) < 10:
            return None

        # Extract and normalize titles
        titles = []
        for article in articles:
            title = article.get("title", "").lower().strip()
            if len(title) > 20:
                titles.append(title)

        if len(titles) < 10:
            return None

        # Calculate pairwise similarity (using simple word overlap)
        def word_overlap(t1: str, t2: str) -> float:
            words1 = set(re.findall(r'\w+', t1))
            words2 = set(re.findall(r'\w+', t2))
            if not words1 or not words2:
                return 0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union

        high_similarity_pairs = []
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                sim = word_overlap(titles[i], titles[j])
                if sim > self.COORDINATION_THRESHOLD:
                    high_similarity_pairs.append({
                        "title1": titles[i][:80],
                        "title2": titles[j][:80],
                        "similarity": round(sim, 3)
                    })

        if not high_similarity_pairs:
            return None

        # Calculate coordination score
        total_pairs = len(titles) * (len(titles) - 1) / 2
        coordination_ratio = len(high_similarity_pairs) / total_pairs

        return AnomalySignal(
            signal_type="narrative_coordination",
            severity=min(1.0, coordination_ratio * 5),
            confidence=min(0.9, 0.3 + len(high_similarity_pairs) * 0.05),
            evidence={
                "high_similarity_pairs": len(high_similarity_pairs),
                "total_pairs_analyzed": int(total_pairs),
                "coordination_ratio": round(coordination_ratio, 4),
                "sample_pairs": high_similarity_pairs[:5]
            },
            detected_at=datetime.utcnow(),
            affected_entities=[]
        )

    def detect_amplification_pattern(self, entity: str, hours: int = 48) -> Optional[AnomalySignal]:
        """
        Detect artificial amplification patterns.
        Compares coverage from high vs low authority sources.
        """
        # Fetch from low-authority sources
        low_auth = self.fetch_articles(entity, hours, max_authority=2)

        # Fetch from high-authority sources
        high_auth = self.fetch_articles(entity, hours, min_authority=5)

        if len(low_auth) < 5:
            return None

        # Suspicious if low-authority dominates significantly
        low_count = len(low_auth)
        high_count = len(high_auth)
        total = low_count + high_count

        if total < 10:
            return None

        amplification_ratio = low_count / total

        # Check temporal patterns (low-auth articles appearing before high-auth)
        low_times = []
        high_times = []

        for a in low_auth:
            try:
                dt = datetime.fromisoformat(a.get("published_at", "").replace("Z", "+00:00"))
                low_times.append(dt)
            except:
                pass

        for a in high_auth:
            try:
                dt = datetime.fromisoformat(a.get("published_at", "").replace("Z", "+00:00"))
                high_times.append(dt)
            except:
                pass

        # Calculate if low-authority appeared first
        low_first = False
        if low_times and high_times:
            low_first = min(low_times) < min(high_times)

        if amplification_ratio < 0.7:
            return None

        return AnomalySignal(
            signal_type="amplification_pattern",
            severity=min(1.0, amplification_ratio),
            confidence=0.7 if low_first else 0.5,
            evidence={
                "low_authority_count": low_count,
                "high_authority_count": high_count,
                "amplification_ratio": round(amplification_ratio, 3),
                "low_authority_first": low_first,
                "entity": entity
            },
            detected_at=datetime.utcnow(),
            affected_entities=[entity]
        )

    def detect_cross_language_coordination(self, query: str, hours: int = 24) -> Optional[AnomalySignal]:
        """
        Detect coordinated campaigns across multiple languages.
        """
        languages = ["en", "es", "fr", "de", "ru", "zh", "ar"]
        language_counts = {}
        language_articles = {}

        for lang in languages:
            params = {
                "api_key": API_KEY,
                "title": query,
                "published_at.start": (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "language.code.eq": lang,
                "per_page": 50,
            }
            response = requests.get(BASE_URL, params=params)
            articles = response.json().get("results", [])
            if articles:
                language_counts[lang] = len(articles)
                language_articles[lang] = articles

        if len(language_counts) < 3:
            return None

        # Check for unusual multi-language concentration
        active_languages = [l for l, c in language_counts.items() if c >= 5]

        if len(active_languages) < 3:
            return None

        # Calculate temporal alignment across languages
        first_times = {}
        for lang, articles in language_articles.items():
            times = []
            for a in articles:
                try:
                    dt = datetime.fromisoformat(a.get("published_at", "").replace("Z", "+00:00"))
                    times.append(dt)
                except:
                    pass
            if times:
                first_times[lang] = min(times)

        # Check if articles appeared within tight window across languages
        if len(first_times) >= 3:
            time_values = list(first_times.values())
            time_spread = (max(time_values) - min(time_values)).total_seconds() / 3600  # hours
            tight_coordination = time_spread < 6  # Within 6 hours
        else:
            tight_coordination = False

        return AnomalySignal(
            signal_type="cross_language_coordination",
            severity=0.8 if tight_coordination else 0.5,
            confidence=min(0.85, 0.3 + len(active_languages) * 0.15),
            evidence={
                "active_languages": active_languages,
                "language_counts": language_counts,
                "tight_temporal_coordination": tight_coordination,
                "query": query
            },
            detected_at=datetime.utcnow(),
            affected_entities=[]
        )

    def run_full_scan(self, query: str, entity: str = None) -> Dict:
        """
        Run complete adversarial detection scan.
        """
        print(f"Running adversarial scan for: {query}")

        # Fetch articles
        articles = self.fetch_articles(query, hours=48)
        print(f"  Fetched {len(articles)} articles")

        signals = []

        # Run all detectors
        print("  Checking temporal burst patterns...")
        signal = self.detect_temporal_burst(articles)
        if signal:
            signals.append(signal)
            print(f"    DETECTED: {signal.signal_type} (severity: {signal.severity:.2f})")

        print("  Checking source network anomalies...")
        signal = self.detect_source_network_anomaly(articles)
        if signal:
            signals.append(signal)
            print(f"    DETECTED: {signal.signal_type} (severity: {signal.severity:.2f})")

        print("  Checking narrative coordination...")
        signal = self.detect_narrative_coordination(articles)
        if signal:
            signals.append(signal)
            print(f"    DETECTED: {signal.signal_type} (severity: {signal.severity:.2f})")

        if entity:
            print(f"  Checking amplification patterns for {entity}...")
            signal = self.detect_amplification_pattern(entity)
            if signal:
                signals.append(signal)
                print(f"    DETECTED: {signal.signal_type} (severity: {signal.severity:.2f})")

        print("  Checking cross-language coordination...")
        signal = self.detect_cross_language_coordination(query)
        if signal:
            signals.append(signal)
            print(f"    DETECTED: {signal.signal_type} (severity: {signal.severity:.2f})")

        # Calculate aggregate threat score
        if signals:
            threat_score = sum(s.severity * s.confidence for s in signals) / len(signals)
            max_severity = max(s.severity for s in signals)
        else:
            threat_score = 0
            max_severity = 0

        return {
            "query": query,
            "entity": entity,
            "scanned_at": datetime.utcnow().isoformat(),
            "articles_analyzed": len(articles),
            "signals_detected": len(signals),
            "threat_score": round(threat_score, 3),
            "max_severity": round(max_severity, 3),
            "threat_level": self._classify_threat(threat_score),
            "signals": [
                {
                    "type": s.signal_type,
                    "severity": s.severity,
                    "confidence": s.confidence,
                    "evidence": s.evidence
                }
                for s in signals
            ],
            "recommendations": self._generate_recommendations(signals)
        }

    def _classify_threat(self, score: float) -> str:
        """Classify threat level based on aggregate score."""
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        return "NONE"

    def _generate_recommendations(self, signals: List[AnomalySignal]) -> List[str]:
        """Generate recommendations based on detected signals."""
        recs = []

        signal_types = {s.signal_type for s in signals}

        if "temporal_burst" in signal_types:
            recs.append("Investigate publication timing - unusual clustering detected")

        if "source_network_anomaly" in signal_types:
            recs.append("Verify source credibility - high concentration of low-authority outlets")

        if "narrative_coordination" in signal_types:
            recs.append("Check for templated messaging - similar content across multiple sources")

        if "amplification_pattern" in signal_types:
            recs.append("Monitor for artificial amplification - low-authority sources leading coverage")

        if "cross_language_coordination" in signal_types:
            recs.append("Assess cross-border coordination - synchronized multi-language campaign possible")

        if not recs:
            recs.append("No significant adversarial signals detected - continue routine monitoring")

        return recs


# Run detection
print("ADVERSARIAL SIGNAL DETECTION")
print("=" * 70)

detector = AdversarialDetector()

# Example scan
results = detector.run_full_scan(
    query="election fraud",
    entity="election"
)

print("\n" + "=" * 70)
print("SCAN RESULTS")
print("-" * 50)
print(f"Query: {results['query']}")
print(f"Articles Analyzed: {results['articles_analyzed']}")
print(f"Signals Detected: {results['signals_detected']}")
print(f"Threat Score: {results['threat_score']:.3f}")
print(f"Threat Level: {results['threat_level']}")

if results["signals"]:
    print("\nDETECTED SIGNALS:")
    for signal in results["signals"]:
        print(f"\n  {signal['type'].upper()}")
        print(f"    Severity: {signal['severity']:.2f}")
        print(f"    Confidence: {signal['confidence']:.2f}")

print("\nRECOMMENDATIONS:")
for rec in results["recommendations"]:
    print(f"  - {rec}")
```

## Detection Signals

| Signal Type | Description | Indicators |
|-------------|-------------|------------|
| `temporal_burst` | Abnormal clustering of publication times | Z-score > 3 in time buckets |
| `source_network_anomaly` | Unusual concentration in low-authority sources | >60% from low/unknown authority |
| `narrative_coordination` | Templated/similar content across sources | >70% title similarity |
| `amplification_pattern` | Low-authority sources leading coverage | Low-auth first, ratio >70% |
| `cross_language_coordination` | Synchronized multi-language campaigns | 3+ languages, <6h spread |

## Common Use Cases

- **Media integrity monitoring** — detect coordinated disinformation campaigns.
- **Platform trust & safety** — identify inauthentic amplification networks.
- **Fact-checking** — flag potentially manipulated narratives for review.
- **Election security** — monitor for foreign influence operations.
- **Brand protection** — detect coordinated attacks on reputation.
- **Academic research** — study information warfare tactics.

## See Also

- [examples.md](./examples.md) — detailed code examples.
