# Adversarial Signal Detection - Advanced Examples

## Real-Time Manipulation Campaign Monitor

Monitor multiple topics for coordinated manipulation attempts in real-time.

```python
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json

@dataclass
class CampaignAlert:
    """Alert for potential manipulation campaign."""
    campaign_id: str
    topics: List[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    signals: List[Dict]
    first_detected: datetime
    last_updated: datetime
    affected_sources: Set[str] = field(default_factory=set)
    status: str = "active"

class RealTimeManipulationMonitor:
    """
    Continuous monitoring for coordinated manipulation campaigns.
    Uses multiple signals to detect inauthentic behavior patterns.
    """

    SEVERITY_THRESHOLDS = {
        "CRITICAL": 0.8,
        "HIGH": 0.6,
        "MEDIUM": 0.4,
        "LOW": 0.2
    }

    def __init__(self, api_key: str, topics: List[str]):
        self.api_key = api_key
        self.topics = topics
        self.base_url = "https://api.apitube.io/v1/news/everything"
        self.active_campaigns: Dict[str, CampaignAlert] = {}
        self.baseline_metrics: Dict[str, Dict] = {}
        self.alert_history = []

    async def fetch_topic_data(self, session: aiohttp.ClientSession,
                                topic: str, hours: int = 6) -> Dict:
        """Fetch recent data for a topic."""
        start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        params = {
            "api_key": self.api_key,
            "title": topic,
            "published_at.start": start,
            "language": "en",
            "per_page": 100,
            "sort.by": "published_at",
            "sort.order": "desc"
        }

        async with session.get(self.base_url, params=params) as response:
            data = await response.json()
            articles = data.get("results", [])

            return {
                "topic": topic,
                "articles": articles,
                "volume": len(articles),
                "timestamp": datetime.utcnow()
            }

    async def update_baselines(self, session: aiohttp.ClientSession):
        """Update baseline metrics for all topics."""
        for topic in self.topics:
            # Fetch 7-day baseline
            daily_volumes = []

            for d in range(1, 8):
                start = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")
                end = (datetime.utcnow() - timedelta(days=d-1)).strftime("%Y-%m-%d")

                params = {
                    "api_key": self.api_key,
                    "title": topic,
                    "published_at.start": start,
                    "published_at.end": end,
                    "language": "en",
                    "per_page": 1
                }

                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    daily_volumes.append(data.get("total_results", 0))

            import numpy as np
            self.baseline_metrics[topic] = {
                "daily_mean": np.mean(daily_volumes),
                "daily_std": np.std(daily_volumes),
                "hourly_mean": np.mean(daily_volumes) / 24,
                "updated_at": datetime.utcnow()
            }

    def calculate_burst_score(self, current_volume: int, topic: str, hours: int) -> float:
        """Calculate burst anomaly score."""
        if topic not in self.baseline_metrics:
            return 0

        baseline = self.baseline_metrics[topic]
        expected = baseline["hourly_mean"] * hours
        std = baseline["daily_std"] * hours / 24

        if std == 0:
            return 0

        z_score = (current_volume - expected) / std

        # Normalize to 0-1
        return min(1.0, max(0, z_score / 5))

    def calculate_source_anomaly_score(self, articles: List[Dict]) -> float:
        """Calculate source distribution anomaly score."""
        if len(articles) < 5:
            return 0

        low_authority = 0
        unknown = 0

        for article in articles:
            opr = article.get("source", {}).get("rankings", {}).get("opr", 0)
            if opr < 0.3:
                low_authority += 1
            elif opr == 0:
                unknown += 1

        suspicious_ratio = (low_authority + unknown) / len(articles)

        return suspicious_ratio

    def calculate_temporal_clustering_score(self, articles: List[Dict]) -> float:
        """Calculate temporal clustering anomaly score."""
        if len(articles) < 10:
            return 0

        timestamps = []
        for article in articles:
            try:
                dt = datetime.fromisoformat(
                    article.get("published_at", "").replace("Z", "+00:00")
                )
                timestamps.append(dt)
            except:
                continue

        if len(timestamps) < 10:
            return 0

        timestamps.sort()

        # Calculate inter-arrival times
        import numpy as np
        deltas = [
            (timestamps[i+1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]

        if not deltas:
            return 0

        # Coefficient of variation - low CV means unnaturally regular timing
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        if mean_delta == 0:
            return 0

        cv = std_delta / mean_delta

        # Very low CV (< 0.5) or very high bursts are suspicious
        if cv < 0.3:  # Too regular
            return 0.8

        # Check for burst clusters
        short_deltas = sum(1 for d in deltas if d < 300)  # < 5 minutes
        burst_ratio = short_deltas / len(deltas)

        return min(1.0, burst_ratio * 1.5)

    def calculate_content_similarity_score(self, articles: List[Dict]) -> float:
        """Calculate content similarity anomaly score."""
        if len(articles) < 5:
            return 0

        import re

        titles = [a.get("title", "").lower() for a in articles if a.get("title")]

        if len(titles) < 5:
            return 0

        # Calculate pairwise similarity
        def word_overlap(t1: str, t2: str) -> float:
            w1 = set(re.findall(r'\w+', t1))
            w2 = set(re.findall(r'\w+', t2))
            if not w1 or not w2:
                return 0
            return len(w1 & w2) / len(w1 | w2)

        high_sim_count = 0
        total_pairs = 0

        for i in range(len(titles)):
            for j in range(i + 1, min(i + 10, len(titles))):  # Check nearby pairs
                sim = word_overlap(titles[i], titles[j])
                total_pairs += 1
                if sim > 0.6:
                    high_sim_count += 1

        if total_pairs == 0:
            return 0

        return min(1.0, high_sim_count / total_pairs * 2)

    def aggregate_signals(self, topic: str, data: Dict) -> Dict:
        """Aggregate all signal scores for a topic."""
        articles = data.get("articles", [])
        volume = data.get("volume", 0)

        signals = {
            "burst": self.calculate_burst_score(volume, topic, hours=6),
            "source_anomaly": self.calculate_source_anomaly_score(articles),
            "temporal_clustering": self.calculate_temporal_clustering_score(articles),
            "content_similarity": self.calculate_content_similarity_score(articles)
        }

        # Weighted aggregate
        weights = {
            "burst": 0.25,
            "source_anomaly": 0.30,
            "temporal_clustering": 0.25,
            "content_similarity": 0.20
        }

        aggregate_score = sum(
            signals[k] * weights[k] for k in signals
        )

        # Determine severity
        severity = "NONE"
        for level, threshold in self.SEVERITY_THRESHOLDS.items():
            if aggregate_score >= threshold:
                severity = level
                break

        return {
            "topic": topic,
            "aggregate_score": round(aggregate_score, 3),
            "severity": severity,
            "signals": {k: round(v, 3) for k, v in signals.items()},
            "volume": volume,
            "timestamp": datetime.utcnow()
        }

    def update_campaign_tracking(self, result: Dict):
        """Update or create campaign tracking based on signals."""
        if result["severity"] == "NONE":
            return

        topic = result["topic"]

        # Create campaign ID based on topic and time window
        campaign_id = hashlib.md5(
            f"{topic}_{datetime.utcnow().strftime('%Y%m%d%H')}".encode()
        ).hexdigest()[:12]

        if campaign_id in self.active_campaigns:
            # Update existing
            campaign = self.active_campaigns[campaign_id]
            campaign.last_updated = datetime.utcnow()
            campaign.signals.append(result)

            # Escalate severity if needed
            if self.SEVERITY_THRESHOLDS.get(result["severity"], 0) > \
               self.SEVERITY_THRESHOLDS.get(campaign.severity, 0):
                campaign.severity = result["severity"]
        else:
            # Create new campaign alert
            self.active_campaigns[campaign_id] = CampaignAlert(
                campaign_id=campaign_id,
                topics=[topic],
                severity=result["severity"],
                signals=[result],
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

    async def monitor_cycle(self) -> List[Dict]:
        """Run one monitoring cycle across all topics."""
        async with aiohttp.ClientSession() as session:
            # Fetch data for all topics concurrently
            tasks = [
                self.fetch_topic_data(session, topic)
                for topic in self.topics
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results = []

        for topic, data in zip(self.topics, results):
            if isinstance(data, Exception):
                continue

            result = self.aggregate_signals(topic, data)
            self.update_campaign_tracking(result)
            cycle_results.append(result)

        return cycle_results

    async def run_continuous_monitoring(self,
                                         interval_minutes: int = 30,
                                         cycles: int = 10):
        """Run continuous monitoring loop."""
        print("Starting continuous manipulation monitoring...")
        print(f"Topics: {self.topics}")
        print(f"Interval: {interval_minutes} minutes")

        # Initialize baselines
        async with aiohttp.ClientSession() as session:
            await self.update_baselines(session)
        print("Baselines initialized")

        for cycle in range(cycles):
            print(f"\n--- Monitoring Cycle {cycle + 1}/{cycles} ---")
            print(f"Time: {datetime.utcnow().isoformat()}")

            results = await self.monitor_cycle()

            # Report findings
            alerts = [r for r in results if r["severity"] != "NONE"]

            if alerts:
                print(f"\nALERTS DETECTED: {len(alerts)}")
                for alert in alerts:
                    print(f"  [{alert['severity']}] {alert['topic']}: "
                          f"score={alert['aggregate_score']:.3f}")
                    for sig, val in alert['signals'].items():
                        if val > 0.3:
                            print(f"    - {sig}: {val:.3f}")
            else:
                print("No significant signals detected")

            # Report active campaigns
            active = [c for c in self.active_campaigns.values()
                     if c.status == "active"]
            if active:
                print(f"\nACTIVE CAMPAIGNS: {len(active)}")
                for camp in active:
                    duration = (datetime.utcnow() - camp.first_detected).total_seconds() / 60
                    print(f"  [{camp.severity}] {camp.campaign_id}: "
                          f"{camp.topics[0]} ({duration:.0f}min)")

            if cycle < cycles - 1:
                await asyncio.sleep(interval_minutes * 60)

        return {
            "monitoring_complete": True,
            "total_cycles": cycles,
            "active_campaigns": len([c for c in self.active_campaigns.values()
                                    if c.status == "active"]),
            "campaigns": [
                {
                    "id": c.campaign_id,
                    "topics": c.topics,
                    "severity": c.severity,
                    "duration_minutes": (c.last_updated - c.first_detected).total_seconds() / 60,
                    "signal_count": len(c.signals)
                }
                for c in self.active_campaigns.values()
            ]
        }


# Usage
async def main():
    monitor = RealTimeManipulationMonitor(
        api_key="YOUR_API_KEY",
        topics=[
            "election fraud",
            "vaccine danger",
            "climate hoax",
            "government conspiracy"
        ]
    )

    results = await monitor.run_continuous_monitoring(
        interval_minutes=15,
        cycles=4
    )

    print("\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print(json.dumps(results, indent=2, default=str))

asyncio.run(main())
```

## Bot Network Detection

Identify potential bot networks through behavioral fingerprinting.

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import hashlib

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class SourceFingerprint:
    """Behavioral fingerprint for a news source."""
    domain: str
    avg_publish_hour: float
    publish_hour_std: float
    avg_articles_per_day: float
    topic_concentration: float  # How focused on specific topics
    authority_score: float
    first_seen: datetime
    article_count: int

class BotNetworkDetector:
    """
    Detect potential bot networks through source behavior analysis.
    Identifies clusters of sources with similar suspicious patterns.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.source_fingerprints: Dict[str, SourceFingerprint] = {}
        self.suspicious_clusters: List[Set[str]] = []

    def fetch_source_articles(self, topic: str, days: int = 7) -> List[Dict]:
        """Fetch articles and group by source."""
        all_articles = []

        for d in range(days):
            start = (datetime.utcnow() - timedelta(days=d+1)).strftime("%Y-%m-%d")
            end = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")

            params = {
                "api_key": self.api_key,
                "title": topic,
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 100
            }

            response = requests.get(BASE_URL, params=params)
            articles = response.json().get("results", [])
            all_articles.extend(articles)

        return all_articles

    def build_source_fingerprints(self, articles: List[Dict]):
        """Build behavioral fingerprints for each source."""
        source_data = defaultdict(list)

        for article in articles:
            source = article.get("source", {})
            domain = source.get("domain")

            if not domain:
                continue

            try:
                pub_time = datetime.fromisoformat(
                    article.get("published_at", "").replace("Z", "+00:00")
                )

                source_data[domain].append({
                    "hour": pub_time.hour,
                    "date": pub_time.date(),
                    "topic": article.get("topic", {}).get("id", "unknown"),
                    "opr": source.get("rankings", {}).get("opr", 0),
                    "published_at": pub_time
                })
            except:
                continue

        for domain, data in source_data.items():
            if len(data) < 3:  # Need minimum data
                continue

            hours = [d["hour"] for d in data]
            dates = set(d["date"] for d in data)
            topics = [d["topic"] for d in data]

            # Calculate metrics
            avg_hour = np.mean(hours)
            hour_std = np.std(hours) if len(hours) > 1 else 0

            articles_per_day = len(data) / max(len(dates), 1)

            # Topic concentration (entropy-based)
            topic_counts = defaultdict(int)
            for t in topics:
                topic_counts[t] += 1

            topic_probs = [c / len(topics) for c in topic_counts.values()]
            topic_entropy = -sum(p * np.log2(p + 1e-10) for p in topic_probs)
            max_entropy = np.log2(len(topic_counts)) if len(topic_counts) > 1 else 1
            topic_concentration = 1 - (topic_entropy / max_entropy) if max_entropy > 0 else 1

            authority = np.mean([d["opr"] for d in data])
            first_seen = min(d["published_at"] for d in data)

            self.source_fingerprints[domain] = SourceFingerprint(
                domain=domain,
                avg_publish_hour=round(avg_hour, 2),
                publish_hour_std=round(hour_std, 2),
                avg_articles_per_day=round(articles_per_day, 2),
                topic_concentration=round(topic_concentration, 3),
                authority_score=round(authority, 3),
                first_seen=first_seen,
                article_count=len(data)
            )

    def calculate_fingerprint_similarity(self, fp1: SourceFingerprint,
                                          fp2: SourceFingerprint) -> float:
        """Calculate similarity between two source fingerprints."""
        # Hour similarity (circular)
        hour_diff = min(
            abs(fp1.avg_publish_hour - fp2.avg_publish_hour),
            24 - abs(fp1.avg_publish_hour - fp2.avg_publish_hour)
        )
        hour_sim = 1 - hour_diff / 12  # Max diff is 12 hours

        # Std similarity
        std_diff = abs(fp1.publish_hour_std - fp2.publish_hour_std)
        std_sim = 1 - min(std_diff / 6, 1)

        # Volume similarity
        vol_ratio = min(fp1.avg_articles_per_day, fp2.avg_articles_per_day) / \
                   max(fp1.avg_articles_per_day, fp2.avg_articles_per_day, 0.1)

        # Topic concentration similarity
        conc_diff = abs(fp1.topic_concentration - fp2.topic_concentration)
        conc_sim = 1 - conc_diff

        # Authority similarity
        auth_diff = abs(fp1.authority_score - fp2.authority_score)
        auth_sim = 1 - auth_diff

        # Weighted combination
        similarity = (
            hour_sim * 0.25 +
            std_sim * 0.15 +
            vol_ratio * 0.20 +
            conc_sim * 0.20 +
            auth_sim * 0.20
        )

        return similarity

    def identify_suspicious_patterns(self, fp: SourceFingerprint) -> List[str]:
        """Identify suspicious patterns in a fingerprint."""
        patterns = []

        # Very regular publishing time (bot-like)
        if fp.publish_hour_std < 2 and fp.article_count > 10:
            patterns.append("highly_regular_timing")

        # High volume from low authority
        if fp.authority_score < 0.3 and fp.avg_articles_per_day > 5:
            patterns.append("high_volume_low_authority")

        # Extreme topic focus
        if fp.topic_concentration > 0.9:
            patterns.append("extreme_topic_focus")

        # Recently appeared with high volume
        days_active = (datetime.utcnow().replace(tzinfo=fp.first_seen.tzinfo) -
                      fp.first_seen).days
        if days_active < 30 and fp.avg_articles_per_day > 3:
            patterns.append("new_source_high_volume")

        return patterns

    def cluster_suspicious_sources(self, similarity_threshold: float = 0.8) -> List[Set[str]]:
        """Cluster sources with similar suspicious patterns."""
        # Filter to suspicious sources
        suspicious = {
            domain: fp for domain, fp in self.source_fingerprints.items()
            if self.identify_suspicious_patterns(fp)
        }

        if len(suspicious) < 2:
            return []

        # Build similarity graph
        domains = list(suspicious.keys())
        clusters = []
        clustered = set()

        for i, d1 in enumerate(domains):
            if d1 in clustered:
                continue

            cluster = {d1}

            for j, d2 in enumerate(domains[i+1:], i+1):
                if d2 in clustered:
                    continue

                sim = self.calculate_fingerprint_similarity(
                    suspicious[d1], suspicious[d2]
                )

                if sim >= similarity_threshold:
                    cluster.add(d2)

            if len(cluster) > 1:
                clusters.append(cluster)
                clustered.update(cluster)

        self.suspicious_clusters = clusters
        return clusters

    def analyze_network(self, topics: List[str]) -> Dict:
        """Run complete bot network analysis."""
        print("Analyzing potential bot networks...")

        # Fetch articles across topics
        all_articles = []
        for topic in topics:
            print(f"  Fetching data for: {topic}")
            articles = self.fetch_source_articles(topic, days=7)
            all_articles.extend(articles)

        print(f"  Total articles: {len(all_articles)}")

        # Build fingerprints
        print("Building source fingerprints...")
        self.build_source_fingerprints(all_articles)
        print(f"  Fingerprinted {len(self.source_fingerprints)} sources")

        # Identify suspicious sources
        suspicious_sources = {}
        for domain, fp in self.source_fingerprints.items():
            patterns = self.identify_suspicious_patterns(fp)
            if patterns:
                suspicious_sources[domain] = {
                    "patterns": patterns,
                    "fingerprint": {
                        "avg_publish_hour": fp.avg_publish_hour,
                        "publish_hour_std": fp.publish_hour_std,
                        "articles_per_day": fp.avg_articles_per_day,
                        "topic_concentration": fp.topic_concentration,
                        "authority_score": fp.authority_score,
                        "article_count": fp.article_count
                    }
                }

        print(f"  Identified {len(suspicious_sources)} suspicious sources")

        # Cluster suspicious sources
        print("Clustering suspicious sources...")
        clusters = self.cluster_suspicious_sources()
        print(f"  Found {len(clusters)} potential bot clusters")

        return {
            "analysis_time": datetime.utcnow().isoformat(),
            "topics_analyzed": topics,
            "total_sources": len(self.source_fingerprints),
            "suspicious_sources": len(suspicious_sources),
            "bot_clusters": len(clusters),
            "suspicious_source_details": suspicious_sources,
            "cluster_details": [
                {
                    "cluster_id": i,
                    "size": len(c),
                    "domains": list(c),
                    "shared_characteristics": self._describe_cluster(c)
                }
                for i, c in enumerate(clusters)
            ]
        }

    def _describe_cluster(self, cluster: Set[str]) -> Dict:
        """Describe shared characteristics of a cluster."""
        fps = [self.source_fingerprints[d] for d in cluster if d in self.source_fingerprints]

        if not fps:
            return {}

        return {
            "avg_authority": round(np.mean([fp.authority_score for fp in fps]), 3),
            "avg_topic_concentration": round(np.mean([fp.topic_concentration for fp in fps]), 3),
            "avg_articles_per_day": round(np.mean([fp.avg_articles_per_day for fp in fps]), 2),
            "publish_hour_range": f"{min(fp.avg_publish_hour for fp in fps):.0f}-{max(fp.avg_publish_hour for fp in fps):.0f}"
        }


# Usage
detector = BotNetworkDetector(api_key="YOUR_API_KEY")

results = detector.analyze_network(
    topics=["vaccine", "election", "climate change"]
)

print("\n" + "=" * 60)
print("BOT NETWORK ANALYSIS RESULTS")
print("=" * 60)
print(f"\nTotal sources analyzed: {results['total_sources']}")
print(f"Suspicious sources: {results['suspicious_sources']}")
print(f"Potential bot clusters: {results['bot_clusters']}")

if results['cluster_details']:
    print("\nCLUSTER DETAILS:")
    for cluster in results['cluster_details']:
        print(f"\n  Cluster {cluster['cluster_id']} ({cluster['size']} sources):")
        print(f"    Domains: {', '.join(cluster['domains'][:5])}...")
        chars = cluster['shared_characteristics']
        print(f"    Avg Authority: {chars.get('avg_authority', 'N/A')}")
        print(f"    Topic Focus: {chars.get('avg_topic_concentration', 'N/A')}")
```

## Cross-Platform Narrative Tracking

Track how narratives spread across different source types and regions.

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

@dataclass
class NarrativeSpread:
    """Tracks spread of a narrative across platforms/regions."""
    narrative: str
    origin_estimate: Dict
    spread_timeline: List[Dict]
    current_reach: Dict
    amplification_score: float
    coordination_indicators: List[str]

class CrossPlatformTracker:
    """
    Track narrative spread across source types, regions, and authority tiers.
    Identifies coordination patterns in cross-platform propagation.
    """

    SOURCE_TIERS = {
        "tier1": {"min_opr": 0.7, "max_opr": 1.0},
        "tier2": {"min_opr": 0.4, "max_opr": 0.7},
        "tier3": {"min_opr": 0.1, "max_opr": 0.4},
        "unknown": {"min_opr": 0, "max_opr": 0.1}
    }

    REGIONS = ["us", "gb", "de", "fr", "ru", "cn", "in", "br"]

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_narrative_data(self, narrative: str, days: int = 3) -> Dict:
        """Fetch comprehensive data about a narrative."""
        data = {
            "by_tier": defaultdict(list),
            "by_region": defaultdict(list),
            "by_language": defaultdict(list),
            "timeline": []
        }

        # Hourly granularity for recent data
        for h in range(days * 24):
            end_time = datetime.utcnow() - timedelta(hours=h)
            start_time = end_time - timedelta(hours=1)

            params = {
                "api_key": self.api_key,
                "title": narrative,
                "published_at.start": start_time.isoformat() + "Z",
                "published_at.end": end_time.isoformat() + "Z",
                "per_page": 100
            }

            response = requests.get(BASE_URL, params=params)
            articles = response.json().get("results", [])

            if articles:
                hour_data = {
                    "hour_offset": h,
                    "timestamp": start_time.isoformat(),
                    "count": len(articles),
                    "tier_breakdown": defaultdict(int),
                    "region_breakdown": defaultdict(int),
                    "language_breakdown": defaultdict(int)
                }

                for article in articles:
                    source = article.get("source", {})
                    opr = source.get("rankings", {}).get("opr", 0)
                    country = source.get("location", {}).get("country_code", "unknown")
                    language = article.get("language", "unknown")

                    # Classify tier
                    tier = "unknown"
                    for t, bounds in self.SOURCE_TIERS.items():
                        if bounds["min_opr"] <= opr < bounds["max_opr"]:
                            tier = t
                            break

                    hour_data["tier_breakdown"][tier] += 1
                    hour_data["region_breakdown"][country.lower()] += 1
                    hour_data["language_breakdown"][language] += 1

                    data["by_tier"][tier].append({
                        "article": article,
                        "hour_offset": h
                    })
                    data["by_region"][country.lower()].append({
                        "article": article,
                        "hour_offset": h
                    })
                    data["by_language"][language].append({
                        "article": article,
                        "hour_offset": h
                    })

                data["timeline"].append(hour_data)

        return data

    def estimate_origin(self, data: Dict) -> Dict:
        """Estimate where the narrative originated."""
        # Find earliest significant coverage
        timeline = sorted(data["timeline"], key=lambda x: -x["hour_offset"])

        earliest_tier = None
        earliest_region = None
        earliest_language = None

        for hour_data in timeline:
            if hour_data["count"] >= 3:  # Minimum threshold
                # Which tier first had coverage?
                tier_counts = hour_data["tier_breakdown"]
                if tier_counts:
                    earliest_tier = max(tier_counts, key=tier_counts.get)

                region_counts = hour_data["region_breakdown"]
                if region_counts:
                    earliest_region = max(region_counts, key=region_counts.get)

                lang_counts = hour_data["language_breakdown"]
                if lang_counts:
                    earliest_language = max(lang_counts, key=lang_counts.get)

                break

        # Check if low-tier sources led (suspicious)
        low_tier_first = False
        for hour_data in timeline[::-1]:  # Oldest first
            if hour_data["count"] >= 2:
                tier3_count = hour_data["tier_breakdown"].get("tier3", 0)
                tier1_count = hour_data["tier_breakdown"].get("tier1", 0)
                if tier3_count > tier1_count:
                    low_tier_first = True
                break

        return {
            "estimated_tier": earliest_tier,
            "estimated_region": earliest_region,
            "estimated_language": earliest_language,
            "low_tier_originated": low_tier_first,
            "hours_ago": timeline[0]["hour_offset"] if timeline else 0
        }

    def calculate_spread_velocity(self, data: Dict) -> Dict:
        """Calculate how fast the narrative spread across dimensions."""
        timeline = data["timeline"]

        if len(timeline) < 2:
            return {"velocity": 0, "acceleration": 0}

        # Sort by time
        timeline = sorted(timeline, key=lambda x: -x["hour_offset"])

        counts = [t["count"] for t in timeline]

        # Velocity: rate of volume increase
        velocities = np.diff(counts)
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0

        # Acceleration: change in velocity
        if len(velocities) > 1:
            accelerations = np.diff(velocities)
            avg_acceleration = np.mean(accelerations)
        else:
            avg_acceleration = 0

        # Regional spread velocity
        regions_over_time = []
        for t in timeline:
            regions_over_time.append(len([r for r, c in t["region_breakdown"].items() if c > 0]))

        region_velocity = np.mean(np.diff(regions_over_time)) if len(regions_over_time) > 1 else 0

        return {
            "volume_velocity": round(avg_velocity, 2),
            "volume_acceleration": round(avg_acceleration, 2),
            "regional_spread_velocity": round(region_velocity, 2),
            "peak_hour": timeline[np.argmax(counts)]["hour_offset"] if counts else 0
        }

    def detect_coordination_indicators(self, data: Dict) -> List[str]:
        """Detect indicators of coordinated spread."""
        indicators = []

        origin = self.estimate_origin(data)
        velocity = self.calculate_spread_velocity(data)

        # Low-tier origin
        if origin["low_tier_originated"]:
            indicators.append("low_authority_origin")

        # Rapid multi-region spread
        if velocity["regional_spread_velocity"] > 0.5:
            indicators.append("rapid_cross_region_spread")

        # Check for synchronized timing across regions
        by_region = data["by_region"]
        if len(by_region) >= 3:
            first_appearances = {}
            for region, articles in by_region.items():
                if articles:
                    first_appearances[region] = min(a["hour_offset"] for a in articles)

            if len(first_appearances) >= 3:
                appearance_times = list(first_appearances.values())
                time_spread = max(appearance_times) - min(appearance_times)
                if time_spread < 6:  # All appeared within 6 hours
                    indicators.append("synchronized_multi_region")

        # Language mismatch (e.g., non-English sources covering English topics)
        by_language = data["by_language"]
        if len(by_language) > 3:
            indicators.append("unusual_language_spread")

        # Volume spike without tier-1 coverage
        timeline = data["timeline"]
        for hour_data in timeline:
            if hour_data["count"] > 20:  # Significant volume
                tier1_ratio = hour_data["tier_breakdown"].get("tier1", 0) / hour_data["count"]
                if tier1_ratio < 0.1:
                    indicators.append("high_volume_no_mainstream")
                    break

        return indicators

    def analyze_narrative(self, narrative: str) -> NarrativeSpread:
        """Run complete narrative spread analysis."""
        print(f"Analyzing narrative: {narrative}")

        # Fetch data
        print("  Fetching cross-platform data...")
        data = self.fetch_narrative_data(narrative, days=3)

        total_articles = sum(t["count"] for t in data["timeline"])
        print(f"  Found {total_articles} articles across {len(data['by_region'])} regions")

        # Estimate origin
        print("  Estimating origin...")
        origin = self.estimate_origin(data)

        # Calculate velocity
        print("  Calculating spread velocity...")
        velocity = self.calculate_spread_velocity(data)

        # Detect coordination
        print("  Checking coordination indicators...")
        coordination = self.detect_coordination_indicators(data)

        # Calculate amplification score
        tier3_count = sum(len(articles) for articles in data["by_tier"].get("tier3", []))
        tier1_count = sum(len(articles) for articles in data["by_tier"].get("tier1", []))

        if tier1_count > 0:
            amplification = tier3_count / tier1_count
        else:
            amplification = tier3_count if tier3_count > 0 else 0

        # Current reach
        current_reach = {
            "total_articles": total_articles,
            "regions": len(data["by_region"]),
            "languages": len(data["by_language"]),
            "tier_distribution": {
                tier: len(articles)
                for tier, articles in data["by_tier"].items()
            }
        }

        return NarrativeSpread(
            narrative=narrative,
            origin_estimate=origin,
            spread_timeline=[
                {
                    "hour": t["hour_offset"],
                    "count": t["count"],
                    "regions": len([r for r, c in t["region_breakdown"].items() if c > 0])
                }
                for t in sorted(data["timeline"], key=lambda x: -x["hour_offset"])[:24]
            ],
            current_reach=current_reach,
            amplification_score=round(amplification, 2),
            coordination_indicators=coordination
        )


# Usage
tracker = CrossPlatformTracker(api_key="YOUR_API_KEY")

narrative = tracker.analyze_narrative("5G health risks")

print("\n" + "=" * 60)
print("NARRATIVE SPREAD ANALYSIS")
print("=" * 60)
print(f"\nNarrative: {narrative.narrative}")
print(f"\nOrigin Estimate:")
print(f"  Tier: {narrative.origin_estimate['estimated_tier']}")
print(f"  Region: {narrative.origin_estimate['estimated_region']}")
print(f"  Low-tier originated: {narrative.origin_estimate['low_tier_originated']}")

print(f"\nCurrent Reach:")
print(f"  Total articles: {narrative.current_reach['total_articles']}")
print(f"  Regions: {narrative.current_reach['regions']}")
print(f"  Languages: {narrative.current_reach['languages']}")

print(f"\nAmplification Score: {narrative.amplification_score}")

if narrative.coordination_indicators:
    print(f"\nCOORDINATION INDICATORS DETECTED:")
    for indicator in narrative.coordination_indicators:
        print(f"  - {indicator}")
else:
    print("\nNo significant coordination indicators detected")
```
