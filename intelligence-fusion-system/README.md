# Intelligence Fusion System

Workflow for multi-source intelligence fusion, cross-referencing validation, source credibility weighting, information cascade detection, misinformation scoring, and consensus building from news streams using the [APITube News API](https://apitube.io).

## Overview

The **Intelligence Fusion System** implements military/intelligence-grade information processing by combining signals from multiple sources, weighting by credibility and track record, detecting information cascades (viral spread vs independent confirmation), identifying potential misinformation through contradiction analysis, and building confidence-weighted consensus assessments. Features Bayesian belief updating, source network analysis, temporal propagation tracking, and automated intelligence reports. Ideal for intelligence analysts, geopolitical researchers, fact-checkers, and strategic decision-makers.

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
import math
import hashlib
import re

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class SourceCredibilityModel:
    """
    Tracks and updates source credibility scores using Bayesian updating.
    """

    def __init__(self):
        # Prior beliefs about source credibility (alpha, beta for Beta distribution)
        self.source_priors = defaultdict(lambda: {"alpha": 2, "beta": 2})  # Neutral prior
        self.source_history = defaultdict(list)
        self.source_metadata = {}

    def get_credibility_score(self, source_domain):
        """Get current credibility score (0-1) for a source."""
        prior = self.source_priors[source_domain]
        # Mean of Beta distribution
        return prior["alpha"] / (prior["alpha"] + prior["beta"])

    def get_credibility_confidence(self, source_domain):
        """Get confidence in credibility estimate."""
        prior = self.source_priors[source_domain]
        n = prior["alpha"] + prior["beta"] - 4  # Subtract initial prior
        return min(1.0, n / 100)  # Confidence increases with observations

    def update_credibility(self, source_domain, was_accurate):
        """Update source credibility based on accuracy observation."""
        prior = self.source_priors[source_domain]

        if was_accurate:
            prior["alpha"] += 1
        else:
            prior["beta"] += 1

        self.source_history[source_domain].append({
            "timestamp": datetime.utcnow().isoformat(),
            "accurate": was_accurate
        })

    def incorporate_external_rating(self, source_domain, opr_score):
        """Incorporate external authority rating (OPR) into prior."""
        # Use OPR to adjust initial prior
        if source_domain not in self.source_metadata:
            # Weight by OPR score
            alpha_boost = opr_score * 5
            self.source_priors[source_domain]["alpha"] += alpha_boost
            self.source_metadata[source_domain] = {"opr": opr_score}

    def get_source_tier(self, source_domain):
        """Classify source into credibility tier."""
        score = self.get_credibility_score(source_domain)
        if score >= 0.8:
            return "tier1_authoritative"
        elif score >= 0.6:
            return "tier2_reliable"
        elif score >= 0.4:
            return "tier3_moderate"
        else:
            return "tier4_questionable"


class ClaimExtractor:
    """
    Extracts and normalizes claims from news articles.
    """

    CLAIM_PATTERNS = {
        "announcement": r"(announced|confirmed|revealed|disclosed|stated)\s+that",
        "allegation": r"(allegedly|reportedly|accused of|claimed|sources say)",
        "prediction": r"(expected to|forecast|predicted|anticipated|projected)",
        "denial": r"(denied|rejected|refuted|dismissed|disputed)",
        "quantitative": r"(\$[\d.]+\s*(billion|million|trillion)|[\d.]+%|\d+\s*(people|employees|units))"
    }

    def __init__(self):
        self.claim_cache = {}

    def extract_claims(self, article):
        """Extract structured claims from article."""
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"

        claims = []

        for claim_type, pattern in self.CLAIM_PATTERNS.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                claims.append({
                    "type": claim_type,
                    "text": title,
                    "confidence": 0.7 if claim_type == "announcement" else 0.5,
                    "entities": [e.get("name") for e in article.get("entities", [])],
                    "source": article.get("source", {}).get("domain"),
                    "timestamp": article.get("published_at")
                })

        # Generate claim fingerprint for deduplication
        fingerprint = self._generate_fingerprint(title, article.get("entities", []))

        return {
            "claims": claims,
            "fingerprint": fingerprint,
            "raw_text": text
        }

    def _generate_fingerprint(self, title, entities):
        """Generate fingerprint for claim deduplication."""
        entity_names = sorted([e.get("name", "") for e in entities])
        key = f"{title[:50]}:{':'.join(entity_names[:3])}"
        return hashlib.md5(key.encode()).hexdigest()[:16]


class InformationCascadeDetector:
    """
    Detects whether information spread is cascade (viral) or independent confirmation.
    """

    def __init__(self):
        self.claim_timeline = defaultdict(list)  # fingerprint -> timeline

    def track_claim(self, claim_fingerprint, source, timestamp, opr_score):
        """Track a claim's appearance."""
        self.claim_timeline[claim_fingerprint].append({
            "source": source,
            "timestamp": timestamp,
            "opr": opr_score
        })

    def analyze_spread_pattern(self, claim_fingerprint):
        """Analyze how a claim spread across sources."""
        timeline = self.claim_timeline.get(claim_fingerprint, [])

        if len(timeline) < 2:
            return {"pattern": "single_source", "confidence": 0.3}

        # Sort by timestamp
        sorted_timeline = sorted(timeline, key=lambda x: x["timestamp"])

        # Analyze spread characteristics
        first_source = sorted_timeline[0]
        sources = [t["source"] for t in sorted_timeline]
        unique_sources = len(set(sources))

        # Time spread
        if len(sorted_timeline) >= 2:
            first_time = datetime.fromisoformat(sorted_timeline[0]["timestamp"].replace("Z", ""))
            last_time = datetime.fromisoformat(sorted_timeline[-1]["timestamp"].replace("Z", ""))
            spread_hours = (last_time - first_time).total_seconds() / 3600
        else:
            spread_hours = 0

        # Source diversity
        opr_scores = [t["opr"] for t in sorted_timeline if t.get("opr")]
        avg_opr = statistics.mean(opr_scores) if opr_scores else 0.5

        # Classification logic
        if unique_sources >= 5 and spread_hours > 24 and avg_opr > 0.6:
            # Multiple independent high-quality sources over time = likely true
            pattern = "independent_confirmation"
            confidence = min(0.95, 0.5 + unique_sources * 0.05 + avg_opr * 0.2)
        elif unique_sources >= 3 and spread_hours < 2:
            # Rapid spread from few sources = potential cascade/viral
            pattern = "viral_cascade"
            confidence = 0.6
        elif unique_sources < 3 and first_source.get("opr", 0) < 0.5:
            # Few low-quality sources = questionable
            pattern = "limited_sourcing"
            confidence = 0.4
        else:
            pattern = "mixed"
            confidence = 0.5 + avg_opr * 0.2

        return {
            "pattern": pattern,
            "confidence": confidence,
            "unique_sources": unique_sources,
            "spread_hours": spread_hours,
            "avg_source_quality": avg_opr,
            "first_source": first_source["source"],
            "timeline_length": len(timeline)
        }


class ContradictionDetector:
    """
    Detects contradictions between claims from different sources.
    """

    CONTRADICTION_SIGNALS = {
        "denial_vs_confirmation": [
            ("confirmed", "denied"),
            ("announced", "refuted"),
            ("will", "won't"),
            ("agreed", "rejected")
        ],
        "quantity_mismatch": r"\$?([\d.]+)\s*(billion|million|%)",
    }

    def __init__(self):
        self.claim_pairs = []

    def find_contradictions(self, claims_by_topic):
        """Find contradicting claims on the same topic."""
        contradictions = []

        for topic, claims in claims_by_topic.items():
            if len(claims) < 2:
                continue

            # Compare all pairs
            for i, claim1 in enumerate(claims):
                for claim2 in claims[i+1:]:
                    contradiction = self._check_contradiction(claim1, claim2)
                    if contradiction:
                        contradictions.append({
                            "topic": topic,
                            "claim1": claim1,
                            "claim2": claim2,
                            "contradiction_type": contradiction["type"],
                            "severity": contradiction["severity"]
                        })

        return contradictions

    def _check_contradiction(self, claim1, claim2):
        """Check if two claims contradict each other."""
        text1 = claim1.get("text", "").lower()
        text2 = claim2.get("text", "").lower()

        # Check denial vs confirmation patterns
        for pos, neg in self.CONTRADICTION_SIGNALS["denial_vs_confirmation"]:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return {"type": "denial_vs_confirmation", "severity": "high"}

        # Check quantity mismatches
        pattern = self.CONTRADICTION_SIGNALS["quantity_mismatch"]
        nums1 = re.findall(pattern, text1)
        nums2 = re.findall(pattern, text2)

        if nums1 and nums2:
            # Compare quantities
            try:
                val1 = float(nums1[0][0])
                val2 = float(nums2[0][0])
                if abs(val1 - val2) / max(val1, val2) > 0.2:  # >20% difference
                    return {"type": "quantity_mismatch", "severity": "medium"}
            except (ValueError, IndexError):
                pass

        return None


class ConsensusBuilder:
    """
    Builds weighted consensus from multiple sources.
    """

    def __init__(self, credibility_model):
        self.credibility_model = credibility_model

    def build_consensus(self, claims, cascade_analysis):
        """Build confidence-weighted consensus assessment."""
        if not claims:
            return {"consensus": None, "confidence": 0}

        # Weight each claim by source credibility
        weighted_scores = []
        source_weights = []

        for claim in claims:
            source = claim.get("source")
            credibility = self.credibility_model.get_credibility_score(source)
            claim_confidence = claim.get("confidence", 0.5)

            # Combine credibility and claim confidence
            weight = credibility * claim_confidence
            weighted_scores.append(weight)
            source_weights.append(credibility)

        # Adjust for cascade pattern
        cascade_multiplier = {
            "independent_confirmation": 1.2,
            "viral_cascade": 0.8,
            "limited_sourcing": 0.6,
            "mixed": 1.0,
            "single_source": 0.5
        }.get(cascade_analysis.get("pattern", "mixed"), 1.0)

        # Calculate consensus confidence
        raw_confidence = statistics.mean(weighted_scores) if weighted_scores else 0
        adjusted_confidence = raw_confidence * cascade_multiplier

        # Determine consensus position
        if adjusted_confidence > 0.7:
            consensus = "high_confidence"
        elif adjusted_confidence > 0.5:
            consensus = "moderate_confidence"
        elif adjusted_confidence > 0.3:
            consensus = "low_confidence"
        else:
            consensus = "insufficient_evidence"

        return {
            "consensus": consensus,
            "confidence": min(1.0, adjusted_confidence),
            "contributing_sources": len(claims),
            "avg_source_credibility": statistics.mean(source_weights) if source_weights else 0,
            "cascade_pattern": cascade_analysis.get("pattern"),
            "cascade_adjustment": cascade_multiplier
        }


class IntelligenceFusionSystem:
    """
    Complete intelligence fusion pipeline combining all components.
    """

    def __init__(self):
        self.credibility_model = SourceCredibilityModel()
        self.claim_extractor = ClaimExtractor()
        self.cascade_detector = InformationCascadeDetector()
        self.contradiction_detector = ContradictionDetector()
        self.consensus_builder = ConsensusBuilder(self.credibility_model)
        self.processed_claims = defaultdict(list)
        self.intelligence_reports = []

    def ingest_article(self, article):
        """Process a single article through the fusion pipeline."""
        source = article.get("source", {})
        source_domain = source.get("domain", "unknown")
        opr = source.get("rank", {}).get("opr", 0.5)

        # Update source credibility model
        self.credibility_model.incorporate_external_rating(source_domain, opr)

        # Extract claims
        extraction = self.claim_extractor.extract_claims(article)

        for claim in extraction["claims"]:
            # Track for cascade detection
            self.cascade_detector.track_claim(
                extraction["fingerprint"],
                source_domain,
                article.get("published_at", datetime.utcnow().isoformat()),
                opr
            )

            # Group by entity/topic
            for entity in claim.get("entities", ["general"]):
                self.processed_claims[entity].append({
                    **claim,
                    "fingerprint": extraction["fingerprint"]
                })

        return extraction

    def fetch_and_process(self, query, days=7, max_articles=200):
        """Fetch articles and process through fusion pipeline."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        articles = []
        page = 1

        while len(articles) < max_articles:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": query,
                "published_at.start": start,
                "language": "en",
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 50,
                "page": page,
            })

            batch = resp.json().get("results", [])
            if not batch:
                break

            articles.extend(batch)
            page += 1

            if len(batch) < 50:
                break

        # Process all articles
        for article in articles[:max_articles]:
            self.ingest_article(article)

        return len(articles)

    def generate_assessment(self, topic):
        """Generate intelligence assessment for a topic."""
        claims = self.processed_claims.get(topic, [])

        if not claims:
            return None

        # Get unique claim fingerprints
        fingerprints = set(c["fingerprint"] for c in claims)

        # Analyze cascade patterns
        cascade_analyses = {}
        for fp in fingerprints:
            cascade_analyses[fp] = self.cascade_detector.analyze_spread_pattern(fp)

        # Find primary cascade pattern
        primary_cascade = max(
            cascade_analyses.values(),
            key=lambda x: x["confidence"]
        ) if cascade_analyses else {"pattern": "unknown", "confidence": 0}

        # Detect contradictions
        contradictions = self.contradiction_detector.find_contradictions({topic: claims})

        # Build consensus
        consensus = self.consensus_builder.build_consensus(claims, primary_cascade)

        # Calculate misinformation risk
        misinfo_risk = self._calculate_misinfo_risk(claims, contradictions, primary_cascade)

        assessment = {
            "topic": topic,
            "generated_at": datetime.utcnow().isoformat(),
            "claim_count": len(claims),
            "unique_claims": len(fingerprints),
            "source_count": len(set(c["source"] for c in claims)),
            "consensus": consensus,
            "cascade_pattern": primary_cascade,
            "contradictions": contradictions,
            "misinformation_risk": misinfo_risk,
            "confidence_level": self._get_confidence_level(consensus["confidence"]),
            "recommendation": self._generate_recommendation(consensus, misinfo_risk)
        }

        self.intelligence_reports.append(assessment)
        return assessment

    def _calculate_misinfo_risk(self, claims, contradictions, cascade):
        """Calculate misinformation risk score."""
        risk_factors = []

        # Factor 1: Source quality
        sources = [c["source"] for c in claims]
        avg_credibility = statistics.mean([
            self.credibility_model.get_credibility_score(s) for s in sources
        ]) if sources else 0.5
        risk_factors.append(1 - avg_credibility)

        # Factor 2: Contradiction presence
        if contradictions:
            risk_factors.append(0.3 * len(contradictions))

        # Factor 3: Cascade pattern
        cascade_risk = {
            "independent_confirmation": 0.1,
            "viral_cascade": 0.4,
            "limited_sourcing": 0.5,
            "single_source": 0.6,
            "mixed": 0.3
        }.get(cascade.get("pattern", "mixed"), 0.3)
        risk_factors.append(cascade_risk)

        # Factor 4: Claim type distribution
        allegation_ratio = sum(1 for c in claims if c.get("type") == "allegation") / max(len(claims), 1)
        risk_factors.append(allegation_ratio * 0.3)

        risk_score = min(1.0, statistics.mean(risk_factors) if risk_factors else 0.5)

        return {
            "score": risk_score,
            "level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
            "factors": {
                "source_quality": 1 - avg_credibility,
                "contradictions": len(contradictions),
                "cascade_risk": cascade_risk,
                "allegation_ratio": allegation_ratio
            }
        }

    def _get_confidence_level(self, confidence):
        """Map confidence score to level."""
        if confidence >= 0.8:
            return "VERY_HIGH"
        elif confidence >= 0.6:
            return "HIGH"
        elif confidence >= 0.4:
            return "MODERATE"
        elif confidence >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"

    def _generate_recommendation(self, consensus, misinfo_risk):
        """Generate actionable recommendation."""
        conf = consensus["confidence"]
        risk = misinfo_risk["score"]

        if conf >= 0.7 and risk < 0.3:
            return "HIGH_CONFIDENCE_ACTIONABLE"
        elif conf >= 0.5 and risk < 0.5:
            return "MODERATE_CONFIDENCE_VERIFY"
        elif risk >= 0.5:
            return "HIGH_RISK_REQUIRES_VERIFICATION"
        else:
            return "INSUFFICIENT_EVIDENCE_MONITOR"

    def generate_briefing(self, topics):
        """Generate multi-topic intelligence briefing."""
        briefing = {
            "generated_at": datetime.utcnow().isoformat(),
            "topics_analyzed": len(topics),
            "assessments": [],
            "high_priority_items": [],
            "contradictions_detected": [],
            "source_quality_summary": {}
        }

        for topic in topics:
            assessment = self.generate_assessment(topic)
            if assessment:
                briefing["assessments"].append(assessment)

                # Flag high priority
                if assessment["consensus"]["confidence"] > 0.7 or assessment["misinformation_risk"]["level"] == "high":
                    briefing["high_priority_items"].append({
                        "topic": topic,
                        "reason": "high_confidence" if assessment["consensus"]["confidence"] > 0.7 else "misinfo_risk",
                        "confidence": assessment["consensus"]["confidence"]
                    })

                # Collect contradictions
                briefing["contradictions_detected"].extend(assessment["contradictions"])

        # Source quality summary
        all_sources = set()
        for claims in self.processed_claims.values():
            for claim in claims:
                all_sources.add(claim.get("source"))

        briefing["source_quality_summary"] = {
            "total_sources": len(all_sources),
            "tier_distribution": defaultdict(int)
        }

        for source in all_sources:
            tier = self.credibility_model.get_source_tier(source)
            briefing["source_quality_summary"]["tier_distribution"][tier] += 1

        return briefing


# Run intelligence fusion
print("INTELLIGENCE FUSION SYSTEM")
print("=" * 70)

fusion = IntelligenceFusionSystem()

# Process articles about a topic
query = "Tesla,Elon Musk"
print(f"\nProcessing intelligence on: {query}")
count = fusion.fetch_and_process(query, days=7, max_articles=100)
print(f"Processed {count} articles")

# Generate assessment
print("\n" + "=" * 70)
print("INTELLIGENCE ASSESSMENT: Tesla")
print("-" * 50)

assessment = fusion.generate_assessment("Tesla")

if assessment:
    print(f"Claims analyzed: {assessment['claim_count']}")
    print(f"Unique claims: {assessment['unique_claims']}")
    print(f"Sources: {assessment['source_count']}")

    print(f"\nCONSENSUS:")
    print(f"  Level: {assessment['consensus']['consensus']}")
    print(f"  Confidence: {assessment['consensus']['confidence']:.1%}")
    print(f"  Contributing sources: {assessment['consensus']['contributing_sources']}")

    print(f"\nINFORMATION SPREAD:")
    print(f"  Pattern: {assessment['cascade_pattern']['pattern']}")
    print(f"  Unique sources: {assessment['cascade_pattern']['unique_sources']}")
    print(f"  Spread hours: {assessment['cascade_pattern'].get('spread_hours', 0):.1f}")

    print(f"\nMISINFORMATION RISK:")
    print(f"  Level: {assessment['misinformation_risk']['level'].upper()}")
    print(f"  Score: {assessment['misinformation_risk']['score']:.2f}")
    print(f"  Factors:")
    for factor, value in assessment['misinformation_risk']['factors'].items():
        print(f"    {factor}: {value:.2f}")

    print(f"\nCONTRADICTIONS: {len(assessment['contradictions'])}")
    for c in assessment['contradictions'][:3]:
        print(f"  [{c['contradiction_type']}] {c['severity']}")

    print(f"\nRECOMMENDATION: {assessment['recommendation']}")

# Generate full briefing
print("\n" + "=" * 70)
print("MULTI-TOPIC BRIEFING")
print("-" * 50)

briefing = fusion.generate_briefing(["Tesla", "Elon Musk", "SpaceX"])

print(f"Topics analyzed: {briefing['topics_analyzed']}")
print(f"High priority items: {len(briefing['high_priority_items'])}")
print(f"Contradictions detected: {len(briefing['contradictions_detected'])}")

print(f"\nSource Quality Distribution:")
for tier, count in briefing['source_quality_summary']['tier_distribution'].items():
    print(f"  {tier}: {count}")

print("\nHigh Priority Items:")
for item in briefing['high_priority_items']:
    print(f"  {item['topic']}: {item['reason']} (conf: {item['confidence']:.1%})")
```

## Components

| Component | Function |
|-----------|----------|
| SourceCredibilityModel | Bayesian credibility tracking with Beta distribution |
| ClaimExtractor | Structured claim extraction with fingerprinting |
| InformationCascadeDetector | Viral vs independent confirmation detection |
| ContradictionDetector | Cross-source contradiction identification |
| ConsensusBuilder | Credibility-weighted consensus assessment |

## Common Use Cases

- **Intelligence analysis** — fuse multi-source information for assessments.
- **Fact-checking** — detect misinformation and contradictions.
- **Due diligence** — verify claims across sources.
- **Crisis monitoring** — assess information quality during events.
- **Strategic planning** — build confidence-weighted intelligence.

## See Also

- [examples.md](./examples.md) — additional code examples.
