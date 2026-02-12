# Media Bias & Source Reliability Analysis

Advanced workflow for analyzing media bias, comparing source coverage, detecting narrative framing, and evaluating source reliability across different outlets.

## Overview

This workflow implements a comprehensive media analysis system that:

- **Bias Detection**: Identify left/right/center bias in coverage through sentiment and framing analysis
- **Narrative Comparison**: Compare how different sources frame the same story
- **Reliability Scoring**: Evaluate source credibility based on consistency and authority metrics
- **Echo Chamber Detection**: Identify clusters of sources with similar framing
- **Fact-Check Integration**: Cross-reference claims across multiple sources
- **Coverage Gap Analysis**: Find stories covered by some sources but ignored by others

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity.name` | string | Subject to analyze across sources |
| `title` | string | Specific story/headline to track |
| `source.domain` | string | Filter to specific sources |
| `source.rank.opr.min` | number | Minimum source authority (0.0-1.0) |
| `source.rank.opr.max` | number | Maximum source authority |
| `sentiment.overall.polarity` | string | Sentiment filter |
| `language` | string | Language filter |
| `published_at.start` | string | Time range start |
| `published_at.end` | string | Time range end |

## Bias Spectrum Categories

| Category | Characteristics | Detection Signals |
|----------|-----------------|-------------------|
| **Far Left** | Strong progressive framing, activist language | Sentiment < -0.3 on conservative topics |
| **Left** | Progressive perspective, social justice focus | Consistent negative sentiment on business/conservative topics |
| **Center-Left** | Balanced with slight progressive lean | Mixed sentiment, fact-focused |
| **Center** | Neutral, factual reporting | Low sentiment variance, balanced coverage |
| **Center-Right** | Balanced with slight conservative lean | Mixed sentiment, market-focused |
| **Right** | Conservative perspective, traditional values | Consistent negative sentiment on progressive topics |
| **Far Right** | Strong conservative framing, nationalist language | Sentiment < -0.3 on progressive topics |

## Quick Start

### Python

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

@dataclass
class SourceProfile:
    domain: str
    opr_rank: float
    articles_analyzed: int
    sentiment_mean: float
    sentiment_std: float
    bias_score: float  # -1 (left) to 1 (right)
    bias_confidence: float
    reliability_score: float
    common_topics: List[str]
    framing_keywords: List[str]

@dataclass
class StoryComparison:
    story_query: str
    sources_covering: int
    sentiment_range: Tuple[float, float]
    framing_clusters: List[Dict]
    outlier_sources: List[str]
    consensus_narrative: str
    divergent_narratives: List[Dict]

class MediaBiasAnalyzer:
    """Comprehensive media bias and reliability analysis."""

    # Reference sources for bias calibration
    REFERENCE_SOURCES = {
        'left': ['msnbc.com', 'huffpost.com', 'dailykos.com'],
        'center_left': ['nytimes.com', 'washingtonpost.com', 'cnn.com'],
        'center': ['reuters.com', 'apnews.com', 'bbc.com'],
        'center_right': ['wsj.com', 'economist.com', 'forbes.com'],
        'right': ['foxnews.com', 'nypost.com', 'washingtonexaminer.com']
    }

    # Political indicator topics
    POLITICAL_TOPICS = {
        'left_indicators': ['climate_change', 'healthcare', 'immigration', 'gun_control'],
        'right_indicators': ['taxes', 'military', 'border_security', 'deregulation']
    }

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apitube.io/v1/news/everything"
        self.source_profiles: Dict[str, SourceProfile] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def fetch_source_articles(self, domain: str, topic: str = None,
                              days: int = 7, limit: int = 100) -> List[dict]:
        """Fetch articles from a specific source."""
        params = {
            "api_key": self.api_key,
            "source.domain": domain,
            "published_at.start": (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z",
            "published_at.end": datetime.utcnow().isoformat() + "Z",
            "per_page": limit,
            "sort.by": "published_at",
            "sort.order": "desc"
        }

        if topic:
            params["topic.id"] = topic

        response = requests.get(self.base_url, params=params)
        return response.json().get("results", [])

    def fetch_story_coverage(self, query: str, days: int = 3) -> List[dict]:
        """Fetch coverage of a specific story across all sources."""
        params = {
            "api_key": self.api_key,
            "title": query,
            "published_at.start": (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z",
            "published_at.end": datetime.utcnow().isoformat() + "Z",
            "per_page": 100,
            "sort.by": "published_at",
            "sort.order": "desc"
        }

        response = requests.get(self.base_url, params=params)
        return response.json().get("results", [])

    def analyze_source_bias(self, domain: str) -> SourceProfile:
        """Build comprehensive bias profile for a source."""
        # Fetch articles on political indicator topics
        left_articles = []
        right_articles = []

        for topic in self.POLITICAL_TOPICS['left_indicators']:
            articles = self.fetch_source_articles(domain, topic=topic, days=30, limit=50)
            left_articles.extend(articles)

        for topic in self.POLITICAL_TOPICS['right_indicators']:
            articles = self.fetch_source_articles(domain, topic=topic, days=30, limit=50)
            right_articles.extend(articles)

        # Also fetch general articles
        general_articles = self.fetch_source_articles(domain, days=14, limit=100)

        all_articles = left_articles + right_articles + general_articles

        if not all_articles:
            return SourceProfile(
                domain=domain, opr_rank=0, articles_analyzed=0,
                sentiment_mean=0, sentiment_std=0, bias_score=0,
                bias_confidence=0, reliability_score=0,
                common_topics=[], framing_keywords=[]
            )

        # Calculate sentiment metrics
        sentiments = [
            a.get('sentiment', {}).get('overall', {}).get('score', 0)
            for a in all_articles
        ]

        # Calculate bias score from political topic coverage
        left_sentiment = np.mean([
            a.get('sentiment', {}).get('overall', {}).get('score', 0)
            for a in left_articles
        ]) if left_articles else 0

        right_sentiment = np.mean([
            a.get('sentiment', {}).get('overall', {}).get('score', 0)
            for a in right_articles
        ]) if right_articles else 0

        # Bias calculation:
        # If more positive about left topics and negative about right = left bias
        # If more positive about right topics and negative about left = right bias
        bias_score = (right_sentiment - left_sentiment) / 2
        bias_score = max(-1, min(1, bias_score))  # Clamp to [-1, 1]

        # Confidence based on article count
        bias_confidence = min(0.9, len(all_articles) / 200)

        # Get OPR rank from first article
        opr_rank = all_articles[0].get('source', {}).get('rankings', {}).get('opr', 0) if all_articles else 0

        # Extract common topics
        topic_counts = defaultdict(int)
        for a in all_articles:
            for t in a.get('topics', []):
                topic_counts[t.get('id', '')] += 1
        common_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:10]

        # Extract framing keywords using TF-IDF
        texts = [a.get('title', '') + ' ' + a.get('description', '') for a in all_articles]
        if texts:
            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                feature_names = self.vectorizer.get_feature_names_out()
                mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_tfidf)[-20:]
                framing_keywords = [feature_names[i] for i in top_indices]
            except:
                framing_keywords = []
        else:
            framing_keywords = []

        # Calculate reliability score
        reliability_score = self._calculate_reliability(all_articles, opr_rank)

        profile = SourceProfile(
            domain=domain,
            opr_rank=round(opr_rank, 3),
            articles_analyzed=len(all_articles),
            sentiment_mean=round(np.mean(sentiments), 4),
            sentiment_std=round(np.std(sentiments), 4),
            bias_score=round(bias_score, 3),
            bias_confidence=round(bias_confidence, 3),
            reliability_score=round(reliability_score, 3),
            common_topics=common_topics,
            framing_keywords=framing_keywords
        )

        self.source_profiles[domain] = profile
        return profile

    def _calculate_reliability(self, articles: List[dict], opr_rank: float) -> float:
        """Calculate source reliability score."""
        if not articles:
            return 0

        # Factors:
        # 1. OPR rank (authority)
        authority_score = opr_rank

        # 2. Consistency (low sentiment variance = more factual)
        sentiments = [a.get('sentiment', {}).get('overall', {}).get('score', 0) for a in articles]
        consistency_score = 1 - min(1, np.std(sentiments) * 2)

        # 3. Coverage breadth (diverse topics)
        topics = set()
        for a in articles:
            for t in a.get('topics', []):
                topics.add(t.get('id'))
        breadth_score = min(1, len(topics) / 20)

        # 4. Article quality indicators
        avg_word_count = np.mean([a.get('words_count', 0) for a in articles])
        quality_score = min(1, avg_word_count / 800)  # 800 words = good article

        # Weighted combination
        reliability = (
            0.35 * authority_score +
            0.25 * consistency_score +
            0.20 * breadth_score +
            0.20 * quality_score
        )

        return reliability

    def compare_story_coverage(self, story_query: str) -> StoryComparison:
        """Compare how different sources cover the same story."""
        articles = self.fetch_story_coverage(story_query)

        if not articles:
            return StoryComparison(
                story_query=story_query, sources_covering=0,
                sentiment_range=(0, 0), framing_clusters=[],
                outlier_sources=[], consensus_narrative='',
                divergent_narratives=[]
            )

        # Group by source
        by_source = defaultdict(list)
        for a in articles:
            domain = a.get('source', {}).get('domain', 'unknown')
            by_source[domain].append(a)

        # Calculate sentiment per source
        source_sentiments = {}
        for domain, arts in by_source.items():
            sents = [a.get('sentiment', {}).get('overall', {}).get('score', 0) for a in arts]
            source_sentiments[domain] = np.mean(sents)

        sentiment_values = list(source_sentiments.values())
        sentiment_range = (min(sentiment_values), max(sentiment_values)) if sentiment_values else (0, 0)

        # Cluster articles by framing using TF-IDF + DBSCAN
        texts = [a.get('title', '') + ' ' + a.get('description', '') for a in articles]

        if len(texts) >= 3:
            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Cluster
                clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
                distances = 1 - similarity_matrix
                labels = clustering.fit_predict(distances)

                # Group by cluster
                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append({
                        'source': articles[i].get('source', {}).get('domain'),
                        'title': articles[i].get('title'),
                        'sentiment': articles[i].get('sentiment', {}).get('overall', {}).get('score', 0)
                    })

                framing_clusters = [
                    {
                        'cluster_id': k,
                        'sources': list(set(a['source'] for a in v)),
                        'sample_titles': [a['title'] for a in v[:3]],
                        'avg_sentiment': round(np.mean([a['sentiment'] for a in v]), 3)
                    }
                    for k, v in clusters.items() if k != -1
                ]

                # Outliers (cluster -1)
                outlier_sources = list(set(
                    articles[i].get('source', {}).get('domain')
                    for i, label in enumerate(labels) if label == -1
                ))
            except:
                framing_clusters = []
                outlier_sources = []
        else:
            framing_clusters = []
            outlier_sources = []

        # Determine consensus narrative (most common framing)
        if framing_clusters:
            largest_cluster = max(framing_clusters, key=lambda x: len(x['sources']))
            consensus_narrative = largest_cluster['sample_titles'][0] if largest_cluster['sample_titles'] else ''

            divergent_narratives = [
                {
                    'sources': c['sources'],
                    'sample': c['sample_titles'][0] if c['sample_titles'] else '',
                    'sentiment_diff': round(c['avg_sentiment'] - largest_cluster['avg_sentiment'], 3)
                }
                for c in framing_clusters if c['cluster_id'] != largest_cluster['cluster_id']
            ]
        else:
            consensus_narrative = ''
            divergent_narratives = []

        return StoryComparison(
            story_query=story_query,
            sources_covering=len(by_source),
            sentiment_range=sentiment_range,
            framing_clusters=framing_clusters,
            outlier_sources=outlier_sources,
            consensus_narrative=consensus_narrative,
            divergent_narratives=divergent_narratives
        )

    def detect_echo_chambers(self, sources: List[str]) -> Dict:
        """Detect echo chambers - clusters of sources with similar bias/framing."""
        profiles = []

        for domain in sources:
            if domain in self.source_profiles:
                profiles.append(self.source_profiles[domain])
            else:
                profile = self.analyze_source_bias(domain)
                profiles.append(profile)

        if len(profiles) < 3:
            return {'detected': False, 'reason': 'insufficient_sources'}

        # Cluster by bias score and framing keywords
        features = []
        for p in profiles:
            # Feature vector: bias_score + keyword similarity
            features.append([p.bias_score, p.sentiment_mean])

        features = np.array(features)

        # Simple clustering by bias proximity
        from sklearn.cluster import KMeans

        n_clusters = min(3, len(profiles))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        chambers = defaultdict(list)
        for i, label in enumerate(labels):
            chambers[int(label)].append(profiles[i].domain)

        # Characterize each chamber
        chamber_profiles = []
        for chamber_id, domains in chambers.items():
            chamber_biases = [self.source_profiles[d].bias_score for d in domains if d in self.source_profiles]

            avg_bias = np.mean(chamber_biases) if chamber_biases else 0

            if avg_bias < -0.3:
                lean = 'left'
            elif avg_bias > 0.3:
                lean = 'right'
            else:
                lean = 'center'

            chamber_profiles.append({
                'chamber_id': chamber_id,
                'sources': domains,
                'average_bias': round(avg_bias, 3),
                'lean': lean,
                'size': len(domains)
            })

        return {
            'detected': len(chambers) > 1,
            'chambers': chamber_profiles,
            'isolation_score': round(np.std([c['average_bias'] for c in chamber_profiles]), 3)
        }

    def generate_bias_report(self, sources: List[str]) -> Dict:
        """Generate comprehensive bias report for multiple sources."""
        profiles = []

        for domain in sources:
            profile = self.analyze_source_bias(domain)
            profiles.append(profile)

        # Sort by bias
        profiles.sort(key=lambda x: x.bias_score)

        # Categorize
        categories = {
            'far_left': [p for p in profiles if p.bias_score < -0.6],
            'left': [p for p in profiles if -0.6 <= p.bias_score < -0.2],
            'center': [p for p in profiles if -0.2 <= p.bias_score <= 0.2],
            'right': [p for p in profiles if 0.2 < p.bias_score <= 0.6],
            'far_right': [p for p in profiles if p.bias_score > 0.6]
        }

        return {
            'analyzed_at': datetime.utcnow().isoformat(),
            'sources_analyzed': len(profiles),
            'bias_spectrum': {
                cat: [{'domain': p.domain, 'bias': p.bias_score, 'confidence': p.bias_confidence}
                      for p in sources_list]
                for cat, sources_list in categories.items()
            },
            'most_reliable': sorted(profiles, key=lambda x: x.reliability_score, reverse=True)[:5],
            'least_reliable': sorted(profiles, key=lambda x: x.reliability_score)[:5],
            'recommendations': self._generate_recommendations(profiles)
        }

    def _generate_recommendations(self, profiles: List[SourceProfile]) -> List[str]:
        """Generate reading recommendations for balanced news diet."""
        recs = []

        biases = [p.bias_score for p in profiles]
        avg_bias = np.mean(biases)

        if avg_bias < -0.2:
            recs.append("📊 Your source mix leans LEFT - consider adding center-right sources")
            right_sources = [p.domain for p in profiles if p.bias_score > 0.2 and p.reliability_score > 0.5]
            if right_sources:
                recs.append(f"   Suggested additions: {', '.join(right_sources[:3])}")

        elif avg_bias > 0.2:
            recs.append("📊 Your source mix leans RIGHT - consider adding center-left sources")
            left_sources = [p.domain for p in profiles if p.bias_score < -0.2 and p.reliability_score > 0.5]
            if left_sources:
                recs.append(f"   Suggested additions: {', '.join(left_sources[:3])}")

        else:
            recs.append("✅ Your source mix is relatively balanced")

        # Reliability recommendations
        low_reliability = [p for p in profiles if p.reliability_score < 0.4]
        if low_reliability:
            recs.append(f"⚠️ {len(low_reliability)} sources have low reliability scores")
            recs.append(f"   Consider replacing: {', '.join(p.domain for p in low_reliability[:3])}")

        return recs


# Usage
analyzer = MediaBiasAnalyzer(api_key="YOUR_API_KEY")

# Analyze bias of multiple sources
sources = [
    "nytimes.com", "wsj.com", "foxnews.com", "cnn.com",
    "bbc.com", "reuters.com", "breitbart.com", "huffpost.com"
]

report = analyzer.generate_bias_report(sources)
print(json.dumps(report, indent=2, default=str))

# Compare story coverage
comparison = analyzer.compare_story_coverage("election results")
print(json.dumps(comparison.__dict__, indent=2))

# Detect echo chambers
echo = analyzer.detect_echo_chambers(sources)
print(json.dumps(echo, indent=2))
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

class MediaBiasAnalyzer {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://api.apitube.io/v1/news/everything';
        this.sourceProfiles = new Map();
    }

    async fetchSourceArticles(domain, days = 7) {
        const endDate = new Date();
        const startDate = new Date(endDate - days * 24 * 60 * 60 * 1000);

        const response = await axios.get(this.baseUrl, {
            params: {
                api_key: this.apiKey,
                'source.domain': domain,
                'published_at.start': startDate.toISOString(),
                'published_at.end': endDate.toISOString(),
                per_page: 100
            }
        });

        return response.data.results || [];
    }

    async analyzeSourceBias(domain) {
        const articles = await this.fetchSourceArticles(domain, 14);

        if (!articles.length) {
            return {
                domain,
                biasScore: 0,
                reliability: 0,
                confidence: 0,
                error: 'no_articles'
            };
        }

        // Calculate sentiment metrics
        const sentiments = articles.map(a =>
            a.sentiment?.overall?.score || 0
        );

        const sentimentMean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
        const sentimentStd = Math.sqrt(
            sentiments.reduce((sum, s) => sum + (s - sentimentMean) ** 2, 0) / sentiments.length
        );

        // Get OPR rank
        const oprRank = articles[0]?.source?.rankings?.opr || 0;

        // Calculate reliability
        const reliability = (oprRank * 0.4) + ((1 - Math.min(sentimentStd, 1)) * 0.3) + 0.3;

        return {
            domain,
            articlesAnalyzed: articles.length,
            sentimentMean: sentimentMean.toFixed(4),
            sentimentStd: sentimentStd.toFixed(4),
            oprRank: oprRank.toFixed(3),
            reliability: reliability.toFixed(3),
            confidence: Math.min(0.9, articles.length / 200).toFixed(3)
        };
    }

    async compareStoryCoverage(query) {
        const response = await axios.get(this.baseUrl, {
            params: {
                api_key: this.apiKey,
                title: query,
                per_page: 100
            }
        });

        const articles = response.data.results || [];

        // Group by source
        const bySource = {};
        articles.forEach(a => {
            const domain = a.source?.domain || 'unknown';
            if (!bySource[domain]) bySource[domain] = [];
            bySource[domain].push(a);
        });

        // Calculate per-source sentiment
        const sourceSentiments = {};
        for (const [domain, arts] of Object.entries(bySource)) {
            const sents = arts.map(a => a.sentiment?.overall?.score || 0);
            sourceSentiments[domain] = {
                count: arts.length,
                avgSentiment: (sents.reduce((a, b) => a + b, 0) / sents.length).toFixed(3),
                sampleTitle: arts[0]?.title
            };
        }

        const sentimentValues = Object.values(sourceSentiments).map(s => parseFloat(s.avgSentiment));

        return {
            query,
            sourcesCovering: Object.keys(bySource).length,
            totalArticles: articles.length,
            sentimentRange: {
                min: Math.min(...sentimentValues).toFixed(3),
                max: Math.max(...sentimentValues).toFixed(3),
                spread: (Math.max(...sentimentValues) - Math.min(...sentimentValues)).toFixed(3)
            },
            bySource: sourceSentiments
        };
    }

    async generateBiasReport(sources) {
        const profiles = [];

        for (const domain of sources) {
            const profile = await this.analyzeSourceBias(domain);
            profiles.push(profile);
        }

        // Sort by sentiment (as proxy for bias in simplified version)
        profiles.sort((a, b) => parseFloat(a.sentimentMean) - parseFloat(b.sentimentMean));

        return {
            analyzedAt: new Date().toISOString(),
            sourcesAnalyzed: profiles.length,
            profiles,
            mostReliable: [...profiles].sort((a, b) =>
                parseFloat(b.reliability) - parseFloat(a.reliability)
            ).slice(0, 3),
            summary: {
                avgSentiment: (profiles.reduce((sum, p) =>
                    sum + parseFloat(p.sentimentMean), 0) / profiles.length).toFixed(4)
            }
        };
    }
}

// Usage
const analyzer = new MediaBiasAnalyzer('YOUR_API_KEY');

(async () => {
    // Analyze sources
    const report = await analyzer.generateBiasReport([
        'nytimes.com', 'wsj.com', 'bbc.com', 'reuters.com'
    ]);
    console.log(JSON.stringify(report, null, 2));

    // Compare story coverage
    const comparison = await analyzer.compareStoryCoverage('climate summit');
    console.log(JSON.stringify(comparison, null, 2));
})();
```

## Common Use Cases

1. **Media Literacy**: Understand bias in news consumption
2. **Research**: Academic study of media framing
3. **PR Monitoring**: Track how different outlets cover your organization
4. **Fact-Checking**: Cross-reference claims across sources
5. **Content Curation**: Build balanced news feeds

## See Also

- [examples.md](./examples.md) - Advanced analysis patterns
- [Multi-Source Monitoring](../multi-source-monitoring/) - Source tracking basics
- [Sentiment Analysis](../sentiment-analysis/) - Sentiment fundamentals
