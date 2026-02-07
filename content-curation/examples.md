# Content Curation — Code Examples

Production-quality examples for automated content curation using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python Examples

### 1. Quality-Scored News Feed

Fetch articles and compute a composite quality score based on source authority, word count, duplication status, paywall status, and sentiment diversity. Sort by quality score and return the top 20 articles.

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def calculate_quality_score(article):
    """
    Composite quality score based on multiple factors:
    - Source authority (OPR): 0-40 points
    - Content depth (word count): 0-30 points
    - Freshness (published date): 0-15 points
    - Not duplicate: +10 points
    - Free content: +5 points
    """
    score = 0

    opr = article.get("source", {}).get("rankings", {}).get("opr", 0)
    score += opr * 4

    words = article.get("words_count", 0)
    if words >= 1000:
        score += 30
    elif words >= 600:
        score += 20
    elif words >= 300:
        score += 10

    published = article.get("published_at", "")
    if published:
        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
        hours_old = (datetime.now(pub_date.tzinfo) - pub_date).total_seconds() / 3600
        if hours_old < 6:
            score += 15
        elif hours_old < 24:
            score += 10
        elif hours_old < 72:
            score += 5

    if not article.get("is_duplicate", True):
        score += 10

    if article.get("is_free", False):
        score += 5

    return score

def fetch_quality_scored_feed(topic_id, language="en", limit=100):
    """Fetch and rank articles by quality score"""
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic_id,
        "language": language,
        "is_duplicate": False,
        "source.rankings.opr.min": 3,
        "per_page": limit,
        "published_at.start": (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z",
    })
    response.raise_for_status()

    articles = response.json().get("results", [])

    scored_articles = []
    for article in articles:
        score = calculate_quality_score(article)
        scored_articles.append({
            "article": article,
            "quality_score": score
        })

    scored_articles.sort(key=lambda x: x["quality_score"], reverse=True)

    print(f"Quality-Scored News Feed: {topic_id.title()}\n")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<60} {'Source':<25}")
    print("-" * 100)

    for idx, item in enumerate(scored_articles[:20], 1):
        article = item["article"]
        score = item["quality_score"]
        title = article["title"][:57] + "..." if len(article["title"]) > 60 else article["title"]
        source = article["source"]["domain"][:22] + "..." if len(article["source"]["domain"]) > 25 else article["source"]["domain"]

        print(f"{idx:<6} {score:<8} {title:<60} {source:<25}")
        print(f"       OPR: {article['source']['rankings']['opr']}, "
              f"Words: {article['words_count']}, "
              f"Sentiment: {article.get('sentiment', {}).get('overall', {}).get('polarity', 'N/A')}")
        print(f"       {article['href']}\n")

    return scored_articles[:20]

top_articles = fetch_quality_scored_feed("technology")
```

### 2. Automated Newsletter Generator

Build a daily digest by selecting the top 3 articles per topic across 5 topics, deduplicate, and format as a Markdown newsletter.

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def fetch_top_articles_by_topic(topic_id, count=3):
    """Fetch top articles for a specific topic"""
    yesterday = datetime.utcnow() - timedelta(days=1)

    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic_id,
        "is_duplicate": False,
        "is_free": True,
        "source.rankings.opr.min": 4,
        "published_at.start": yesterday.isoformat() + "Z",
        "per_page": count,
        "sort.by": "published_at",
        "sort.order": "desc"
    })
    response.raise_for_status()

    return response.json().get("results", [])

def generate_newsletter(topics):
    """Generate a Markdown newsletter from multiple topics"""
    newsletter_date = datetime.utcnow().strftime("%B %d, %Y")

    newsletter = f"""# Daily News Digest
## {newsletter_date}

---

"""

    seen_urls = set()
    total_articles = 0

    for topic_id, topic_name in topics.items():
        articles = fetch_top_articles_by_topic(topic_id, count=3)

        unique_articles = []
        for article in articles:
            if article["href"] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article["href"])

        if not unique_articles:
            continue

        newsletter += f"## {topic_name}\n\n"

        for idx, article in enumerate(unique_articles, 1):
            title = article["title"]
            description = article.get("description", "")[:200]
            source = article["source"]["domain"]
            href = article["href"]
            sentiment = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
            words = article.get("words_count", 0)
            read_time = article.get("read_time", 0)

            newsletter += f"### {idx}. {title}\n\n"
            newsletter += f"**Source:** [{source}]({article['source']['home_page_url']})  \n"
            newsletter += f"**Sentiment:** {sentiment.capitalize()} | "
            newsletter += f"**Read time:** {read_time} min ({words} words)  \n\n"
            newsletter += f"{description}...\n\n"
            newsletter += f"[Read full article →]({href})\n\n"
            newsletter += "---\n\n"

            total_articles += 1

    newsletter += f"\n*Newsletter generated with {total_articles} articles from {len(topics)} topics*\n"

    return newsletter

topics = {
    "technology": "Technology",
    "business": "Business",
    "science": "Science",
    "health": "Health",
    "climate_change": "Climate Change"
}

newsletter_content = generate_newsletter(topics)

with open(f"newsletter_{datetime.utcnow().strftime('%Y%m%d')}.md", "w", encoding="utf-8") as f:
    f.write(newsletter_content)

print(newsletter_content)
print(f"\nNewsletter saved to newsletter_{datetime.utcnow().strftime('%Y%m%d')}.md")
```

### 3. Balanced Perspective Curator

For a given topic, fetch positive and negative articles separately, then pair them side-by-side to present balanced viewpoints.

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def fetch_articles_by_sentiment(topic_id, polarity, count=10):
    """Fetch articles filtered by sentiment polarity"""
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic_id,
        "sentiment.overall.polarity": polarity,
        "is_duplicate": False,
        "is_free": True,
        "source.rankings.opr.min": 4,
        "published_at.start": (datetime.utcnow() - timedelta(days=3)).isoformat() + "Z",
        "per_page": count,
        "sort.by": "published_at",
        "sort.order": "desc"
    })
    response.raise_for_status()

    return response.json().get("results", [])

def curate_balanced_perspectives(topic_id, topic_name):
    """Curate balanced perspectives on a topic"""
    print(f"\n{'='*100}")
    print(f"BALANCED PERSPECTIVE: {topic_name}")
    print(f"{'='*100}\n")

    positive_articles = fetch_articles_by_sentiment(topic_id, "positive", count=10)
    negative_articles = fetch_articles_by_sentiment(topic_id, "negative", count=10)

    max_pairs = min(len(positive_articles), len(negative_articles))

    if max_pairs == 0:
        print("Insufficient articles for balanced perspective.")
        return

    print(f"Found {len(positive_articles)} positive and {len(negative_articles)} negative articles.\n")
    print(f"Presenting {max_pairs} balanced pairs:\n")

    for i in range(max_pairs):
        pos_article = positive_articles[i]
        neg_article = negative_articles[i]

        print(f"{'─'*100}")
        print(f"PAIR #{i+1}")
        print(f"{'─'*100}\n")

        print(f"✓ POSITIVE VIEW (Sentiment: {pos_article['sentiment']['overall']['score']:.2f})")
        print(f"  Title: {pos_article['title']}")
        print(f"  Source: {pos_article['source']['domain']} (OPR: {pos_article['source']['rankings']['opr']})")
        print(f"  Published: {pos_article['published_at']}")
        print(f"  Summary: {pos_article.get('description', 'N/A')[:150]}...")
        print(f"  Link: {pos_article['href']}\n")

        print(f"✗ NEGATIVE VIEW (Sentiment: {neg_article['sentiment']['overall']['score']:.2f})")
        print(f"  Title: {neg_article['title']}")
        print(f"  Source: {neg_article['source']['domain']} (OPR: {neg_article['source']['rankings']['opr']})")
        print(f"  Published: {neg_article['published_at']}")
        print(f"  Summary: {neg_article.get('description', 'N/A')[:150]}...")
        print(f"  Link: {neg_article['href']}\n")

    avg_pos_sentiment = sum(a['sentiment']['overall']['score'] for a in positive_articles[:max_pairs]) / max_pairs
    avg_neg_sentiment = sum(a['sentiment']['overall']['score'] for a in negative_articles[:max_pairs]) / max_pairs

    print(f"{'─'*100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'─'*100}")
    print(f"Average Positive Sentiment: {avg_pos_sentiment:.3f}")
    print(f"Average Negative Sentiment: {avg_neg_sentiment:.3f}")
    print(f"Sentiment Spread: {avg_pos_sentiment - avg_neg_sentiment:.3f}")
    print(f"{'='*100}\n")

curate_balanced_perspectives("artificial_intelligence", "Artificial Intelligence")
```

### 4. Executive Briefing Builder

Fetch today's top articles from high-authority sources, group by topic, summarize using the API's summary field, and generate a concise briefing document.

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def fetch_todays_top_articles(min_opr=6, limit=50):
    """Fetch today's top articles from high-authority sources"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "source.rankings.opr.min": min_opr,
        "is_duplicate": False,
        "is_free": True,
        "published_at.start": today_start.isoformat() + "Z",
        "per_page": limit,
        "sort.by": "published_at",
        "sort.order": "desc"
    })
    response.raise_for_status()

    return response.json().get("results", [])

def generate_executive_briefing():
    """Generate an executive briefing document"""
    briefing_date = datetime.utcnow().strftime("%A, %B %d, %Y")

    briefing = f"""
{'='*100}
EXECUTIVE BRIEFING
{briefing_date}
{'='*100}

"""

    articles = fetch_todays_top_articles(min_opr=6, limit=50)

    if not articles:
        briefing += "No high-priority articles found for today.\n"
        return briefing

    topics_map = defaultdict(list)

    for article in articles:
        topics = article.get("topics", [])
        if topics:
            primary_topic = topics[0]["name"]
            topics_map[primary_topic].append(article)

    sorted_topics = sorted(topics_map.items(), key=lambda x: len(x[1]), reverse=True)

    briefing += f"OVERVIEW: {len(articles)} high-priority articles across {len(topics_map)} topics\n\n"
    briefing += f"{'─'*100}\n\n"

    for topic_name, topic_articles in sorted_topics[:8]:
        briefing += f"## {topic_name.upper()} ({len(topic_articles)} articles)\n\n"

        for idx, article in enumerate(topic_articles[:3], 1):
            title = article["title"]
            source = article["source"]["domain"]
            opr = article["source"]["rankings"]["opr"]
            sentiment = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")

            briefing += f"{idx}. **{title}**\n"
            briefing += f"   Source: {source} (Authority: {opr}/10) | Sentiment: {sentiment.capitalize()}\n"

            summary_sentences = article.get("summary", [])
            if summary_sentences:
                key_sentence = summary_sentences[0]["sentence"]
                briefing += f"   Summary: {key_sentence}\n"
            else:
                description = article.get("description", "")[:150]
                briefing += f"   Summary: {description}...\n"

            entities = article.get("entities", [])
            if entities:
                top_entities = [e["name"] for e in entities[:3]]
                briefing += f"   Key Entities: {', '.join(top_entities)}\n"

            briefing += f"   Link: {article['href']}\n\n"

        if len(topic_articles) > 3:
            briefing += f"   ... and {len(topic_articles) - 3} more articles in this topic\n\n"

        briefing += f"{'─'*100}\n\n"

    sentiment_distribution = defaultdict(int)
    for article in articles:
        polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
        sentiment_distribution[polarity] += 1

    briefing += "SENTIMENT ANALYSIS\n"
    briefing += f"Positive: {sentiment_distribution['positive']} articles\n"
    briefing += f"Neutral:  {sentiment_distribution['neutral']} articles\n"
    briefing += f"Negative: {sentiment_distribution['negative']} articles\n\n"

    briefing += f"{'='*100}\n"
    briefing += "End of Executive Briefing\n"
    briefing += f"{'='*100}\n"

    return briefing

briefing_content = generate_executive_briefing()

with open(f"executive_briefing_{datetime.utcnow().strftime('%Y%m%d')}.txt", "w", encoding="utf-8") as f:
    f.write(briefing_content)

print(briefing_content)
print(f"\nBriefing saved to executive_briefing_{datetime.utcnow().strftime('%Y%m%d')}.txt")
```

### 5. Content Calendar Planner

Analyze topic volume trends over 30 days to identify optimal publishing times and underserved topics.

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def fetch_articles_for_date_range(topic_id, start_date, end_date):
    """Fetch articles for a specific date range"""
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic_id,
        "published_at.start": start_date.isoformat() + "Z",
        "published_at.end": end_date.isoformat() + "Z",
        "per_page": 100
    })
    response.raise_for_status()

    return response.json().get("results", [])

def analyze_topic_trends(topics, days=30):
    """Analyze topic volume trends over a specified period"""
    print(f"\n{'='*100}")
    print(f"CONTENT CALENDAR PLANNER - {days}-Day Topic Trend Analysis")
    print(f"{'='*100}\n")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    topic_daily_volumes = defaultdict(lambda: defaultdict(int))
    topic_total_volumes = defaultdict(int)
    topic_hourly_distribution = defaultdict(lambda: defaultdict(int))

    for topic_id, topic_name in topics.items():
        print(f"Analyzing {topic_name}...", end=" ")
        articles = fetch_articles_for_date_range(topic_id, start_date, end_date)
        print(f"{len(articles)} articles found.")

        for article in articles:
            pub_date_str = article.get("published_at", "")
            if pub_date_str:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                day_key = pub_date.strftime("%Y-%m-%d")
                hour_key = pub_date.hour

                topic_daily_volumes[topic_id][day_key] += 1
                topic_hourly_distribution[topic_id][hour_key] += 1
                topic_total_volumes[topic_id] += 1

    print(f"\n{'─'*100}")
    print("TOPIC VOLUME SUMMARY")
    print(f"{'─'*100}\n")

    sorted_topics = sorted(topic_total_volumes.items(), key=lambda x: x[1], reverse=True)

    for topic_id, total_count in sorted_topics:
        topic_name = topics[topic_id]
        avg_per_day = total_count / days
        print(f"{topic_name:<30} Total: {total_count:>5} | Avg/day: {avg_per_day:>6.1f}")

    print(f"\n{'─'*100}")
    print("OPTIMAL PUBLISHING TIMES (by hour of day)")
    print(f"{'─'*100}\n")

    for topic_id, topic_name in topics.items():
        hourly_dist = topic_hourly_distribution[topic_id]
        if not hourly_dist:
            continue

        sorted_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)
        top_hours = sorted_hours[:3]

        print(f"{topic_name}:")
        for hour, count in top_hours:
            print(f"  {hour:02d}:00 - {count} articles ({count/topic_total_volumes[topic_id]*100:.1f}%)")
        print()

    print(f"{'─'*100}")
    print("UNDERSERVED TOPICS (lowest competition)")
    print(f"{'─'*100}\n")

    underserved = sorted_topics[-3:]
    for topic_id, total_count in underserved:
        topic_name = topics[topic_id]
        avg_per_day = total_count / days
        print(f"{topic_name:<30} Only {total_count} articles ({avg_per_day:.1f}/day)")
        print(f"  → Opportunity for content creation\n")

    print(f"{'─'*100}")
    print("TRENDING TOPICS (highest volume)")
    print(f"{'─'*100}\n")

    trending = sorted_topics[:3]
    for topic_id, total_count in trending:
        topic_name = topics[topic_id]
        daily_volumes = topic_daily_volumes[topic_id]

        recent_7days = sum(daily_volumes.get((end_date - timedelta(days=i)).strftime("%Y-%m-%d"), 0) for i in range(7))
        prev_7days = sum(daily_volumes.get((end_date - timedelta(days=i)).strftime("%Y-%m-%d"), 0) for i in range(7, 14))

        if prev_7days > 0:
            growth = ((recent_7days - prev_7days) / prev_7days) * 100
            print(f"{topic_name:<30} {total_count} articles | 7-day growth: {growth:+.1f}%")
        else:
            print(f"{topic_name:<30} {total_count} articles | New trending topic")

    print(f"\n{'='*100}\n")

    return {
        "topic_volumes": topic_total_volumes,
        "daily_volumes": topic_daily_volumes,
        "hourly_distribution": topic_hourly_distribution
    }

topics = {
    "technology": "Technology",
    "artificial_intelligence": "Artificial Intelligence",
    "climate_change": "Climate Change",
    "cryptocurrency": "Cryptocurrency",
    "health": "Health",
    "politics": "Politics",
    "business": "Business",
    "science": "Science"
}

trend_data = analyze_topic_trends(topics, days=30)
```

---

## JavaScript Examples

### 1. Quality-Scored News Feed

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

function calculateQualityScore(article) {
    let score = 0;

    const opr = article.source?.rankings?.opr ?? 0;
    score += opr * 4;

    const words = article.words_count ?? 0;
    if (words >= 1000) score += 30;
    else if (words >= 600) score += 20;
    else if (words >= 300) score += 10;

    const published = article.published_at;
    if (published) {
        const pubDate = new Date(published);
        const hoursOld = (Date.now() - pubDate.getTime()) / (1000 * 60 * 60);
        if (hoursOld < 6) score += 15;
        else if (hoursOld < 24) score += 10;
        else if (hoursOld < 72) score += 5;
    }

    if (!article.is_duplicate) score += 10;
    if (article.is_free) score += 5;

    return score;
}

async function fetchQualityScoredFeed(topicId, language = "en", limit = 100) {
    const twoDaysAgo = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topicId,
        language,
        is_duplicate: "false",
        "source.rankings.opr.min": "3",
        per_page: String(limit),
        "published_at.start": twoDaysAgo
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    const articles = data.results ?? [];

    const scoredArticles = articles.map(article => ({
        article,
        quality_score: calculateQualityScore(article)
    }));

    scoredArticles.sort((a, b) => b.quality_score - a.quality_score);

    console.log(`Quality-Scored News Feed: ${topicId}\n`);
    console.log(`${"Rank".padEnd(6)} ${"Score".padEnd(8)} ${"Title".padEnd(60)} ${"Source".padEnd(25)}`);
    console.log("-".repeat(100));

    scoredArticles.slice(0, 20).forEach((item, idx) => {
        const { article, quality_score } = item;
        const title = article.title.length > 60 ? article.title.slice(0, 57) + "..." : article.title;
        const source = article.source.domain.length > 25 ? article.source.domain.slice(0, 22) + "..." : article.source.domain;

        console.log(`${String(idx + 1).padEnd(6)} ${String(quality_score).padEnd(8)} ${title.padEnd(60)} ${source.padEnd(25)}`);
        console.log(`       OPR: ${article.source.rankings.opr}, Words: ${article.words_count}, Sentiment: ${article.sentiment?.overall?.polarity ?? "N/A"}`);
        console.log(`       ${article.href}\n`);
    });

    return scoredArticles.slice(0, 20);
}

await fetchQualityScoredFeed("technology");
```

### 2. Automated Newsletter Generator

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const fs = require("fs").promises;

async function fetchTopArticlesByTopic(topicId, count = 3) {
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topicId,
        is_duplicate: "false",
        is_free: "true",
        "source.rankings.opr.min": "4",
        "published_at.start": yesterday,
        per_page: String(count),
        "sort.by": "published_at",
        "sort.order": "desc"
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    return data.results ?? [];
}

async function generateNewsletter(topics) {
    const newsletterDate = new Date().toLocaleDateString("en-US", {
        year: "numeric", month: "long", day: "numeric"
    });

    let newsletter = `# Daily News Digest\n## ${newsletterDate}\n\n---\n\n`;

    const seenUrls = new Set();
    let totalArticles = 0;

    for (const [topicId, topicName] of Object.entries(topics)) {
        const articles = await fetchTopArticlesByTopic(topicId, 3);

        const uniqueArticles = articles.filter(article => {
            if (seenUrls.has(article.href)) return false;
            seenUrls.add(article.href);
            return true;
        });

        if (uniqueArticles.length === 0) continue;

        newsletter += `## ${topicName}\n\n`;

        uniqueArticles.forEach((article, idx) => {
            const title = article.title;
            const description = (article.description ?? "").slice(0, 200);
            const source = article.source.domain;
            const href = article.href;
            const sentiment = article.sentiment?.overall?.polarity ?? "neutral";
            const words = article.words_count ?? 0;
            const readTime = article.read_time ?? 0;

            newsletter += `### ${idx + 1}. ${title}\n\n`;
            newsletter += `**Source:** [${source}](${article.source.home_page_url})  \n`;
            newsletter += `**Sentiment:** ${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)} | `;
            newsletter += `**Read time:** ${readTime} min (${words} words)  \n\n`;
            newsletter += `${description}...\n\n`;
            newsletter += `[Read full article →](${href})\n\n`;
            newsletter += "---\n\n";

            totalArticles++;
        });
    }

    newsletter += `\n*Newsletter generated with ${totalArticles} articles from ${Object.keys(topics).length} topics*\n`;

    return newsletter;
}

const topics = {
    technology: "Technology",
    business: "Business",
    science: "Science",
    health: "Health",
    climate_change: "Climate Change"
};

const newsletterContent = await generateNewsletter(topics);

const filename = `newsletter_${new Date().toISOString().split("T")[0].replace(/-/g, "")}.md`;
await fs.writeFile(filename, newsletterContent, "utf-8");

console.log(newsletterContent);
console.log(`\nNewsletter saved to ${filename}`);
```

### 3. Balanced Perspective Curator

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function fetchArticlesBySentiment(topicId, polarity, count = 10) {
    const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topicId,
        "sentiment.overall.polarity": polarity,
        is_duplicate: "false",
        is_free: "true",
        "source.rankings.opr.min": "4",
        "published_at.start": threeDaysAgo,
        per_page: String(count),
        "sort.by": "published_at",
        "sort.order": "desc"
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    return data.results ?? [];
}

async function curateBalancedPerspectives(topicId, topicName) {
    console.log("\n" + "=".repeat(100));
    console.log(`BALANCED PERSPECTIVE: ${topicName}`);
    console.log("=".repeat(100) + "\n");

    const positiveArticles = await fetchArticlesBySentiment(topicId, "positive", 10);
    const negativeArticles = await fetchArticlesBySentiment(topicId, "negative", 10);

    const maxPairs = Math.min(positiveArticles.length, negativeArticles.length);

    if (maxPairs === 0) {
        console.log("Insufficient articles for balanced perspective.");
        return;
    }

    console.log(`Found ${positiveArticles.length} positive and ${negativeArticles.length} negative articles.\n`);
    console.log(`Presenting ${maxPairs} balanced pairs:\n`);

    for (let i = 0; i < maxPairs; i++) {
        const posArticle = positiveArticles[i];
        const negArticle = negativeArticles[i];

        console.log("─".repeat(100));
        console.log(`PAIR #${i + 1}`);
        console.log("─".repeat(100) + "\n");

        console.log(`✓ POSITIVE VIEW (Sentiment: ${posArticle.sentiment.overall.score.toFixed(2)})`);
        console.log(`  Title: ${posArticle.title}`);
        console.log(`  Source: ${posArticle.source.domain} (OPR: ${posArticle.source.rankings.opr})`);
        console.log(`  Published: ${posArticle.published_at}`);
        console.log(`  Summary: ${(posArticle.description ?? "N/A").slice(0, 150)}...`);
        console.log(`  Link: ${posArticle.href}\n`);

        console.log(`✗ NEGATIVE VIEW (Sentiment: ${negArticle.sentiment.overall.score.toFixed(2)})`);
        console.log(`  Title: ${negArticle.title}`);
        console.log(`  Source: ${negArticle.source.domain} (OPR: ${negArticle.source.rankings.opr})`);
        console.log(`  Published: ${negArticle.published_at}`);
        console.log(`  Summary: ${(negArticle.description ?? "N/A").slice(0, 150)}...`);
        console.log(`  Link: ${negArticle.href}\n`);
    }

    const avgPosSentiment = positiveArticles.slice(0, maxPairs).reduce((sum, a) => sum + a.sentiment.overall.score, 0) / maxPairs;
    const avgNegSentiment = negativeArticles.slice(0, maxPairs).reduce((sum, a) => sum + a.sentiment.overall.score, 0) / maxPairs;

    console.log("─".repeat(100));
    console.log("SUMMARY STATISTICS");
    console.log("─".repeat(100));
    console.log(`Average Positive Sentiment: ${avgPosSentiment.toFixed(3)}`);
    console.log(`Average Negative Sentiment: ${avgNegSentiment.toFixed(3)}`);
    console.log(`Sentiment Spread: ${(avgPosSentiment - avgNegSentiment).toFixed(3)}`);
    console.log("=".repeat(100) + "\n");
}

await curateBalancedPerspectives("artificial_intelligence", "Artificial Intelligence");
```

---

## PHP Examples

### 1. Quality-Scored News Feed

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function calculateQualityScore(array $article): int
{
    $score = 0;

    $opr = $article['source']['rankings']['opr'] ?? 0;
    $score += $opr * 4;

    $words = $article['words_count'] ?? 0;
    if ($words >= 1000) {
        $score += 30;
    } elseif ($words >= 600) {
        $score += 20;
    } elseif ($words >= 300) {
        $score += 10;
    }

    $published = $article['published_at'] ?? null;
    if ($published) {
        $pubDate = new DateTimeImmutable($published);
        $now = new DateTimeImmutable('now', new DateTimeZone('UTC'));
        $hoursOld = ($now->getTimestamp() - $pubDate->getTimestamp()) / 3600;

        if ($hoursOld < 6) {
            $score += 15;
        } elseif ($hoursOld < 24) {
            $score += 10;
        } elseif ($hoursOld < 72) {
            $score += 5;
        }
    }

    if (!($article['is_duplicate'] ?? true)) {
        $score += 10;
    }

    if ($article['is_free'] ?? false) {
        $score += 5;
    }

    return $score;
}

function fetchQualityScoredFeed(string $topicId, string $language = "en", int $limit = 100): array
{
    global $apiKey, $baseUrl;

    $twoDaysAgo = (new DateTimeImmutable('now', new DateTimeZone('UTC')))
        ->modify('-2 days')
        ->format('c');

    $query = http_build_query([
        "api_key" => $apiKey,
        "topic.id" => $topicId,
        "language" => $language,
        "is_duplicate" => false,
        "source.rankings.opr.min" => 3,
        "per_page" => $limit,
        "published_at.start" => $twoDaysAgo,
    ]);

    $response = file_get_contents("{$baseUrl}?{$query}");
    $data = json_decode($response, true);
    $articles = $data['results'] ?? [];

    $scoredArticles = [];
    foreach ($articles as $article) {
        $scoredArticles[] = [
            'article' => $article,
            'quality_score' => calculateQualityScore($article),
        ];
    }

    usort($scoredArticles, fn($a, $b) => $b['quality_score'] <=> $a['quality_score']);

    echo "Quality-Scored News Feed: " . ucfirst($topicId) . "\n\n";
    echo str_pad("Rank", 6) . str_pad("Score", 8) . str_pad("Title", 60) . str_pad("Source", 25) . "\n";
    echo str_repeat("-", 100) . "\n";

    foreach (array_slice($scoredArticles, 0, 20) as $idx => $item) {
        $article = $item['article'];
        $score = $item['quality_score'];
        $title = mb_strlen($article['title']) > 60 ? mb_substr($article['title'], 0, 57) . "..." : $article['title'];
        $source = mb_strlen($article['source']['domain']) > 25 ? mb_substr($article['source']['domain'], 0, 22) . "..." : $article['source']['domain'];

        echo str_pad($idx + 1, 6) . str_pad($score, 8) . str_pad($title, 60) . str_pad($source, 25) . "\n";
        echo "       OPR: {$article['source']['rankings']['opr']}, ";
        echo "Words: {$article['words_count']}, ";
        echo "Sentiment: " . ($article['sentiment']['overall']['polarity'] ?? 'N/A') . "\n";
        echo "       {$article['href']}\n\n";
    }

    return array_slice($scoredArticles, 0, 20);
}

$topArticles = fetchQualityScoredFeed("technology");
```

### 2. Automated Newsletter Generator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function fetchTopArticlesByTopic(string $topicId, int $count = 3): array
{
    global $apiKey, $baseUrl;

    $yesterday = (new DateTimeImmutable('now', new DateTimeZone('UTC')))
        ->modify('-1 day')
        ->format('c');

    $query = http_build_query([
        "api_key" => $apiKey,
        "topic.id" => $topicId,
        "is_duplicate" => false,
        "is_free" => true,
        "source.rankings.opr.min" => 4,
        "published_at.start" => $yesterday,
        "per_page" => $count,
        "sort.by" => "published_at",
        "sort.order" => "desc",
    ]);

    $response = file_get_contents("{$baseUrl}?{$query}");
    $data = json_decode($response, true);

    return $data['results'] ?? [];
}

function generateNewsletter(array $topics): string
{
    $newsletterDate = (new DateTimeImmutable('now'))->format('F d, Y');

    $newsletter = "# Daily News Digest\n";
    $newsletter .= "## {$newsletterDate}\n\n";
    $newsletter .= "---\n\n";

    $seenUrls = [];
    $totalArticles = 0;

    foreach ($topics as $topicId => $topicName) {
        $articles = fetchTopArticlesByTopic($topicId, 3);

        $uniqueArticles = [];
        foreach ($articles as $article) {
            if (!in_array($article['href'], $seenUrls, true)) {
                $uniqueArticles[] = $article;
                $seenUrls[] = $article['href'];
            }
        }

        if (empty($uniqueArticles)) {
            continue;
        }

        $newsletter .= "## {$topicName}\n\n";

        foreach ($uniqueArticles as $idx => $article) {
            $title = $article['title'];
            $description = mb_substr($article['description'] ?? '', 0, 200);
            $source = $article['source']['domain'];
            $href = $article['href'];
            $sentiment = $article['sentiment']['overall']['polarity'] ?? 'neutral';
            $words = $article['words_count'] ?? 0;
            $readTime = $article['read_time'] ?? 0;

            $newsletter .= "### " . ($idx + 1) . ". {$title}\n\n";
            $newsletter .= "**Source:** [{$source}]({$article['source']['home_page_url']})  \n";
            $newsletter .= "**Sentiment:** " . ucfirst($sentiment) . " | ";
            $newsletter .= "**Read time:** {$readTime} min ({$words} words)  \n\n";
            $newsletter .= "{$description}...\n\n";
            $newsletter .= "[Read full article →]({$href})\n\n";
            $newsletter .= "---\n\n";

            $totalArticles++;
        }
    }

    $topicCount = count($topics);
    $newsletter .= "\n*Newsletter generated with {$totalArticles} articles from {$topicCount} topics*\n";

    return $newsletter;
}

$topics = [
    'technology' => 'Technology',
    'business' => 'Business',
    'science' => 'Science',
    'health' => 'Health',
    'climate_change' => 'Climate Change',
];

$newsletterContent = generateNewsletter($topics);

$filename = 'newsletter_' . (new DateTimeImmutable('now'))->format('Ymd') . '.md';
file_put_contents($filename, $newsletterContent);

echo $newsletterContent;
echo "\nNewsletter saved to {$filename}\n";
```

### 3. Executive Briefing Builder

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function fetchTodaysTopArticles(int $minOpr = 6, int $limit = 50): array
{
    global $apiKey, $baseUrl;

    $todayStart = (new DateTimeImmutable('now', new DateTimeZone('UTC')))
        ->setTime(0, 0, 0)
        ->format('c');

    $query = http_build_query([
        "api_key" => $apiKey,
        "source.rankings.opr.min" => $minOpr,
        "is_duplicate" => false,
        "is_free" => true,
        "published_at.start" => $todayStart,
        "per_page" => $limit,
        "sort.by" => "published_at",
        "sort.order" => "desc",
    ]);

    $response = file_get_contents("{$baseUrl}?{$query}");
    $data = json_decode($response, true);

    return $data['results'] ?? [];
}

function generateExecutiveBriefing(): string
{
    $briefingDate = (new DateTimeImmutable('now'))->format('l, F d, Y');

    $briefing = "\n" . str_repeat("=", 100) . "\n";
    $briefing .= "EXECUTIVE BRIEFING\n";
    $briefing .= "{$briefingDate}\n";
    $briefing .= str_repeat("=", 100) . "\n\n";

    $articles = fetchTodaysTopArticles(6, 50);

    if (empty($articles)) {
        $briefing .= "No high-priority articles found for today.\n";
        return $briefing;
    }

    $topicsMap = [];

    foreach ($articles as $article) {
        $topics = $article['topics'] ?? [];
        if (!empty($topics)) {
            $primaryTopic = $topics[0]['name'];
            $topicsMap[$primaryTopic][] = $article;
        }
    }

    uasort($topicsMap, fn($a, $b) => count($b) <=> count($a));

    $topicCount = count($topicsMap);
    $articleCount = count($articles);
    $briefing .= "OVERVIEW: {$articleCount} high-priority articles across {$topicCount} topics\n\n";
    $briefing .= str_repeat("─", 100) . "\n\n";

    $topicIdx = 0;
    foreach ($topicsMap as $topicName => $topicArticles) {
        if (++$topicIdx > 8) break;

        $topicArticleCount = count($topicArticles);
        $briefing .= "## " . strtoupper($topicName) . " ({$topicArticleCount} articles)\n\n";

        foreach (array_slice($topicArticles, 0, 3) as $idx => $article) {
            $title = $article['title'];
            $source = $article['source']['domain'];
            $opr = $article['source']['rankings']['opr'];
            $sentiment = $article['sentiment']['overall']['polarity'] ?? 'neutral';

            $briefing .= ($idx + 1) . ". **{$title}**\n";
            $briefing .= "   Source: {$source} (Authority: {$opr}/10) | Sentiment: " . ucfirst($sentiment) . "\n";

            $summarySentences = $article['summary'] ?? [];
            if (!empty($summarySentences)) {
                $keySentence = $summarySentences[0]['sentence'];
                $briefing .= "   Summary: {$keySentence}\n";
            } else {
                $description = mb_substr($article['description'] ?? '', 0, 150);
                $briefing .= "   Summary: {$description}...\n";
            }

            $entities = $article['entities'] ?? [];
            if (!empty($entities)) {
                $topEntities = array_map(fn($e) => $e['name'], array_slice($entities, 0, 3));
                $briefing .= "   Key Entities: " . implode(', ', $topEntities) . "\n";
            }

            $briefing .= "   Link: {$article['href']}\n\n";
        }

        if (count($topicArticles) > 3) {
            $remaining = count($topicArticles) - 3;
            $briefing .= "   ... and {$remaining} more articles in this topic\n\n";
        }

        $briefing .= str_repeat("─", 100) . "\n\n";
    }

    $sentimentDistribution = ['positive' => 0, 'neutral' => 0, 'negative' => 0];
    foreach ($articles as $article) {
        $polarity = $article['sentiment']['overall']['polarity'] ?? 'neutral';
        $sentimentDistribution[$polarity]++;
    }

    $briefing .= "SENTIMENT ANALYSIS\n";
    $briefing .= "Positive: {$sentimentDistribution['positive']} articles\n";
    $briefing .= "Neutral:  {$sentimentDistribution['neutral']} articles\n";
    $briefing .= "Negative: {$sentimentDistribution['negative']} articles\n\n";

    $briefing .= str_repeat("=", 100) . "\n";
    $briefing .= "End of Executive Briefing\n";
    $briefing .= str_repeat("=", 100) . "\n";

    return $briefing;
}

$briefingContent = generateExecutiveBriefing();

$filename = 'executive_briefing_' . (new DateTimeImmutable('now'))->format('Ymd') . '.txt';
file_put_contents($filename, $briefingContent);

echo $briefingContent;
echo "\nBriefing saved to {$filename}\n";
```
