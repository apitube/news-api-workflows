# News API Workflows

A collection of workflows and code examples for integrating with the [APITube News API](https://apitube.io) — a real-time news aggregation API providing access to over 500,000 sources across 200 countries and 60 languages.

## About APITube News API

APITube News API is a simple HTTP REST API for searching and retrieving live news articles from all over the web. It offers:

- **Massive coverage** — 500,000+ news sources, 200 countries, 60 languages
- **Rich filtering** — 65+ filters including keywords, date range, language, country, category, source, and more
- **Structured data** — each article includes title, description, body, URL, source, author, image, sentiment, categories, and publication date
- **Historical access** — search news archives going back up to 10 years
- **Up to 50 articles per request** with pagination support
- **Sorting** — by publication date or relevance

## Workflows

| Workflow | Description |
|----------|-------------|
| [Get Latest News](./get-latest-news/) | Retrieve the most recent news articles with flexible filtering by topic, language, country, and more. |
| [Entity Tracking](./entity-tracking/) | Track mentions of companies, people, organizations, and other named entities across news sources. |
| [Sentiment Analysis](./sentiment-analysis/) | Filter and analyze news articles by emotional tone — positive, negative, or neutral. |
| [Competitive Intelligence](./competitive-intelligence/) | Monitor competitors, compare brand coverage, and analyze market positioning. |
| [Topic-Based Aggregation](./topic-based-aggregation/) | Aggregate and analyze news by predefined topics, build thematic feeds, and detect trends. |
| [Multi-Source Monitoring](./multi-source-monitoring/) | Monitor and compare news from specific sources, filter by source rank and country. |
| [Export and Analytics](./export-and-analytics/) | Export news data to CSV/JSON/JSONL and build analytics pipelines for reporting. |
| [Real-Time Alerts](./real-time-alerts/) | Build automated alerting pipelines that detect breaking news, sentiment spikes, and anomalies. |
| [Multilingual Analysis](./multilingual-analysis/) | Cross-language news monitoring, comparison, and translation-aware analytics. |
| [Content Curation](./content-curation/) | Automated content curation with quality scoring, deduplication, and newsletter generation. |

Each workflow includes a **README** with parameter reference and quick start examples, plus an **examples.md** with detailed code samples in Python, JavaScript, and PHP.

## Quick Start

```bash
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language=en&limit=10"
```

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "language": "en",
    "limit": 10
})

for article in response.json()["results"]:
    print(article["title"])
```

```javascript
const response = await fetch(
  "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language=en&limit=10"
);
const data = await response.json();
data.results.forEach((a) => console.log(a.title));
```

## Use Cases

- **News aggregators** — collect articles from multiple sources in one feed
- **Monitoring dashboards** — track news about specific topics or companies
- **Chatbots and assistants** — provide users with up-to-date news on demand
- **Research and analytics** — gather large datasets of news articles for analysis
- **Marketing intelligence** — track competitor activity and industry trends

## Links

- [APITube Website](https://apitube.io)
- [API Documentation](https://docs.apitube.io)
- [Postman Collection](https://www.postman.com/apitube/apitube/documentation/405xvjo/apitube-news-api)
- [GitHub](https://github.com/apitube)
