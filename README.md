# News API Workflows

A collection of workflows and code examples for integrating with the [APITube News API](https://apitube.io) — a real-time news aggregation API providing access to over 500,000 sources across 200 countries and 60 languages.

## About APITube News API

APITube News API is a simple HTTP REST API for searching and retrieving live news articles from all over the web. It offers:

- **Massive coverage** — 500,000+ news sources, 200 countries, 60 languages
- **Rich filtering** — 65+ filters including keywords, date range, language, country, category, source, and more
- **Structured data** — each article includes title, description, content, URL, source, author, image, sentiment, category, and publication date
- **Historical access** — search news archives going back up to 10 years
- **Up to 50 articles per request** with pagination support
- **Sorting** — by publication date or relevance

## Workflows

### [Get Latest News](./get-latest-news/)

Retrieve the most recent news articles with flexible filtering by topic, language, country, and more.

- **Endpoint:** `GET https://api.apitube.io/v1/news/everything`
- **Authentication:** API key via `api_key` query parameter
- **[README](./get-latest-news/README.md)** — overview, parameters, quick start
- **[Code Examples](./get-latest-news/examples.md)** — detailed examples in Python, JavaScript, and PHP

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

for article in response.json()["articles"]:
    print(article["title"])
```

```javascript
const response = await fetch(
  "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&language=en&limit=10"
);
const data = await response.json();
data.articles.forEach((a) => console.log(a.title));
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
