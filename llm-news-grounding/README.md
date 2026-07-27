# LLM News Grounding

Workflow for grounding large language models in live news — retrieval, highlighted snippets, citation-ready context, and claim verification using the [APITube News API](https://apitube.io).

## Overview

The **LLM News Grounding** workflow turns the news corpus into a retrieval layer for language models. It retrieves relevant articles for a user question, extracts highlighted snippets that are already trimmed to the matching passage, assembles a citation-ready context block within a token budget, and verifies the model's output against real coverage with the fact-check endpoint. Uses server-side highlighting instead of client-side chunking, so the API returns the passages that matched rather than whole article bodies. Ideal for RAG application developers, AI assistant builders, research tooling teams, and anyone who needs a model to answer with sources instead of memory.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/fact-check
```

## Key Parameters

| Parameter             | Type    | Description                                                                     |
|-----------------------|---------|---------------------------------------------------------------------------------|
| `api_key`             | string  | **Required.** Your API key.                                                     |
| `title`               | string  | Retrieval keywords. Multiple words are matched with AND regardless of order.    |
| `sort.by`             | string  | Ranking mode. Use `trust` for source credibility; `quality` for editorial weight. |
| `hl`                  | boolean | Enable highlighting. Returns matched passages instead of whole bodies.          |
| `hl.fl`               | string  | Fields to highlight: `title`, `description`, `body`. Max 5.                     |
| `hl.fragsize`         | integer | Snippet length in characters (50–500). Set this to your chunk size.             |
| `hl.snippets`         | integer | Snippets per field (1–10).                                                      |
| `hl.tag.pre`          | string  | Opening tag for matches. Defaults to `<em>`; an empty value is ignored.         |
| `hl.tag.post`         | string  | Closing tag for matches. Defaults to `</em>`.                                   |
| `fl`                  | string  | Field selection — trim the payload to what the prompt actually needs.           |
| `source.rank.opr.min` | number  | Minimum source authority (0–10). Raise it to keep low-quality domains out.      |
| `is_duplicate`        | integer | Set to `0` to drop near-duplicate coverage before it wastes context.            |
| `published_at.start`  | string  | Start date (ISO 8601, `YYYY-MM-DD`, or `NOW-7DAYS`).                            |
| `language.code`       | string  | Filter by language code.                                                        |
| `per_page`            | integer | Number of articles to retrieve.                                                 |
| `claim`               | string  | Fact-check: a single statement to verify (5–500 characters).                    |
| `text`                | string  | Fact-check: a block of text; claims are extracted automatically.                |
| `id`                  | integer | Fact-check: an APITube article id to verify.                                    |
| `evidence_per_claim`  | integer | Fact-check: evidence articles retrieved per claim.                              |

## Quick Start

### cURL

```bash
# Retrieve with highlighted snippets sized for a RAG chunk
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=inflation&hl=true&hl.fl=title,description,body&hl.fragsize=300&hl.snippets=2&per_page=8"

# Retrieve only high-authority, non-duplicate coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=inflation&sort.by=trust&source.rank.opr.min=6&is_duplicate=0&per_page=8"

# Verify a single claim against the corpus
curl -s "https://api.apitube.io/v1/fact-check?api_key=YOUR_API_KEY&claim=Inflation%20in%20Nigeria%20fell%20below%2016%20percent%20in%20June%202026&evidence_per_claim=3"
```

### Python

```python
import requests
import re

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check"


class NewsGrounder:
    """Retrieval layer that returns citation-ready context for an LLM prompt."""

    def __init__(self, api_key, chunk_chars=300, snippets_per_field=2):
        self.api_key = api_key
        self.chunk_chars = chunk_chars
        self.snippets_per_field = snippets_per_field

    def retrieve(self, query, limit=8, min_authority=5, days=30, language="en"):
        """Retrieve articles with server-side highlighted passages."""
        params = {
            "api_key": self.api_key,
            "title": query,
            "sort.by": "trust",
            "source.rank.opr.min": min_authority,
            "is_duplicate": 0,
            "published_at.start": f"NOW-{days}DAYS",
            "language.code": language,
            "per_page": limit,
            "hl": "true",
            "hl.fl": "title,description,body",
            "hl.fragsize": self.chunk_chars,
            "hl.snippets": self.snippets_per_field,
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        payload = response.json()

        if payload.get("status") != "ok":
            return []

        articles = payload.get("results", [])
        highlighting = payload.get("highlighting", {})

        return [self._merge(article, highlighting) for article in articles]

    def _merge(self, article, highlighting):
        """Attach highlighted passages to an article, falling back to description."""
        marked = highlighting.get(str(article["id"]), {})

        passages = []
        for field in ("title", "description", "body"):
            passages.extend(marked.get(field, []))

        if not passages:
            passages = [article.get("description") or ""]

        return {
            "id": article["id"],
            "title": article["title"],
            "url": article["href"],
            "source": article["source"]["domain"],
            "authority": (article["source"].get("rankings") or {}).get("opr"),
            "published_at": article["published_at"],
            "passages": [self._clean(p) for p in passages if p],
        }

    def _clean(self, text):
        """Collapse whitespace and strip the ellipsis artifacts highlighting leaves."""
        text = re.sub(r"</?em>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"(\.\s){2,}\.?", "... ", text).strip()

    def build_context(self, query, token_budget=2000, max_per_domain=2, **kwargs):
        """Assemble a numbered context block that fits a rough token budget."""
        char_budget = token_budget * 4
        documents = self.diversify(self.retrieve(query, **kwargs), max_per_domain)

        blocks = []
        citations = []
        used = 0

        for document in documents:
            body = " ".join(document["passages"])
            block = (
                f"[{len(blocks) + 1}] {document['title']}\n"
                f"Source: {document['source']} ({document['published_at']})\n"
                f"URL: {document['url']}\n"
                f"{body}\n"
            )

            if used + len(block) > char_budget:
                break

            blocks.append(block)
            citations.append(document)
            used += len(block)

        return {
            "context": "\n".join(blocks),
            "citations": citations,
            "chars_used": used,
        }

    def diversify(self, documents, max_per_domain=2):
        """Cap how many articles one domain contributes and drop repeated headlines.

        Authority-based sorting concentrates results on a handful of strong domains,
        and `is_duplicate=0` only removes near-identical bodies, not re-runs of the
        same story under the same headline. Both hurt a grounding context.
        """
        per_domain = {}
        seen_titles = set()
        kept = []

        for document in documents:
            title_key = document["title"].strip().lower()
            domain = document["source"]

            if title_key in seen_titles:
                continue
            if per_domain.get(domain, 0) >= max_per_domain:
                continue

            seen_titles.add(title_key)
            per_domain[domain] = per_domain.get(domain, 0) + 1
            kept.append(document)

        return kept

    def verify(self, statement, evidence_per_claim=3):
        """Check a statement against the live corpus and return the verdict."""
        response = requests.get(
            FACT_CHECK_URL,
            params={
                "api_key": self.api_key,
                "claim": statement,
                "evidence_per_claim": evidence_per_claim,
            },
            timeout=90,
        )
        payload = response.json()

        if payload.get("status") != "ok":
            return None

        claim = payload["claims"][0]
        return {
            "verdict": claim["verdict"],
            "confidence": claim["confidence"],
            "explanation": claim["explanation"],
            "supporting": [e for e in claim["evidence"] if e["stance"] == "supports"],
            "refuting": [e for e in claim["evidence"] if e["stance"] == "refutes"],
        }


PROMPT_TEMPLATE = """Answer the question using only the numbered sources below.
Cite every factual statement with its number, like [1] or [2, 3].
If the sources do not contain the answer, say so plainly.

Sources:
{context}

Question: {question}
"""


grounder = NewsGrounder(API_KEY)

pack = grounder.build_context("inflation", token_budget=1500, limit=8, days=14)
prompt = PROMPT_TEMPLATE.format(context=pack["context"], question="What is happening with inflation?")

print(f"Context: {pack['chars_used']} chars across {len(pack['citations'])} sources")
for index, citation in enumerate(pack["citations"], start=1):
    print(f"  [{index}] {citation['source']} — {citation['title'][:60]}")

verdict = grounder.verify("Inflation in Nigeria fell below 16 percent in June 2026")
if verdict:
    print(f"\nVerdict: {verdict['verdict']} (confidence {verdict['confidence']})")
    print(f"Supporting: {len(verdict['supporting'])}, refuting: {len(verdict['refuting'])}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check";

class NewsGrounder {
  constructor(apiKey, chunkChars = 300, snippetsPerField = 2) {
    this.apiKey = apiKey;
    this.chunkChars = chunkChars;
    this.snippetsPerField = snippetsPerField;
  }

  async retrieve(query, { limit = 8, minAuthority = 5, days = 30, language = "en" } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      title: query,
      "sort.by": "trust",
      "source.rank.opr.min": String(minAuthority),
      is_duplicate: "0",
      "published_at.start": `NOW-${days}DAYS`,
      "language.code": language,
      per_page: String(limit),
      hl: "true",
      "hl.fl": "title,description,body",
      "hl.fragsize": String(this.chunkChars),
      "hl.snippets": String(this.snippetsPerField)
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const payload = await response.json();

    if (payload.status !== "ok") return [];

    const highlighting = payload.highlighting || {};
    return (payload.results || []).map((article) => this.merge(article, highlighting));
  }

  merge(article, highlighting) {
    const marked = highlighting[String(article.id)] || {};
    let passages = [];

    for (const field of ["title", "description", "body"]) {
      passages = passages.concat(marked[field] || []);
    }

    if (passages.length === 0) {
      passages = [article.description || ""];
    }

    return {
      id: article.id,
      title: article.title,
      url: article.href,
      source: article.source.domain,
      authority: article.source.rankings?.opr,
      publishedAt: article.published_at,
      passages: passages.filter(Boolean).map((p) => this.clean(p))
    };
  }

  clean(text) {
    return text
      .replace(/<\/?em>/g, "")
      .replace(/\s+/g, " ")
      .replace(/(\.\s){2,}\.?/g, "... ")
      .trim();
  }

  async buildContext(query, { tokenBudget = 2000, maxPerDomain = 2, ...options } = {}) {
    const charBudget = tokenBudget * 4;
    const documents = this.diversify(await this.retrieve(query, options), maxPerDomain);

    const blocks = [];
    const citations = [];
    let used = 0;

    for (const document of documents) {
      const body = document.passages.join(" ");
      const block =
        `[${blocks.length + 1}] ${document.title}\n` +
        `Source: ${document.source} (${document.publishedAt})\n` +
        `URL: ${document.url}\n` +
        `${body}\n`;

      if (used + block.length > charBudget) break;

      blocks.push(block);
      citations.push(document);
      used += block.length;
    }

    return {
      context: blocks.join("\n"),
      citations,
      charsUsed: used
    };
  }

  // Authority sorting concentrates results on a few strong domains, and
  // is_duplicate=0 only drops near-identical bodies — not re-runs of the same
  // headline. Cap both before they eat the context window.
  diversify(documents, maxPerDomain = 2) {
    const perDomain = new Map();
    const seenTitles = new Set();
    const kept = [];

    for (const document of documents) {
      const titleKey = document.title.trim().toLowerCase();
      const count = perDomain.get(document.source) || 0;

      if (seenTitles.has(titleKey)) continue;
      if (count >= maxPerDomain) continue;

      seenTitles.add(titleKey);
      perDomain.set(document.source, count + 1);
      kept.push(document);
    }

    return kept;
  }

  async verify(statement, evidencePerClaim = 3) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      claim: statement,
      evidence_per_claim: String(evidencePerClaim)
    });

    const response = await fetch(`${FACT_CHECK_URL}?${params}`);
    const payload = await response.json();

    if (payload.status !== "ok") return null;

    const claim = payload.claims[0];
    return {
      verdict: claim.verdict,
      confidence: claim.confidence,
      explanation: claim.explanation,
      supporting: claim.evidence.filter((e) => e.stance === "supports"),
      refuting: claim.evidence.filter((e) => e.stance === "refutes")
    };
  }
}

const grounder = new NewsGrounder(API_KEY);

const pack = await grounder.buildContext("inflation", { tokenBudget: 1500, limit: 8, days: 14 });
console.log(`Context: ${pack.charsUsed} chars across ${pack.citations.length} sources`);
pack.citations.forEach((citation, index) => {
  console.log(`  [${index + 1}] ${citation.source} — ${citation.title.slice(0, 60)}`);
});

const verdict = await grounder.verify("Inflation in Nigeria fell below 16 percent in June 2026");
if (verdict) {
  console.log(`\nVerdict: ${verdict.verdict} (confidence ${verdict.confidence})`);
  console.log(`Supporting: ${verdict.supporting.length}, refuting: ${verdict.refuting.length}`);
}
```

### PHP

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check";

function apitubeGet(string $url, array $params): array
{
    $handle = curl_init($url . "?" . http_build_query($params));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, 90);
    $body = curl_exec($handle);
    curl_close($handle);

    return json_decode($body, true) ?: [];
}

function cleanPassage(string $text): string
{
    $text = preg_replace('#</?em>#', '', $text);
    $text = preg_replace('/\s+/', ' ', $text);
    $text = preg_replace('/(\.\s){2,}\.?/', '... ', $text);

    return trim($text);
}

function retrieveGrounded(string $query, int $limit = 8, int $days = 30): array
{
    $payload = apitubeGet(BASE_URL, [
        "api_key"             => API_KEY,
        "title"               => $query,
        "sort.by"             => "trust",
        "source.rank.opr.min" => 5,
        "is_duplicate"        => 0,
        "published_at.start"  => "NOW-{$days}DAYS",
        "language.code"       => "en",
        "per_page"            => $limit,
        "hl"                  => "true",
        "hl.fl"               => "title,description,body",
        "hl.fragsize"         => 300,
        "hl.snippets"         => 2,
    ]);

    if (($payload["status"] ?? "") !== "ok") {
        return [];
    }

    $highlighting = $payload["highlighting"] ?? [];
    $documents = [];

    foreach ($payload["results"] ?? [] as $article) {
        $marked = $highlighting[(string) $article["id"]] ?? [];
        $passages = [];

        foreach (["title", "description", "body"] as $field) {
            $passages = array_merge($passages, $marked[$field] ?? []);
        }

        if (empty($passages)) {
            $passages = [$article["description"] ?? ""];
        }

        $documents[] = [
            "id"           => $article["id"],
            "title"        => $article["title"],
            "url"          => $article["href"],
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "passages"     => array_map("cleanPassage", array_filter($passages)),
        ];
    }

    return $documents;
}

/**
 * Authority sorting concentrates results on a few strong domains, and
 * is_duplicate=0 only drops near-identical bodies — not re-runs of the same
 * headline. Cap both before they eat the context window.
 */
function diversify(array $documents, int $maxPerDomain = 2): array
{
    $perDomain = [];
    $seenTitles = [];
    $kept = [];

    foreach ($documents as $document) {
        $titleKey = strtolower(trim($document["title"]));
        $domain = $document["source"];

        if (isset($seenTitles[$titleKey])) {
            continue;
        }

        if (($perDomain[$domain] ?? 0) >= $maxPerDomain) {
            continue;
        }

        $seenTitles[$titleKey] = true;
        $perDomain[$domain] = ($perDomain[$domain] ?? 0) + 1;
        $kept[] = $document;
    }

    return $kept;
}

function buildContext(string $query, int $tokenBudget = 2000, int $maxPerDomain = 2): array
{
    $charBudget = $tokenBudget * 4;
    $documents = diversify(retrieveGrounded($query), $maxPerDomain);

    $blocks = [];
    $used = 0;
    $index = 1;

    foreach ($documents as $document) {
        $body = implode(" ", $document["passages"]);
        $block = sprintf(
            "[%d] %s\nSource: %s (%s)\nURL: %s\n%s\n",
            $index,
            $document["title"],
            $document["source"],
            $document["published_at"],
            $document["url"],
            $body
        );

        if ($used + strlen($block) > $charBudget) {
            break;
        }

        $blocks[] = $block;
        $used += strlen($block);
        $index++;
    }

    return [
        "context"    => implode("\n", $blocks),
        "citations"  => array_slice($documents, 0, count($blocks)),
        "chars_used" => $used,
    ];
}

function verifyClaim(string $statement, int $evidencePerClaim = 3): ?array
{
    $payload = apitubeGet(FACT_CHECK_URL, [
        "api_key"            => API_KEY,
        "claim"              => $statement,
        "evidence_per_claim" => $evidencePerClaim,
    ]);

    if (($payload["status"] ?? "") !== "ok") {
        return null;
    }

    $claim = $payload["claims"][0];
    $supporting = array_filter($claim["evidence"], fn($e) => $e["stance"] === "supports");
    $refuting = array_filter($claim["evidence"], fn($e) => $e["stance"] === "refutes");

    return [
        "verdict"     => $claim["verdict"],
        "confidence"  => $claim["confidence"],
        "explanation" => $claim["explanation"],
        "supporting"  => count($supporting),
        "refuting"    => count($refuting),
    ];
}

$pack = buildContext("inflation", 1500);
printf("Context: %d chars across %d sources\n", $pack["chars_used"], count($pack["citations"]));

foreach ($pack["citations"] as $i => $citation) {
    printf("  [%d] %s — %s\n", $i + 1, $citation["source"], substr($citation["title"], 0, 60));
}

$verdict = verifyClaim("Inflation in Nigeria fell below 16 percent in June 2026");
if ($verdict) {
    printf("\nVerdict: %s (confidence %.2f)\n", $verdict["verdict"], $verdict["confidence"]);
    printf("Supporting: %d, refuting: %d\n", $verdict["supporting"], $verdict["refuting"]);
}
```

## Why Highlighting Instead of Client-Side Chunking

A conventional RAG pipeline downloads full article bodies, splits them into chunks, embeds every chunk, and searches the embeddings. Highlighting moves the passage selection server-side: `hl=true` returns only the fragments that matched the query, already sized by `hl.fragsize`.

| Concern            | Full-body retrieval                    | `hl=true` retrieval                              |
|--------------------|----------------------------------------|--------------------------------------------------|
| Payload            | Whole `body` per article               | Matching passages only                           |
| Chunk size         | Decided in your code after download    | Decided by `hl.fragsize` before transfer         |
| Passage selection  | Requires embeddings or a reranker      | Handled by the search index                      |
| Term variants      | Exact string matching unless you add it| Synonyms and morphology expanded automatically   |
| Multilingual       | Needs per-language tokenizers          | Language-specific dictionaries applied server-side |

Highlighting expands search terms using the dictionary, so a query for `run` also marks `running`, `runner` and `ran`. That is why a highlighted passage sometimes contains no literal occurrence of your query string.

Matches come back wrapped in `<em>` markup, which wastes tokens and confuses some models. Passing empty values for `hl.tag.pre` and `hl.tag.post` does **not** turn it off — the API ignores empty tag values and falls back to `<em>`. Either strip the markup client-side, as every example here does, or pass a harmless non-empty delimiter of your own.

## Verdict Scale

`/v1/fact-check` returns one of eight verdicts per claim, plus a calibrated `confidence` from 0 to 1:

| Verdict        | Meaning                                                     |
|----------------|-------------------------------------------------------------|
| `true`         | Supported by the evidence.                                   |
| `mostly_true`  | Broadly supported with minor inaccuracies.                   |
| `mixed`        | Evidence both supports and refutes parts of the claim.       |
| `misleading`   | Technically accurate but framed to mislead.                  |
| `mostly_false` | Largely contradicted with a kernel of truth.                 |
| `false`        | Contradicted by the evidence.                                |
| `unverified`   | Not enough evidence in the corpus to confirm or refute.      |
| `outdated`     | Once true, contradicted by more recent evidence.             |

Each claim carries an `evidence` array where every item has a `stance` (`supports`, `refutes`, `neutral`), a `relevance` score from 0 to 1, a `source_authority` value, and a `snippet`. Treat `unverified` as "the corpus cannot answer this", not as "false" — it is the expected verdict for claims about private facts, opinions, or events too recent to have been covered.

## Common Use Cases

- **Grounded chat assistants** — answer questions about current events with inline citations.
- **Hallucination guards** — run a model's factual statements through `/v1/fact-check` before showing them.
- **Research copilots** — assemble a sourced briefing on a topic within a fixed token budget.
- **Editorial verification** — check quotes and figures in a draft against live coverage.
- **Agent tool backends** — expose retrieval and verification as two tools an agent can call.
- **Newsroom fact desks** — triage claims by verdict and confidence before human review.
- **Compliance review** — verify marketing claims against what the press actually reported.

## See Also

- [examples.md](./examples.md) — detailed code examples for LLM news grounding.
