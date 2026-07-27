# LLM News Grounding — Examples

Advanced code examples for multi-query retrieval, citation-bound answer assembly, and post-generation hallucination checks.

---

## Python — Multi-Query RAG Pipeline

```python
import requests
import re
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check"


class MultiQueryRAG:
    """Retrieval pipeline that fans a question out into several sub-queries.

    A single keyword query misses coverage that describes the same event in
    different words. Running several narrow queries and merging the results
    covers more ground than one broad query with a high per_page.
    """

    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, keywords, limit=10, min_authority=5, days=30, language="en"):
        """One retrieval pass with highlighted passages."""
        params = {
            "api_key": self.api_key,
            "title": keywords,
            "sort.by": "trust",
            "source.rank.opr.min": min_authority,
            "is_duplicate": 0,
            "published_at.start": f"NOW-{days}DAYS",
            "language.code": language,
            "per_page": limit,
            "hl": "true",
            "hl.fl": "title,description,body",
            "hl.fragsize": 280,
            "hl.snippets": 2,
        }

        try:
            payload = requests.get(BASE_URL, params=params, timeout=30).json()
        except requests.RequestException:
            return []

        if payload.get("status") != "ok":
            return []

        highlighting = payload.get("highlighting", {})
        documents = []

        for article in payload.get("results", []):
            marked = highlighting.get(str(article["id"]), {})
            passages = []

            for field in ("title", "description", "body"):
                passages.extend(marked.get(field, []))

            if not passages:
                passages = [article.get("description") or ""]

            documents.append({
                "id": article["id"],
                "title": article["title"],
                "url": article["href"],
                "source": article["source"]["domain"],
                "authority": (article["source"].get("rankings") or {}).get("opr") or 0,
                "published_at": article["published_at"],
                "passages": [self._clean(p) for p in passages if p],
                "matched_query": keywords,
            })

        return documents

    def _clean(self, text):
        text = re.sub(r"</?em>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return re.sub(r"(\.\s){2,}\.?", "... ", text).strip()

    def multi_search(self, sub_queries, per_query=6, **kwargs):
        """Run every sub-query and merge on article id, keeping match counts."""
        by_id = {}
        hit_counts = defaultdict(int)

        for query in sub_queries:
            for document in self.search(query, limit=per_query, **kwargs):
                hit_counts[document["id"]] += 1
                if document["id"] not in by_id:
                    by_id[document["id"]] = document

        for article_id, document in by_id.items():
            document["query_hits"] = hit_counts[article_id]

        return list(by_id.values())

    def rank(self, documents, max_per_domain=2):
        """Rank by cross-query agreement, then authority, then recency."""
        ordered = sorted(
            documents,
            key=lambda d: (d["query_hits"], d["authority"], d["published_at"]),
            reverse=True,
        )

        per_domain = defaultdict(int)
        seen_titles = set()
        kept = []

        for document in ordered:
            title_key = document["title"].strip().lower()

            if title_key in seen_titles:
                continue
            if per_domain[document["source"]] >= max_per_domain:
                continue

            seen_titles.add(title_key)
            per_domain[document["source"]] += 1
            kept.append(document)

        return kept

    def context_pack(self, sub_queries, token_budget=2500, **kwargs):
        """Assemble a numbered, budget-bound context block."""
        documents = self.rank(self.multi_search(sub_queries, **kwargs))
        char_budget = token_budget * 4

        blocks = []
        citations = []
        used = 0

        for document in documents:
            block = (
                f"[{len(blocks) + 1}] {document['title']}\n"
                f"Source: {document['source']} | Published: {document['published_at']}\n"
                f"URL: {document['url']}\n"
                f"{' '.join(document['passages'])}\n"
            )

            if used + len(block) > char_budget:
                break

            blocks.append(block)
            citations.append(document)
            used += len(block)

        return {"context": "\n".join(blocks), "citations": citations, "chars_used": used}

    def audit_answer(self, answer_text, max_claims=5, evidence_per_claim=3):
        """Extract claims from a generated answer and verify each one."""
        try:
            payload = requests.get(
                FACT_CHECK_URL,
                params={
                    "api_key": self.api_key,
                    "text": answer_text,
                    "max_claims": max_claims,
                    "evidence_per_claim": evidence_per_claim,
                },
                timeout=120,
            ).json()
        except requests.RequestException:
            return None

        if payload.get("status") != "ok":
            return None

        contradicted = {"false", "mostly_false", "misleading", "outdated"}
        claims = []

        for claim in payload.get("claims", []):
            claims.append({
                "claim": claim["claim"],
                "verdict": claim["verdict"],
                "confidence": claim["confidence"],
                "explanation": claim["explanation"],
                "supports": sum(1 for e in claim["evidence"] if e["stance"] == "supports"),
                "refutes": sum(1 for e in claim["evidence"] if e["stance"] == "refutes"),
            })

        return {
            "overall": payload["summary"]["overall_verdict"],
            "confidence": payload["summary"]["overall_confidence"],
            "breakdown": payload["summary"]["verdict_breakdown"],
            "claims": claims,
            "blocking": [c for c in claims if c["verdict"] in contradicted],
        }


ANSWER_PROMPT = """Answer the question using only the numbered sources below.
Cite every factual statement with its source number, like [1] or [2, 3].
Do not state anything the sources do not support. If they do not answer the
question, say so.

Sources:
{context}

Question: {question}
"""


def decompose(question):
    """Turn one question into narrow retrieval queries.

    In production a language model writes these. Keep each query to the words
    that would plausibly appear in a headline — `title` matches headlines only.
    """
    return [question, f"{question} forecast", f"{question} data"]


rag = MultiQueryRAG(API_KEY)

question = "inflation"
pack = rag.context_pack(decompose(question), token_budget=2000, days=21)

print(f"Retrieved {len(pack['citations'])} sources, {pack['chars_used']} chars")
for index, citation in enumerate(pack["citations"], start=1):
    print(f"  [{index}] {citation['source']:<24} hits={citation['query_hits']} "
          f"opr={citation['authority']} — {citation['title'][:52]}")

prompt = ANSWER_PROMPT.format(context=pack["context"], question=question)
print(f"\nPrompt is {len(prompt)} chars — send it to your model of choice.")

draft = (
    "UK inflation eased to a 15-month low in June 2026. "
    "Food inflation hit its lowest rate in almost two years."
)

audit = rag.audit_answer(draft)
if audit:
    print(f"\nAudit: {audit['overall']} (confidence {audit['confidence']})")
    print(f"Breakdown: {audit['breakdown']}")
    for claim in audit["claims"]:
        print(f"  {claim['verdict']:<12} {claim['confidence']:.2f} "
              f"(+{claim['supports']}/-{claim['refutes']}) {claim['claim'][:60]}")
    if audit["blocking"]:
        print(f"\nBLOCKED: {len(audit['blocking'])} claim(s) contradicted by evidence")
```

---

## JavaScript — Citation-Bound Answer Assembler

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check";

class CitationAssembler {
  constructor(apiKey) {
    this.apiKey = apiKey;
  }

  async search(keywords, { limit = 10, minAuthority = 5, days = 30, language = "en" } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      title: keywords,
      "sort.by": "trust",
      "source.rank.opr.min": String(minAuthority),
      is_duplicate: "0",
      "published_at.start": `NOW-${days}DAYS`,
      "language.code": language,
      per_page: String(limit),
      hl: "true",
      "hl.fl": "title,description,body",
      "hl.fragsize": "280",
      "hl.snippets": "2"
    });

    let payload;
    try {
      const response = await fetch(`${BASE_URL}?${params}`);
      payload = await response.json();
    } catch {
      return [];
    }

    if (payload.status !== "ok") return [];

    const highlighting = payload.highlighting || {};

    return (payload.results || []).map((article) => {
      const marked = highlighting[String(article.id)] || {};
      let passages = [];

      for (const field of ["title", "description", "body"]) {
        passages = passages.concat(marked[field] || []);
      }

      if (passages.length === 0) passages = [article.description || ""];

      return {
        id: article.id,
        title: article.title,
        url: article.href,
        source: article.source.domain,
        authority: article.source.rankings?.opr || 0,
        publishedAt: article.published_at,
        passages: passages.filter(Boolean).map((p) => this.clean(p)),
        matchedQuery: keywords
      };
    });
  }

  clean(text) {
    return text
      .replace(/<\/?em>/g, "")
      .replace(/\s+/g, " ")
      .replace(/(\.\s){2,}\.?/g, "... ")
      .trim();
  }

  async multiSearch(subQueries, { perQuery = 6, ...options } = {}) {
    const byId = new Map();
    const hits = new Map();

    for (const query of subQueries) {
      const documents = await this.search(query, { limit: perQuery, ...options });

      for (const document of documents) {
        hits.set(document.id, (hits.get(document.id) || 0) + 1);
        if (!byId.has(document.id)) byId.set(document.id, document);
      }
    }

    return [...byId.values()].map((document) => ({
      ...document,
      queryHits: hits.get(document.id)
    }));
  }

  rank(documents, maxPerDomain = 2) {
    const ordered = [...documents].sort((a, b) => {
      if (b.queryHits !== a.queryHits) return b.queryHits - a.queryHits;
      if (b.authority !== a.authority) return b.authority - a.authority;
      return b.publishedAt.localeCompare(a.publishedAt);
    });

    const perDomain = new Map();
    const seenTitles = new Set();
    const kept = [];

    for (const document of ordered) {
      const titleKey = document.title.trim().toLowerCase();
      const count = perDomain.get(document.source) || 0;

      if (seenTitles.has(titleKey) || count >= maxPerDomain) continue;

      seenTitles.add(titleKey);
      perDomain.set(document.source, count + 1);
      kept.push(document);
    }

    return kept;
  }

  async contextPack(subQueries, { tokenBudget = 2500, ...options } = {}) {
    const documents = this.rank(await this.multiSearch(subQueries, options));
    const charBudget = tokenBudget * 4;

    const blocks = [];
    const citations = [];
    let used = 0;

    for (const document of documents) {
      const block =
        `[${blocks.length + 1}] ${document.title}\n` +
        `Source: ${document.source} | Published: ${document.publishedAt}\n` +
        `URL: ${document.url}\n` +
        `${document.passages.join(" ")}\n`;

      if (used + block.length > charBudget) break;

      blocks.push(block);
      citations.push(document);
      used += block.length;
    }

    return { context: blocks.join("\n"), citations, charsUsed: used };
  }

  // Verify that every [n] the model emitted points at a real source, and that
  // no source went uncited. A model citing [7] over a five-source context is
  // the clearest hallucination signal you get for free.
  checkCitations(answer, citations) {
    const cited = new Set();
    const invalid = [];

    for (const match of answer.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
      for (const part of match[1].split(",")) {
        const index = Number(part.trim());
        if (index >= 1 && index <= citations.length) cited.add(index);
        else invalid.push(index);
      }
    }

    const unused = citations.map((_, i) => i + 1).filter((i) => !cited.has(i));

    return {
      valid: invalid.length === 0,
      invalidRefs: invalid,
      unusedSources: unused,
      coverage: citations.length ? cited.size / citations.length : 0
    };
  }

  async auditAnswer(answerText, { maxClaims = 5, evidencePerClaim = 3 } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      text: answerText,
      max_claims: String(maxClaims),
      evidence_per_claim: String(evidencePerClaim)
    });

    let payload;
    try {
      const response = await fetch(`${FACT_CHECK_URL}?${params}`);
      payload = await response.json();
    } catch {
      return null;
    }

    if (payload.status !== "ok") return null;

    const contradicted = new Set(["false", "mostly_false", "misleading", "outdated"]);

    const claims = payload.claims.map((claim) => ({
      claim: claim.claim,
      verdict: claim.verdict,
      confidence: claim.confidence,
      supports: claim.evidence.filter((e) => e.stance === "supports").length,
      refutes: claim.evidence.filter((e) => e.stance === "refutes").length
    }));

    return {
      overall: payload.summary.overall_verdict,
      confidence: payload.summary.overall_confidence,
      breakdown: payload.summary.verdict_breakdown,
      claims,
      blocking: claims.filter((c) => contradicted.has(c.verdict))
    };
  }
}

const assembler = new CitationAssembler(API_KEY);

const pack = await assembler.contextPack(["inflation", "inflation forecast", "inflation data"], {
  tokenBudget: 2000,
  days: 21
});

console.log(`Retrieved ${pack.citations.length} sources, ${pack.charsUsed} chars`);
pack.citations.forEach((citation, index) => {
  console.log(
    `  [${index + 1}] ${citation.source.padEnd(24)} hits=${citation.queryHits} ` +
      `opr=${citation.authority} — ${citation.title.slice(0, 52)}`
  );
});

const modelAnswer = "Inflation eased in June [1], with food prices slowing [2].";
const citationCheck = assembler.checkCitations(modelAnswer, pack.citations);
console.log(`\nCitations valid: ${citationCheck.valid}`);
console.log(`Invalid refs: ${citationCheck.invalidRefs.join(", ") || "none"}`);
console.log(`Source coverage: ${(citationCheck.coverage * 100).toFixed(0)}%`);

const audit = await assembler.auditAnswer(
  "UK inflation eased to a 15-month low in June 2026. Food inflation hit its lowest rate in almost two years."
);

if (audit) {
  console.log(`\nAudit: ${audit.overall} (confidence ${audit.confidence})`);
  audit.claims.forEach((claim) => {
    console.log(
      `  ${claim.verdict.padEnd(12)} ${claim.confidence.toFixed(2)} ` +
        `(+${claim.supports}/-${claim.refutes}) ${claim.claim.slice(0, 60)}`
    );
  });
}
```

---

## PHP — Claim Verification Service

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const FACT_CHECK_URL = "https://api.apitube.io/v1/fact-check";

const CONTRADICTED = ["false", "mostly_false", "misleading", "outdated"];

/**
 * Highlighting wraps matches in <em> by default. Passing empty hl.tag.pre /
 * hl.tag.post does NOT disable it — the API falls back to <em> — so strip the
 * markup here before the text reaches a prompt.
 */
function cleanPassage(string $text): string
{
    $text = preg_replace('#</?em>#', '', $text);
    $text = preg_replace('/\s+/', ' ', $text);
    $text = preg_replace('/(\.\s){2,}\.?/', '... ', $text);

    return trim($text);
}

function apiGet(string $url, array $params, int $timeout = 120): array
{
    $handle = curl_init($url . "?" . http_build_query($params));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, $timeout);

    $body = curl_exec($handle);
    $failed = curl_errno($handle) !== 0;
    curl_close($handle);

    if ($failed) {
        return [];
    }

    return json_decode($body, true) ?: [];
}

/**
 * Verify a block of generated text. Claims are extracted automatically when
 * `text` is used; pass `claim` instead to check one statement verbatim.
 */
function verifyText(string $text, int $maxClaims = 5, int $evidencePerClaim = 3): ?array
{
    $payload = apiGet(FACT_CHECK_URL, [
        "api_key"            => API_KEY,
        "text"               => $text,
        "max_claims"         => $maxClaims,
        "evidence_per_claim" => $evidencePerClaim,
    ]);

    if (($payload["status"] ?? "") !== "ok") {
        return null;
    }

    $claims = [];

    foreach ($payload["claims"] ?? [] as $claim) {
        $supports = 0;
        $refutes = 0;
        $topEvidence = null;

        foreach ($claim["evidence"] ?? [] as $evidence) {
            if ($evidence["stance"] === "supports") {
                $supports++;
            } elseif ($evidence["stance"] === "refutes") {
                $refutes++;
            }

            if ($topEvidence === null || $evidence["relevance"] > $topEvidence["relevance"]) {
                $topEvidence = $evidence;
            }
        }

        $claims[] = [
            "claim"       => $claim["claim"],
            "verdict"     => $claim["verdict"],
            "confidence"  => $claim["confidence"],
            "explanation" => $claim["explanation"],
            "checkworthy" => $claim["checkworthy"],
            "as_of"       => $claim["as_of"],
            "supports"    => $supports,
            "refutes"     => $refutes,
            "top_source"  => $topEvidence["source"] ?? null,
            "top_url"     => $topEvidence["url"] ?? null,
        ];
    }

    $blocking = array_values(array_filter(
        $claims,
        fn($c) => in_array($c["verdict"], CONTRADICTED, true)
    ));

    return [
        "overall"    => $payload["summary"]["overall_verdict"],
        "confidence" => $payload["summary"]["overall_confidence"],
        "breakdown"  => $payload["summary"]["verdict_breakdown"],
        "claims"     => $claims,
        "blocking"   => $blocking,
    ];
}

/**
 * Verify a single statement and return the evidence split by stance.
 */
function verifySingleClaim(string $statement, int $evidencePerClaim = 5): ?array
{
    $payload = apiGet(FACT_CHECK_URL, [
        "api_key"            => API_KEY,
        "claim"              => $statement,
        "evidence_per_claim" => $evidencePerClaim,
    ]);

    if (($payload["status"] ?? "") !== "ok") {
        return null;
    }

    $claim = $payload["claims"][0];
    $grouped = ["supports" => [], "refutes" => [], "neutral" => []];

    foreach ($claim["evidence"] ?? [] as $evidence) {
        $grouped[$evidence["stance"]][] = [
            "title"     => $evidence["title"],
            "source"    => $evidence["source"],
            "url"       => $evidence["url"],
            "relevance" => $evidence["relevance"],
            "authority" => $evidence["source_authority"],
            "snippet"   => $evidence["snippet"],
        ];
    }

    return [
        "verdict"    => $claim["verdict"],
        "confidence" => $claim["confidence"],
        "evidence"   => $grouped,
    ];
}

/**
 * Retrieve grounding passages for a claim so a reviewer can read the context
 * that the verdict was based on.
 */
function groundingFor(string $keywords, int $limit = 5): array
{
    $payload = apiGet(BASE_URL, [
        "api_key"             => API_KEY,
        "title"               => $keywords,
        "sort.by"             => "trust",
        "source.rank.opr.min" => 6,
        "is_duplicate"        => 0,
        "per_page"            => $limit,
        "hl"                  => "true",
        "hl.fl"               => "title,body",
        "hl.fragsize"         => 250,
        "hl.snippets"         => 1,
    ], 30);

    if (($payload["status"] ?? "") !== "ok") {
        return [];
    }

    $highlighting = $payload["highlighting"] ?? [];
    $out = [];

    foreach ($payload["results"] ?? [] as $article) {
        $marked = $highlighting[(string) $article["id"]] ?? [];
        $passages = array_merge($marked["title"] ?? [], $marked["body"] ?? []);

        $out[] = [
            "source"  => $article["source"]["domain"],
            "title"   => $article["title"],
            "url"     => $article["href"],
            "passage" => cleanPassage($passages[0] ?? ($article["description"] ?? "")),
        ];
    }

    return $out;
}

$draft = "UK inflation eased to a 15-month low in June 2026. "
    . "Food inflation hit its lowest rate in almost two years.";

$audit = verifyText($draft);

if ($audit) {
    printf("Overall: %s (confidence %.2f)\n", $audit["overall"], $audit["confidence"]);
    printf("Breakdown: %s\n\n", json_encode($audit["breakdown"]));

    foreach ($audit["claims"] as $claim) {
        printf(
            "  %-12s %.2f (+%d/-%d) %s\n",
            $claim["verdict"],
            $claim["confidence"],
            $claim["supports"],
            $claim["refutes"],
            substr($claim["claim"], 0, 60)
        );

        if ($claim["top_source"]) {
            printf("      top evidence: %s\n", $claim["top_source"]);
        }
    }

    if (!empty($audit["blocking"])) {
        printf("\nBLOCKED: %d claim(s) contradicted by evidence\n", count($audit["blocking"]));
    }
}

$single = verifySingleClaim("Food inflation in the UK slowed in June 2026");

if ($single) {
    printf("\nSingle claim verdict: %s (%.2f)\n", $single["verdict"], $single["confidence"]);

    foreach (["supports", "refutes"] as $stance) {
        printf("  %s: %d article(s)\n", $stance, count($single["evidence"][$stance]));
    }
}

foreach (groundingFor("inflation", 3) as $item) {
    printf("\n%s — %s\n  %s\n", $item["source"], substr($item["title"], 0, 60), substr($item["passage"], 0, 120));
}
```

---

## Notes on Behaviour

- **`sort.by=trust` concentrates on strong domains.** The ranking is heavily weighted toward source authority, so consecutive results often come from the same publisher. Cap per-domain contributions before building context — every example above does.
- **`is_duplicate=0` is not headline deduplication.** It removes near-identical bodies, not the same story re-published under the same headline by the same outlet. Deduplicate on normalised title as well.
- **`title` matches headlines only.** Sub-queries should use words that would plausibly appear in a headline. A long natural-language question passed straight into `title` matches nothing, because every word must appear.
- **Highlighting expands terms.** Synonyms and morphology come from the dictionary, so `hl` marks `running` when you searched `run`. That is why highlighted passages sometimes contain no literal occurrence of your query string.
- **Empty `hl.tag.pre` / `hl.tag.post` are ignored.** The API falls back to `<em>` markup rather than emitting bare text, so strip the tags client-side before the passage reaches a prompt. A non-empty custom delimiter is honoured.
- **`unverified` is the common verdict.** It means the corpus lacks evidence either way — expected for opinions, private facts, and very recent events. Do not surface it to users as "false".

## See Also

- [README.md](./README.md) — LLM News Grounding workflow overview and quick start.
