# Journalist Beat Mapping — Code Examples

Detailed examples for inferring journalist beats from coverage profiles using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Classify a Beat from Top Topics

```python
import requests

API_KEY = "YOUR_API_KEY"

def get_coverage(journalist_id):
    response = requests.get(
        f"https://api.apitube.io/v1/journalists/{journalist_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    return response.json()

profile = get_coverage(88123)
coverage = profile["coverage"]
topics = coverage["top_topics"]
total = sum(t["count"] for t in topics) or 1

print(f"Beat profile for {profile['name']}:\n")
for topic in topics:
    share = topic["count"] / total * 100
    bar = "#" * int(share / 2)
    print(f"  {topic['name']:<18} {topic['count']:>5} ({share:5.1f}%) {bar}")

primary = topics[0]
label = "specialist" if primary["count"] / total > 0.5 else "generalist"
print(f"\nPrimary beat: {primary['name']} ({label})")
```

### Build an Expertise Fingerprint

```python
import requests

API_KEY = "YOUR_API_KEY"

def fingerprint(journalist_id):
    response = requests.get(
        f"https://api.apitube.io/v1/journalists/{journalist_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    coverage = profile["coverage"]

    return {
        "name": profile["name"],
        "topics": [t["name"] for t in coverage["top_topics"][:3]],
        "entities": [e["name"] for e in coverage["top_entities"][:3]],
        "countries": [c["name"] for c in coverage["top_countries"][:3]],
        "languages": [l["code"] for l in coverage["top_languages"][:3]],
    }

fp = fingerprint(88123)
print(f"{fp['name']}")
print(f"  Beats:      {', '.join(fp['topics'])}")
print(f"  Covers:     {', '.join(fp['entities'])}")
print(f"  Regions:    {', '.join(fp['countries'])}")
print(f"  Languages:  {', '.join(fp['languages'])}")
```

### Read the Sentiment Lean

```python
import requests

API_KEY = "YOUR_API_KEY"

response = requests.get(
    "https://api.apitube.io/v1/journalists/88123",
    params={"api_key": "YOUR_API_KEY"},
)
response.raise_for_status()
profile = response.json()
sentiment = profile["coverage"]["sentiment"]

total = sentiment["positive"] + sentiment["neutral"] + sentiment["negative"] or 1

print(f"Tone profile for {profile['name']}:\n")
for label in ["positive", "neutral", "negative"]:
    share = sentiment[label] / total * 100
    bar = "#" * int(share / 2)
    print(f"  {label:>8}: {sentiment[label]:>6} ({share:5.1f}%) {bar}")

lean = max(sentiment, key=sentiment.get)
print(f"\nTonal lean: {lean}")
```

### Compare Several Journalists by Topic

```python
import requests

API_KEY = "YOUR_API_KEY"

def topic_share(journalist_id, topic_id):
    response = requests.get(
        f"https://api.apitube.io/v1/journalists/{journalist_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    coverage = response.json()["coverage"]
    total = sum(t["count"] for t in coverage["top_topics"]) or 1

    for topic in coverage["top_topics"]:
        if topic["id"] == topic_id:
            return topic["count"] / total
    return 0.0

JOURNALISTS = {88123: "Jane Doe", 88200: "John Roe", 88451: "Mara Vance"}
TOPIC = "politics"

print(f"Who covers '{TOPIC}' most heavily?\n")
ranked = sorted(
    ((name, topic_share(jid, TOPIC)) for jid, name in JOURNALISTS.items()),
    key=lambda x: x[1],
    reverse=True,
)

for name, share in ranked:
    bar = "#" * int(share * 50)
    print(f"  {name:<14} {share:6.1%} {bar}")

print(f"\nBest contact for a '{TOPIC}' story: {ranked[0][0]}")
```

---

## JavaScript

### Classify a Beat from Top Topics

```javascript
const API_KEY = "YOUR_API_KEY";

async function getCoverage(journalistId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/journalists/${journalistId}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const profile = await getCoverage(88123);
const topics = profile.coverage.top_topics;
const total = topics.reduce((sum, t) => sum + t.count, 0) || 1;

console.log(`Beat profile for ${profile.name}:\n`);
topics.forEach((topic) => {
  const share = (topic.count / total) * 100;
  const bar = "#".repeat(Math.round(share / 2));
  console.log(`  ${topic.name.padEnd(18)} ${String(topic.count).padStart(5)} (${share.toFixed(1).padStart(5)}%) ${bar}`);
});

const primary = topics[0];
const label = primary.count / total > 0.5 ? "specialist" : "generalist";
console.log(`\nPrimary beat: ${primary.name} (${label})`);
```

### Build an Expertise Fingerprint

```javascript
const API_KEY = "YOUR_API_KEY";

async function fingerprint(journalistId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/journalists/${journalistId}?${params}`);
  const profile = await response.json();
  const c = profile.coverage;

  return {
    name: profile.name,
    topics: c.top_topics.slice(0, 3).map((t) => t.name),
    entities: c.top_entities.slice(0, 3).map((e) => e.name),
    countries: c.top_countries.slice(0, 3).map((x) => x.name),
    languages: c.top_languages.slice(0, 3).map((l) => l.code),
  };
}

const fp = await fingerprint(88123);
console.log(fp.name);
console.log(`  Beats:      ${fp.topics.join(", ")}`);
console.log(`  Covers:     ${fp.entities.join(", ")}`);
console.log(`  Regions:    ${fp.countries.join(", ")}`);
console.log(`  Languages:  ${fp.languages.join(", ")}`);
```

### Read the Sentiment Lean

```javascript
const API_KEY = "YOUR_API_KEY";

const params = new URLSearchParams({ api_key: API_KEY });
const response = await fetch(`https://api.apitube.io/v1/journalists/88123?${params}`);
const profile = await response.json();
const sentiment = profile.coverage.sentiment;

const total = sentiment.positive + sentiment.neutral + sentiment.negative || 1;

console.log(`Tone profile for ${profile.name}:\n`);
["positive", "neutral", "negative"].forEach((label) => {
  const share = (sentiment[label] / total) * 100;
  const bar = "#".repeat(Math.round(share / 2));
  console.log(`  ${label.padStart(8)}: ${String(sentiment[label]).padStart(6)} (${share.toFixed(1).padStart(5)}%) ${bar}`);
});

const lean = ["positive", "neutral", "negative"].reduce((a, b) =>
  sentiment[b] > sentiment[a] ? b : a
);
console.log(`\nTonal lean: ${lean}`);
```

### Compare Several Journalists by Topic

```javascript
const API_KEY = "YOUR_API_KEY";

async function topicShare(journalistId, topicId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/journalists/${journalistId}?${params}`);
  const coverage = (await response.json()).coverage;
  const total = coverage.top_topics.reduce((s, t) => s + t.count, 0) || 1;

  const match = coverage.top_topics.find((t) => t.id === topicId);
  return match ? match.count / total : 0;
}

const JOURNALISTS = { 88123: "Jane Doe", 88200: "John Roe", 88451: "Mara Vance" };
const TOPIC = "politics";

console.log(`Who covers '${TOPIC}' most heavily?\n`);

const ranked = (
  await Promise.all(
    Object.entries(JOURNALISTS).map(async ([jid, name]) => ({
      name,
      share: await topicShare(jid, TOPIC),
    }))
  )
).sort((a, b) => b.share - a.share);

ranked.forEach(({ name, share }) => {
  const bar = "#".repeat(Math.round(share * 50));
  console.log(`  ${name.padEnd(14)} ${(share * 100).toFixed(1).padStart(5)}% ${bar}`);
});

console.log(`\nBest contact for a '${TOPIC}' story: ${ranked[0].name}`);
```

---

## PHP

### Classify a Beat from Top Topics

```php
<?php

$apiKey = "YOUR_API_KEY";

function getCoverage(int $journalistId): array
{
    global $apiKey;

    $query = http_build_query(["api_key" => $apiKey]);
    return json_decode(file_get_contents(
        "https://api.apitube.io/v1/journalists/{$journalistId}?{$query}"
    ), true);
}

$profile = getCoverage(88123);
$topics  = $profile["coverage"]["top_topics"];
$total   = array_sum(array_column($topics, "count")) ?: 1;

echo "Beat profile for {$profile['name']}:\n\n";
foreach ($topics as $topic) {
    $share = $topic["count"] / $total * 100;
    $bar   = str_repeat("#", (int) ($share / 2));
    printf("  %-18s %5d (%5.1f%%) %s\n", $topic["name"], $topic["count"], $share, $bar);
}

$primary = $topics[0];
$label   = ($primary["count"] / $total > 0.5) ? "specialist" : "generalist";
printf("\nPrimary beat: %s (%s)\n", $primary["name"], $label);
```

### Build an Expertise Fingerprint

```php
<?php

$apiKey = "YOUR_API_KEY";

function fingerprint(int $journalistId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/journalists/{$journalistId}?{$query}"
    ), true);
    $c = $profile["coverage"];

    return [
        "name"      => $profile["name"],
        "topics"    => array_column(array_slice($c["top_topics"], 0, 3), "name"),
        "entities"  => array_column(array_slice($c["top_entities"], 0, 3), "name"),
        "countries" => array_column(array_slice($c["top_countries"], 0, 3), "name"),
        "languages" => array_column(array_slice($c["top_languages"], 0, 3), "code"),
    ];
}

$fp = fingerprint(88123);
echo "{$fp['name']}\n";
echo "  Beats:      " . implode(", ", $fp["topics"]) . "\n";
echo "  Covers:     " . implode(", ", $fp["entities"]) . "\n";
echo "  Regions:    " . implode(", ", $fp["countries"]) . "\n";
echo "  Languages:  " . implode(", ", $fp["languages"]) . "\n";
```

### Read the Sentiment Lean

```php
<?php

$apiKey = "YOUR_API_KEY";

$query   = http_build_query(["api_key" => $apiKey]);
$profile = json_decode(file_get_contents(
    "https://api.apitube.io/v1/journalists/88123?{$query}"
), true);
$sentiment = $profile["coverage"]["sentiment"];

$total = $sentiment["positive"] + $sentiment["neutral"] + $sentiment["negative"] ?: 1;

echo "Tone profile for {$profile['name']}:\n\n";
foreach (["positive", "neutral", "negative"] as $label) {
    $share = $sentiment[$label] / $total * 100;
    $bar   = str_repeat("#", (int) ($share / 2));
    printf("  %8s: %6d (%5.1f%%) %s\n", $label, $sentiment[$label], $share, $bar);
}

$lean = array_keys($sentiment, max($sentiment))[0];
echo "\nTonal lean: {$lean}\n";
```

### Compare Several Journalists by Topic

```php
<?php

$apiKey = "YOUR_API_KEY";

function topicShare(int $journalistId, string $topicId): float
{
    global $apiKey;

    $query    = http_build_query(["api_key" => $apiKey]);
    $coverage = json_decode(file_get_contents(
        "https://api.apitube.io/v1/journalists/{$journalistId}?{$query}"
    ), true)["coverage"];

    $total = array_sum(array_column($coverage["top_topics"], "count")) ?: 1;

    foreach ($coverage["top_topics"] as $topic) {
        if ($topic["id"] === $topicId) {
            return $topic["count"] / $total;
        }
    }
    return 0.0;
}

$journalists = [88123 => "Jane Doe", 88200 => "John Roe", 88451 => "Mara Vance"];
$topic       = "politics";

echo "Who covers '{$topic}' most heavily?\n\n";

$ranked = [];
foreach ($journalists as $jid => $name) {
    $ranked[] = ["name" => $name, "share" => topicShare($jid, $topic)];
}

usort($ranked, fn($a, $b) => $b["share"] <=> $a["share"]);

foreach ($ranked as $row) {
    $bar = str_repeat("#", (int) ($row["share"] * 50));
    printf("  %-14s %5.1f%% %s\n", $row["name"], $row["share"] * 100, $bar);
}

echo "\nBest contact for a '{$topic}' story: {$ranked[0]['name']}\n";
```
