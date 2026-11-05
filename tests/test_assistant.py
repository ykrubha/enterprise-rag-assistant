"""Tests with small fake models, so they need no downloads and run in seconds."""

import json
import re

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import kb_assistant.assistant as assistant_module
from kb_assistant.api import create_app
from kb_assistant.assistant import (
    AbstainPolicy, Assistant, ExtractiveGenerator, GenerationError, LLMGenerator, NOT_FOUND,
    build_user_prompt, normalize_citations, post_with_retry,
)
from kb_assistant.index import HybridIndex, tokenize
from kb_assistant.ingest import (
    Chunk, clean_markup, doc_url, load_chunks, parse_front_matter, split_sections, split_sentences,
    window_words,
)


class FakeEmbedder:
    """Hashes words into a small vector, so texts that share words end up similar."""

    def embed(self, texts):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, sum(map(ord, word)) % 64] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)


class FakeReranker:
    """Scores by shared words: 2 points each, minus 3, so unrelated text scores below zero."""

    def score(self, query, texts):
        terms = set(tokenize(query))
        return np.array([2.0 * len(terms & set(tokenize(text))) - 3.0 for text in texts])


def chunk(doc, section, text, category="storage"):
    return Chunk(f"{doc}#0", doc, doc.split(".")[0].title(), section, text,
                 f"https://example.com/{doc}", category)


CHUNKS = [
    chunk("volumes.md", "emptyDir", "An emptyDir volume provides scratch space. It vanishes when the pod is removed from the node."),
    chunk("secrets.md", "Encryption", "Secrets are stored unencrypted in etcd by default. Enable encryption at rest to protect them.", "security"),
    chunk("service.md", "NodePort", "A NodePort service opens a fixed port on every node. The default range is 30000-32767.", "networking"),
]


@pytest.fixture
def index():
    return HybridIndex(CHUNKS, FakeEmbedder(), reranker=FakeReranker())


def permissive():
    return AbstainPolicy(min_rerank=-100.0, min_coverage=0.0)


class Scripted:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, question, hits, index):
        return self.reply


# --- Ingestion ---

def test_shortcodes_links_and_html_become_plain_text():
    raw = ('See {{< glossary_tooltip text="Pods" term_id="pod" >}} and [the docs](http://x.io). '
           '{{< feature-state for_k8s_version="v1.20" state="stable" >}} <b>Bold</b> <!-- hidden -->')
    cleaned = clean_markup(raw)
    assert "Pods" in cleaned and "the docs" in cleaned and "Bold" in cleaned
    for leftover in ("{{", "http://x.io", "<b>", "hidden", "term_id"):
        assert leftover not in cleaned


def test_front_matter_is_parsed_and_removed():
    meta, body = parse_front_matter("---\ntitle: Volumes\nweight: 10\n---\nBody text")
    assert meta["title"] == "Volumes" and body == "Body text"
    assert parse_front_matter("No front matter")[0] == {}


def test_sections_track_nested_headings_and_ignore_code_fences():
    body = "Intro text\n\n## Top\nA\n\n### Inner\nB\n\n```\n# not a heading\n```\n\n## Next\nC"
    sections = dict(split_sections(body))
    assert sections[""] == "Intro text"
    assert "Top" in sections and "Top > Inner" in sections and "Next" in sections
    assert "# not a heading" in sections["Top > Inner"]


def test_windows_respect_the_limit_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    windows = window_words(text, max_words=50, overlap_words=14)
    assert len(windows) > 1
    assert all(len(w.split()) <= 50 for w in windows)
    assert windows[0].split(". ")[-1] in windows[1]
    assert "Sentence number 39" in windows[-1]


def test_one_giant_sentence_is_still_split():
    windows = window_words(" ".join(["word"] * 500), max_words=100, overlap_words=20)
    assert all(len(w.split()) <= 100 for w in windows)
    assert len(windows) >= 5


def test_sentence_splitting():
    assert split_sentences("One two. Three four!\n\nFive six?") == ["One two.", "Three four!", "Five six?"]


def test_urls_drop_index_pages():
    assert doc_url("storage/volumes.md", "https://k8s.io/docs") == "https://k8s.io/docs/storage/volumes/"
    assert doc_url("storage/_index.md", "https://k8s.io/docs") == "https://k8s.io/docs/storage/"


def test_load_chunks_from_a_folder(tmp_path):
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage" / "volumes.md").write_text(
        "---\ntitle: Volumes\n---\n\n## emptyDir\n\nAn emptyDir volume gives scratch space for containers in a pod.\n")
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].title == "Volumes" and chunks[0].section == "emptyDir"
    assert chunks[0].category == "storage"
    assert "Volumes > emptyDir" in chunks[0].indexed_text


# --- Retrieval ---

def test_tokenize_splits_camel_case_and_removes_stopwords():
    tokens = tokenize("The NetworkPolicy is for pods")
    assert {"network", "policy", "networkpolicy", "pod"} <= set(tokens)
    assert "the" not in tokens


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_every_mode_finds_the_obvious_document(index, mode):
    hits = index.search("NodePort fixed port range", k=2, mode=mode)
    assert hits[0].chunk.doc_id == "service.md"


def test_unknown_mode_is_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")


def test_rerank_needs_a_reranker():
    with pytest.raises(ValueError):
        HybridIndex(CHUNKS, FakeEmbedder()).search("x", mode="rerank")


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_category_filter_is_never_crossed(index, mode):
    for hit in index.search("secrets encryption etcd", k=3, mode=mode, categories=["storage"]):
        assert hit.chunk.category == "storage"


def test_coverage_is_low_for_terms_the_corpus_lacks(index):
    hits = index.search("emptyDir scratch volume", k=2)
    assert index.term_coverage("emptyDir scratch volume", hits) == pytest.approx(1.0)
    assert index.term_coverage("parental leave benefits", hits) < 0.2


def test_index_saves_and_loads(index, tmp_path):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, FakeEmbedder(), FakeReranker())
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert np.allclose(loaded.vectors, index.vectors)


# --- Answers and citations ---

def test_extractive_answer_quotes_the_source_with_a_citation(index):
    answer = Assistant(index, ExtractiveGenerator(max_sentences=1), permissive()).ask("default range")
    assert not answer.abstained
    assert "30000-32767" in answer.answer and answer.answer.endswith("[1]")
    assert answer.citations[0].doc_id == "service.md"


def test_low_support_questions_are_declined_without_calling_the_generator(index):
    class Fails:
        def generate(self, *args):
            raise AssertionError("The generator should not run.")

    answer = Assistant(index, Fails(), AbstainPolicy(min_rerank=5.0)).ask("parental leave policy")
    assert answer.abstained and answer.answer == NOT_FOUND and answer.citations == []


def test_citations_are_renumbered_in_order_of_use(index):
    hits = index.search("volume secrets", k=3)
    answer = Assistant(index, Scripted("First [3]. Second [1]. Third [3]."), permissive()).ask("volume secrets")
    assert answer.answer == "First [1]. Second [2]. Third [1]."
    assert [c.number for c in answer.citations] == [1, 2]
    assert answer.citations[0].doc_id == hits[2].chunk.doc_id
    assert answer.citations[1].doc_id == hits[0].chunk.doc_id


@pytest.mark.parametrize("reply", [
    "An answer with no citation.",
    "An answer citing a source that doesn't exist [9].",
    "Citing source zero [0].",
    "INSUFFICIENT",
    "INSUFFICIENT\n",
])
def test_untraceable_or_refused_answers_are_never_shown(index, reply):
    answer = Assistant(index, Scripted(reply), permissive()).ask("volume scratch")
    assert answer.abstained and answer.answer == NOT_FOUND


@pytest.mark.parametrize("reply,reason", [
    ("", "empty_reply"),
    ("INSUFFICIENT", "model_declined"),
    ("No citation here.", "no_citation"),
    ("A bad citation [9].", "invalid_citation"),
])
def test_decline_reason_and_raw_reply_are_recorded(index, reply, reason):
    answer = Assistant(index, Scripted(reply), permissive()).ask("volume scratch")
    assert answer.abstained and answer.decline_reason == reason and answer.raw_reply == reply


def test_low_support_decline_has_a_reason(index):
    answer = Assistant(index, Scripted("x [1]."), AbstainPolicy(min_rerank=99.0)).ask("volume scratch")
    assert answer.decline_reason == "low_support" and answer.raw_reply is None


def test_api_does_not_expose_raw_replies(index):
    app = create_app(Assistant(index, Scripted("Answer [1]."), permissive()), api_key="")
    with TestClient(app) as test_client:
        body = test_client.post("/ask", json={"question": "volume scratch"}).json()
    assert "raw_reply" not in body and body["answer"] == "Answer [1]."


@pytest.mark.parametrize("reply,expected", [
    ("Fact [1, 2].", "Fact [1][2]."),
    ("Fact [Source 1].", "Fact [1]."),
    ("Fact [sources 1, 3].", "Fact [1][3]."),
    ("Fact [ 2 ].", "Fact [2]."),
    ("Fact [1][2].", "Fact [1][2]."),
    ("Use `items[0, 1]` here [1].", "Use `items[0, 1]` here [1]."),
    ("Not a citation [see above].", "Not a citation [see above]."),
])
def test_citation_forms_are_normalized(reply, expected):
    assert normalize_citations(reply) == expected


def test_comma_separated_citations_are_accepted_and_renumbered(index):
    answer = Assistant(index, Scripted("Fact one [2, 3]. Fact two [3]."), permissive()).ask("volume secrets")
    assert not answer.abstained
    assert answer.answer == "Fact one [1][2]. Fact two [2]."
    assert answer.raw_reply == "Fact one [2, 3]. Fact two [3]."


def test_numbers_inside_code_are_not_citations(index):
    answer = Assistant(index, Scripted("Use `items[9]` for this [1]."), permissive()).ask("volume")
    assert not answer.abstained
    assert "`items[9]`" in answer.answer
    assert len(answer.citations) == 1


def test_code_alone_does_not_count_as_a_citation(index):
    answer = Assistant(index, Scripted("Use `items[1]` for this."), permissive()).ask("volume")
    assert answer.abstained and answer.decline_reason == "no_citation"


def test_prompt_marks_sources_as_data(index):
    prompt = build_user_prompt("q", index.search("volume", k=1))
    assert '<source id="1"' in prompt and "</source>" in prompt and "Question: q" in prompt


# --- LLM adapters ---

def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_wire_format(index):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-api-key"]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Answer [1]."}]})

    generator = LLMGenerator("anthropic", "m", client(handler), api_key="k")
    assert generator.generate("q", index.search("volume", k=1), index) == "Answer [1]."
    assert seen["key"] == "k" and "system" in seen["body"]
    assert seen["body"]["messages"][0]["role"] == "user"


def test_openai_wire_format(index):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer [1]."}}]})

    generator = LLMGenerator("openai", "m", client(handler), api_key="k", base_url="http://x/v1/")
    assert generator.generate("q", index.search("volume", k=1), index) == "Answer [1]."
    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["body"]["messages"][0]["role"] == "system"


def test_retries_rate_limits_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(assistant_module.time, "sleep", sleeps.append)
    replies = [httpx.Response(429, headers={"retry-after": "30"}), httpx.Response(503),
               httpx.Response(200, json={"ok": True})]
    data = post_with_retry(client(lambda request: replies.pop(0)), "http://x", {}, {})
    assert data == {"ok": True}
    assert sleeps == [30.0, 4.0]


def test_bad_requests_fail_at_once(monkeypatch):
    monkeypatch.setattr(assistant_module.time, "sleep", lambda _: pytest.fail("Should not retry."))
    with pytest.raises(GenerationError, match="400"):
        post_with_retry(client(lambda request: httpx.Response(400, text="bad")), "http://x", {}, {})


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMGenerator("anthropic")


# --- API ---

@pytest.fixture
def api(index):
    app = create_app(Assistant(index, ExtractiveGenerator(), permissive()), api_key="")
    with TestClient(app) as test_client:
        yield test_client


def test_health(api):
    assert api.get("/health").json() == {"status": "ok", "chunks": 3, "documents": 3}


def test_ask_returns_an_answer_with_citations(api):
    body = api.post("/ask", json={"question": "What is the default NodePort range?"}).json()
    assert body["abstained"] is False
    assert any(c["doc_id"] == "service.md" for c in body["citations"])
    assert body["latency_ms"] >= 0


def test_ask_applies_the_category_filter(api):
    body = api.post("/ask", json={"question": "secrets encryption", "categories": ["storage"]}).json()
    assert all(c["doc_id"] == "volumes.md" for c in body["citations"])


def test_bad_input_is_rejected(api):
    assert api.post("/ask", json={"question": "x"}).status_code == 422
    assert api.post("/ask", json={"question": "valid question", "categories": ["nope"]}).status_code == 422
    assert api.post("/search", json={"question": "valid question", "mode": "magic"}).status_code == 422
    assert api.post("/search", json={"question": "valid question", "k": 1000}).status_code == 422


def test_search_returns_ranked_chunks(api):
    body = api.post("/search", json={"question": "emptyDir scratch", "k": 2}).json()
    assert body["results"][0]["chunk_id"] == "volumes.md#0"


def test_api_key_is_enforced(index):
    app = create_app(Assistant(index, ExtractiveGenerator(), permissive()), api_key="secret")
    with TestClient(app) as test_client:
        payload = {"question": "volume scratch"}
        assert test_client.post("/ask", json=payload).status_code == 401
        assert test_client.post("/ask", json=payload, headers={"x-api-key": "wrong"}).status_code == 401
        assert test_client.post("/ask", json=payload, headers={"x-api-key": "secret"}).status_code == 200
        assert test_client.get("/health").status_code == 200


def test_generation_failures_return_a_clean_error(index):
    class Fails:
        def generate(self, *args):
            raise GenerationError("secret provider detail")

    app = create_app(Assistant(index, Fails(), permissive()), api_key="")
    with TestClient(app) as test_client:
        response = test_client.post("/ask", json={"question": "volume scratch"})
    assert response.status_code == 502
    assert "secret" not in response.text


# --- Evaluation data ---

def test_questions_are_well_formed_and_splits_do_not_overlap():
    from eval.questions import DEV
    from eval.test_questions import TEST

    for question in DEV + TEST:
        if question.answerable:
            assert question.fact and re.compile(question.fact)
        else:
            assert not question.fact
    assert not {q.text for q in DEV} & {q.text for q in TEST}
