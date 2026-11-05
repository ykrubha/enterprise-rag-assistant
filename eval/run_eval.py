"""Evaluation pipeline: retrieval, answers, and abstention.

    python -m eval.run_eval --split dev                      # develop without touching the test set
    python -m eval.run_eval --split test                     # final scores, with extractive answers
    python -m eval.run_eval --split test --generator openai --model gemini-3.5-flash-lite

Decline thresholds are tuned on the dev split only, then frozen for the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kb_assistant.assistant import (
    AbstainPolicy, Assistant, ExtractiveGenerator, GenerationError, LLMGenerator,
)
from kb_assistant.index import HybridIndex, MiniLmEmbedder, MsMarcoReranker

from .questions import DEV, Q

MODES = ("bm25", "dense", "hybrid", "rerank")
MODE_NAMES = {"bm25": "BM25", "dense": "Embeddings", "hybrid": "Hybrid (RRF)", "rerank": "Hybrid + reranker"}


def retrieval_row(index: HybridIndex, question: Q, mode: str, k: int = 10) -> dict:
    hits = index.search(question.text, k=k, mode=mode)
    relevant = [hit.chunk.doc_id in question.gold for hit in hits]
    first = next((rank for rank, flag in enumerate(relevant, start=1) if flag), None)
    # A document can match through several chunks, so nDCG counts only its first hit.
    gains = []
    seen = set()
    for hit in hits:
        is_new = hit.chunk.doc_id in question.gold and hit.chunk.doc_id not in seen
        gains.append(1.0 if is_new else 0.0)
        if is_new:
            seen.add(hit.chunk.doc_id)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(question.gold), k)))
    fact = re.compile(question.fact, re.IGNORECASE)
    return {
        "hit1": float(first is not None and first <= 1),
        "hit3": float(first is not None and first <= 3),
        "hit5": float(first is not None and first <= 5),
        "mrr10": 1.0 / first if first else 0.0,
        "ndcg10": dcg / ideal,
        "fact_in_top5": float(any(fact.search(h.chunk.indexed_text) for h in hits[:5])),
    }


def bootstrap_ci(values: list[float], samples: int = 2000, seed: int = 7) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), (samples, len(data)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_retrieval(index: HybridIndex, questions: list[Q]) -> dict:
    answerable = [q for q in questions if q.answerable]
    report = {}
    for mode in MODES:
        rows = [retrieval_row(index, q, mode) for q in answerable]
        summary = {m: round(statistics.mean(r[m] for r in rows), 4) for m in rows[0]}
        low, high = bootstrap_ci([r["mrr10"] for r in rows])
        summary["mrr10_ci95"] = [round(low, 3), round(high, 3)]
        report[mode] = summary
    return report


def tune_policy(index: HybridIndex, questions: list[Q]) -> tuple[AbstainPolicy, dict]:
    """Pick thresholds that maximize balanced accuracy on dev. Ties keep the more permissive one."""
    signals = []
    for question in questions:
        hits = index.search(question.text, k=5, mode="rerank")
        rerank = hits[0].score if hits else float("-inf")
        signals.append((question.answerable, rerank, index.term_coverage(question.text, hits)))

    positives = sum(1 for a, _, _ in signals if a)
    negatives = len(signals) - positives
    best, best_score = AbstainPolicy(-20.0, 0.0), (-1.0, 0.0, 0.0)
    for min_rerank in np.arange(-12.0, 8.01, 0.5):
        for min_coverage in np.arange(0.0, 1.001, 0.1):
            policy = AbstainPolicy(round(float(min_rerank), 1), round(float(min_coverage), 1))
            kept = sum(1 for a, r, c in signals if a and not policy.should_abstain(r, c))
            declined = sum(1 for a, r, c in signals if not a and policy.should_abstain(r, c))
            balanced = (kept / positives + declined / negatives) / 2
            score = (balanced, -policy.min_rerank, -policy.min_coverage)
            if score > best_score:
                best, best_score = policy, score
    return best, {"dev_balanced_accuracy": round(best_score[0], 4)}


def evaluate_answers(assistant: Assistant, questions: list[Q], pause_s: float = 0.0) -> dict:
    rows = []
    for number, question in enumerate(questions, start=1):
        answer = assistant.ask(question.text)
        if pause_s and not answer.abstained:
            time.sleep(pause_s)
        outcome = f"declined ({answer.decline_reason})" if answer.abstained else "answered"
        print(f"{number}/{len(questions)} {outcome}: {question.text[:70]}", flush=True)
        if answer.abstained and question.answerable and answer.raw_reply is not None:
            print(f"    raw reply: {answer.raw_reply[:300]!r}", flush=True)
        row = {
            "question": question.text, "answerable": question.answerable,
            "abstained": answer.abstained, "answer": answer.answer,
            "cited_docs": [c.doc_id for c in answer.citations],
            "confidence": answer.confidence, "latency_ms": answer.latency_ms,
            "decline_reason": answer.decline_reason, "raw_reply": answer.raw_reply,
        }
        if question.answerable:
            row["gold"] = list(question.gold)
            row["fact_found"] = (not answer.abstained
                                 and bool(re.search(question.fact, answer.answer, re.IGNORECASE)))
            row["cites_gold"] = any(d in question.gold for d in row["cited_docs"])
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    cited = [d in r["gold"] for r in answered for d in r["cited_docs"]]
    all_answered = [r for r in rows if not r["abstained"]]
    return {
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "answer_rate": round(len(answered) / len(answerable), 4),
        "key_fact_recall": round(sum(r["fact_found"] for r in answerable) / len(answerable), 4),
        "key_fact_recall_when_answered": round(
            sum(r["fact_found"] for r in answered) / max(len(answered), 1), 4),
        "cites_gold_doc_rate": round(sum(r["cites_gold"] for r in answered) / max(len(answered), 1), 4),
        "citation_precision": round(sum(cited) / max(len(cited), 1), 4),
        "abstain_recall": round(sum(r["abstained"] for r in unanswerable) / len(unanswerable), 4),
        "abstain_precision": round(
            sum(r["abstained"] for r in unanswerable)
            / max(sum(r["abstained"] for r in rows), 1), 4),
        "latency_ms_p50": round(float(np.percentile([r["latency_ms"] for r in rows], 50)), 1),
        "latency_ms_p95": round(float(np.percentile([r["latency_ms"] for r in rows], 95)), 1),
        "answered_total": len(all_answered),
        "decline_reasons_on_answerable": dict(Counter(r["decline_reason"] for r in answerable if r["abstained"])),
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    def retrieval_table(data: dict) -> list[str]:
        lines = ["| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 (95% CI) | nDCG@10 | Key fact in top 5 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for mode in MODES:
            s = data[mode]
            name = f"**{MODE_NAMES[mode]}**" if mode == "rerank" else MODE_NAMES[mode]
            lines.append(
                f"| {name} | {s['hit1']:.2f} | {s['hit3']:.2f} | {s['hit5']:.2f} | "
                f"{s['mrr10']:.2f} ({s['mrr10_ci95'][0]:.2f}-{s['mrr10_ci95'][1]:.2f}) | "
                f"{s['ndcg10']:.2f} | {s['fact_in_top5']:.2f} |")
        return lines

    def answer_table(data: dict) -> list[str]:
        return [
            "| Metric | Score |", "|---|---:|",
            f"| Answer rate on answerable questions | {data['answer_rate']:.2f} |",
            f"| Key fact recall, all answerable questions | {data['key_fact_recall']:.2f} |",
            f"| Key fact recall, when it answered | {data['key_fact_recall_when_answered']:.2f} |",
            f"| Answers citing a correct document | {data['cites_gold_doc_rate']:.2f} |",
            f"| Citation precision | {data['citation_precision']:.2f} |",
            f"| Unanswerable questions correctly declined | {data['abstain_recall']:.2f} |",
            f"| Declines that were correct | {data['abstain_precision']:.2f} |",
            f"| Latency p50 / p95 (ms, {report['generator']}) | "
            f"{data['latency_ms_p50']} / {data['latency_ms_p95']} |",
            "",
            f"Why answerable questions were declined: {data['decline_reasons_on_answerable']}",
        ]

    test = report[report["split"]]
    return "\n".join([
        f"Corpus: {report['documents']} documents, {report['chunks']} chunks. "
        f"Split `{report['split']}`: {test['answers']['answerable_questions']} answerable and "
        f"{test['answers']['unanswerable_questions']} unanswerable questions.",
        "",
        "**Retrieval (document level)**",
        "",
        *(retrieval_table(test["retrieval"]) if test["retrieval"] else ["Not rerun for this generator."]),
        "",
        f"**Answers (`{report['generator']}` generator, reranked retrieval)**",
        "",
        *answer_table(test["answers"]),
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--out", default="results")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--generator", default="extractive", choices=["extractive", "anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-seconds-between-calls", type=float, default=0.0,
                        help="Pace LLM requests to stay under free-tier rate limits.")
    parser.add_argument("--retrieval", action=argparse.BooleanOptionalAction, default=None,
                        help="Also compare retrieval methods. On by default for the extractive generator.")
    parser.add_argument("--fail-under-hit5", type=float, default=None)
    args = parser.parse_args()

    index = HybridIndex.load(args.index_dir, MiniLmEmbedder(), MsMarcoReranker())
    questions = DEV
    if args.split == "test":
        from .test_questions import TEST
        questions = TEST
    policy, tuning = tune_policy(index, DEV)
    print("Thresholds tuned on dev:", policy, tuning, flush=True)
    generator = (ExtractiveGenerator() if args.generator == "extractive"
                 else LLMGenerator(provider=args.generator, model=args.model))
    assistant = Assistant(index, generator=generator, policy=policy)

    name = "extractive" if args.generator == "extractive" else generator.model
    compare = args.retrieval if args.retrieval is not None else args.generator == "extractive"
    try:
        answers = evaluate_answers(assistant, questions, args.min_seconds_between_calls)
    except GenerationError as err:
        print(f"\nStopped, and no score was written. The LLM provider failed: {err}")
        return 1
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "generator": name,
        "documents": len({c.doc_id for c in index.chunks}),
        "chunks": len(index.chunks),
        "abstain_policy": {"min_rerank": policy.min_rerank, "min_coverage": policy.min_coverage},
        "tuning": tuning,
        args.split: {"answers": answers,
                     "retrieval": evaluate_retrieval(index, questions) if compare else None},
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = args.split + ("" if args.generator == "extractive" else "_llm")
    (out / f"eval_{suffix}.json").write_text(json.dumps(report, indent=2))
    (out / f"eval_{suffix}.md").write_text(to_markdown(report))
    policy_path = Path(args.index_dir) / "abstain_policy.json"
    policy_path.write_text(json.dumps(report["abstain_policy"]))
    print(to_markdown(report))

    if compare and args.fail_under_hit5 is not None:
        hit5 = report[args.split]["retrieval"]["rerank"]["hit5"]
        if hit5 < args.fail_under_hit5:
            print(f"FAIL: Hit@5 {hit5} is below {args.fail_under_hit5}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
