"""LangSmith experiment: how well does the matching prompt decide "fit / not fit"?

The dataset (evals/dataset.json) holds job + seeker pairs whose correct answer is known.
Each run scores every pair with Gemini and an automatic evaluator compares the decision
(score >= MATCH_MIN_SCORE) with the known answer. Results appear in LangSmith under
Datasets & Experiments -> job-matcher-fit.

Usage (from the job-matcher folder):
    .venv\\Scripts\\python -m evals.run_eval --prompt basic   # before: "just give a score"
    .venv\\Scripts\\python -m evals.run_eval --prompt rubric  # after: explicit rubric
"""
import argparse
import json
from pathlib import Path

from langsmith import Client

from app import config
from app.ai import PROMPTS, ask_ai, job_text, seeker_text

DATASET_NAME = "job-matcher-fit"
DATA_FILE = Path(__file__).with_name("dataset.json")


def ensure_dataset(client: Client) -> None:
    """Creates the dataset in LangSmith and uploads cases from dataset.json that it does not have yet."""
    rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        have = {(e.metadata or {}).get("case") for e in client.list_examples(dataset_id=ds.id)}
    else:
        ds = client.create_dataset(DATASET_NAME, description="Job / seeker pairs with a known fit decision")
        have = set()
    new = [r for r in rows if r["id"] not in have]
    if new:
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"job": r["job"], "seeker": r["seeker"]} for r in new],
            outputs=[{"fit": r["fit"], "note": r["note"]} for r in new],
            metadata=[{"case": r["id"]} for r in new],
        )
        print(f"Uploaded {len(new)} new examples to {DATASET_NAME}")


def make_target(prompt_version: str):
    def predict(inputs: dict) -> dict:
        job = {"field_other": "", "workplace_other": "", "age_min": None, "age_max": None, **inputs["job"]}
        seeker = {"id": 1, **inputs["seeker"]}
        matches = ask_ai(job_text(job), seeker_text(seeker), 0, prompt_version=prompt_version)
        m = next((m for m in matches if m.get("seeker_id") == 1), None)
        score = int(m["score"]) if m else 0
        return {"score": score, "fit": score >= config.MATCH_MIN_SCORE, "reason": m["reason"] if m else ""}
    return predict


# ---- Evaluators: compare the decision with the known answer ----

# A run that failed (e.g. Gemini unavailable) has no "fit" and counts as a wrong answer.

def correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "correct", "score": int(outputs.get("fit") == reference_outputs["fit"])}


def unsuitable_suggested(outputs: dict, reference_outputs: dict) -> dict:
    """1 = the model would show the employer someone who does not fit (the costly mistake)."""
    return {"key": "unsuitable_suggested",
            "score": int(outputs.get("fit") is True and not reference_outputs["fit"])}


def missed_fit(outputs: dict, reference_outputs: dict) -> dict:
    """1 = a suitable seeker was left out."""
    return {"key": "missed_fit", "score": int(reference_outputs["fit"] and outputs.get("fit") is False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    args = parser.parse_args()

    client = Client()
    ensure_dataset(client)
    results = client.evaluate(
        make_target(args.prompt),
        data=DATASET_NAME,
        evaluators=[correct, unsuitable_suggested, missed_fit],
        experiment_prefix=f"prompt-{args.prompt}",
        metadata={"prompt_version": args.prompt, "model": config.GEMINI_MODEL,
                  "min_score": config.MATCH_MIN_SCORE},
        max_concurrency=1,  # the Gemini free tier allows only a few requests per minute
    )

    totals = {"correct": 0, "unsuitable_suggested": 0, "missed_fit": 0}
    wrong = []
    n = failed = 0
    for r in results:
        n += 1
        for ev in r["evaluation_results"]["results"]:
            totals[ev.key] += ev.score or 0
        out, ref = r["run"].outputs or {}, r["example"].outputs
        if "fit" not in out:
            failed += 1
            wrong.append(f'  {r["example"].metadata.get("case")}: RUN FAILED - {r["run"].error}'[:200])
        elif out["fit"] != ref["fit"]:
            wrong.append(f'  {r["example"].metadata.get("case")}: expected fit={ref["fit"]}, '
                         f'got score {out.get("score")} - {ref["note"]}')
    print(f"\nprompt {args.prompt} on {n} examples ({config.GEMINI_MODEL}):")
    print(f"  correct:              {totals['correct']}/{n} ({totals['correct'] / max(n, 1):.0%})")
    print(f"  unsuitable suggested: {totals['unsuitable_suggested']}")
    print(f"  missed fits:          {totals['missed_fit']}")
    print(f"  failed runs:          {failed}")
    if wrong:
        print("Mistakes:")
        print("\n".join(wrong))


if __name__ == "__main__":
    main()
