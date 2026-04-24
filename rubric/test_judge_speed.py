"""
Judge model speed & cost benchmark.
Tests claude-sonnet-4-6 vs gemini-3-flash-preview (with/without thinking).
Usage: python test_judge_speed.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

URL = os.getenv("LLM_API_URL")
TOKEN = os.getenv("LLM_TOKEN")
APP = os.getenv("LLM_APP", "quark_gen")
QUOTA_ID = os.getenv("LLM_QUOTA_ID", "")
USER_ID = os.getenv("LLM_USER_ID", "")
ACCESS_KEY = os.getenv("LLM_ACCESS_KEY", "")

HEADERS = {"content-type": "application/json", "token": TOKEN}

# ── realistic judge prompt ──────────────────────────────────────────────────
RUBRICS_TEXT = """1. Final Answer Correctness: Score 1 iff the final mathematical answer (the value or expression contained within the last \\boxed{}) is equivalent to the ground truth provided in the task; otherwise 0.
2. Boxed Format Adherence: Score 1 iff the response contains exactly one final answer enclosed within a \\boxed{} command; otherwise 0.
3. Internal Logical Consistency: Score 1 iff the reasoning steps are free of internal contradictions and do not contain mathematically invalid operations; otherwise 0.
4. Solution Completeness: Score 1 iff the reasoning process reaches a definitive numerical or symbolic conclusion and does not terminate abruptly mid-thought or mid-calculation; otherwise 0.
5. Conciseness and Non-Redundancy: Score 1 iff the thinking process avoids repeating the same logical cycles or performing identical calculations multiple times without progress; otherwise 0."""

ROLLOUT_TEXT = """Prompt:
Find the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square.

Response:
<think>
Let me denote n^2 - 19n + 99 = k^2 for some non-negative integer k.

Rearranging: n^2 - 19n + 99 - k^2 = 0

This is a quadratic in n. Let me complete the square.

n^2 - 19n + (19/2)^2 = k^2 - 99 + (19/2)^2
(n - 19/2)^2 = k^2 - 99 + 361/4
(n - 19/2)^2 = k^2 + (361 - 396)/4
(n - 19/2)^2 = k^2 - 35/4

Multiply both sides by 4:
(2n - 19)^2 = 4k^2 - 35
(2n - 19)^2 - (2k)^2 = -35
(2n - 19 - 2k)(2n - 19 + 2k) = -35

Let a = 2n - 19 - 2k and b = 2n - 19 + 2k, so ab = -35.

Also a + b = 2(2n - 19) = 4n - 38, which is even.
And b - a = 4k, which is also even.

Since ab = -35 and a+b must be even, both a and b must have the same parity (both even or both odd). Since 35 = 5 × 7, the factor pairs of -35 are:
(-1, 35), (1, -35), (-5, 7), (5, -7), (-7, 5), (7, -5), (-35, 1), (35, -1)

For a and b to have same parity, we need both odd (since -35 is odd, we can't have both even).
All pairs above have both odd factors. Good.

For each pair (a, b):
n = (a + b + 38) / 4

(-1, 35): a+b = 34, n = 72/4 = 18, k = (b-a)/4 = 36/4 = 9. Check: 18^2 - 19(18) + 99 = 324 - 342 + 99 = 81 = 9^2. ✓
(1, -35): a+b = -34, n = 4/4 = 1, k = (-35-1)/4 = -9. |k|=9. Check: 1 - 19 + 99 = 81 = 9^2. ✓
(-5, 7): a+b = 2, n = 40/4 = 10, k = (7-(-5))/4 = 3. Check: 100 - 190 + 99 = 9 = 3^2. ✓
(5, -7): a+b = -2, n = 36/4 = 9, k = (-7-5)/4 = -3. Check: 81 - 171 + 99 = 9 = 3^2. ✓
(-7, 5): same as (-5,7) rearranged → same n=10 or gives n that repeats.

Wait let me redo: (a,b) with ab=-35:
(-7, 5): n = (-7+5+38)/4 = 36/4 = 9. Same as (5,-7). k = (5-(-7))/4 = 3.
(7, -5): n = (7-5+38)/4 = 40/4 = 10. Same as (-5,7).
(-35, 1): n = (-35+1+38)/4 = 4/4 = 1. Same as (1,-35).
(35, -1): n = (35-1+38)/4 = 72/4 = 18. Same as (-1,35).

So distinct positive integer solutions: n = 1, 9, 10, 18.

Sum = 1 + 9 + 10 + 18 = 38.
</think>

The sum of all positive integers n is $\\boxed{38}$."""

SYSTEM_PROMPT = "You are a strict binary judge for RL training."


def build_judge_prompt(rubrics: str, rollout: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), "judge.txt")) as f:
        template = f.read()
    return template.replace("[INSERT RUBRICS HERE]", rubrics).replace("[INSERT CANDIDATE ROLLOUT HERE]", rollout)


JUDGE_PROMPT = build_judge_prompt(RUBRICS_TEXT, ROLLOUT_TEXT)
print(f"Judge prompt length: {len(JUDGE_PROMPT)} chars\n")


def call_api(model: str, extra_params: dict = None, tag: str = "") -> dict:
    params = {"temperature": 0.0, "max_tokens": 64}
    if extra_params:
        params.update(extra_params)

    data = {
        "business_unit": "",
        "app": APP,
        "quota_id": QUOTA_ID,
        "model": model,
        "user_id": USER_ID,
        "access_key": ACCESS_KEY,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_PROMPT},
        ],
        "params": params,
        "cache": 0,
        "tag": tag or model,
    }

    t0 = time.time()
    try:
        resp = requests.post(URL, headers=HEADERS,
                             data=json.dumps(data, separators=(",", ":"), ensure_ascii=False),
                             verify=False, timeout=60)
        resp.raise_for_status()
        elapsed = time.time() - t0
        rdata = resp.json()

        # extract content
        content = None
        try:
            content = rdata["data"]["data"]["message"]  # gemini format
        except Exception:
            pass
        if content is None:
            try:
                c = rdata["data"]["completion"]["choices"][0]["message"]["content"]
                content = c if isinstance(c, str) else "".join(p.get("text","") for p in c)
            except Exception:
                pass

        # extract usage
        usage = {}
        try:
            usage = rdata["data"]["completion"]["usage"]
        except Exception:
            pass
        try:
            cost = rdata["data"]["cost_info"]
        except Exception:
            cost = {}
        try:
            thinking_tokens = rdata["data"]["data"].get("thinking_tokens", 0) or usage.get("thoughtsTokenCount", 0) or usage.get("reasoning_tokens", 0) or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        except Exception:
            thinking_tokens = 0

        return {
            "ok": True, "elapsed": elapsed, "content": content,
            "thinking_tokens": thinking_tokens,
            "prompt_tokens": usage.get("promptTokenCount") or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("candidatesTokenCount") or usage.get("completion_tokens", 0),
            "cost": cost.get("total_cost", 0),
        }
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "error": str(e)}


def run_n(model: str, extra_params: dict, n: int, tag: str) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(call_api, model, extra_params, tag) for _ in range(n)]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def report(label: str, results: list):
    ok = [r for r in results if r["ok"]]
    fail = len(results) - len(ok)
    if not ok:
        print(f"  {label}: ALL FAILED")
        return
    times = [r["elapsed"] for r in ok]
    think = [r["thinking_tokens"] for r in ok]
    out = [r["output_tokens"] for r in ok]
    costs = [r["cost"] for r in ok]
    contents = [r.get("content", "") for r in ok]
    print(f"\n  [{label}]")
    print(f"    calls: {len(ok)} ok, {fail} failed")
    print(f"    latency: avg={sum(times)/len(times):.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")
    print(f"    thinking tokens: avg={sum(think)/len(think):.0f}  max={max(think)}")
    print(f"    output tokens:   avg={sum(out)/len(out):.1f}")
    print(f"    cost/call (¥):   avg={sum(costs)/len(costs):.5f}  total={sum(costs):.4f}")
    print(f"    sample output:   {repr(contents[0][:80]) if contents[0] else 'None'}")


N = 8  # 并发调用数，模拟真实压力

print("=" * 60)
print(f"Testing with {N} concurrent calls each config")
print("=" * 60)

configs = [
    ("gemini-3-flash | default (temp=0)", "gemini-3-flash-preview", {}),
    ("gemini-3-flash | budget_tokens=0", "gemini-3-flash-preview", {"budget_tokens": 0}),
    ("gemini-3-flash | thinking=disabled", "gemini-3-flash-preview", {"thinking": {"type": "disabled"}}),
    ("gemini-3-flash | enable_thinking=false", "gemini-3-flash-preview", {"enable_thinking": False}),
    ("claude-sonnet-4-6 | default", "claude-sonnet-4-6", {}),
]

for label, model, extra in configs:
    print(f"\nRunning: {label} ...")
    results = run_n(model, extra, N, label)
    report(label, results)

print("\n" + "=" * 60)
print("Done.")
