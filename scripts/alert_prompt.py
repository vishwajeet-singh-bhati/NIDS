"""Turn a device-alarm evidence pack (android: files/evidence_packs.jsonl) into a prompt for a local Llama model,
and check the model's text before it is shown to the user.

    python scripts/alert_prompt.py --packs evidence_packs.jsonl [--index -1] [--ollama llama3.2] [--text "<model reply to check>"]

Without --ollama it only prints the prompt (paste it into any local Llama runner). With --ollama it calls a local
Ollama server (http://localhost:11434) and then runs the same checks on the reply.

Design: the model only REWRITES measured facts in simple words. It is told not to name a malware type, not to claim
certainty, and not to invent apps, data theft or causes. `check_alert` rejects replies that break those rules.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

SYSTEM = """You write short security alerts for an Android phone user who is not technical.
Use ONLY the facts in the JSON evidence below. Do not add facts, app names, causes, or outcomes that are not in it.
Rules:
- 3 to 4 short sentences, at most 80 words, plain everyday words, no jargon and no IP addresses.
- This is a WARNING, not a confirmed infection. Say the phone's network traffic looks like traffic seen from harmful apps.
- Mention, using the numbers in the evidence: how many suspicious connections, within about how many minutes, and to how many different servers.
- The evidence says malware_type is null: do NOT name a type of malware (no 'ransomware', 'adware', 'spyware', 'virus', etc.).
- Never say data was stolen, files were encrypted, money was lost, or that the phone is definitely infected.
- End with one safe, general step: review recently installed apps and run a trusted security scan; if unsure, turn off mobile data and Wi-Fi.
- Say that nothing was blocked and the check happened only on the phone."""

BANNED = [r"ransomware", r"adware", r"spyware", r"scareware", r"trojan", r"virus", r"botnet", r"stolen", r"steal", r"encrypted your",
          r"definitely", r"certainly", r"confirmed (infection|attack|malware)", r"hacked", r"your (money|bank)"]


def build_prompt(pack: dict) -> str:
    return SYSTEM + "\n\nEVIDENCE (JSON):\n" + json.dumps(pack, indent=1) + "\n\nWrite the alert now."


def check_alert(text: str, pack: dict) -> list[str]:
    """Return a list of problems (empty = acceptable)."""
    problems = []
    low = text.lower()
    for pat in BANNED:
        if re.search(pat, low):
            problems.append(f"contains forbidden claim: /{pat}/")
    if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text):
        problems.append("contains an IP address")
    if len(text.split()) > 95:
        problems.append("longer than ~80 words")
    ev = pack["evidence"]
    allowed = {str(ev["distinct_flow_alerts_logged"]), str(ev["flow_alerts_counted_by_rule"]), str(ev["distinct_servers"]),
               str(pack["rule"]["min_flow_alerts"]), str(int(pack["rule"]["within_seconds"])), str(int(round(pack["rule"]["within_seconds"] / 60))),
               str(max(1, int(round(ev["span_seconds"] / 60)))), "1", "2", "3", "4"}
    for n in re.findall(r"\b\d+\b", text):
        if n not in allowed:
            problems.append(f"number {n} is not in the evidence")
    if "not" not in low and "no " not in low:
        problems.append("does not say it is unconfirmed / nothing was blocked")
    return problems


def ollama(model: str, prompt: str) -> str:
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["response"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--ollama", default="")
    ap.add_argument("--text", default="", help="a model reply to check against the pack")
    a = ap.parse_args()
    packs = [json.loads(l) for l in open(a.packs) if l.strip()]
    if not packs:
        sys.exit("no evidence packs (no device alarm happened)")
    pack = packs[a.index]
    prompt = build_prompt(pack)
    print(prompt)
    reply = a.text or (ollama(a.ollama, prompt) if a.ollama else "")
    if reply:
        print("\n--- model reply ---\n" + reply)
        probs = check_alert(reply, pack)
        print("\nCHECK:", "OK" if not probs else "REJECTED: " + "; ".join(probs))


if __name__ == "__main__":
    main()
