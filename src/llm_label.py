import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM = "You are a bank customer support request classifier. Respond ONLY with valid JSON, without any explanations."

SCHEMA_TEXT = (
    'Return JSON: {"intent": "<exactly one class from the list>", '
    '"sentiment": "negative|neutral|positive", '
    '"confidence": <number from 0 to 1>}'
)

def build_prompt(text, labels, version, examples=None):
    parts = ["Classes:\n" + "\n".join(f"- {l}" for l in labels)]
    if version == "v2":
        parts.append("Rules: choose the most specific class; if you are unsure between two, choose the one that best describes the customer's main request.")
        parts.append("Examples:\n" + "\n".join(f'Request: "{t}" -> {{"intent": "{l}"}}' for t, l in examples))
    parts.append(SCHEMA_TEXT)
    parts.append(f'Request: "{text}"')
    return "\n\n".join(parts)

def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}

def ask(prompt, model=MODEL):
    resp = client.chat.completions.create(
        model=model, max_tokens=200, temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--out", default="results/llm_v1.csv")
    args = ap.parse_args()

    out_path = ROOT / args.out
    test = pd.read_csv(ROOT / "data" / "raw" / "test_sample.csv") 

    if out_path.exists():
        existing_out = pd.read_csv(out_path)
        existing_out = existing_out.dropna(subset=["pred_intent"]) 
        processed_texts = set(existing_out["text"])
        
        test = test[~test["text"].isin(processed_texts)]
        print(f"Found existing results. Skipping {len(processed_texts)} rows. Remaining: {len(test)}")
    else:
        existing_out = pd.DataFrame()

    if len(test) == 0:
        print("All rows have been processed!")
        return

    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    labels = sorted(train["label_text"].unique().tolist())

    examples = None
    if args.prompt == "v2":
        ex = train.groupby("label_text").head(1)
        examples = list(zip(ex["text"], ex["label_text"]))

    rows = []
    limit_reached = False
    
    for text in tqdm(test["text"]):
        prompt = build_prompt(text, labels, args.prompt, examples)
        
        try:
            raw = ask(prompt, model=args.model)
            d = parse_json(raw)
            
            retries = 0
            while not d.get("intent") and retries < 2:
                raw = ask(prompt, model=args.model)
                d = parse_json(raw)
                retries += 1

            rows.append({
                "pred_intent": d.get("intent"),
                "pred_sentiment": d.get("sentiment"),
                "confidence": d.get("confidence"),
                "raw": raw,
            })
            
        except RateLimitError:
            print("\n[!] Rate limit reached (429). Saving progress...")
            limit_reached = True
            break 
            
        except Exception as e:
            print(f"\n[!] Error: {e}. Saving progress...")
            limit_reached = True
            break

    if rows:
        valid_test = test.iloc[:len(rows)].reset_index(drop=True)
        new_out = pd.concat([valid_test, pd.DataFrame(rows)], axis=1)
        
        final_out = pd.concat([existing_out, new_out], ignore_index=True)
        final_out.to_csv(out_path, index=False)
        print(f"Saved {len(new_out)} new rows. Total in file: {len(final_out)}")
    
    if limit_reached:
        print("Please run the command again; the script will resume from where it stopped.")

if __name__ == "__main__":
    main()