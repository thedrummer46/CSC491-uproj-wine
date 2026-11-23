"""Run ONE LLM trial: one treatment, one repeat, one step.

    python scripts/run_trial.py --t 5 --step 1 --repeat 1

Builds the prompt (background library + this trial's earlier steps + the
step's asks), runs `claude -p` in a fresh empty folder outside the repo,
saves the code to treatments/tK/rNN/stepN.py and logs cost to
results/cost.csv. Never hand-edit the saved code.
"""
import argparse, csv, json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COST = ROOT / "results" / "cost.csv"
FIELDS = ["treatment", "repeat", "step", "skill", "model", "input_tokens",
          "output_tokens", "cost_usd", "seconds", "loc", "ok", "when"]

# treatment -> (background, told to reuse?)
TREATMENTS = {1: ("plain", False), 2: ("plain", True),
              3: ("sklearn", False), 4: ("sklearn", True),
              5: ("ezr", False), 6: ("ezr", True)}

def header():
    with open(ROOT / "data" / "Wine_quality.csv", newline="") as f:
        return next(csv.reader(f))

def background(kind):
    if kind == "plain":
        return "Use plain Python 3 and its standard library only.\n"
    if kind == "sklearn":
        return "Use Python 3 with scikit-learn (and numpy/pandas if you need them).\n"
    ezr = (ROOT / "lib" / "ezr.py").read_text()
    return ("Build on this library, ezr.py. It is importable as `import ezr`.\n"
            "```python\n" + ezr + "\n```\n")

def prompt(t, step, repeat, spec):
    kind, reuse = TREATMENTS[t]
    folder = ROOT / "treatments" / f"t{t}" / f"r{repeat:02d}"
    p = ["You are writing one Python file for a winemaker's AI assistant.\n",
         "Data: a CSV whose header row is:\n" + ",".join(header()) + "\n"
         "Column names ending in + are goals to maximize, ending in - to minimize; "
         "the rest are lab readings. Every function receives rows as a list of "
         "lists: the first row is the header (strings), the rest are numbers.\n",
         background(kind)]
    for k in range(1, step):  # this trial's own earlier steps
        f = folder / f"step{k}.py"
        if not f.exists():
            sys.exit(f"run step{k} first: {f.relative_to(ROOT)} is missing")
        p.append(f"Earlier step {k}, importable as `import step{k}`:\n"
                 f"```python\n{f.read_text()}\n```\n")
    if reuse and step > 1:
        p.append("Reuse the library and the earlier steps wherever you can: "
                 "import their functions instead of rewriting them.\n")
    p.append(f"Write step{step}.py. It must answer these requests from the winemaker:\n")
    p += [f"- \"{a}\"\n" for a in spec["asks"]]
    p.append("It must define:\n")
    p += [f"- {a}\n" for a in spec["api"]]
    p.append(f"Workload: {spec['calls']} Everything for one train/test split "
             "must finish within 90 seconds on an ordinary laptop; slower code fails.\n")
    p.append("Do not use any tools or write any files. Reply with ONLY the contents "
             "of step%d.py as plain text: no explanation, no markdown fences.\n" % step)
    return "".join(p)

def code_of(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)  # strip fences if it added them
    return (m.group(1) if m else text).strip() + "\n"

def spent():
    if not COST.exists():
        return 0.0
    with open(COST, newline="") as f:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))

def log(row):
    new = not COST.exists()
    with open(COST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--t", type=int, required=True, choices=TREATMENTS)
    a.add_argument("--step", type=int, required=True)
    a.add_argument("--repeat", type=int, default=1)
    a.add_argument("--model", default="opus")
    a.add_argument("--max-call", type=float, default=2.0)  # $ cap for one call
    a.add_argument("--cap", type=float, default=30.0)      # $ cap for everything in cost.csv
    a.add_argument("--dry", action="store_true")           # print the prompt, spend nothing
    x = a.parse_args()

    spec = json.loads((ROOT / "scripts" / "skills.json").read_text())[str(x.step)]
    text = prompt(x.t, x.step, x.repeat, spec)
    if x.dry:
        return print(text)
    if spent() >= x.cap:
        sys.exit(f"stop: ${spent():.2f} spent, cap is ${x.cap:.2f}")

    claude = shutil.which("claude") or sys.exit("claude not found on PATH")
    fresh = tempfile.mkdtemp(prefix="trial-")  # outside the repo: no CLAUDE.md, no history
    t0 = time.time()
    r = subprocess.run([claude, "-p", "--output-format", "json", "--model", x.model,
                        "--tools", "",  # no tools: the code must come back as text
                        "--max-turns", "1", "--max-budget-usd", str(x.max_call)],
                       input=text, capture_output=True, text=True, cwd=fresh,
                       encoding="utf-8")
    secs = round(time.time() - t0, 1)
    shutil.rmtree(fresh, ignore_errors=True)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.exit(f"claude failed:\n{r.stderr or r.stdout}")

    code = code_of(out.get("result", ""))
    dest = ROOT / "treatments" / f"t{x.t}" / f"r{x.repeat:02d}" / f"step{x.step}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(code, encoding="utf-8")
    raw = ROOT / "results" / "raw" / f"t{x.t}-r{x.repeat:02d}-step{x.step}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(r.stdout, encoding="utf-8")  # full reply, for debugging

    u = out.get("usage", {})
    tin = (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
           + u.get("cache_read_input_tokens", 0))
    loc = sum(1 for l in code.splitlines() if l.strip())
    log(dict(treatment=f"t{x.t}", repeat=x.repeat, step=x.step, skill=spec["skill"],
             model=x.model, input_tokens=tin, output_tokens=u.get("output_tokens", 0),
             cost_usd=out.get("total_cost_usd", 0), seconds=secs, loc=loc,
             ok=not out.get("is_error", False), when=time.strftime("%Y-%m-%d %H:%M")))
    print(f"t{x.t} r{x.repeat:02d} step{x.step}: {loc} loc, {tin}+{u.get('output_tokens',0)} tokens, "
          f"${out.get('total_cost_usd',0):.3f}, {secs}s -> {dest.relative_to(ROOT)}")
    print(f"total spent so far: ${spent():.2f}")
    if not code.strip():
        print(f"WARNING: no code came back (subtype={out.get('subtype')}). "
              f"See {raw.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
