# Installation Guide for PyCharm

If `pip install -r requirements.txt` is not "sticking" — i.e. you keep
getting `ModuleNotFoundError` after what looks like a successful install,
or you find yourself reinstalling the same packages over and over — the
problem is almost always one of the four causes below. This guide walks
through them in the order they're most likely to bite you.

> **TL;DR** — open `check_environment.py` in PyCharm, right-click, **Run**.
> It prints which interpreter you're actually using, what's installed, what's
> missing, and the exact `pip install` command that fixes it.

---

## Step 1: open the project the right way

In PyCharm: **File > Open** and select the `causal_bias_audit/` folder
**itself** (not its parent). PyCharm needs the project root to be the
folder that contains `requirements.txt`, otherwise the per-project
interpreter ends up in the wrong place and your imports won't resolve.

## Step 2: create a project virtualenv (the interpreter trap)

This is the #1 reason packages appear to "go missing" between installs.

1. **File > Settings > Project: causal_bias_audit > Python Interpreter**
   (or **PyCharm > Settings** on macOS).
2. Click the gear icon > **Add Interpreter > Add Local Interpreter**.
3. Select **Virtualenv Environment**, **New**, location
   `causal_bias_audit/.venv`, base interpreter Python **3.10, 3.11, or 3.12**
   (3.13 currently has wheel gaps for some scientific packages — avoid for
   this project).
4. Click **OK**. PyCharm will spin up the venv and select it as the project
   interpreter.

You should now see `Python 3.x (causal_bias_audit)` in the bottom-right
status bar. **If you see anything else** (e.g. `Python 3.x` with no project
name, or a system path), nothing else in this guide will work — fix the
interpreter first.

## Step 3: install dependencies in the *built-in* terminal

The single most common mistake: opening a system terminal (Terminal.app,
PowerShell, Windows Terminal) and running `pip install` there. That `pip`
points at your system Python or some other venv, so the packages land
somewhere PyCharm's project interpreter can't see. PyCharm then runs your
script and reports modules missing — because for *its* interpreter, they
are.

Always use **PyCharm's built-in terminal**, which auto-activates the
project venv:

1. **View > Tool Windows > Terminal** (or `Alt+F12` on Windows/Linux,
   `Option+F12` on macOS).
2. The prompt should show `(.venv)` at the start. **If it doesn't, stop
   and fix Step 2** — running pip without that prefix puts packages in the
   wrong Python.
3. Run:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

The first install takes 1–3 minutes; `causal-learn`, `statsmodels`, and
their transitive deps account for most of it.

## Step 4: verify with check_environment.py

In the Project pane, right-click `check_environment.py` > **Run
'check_environment'**. You should see green-style `[OK]` lines for every
package. If anything is `[MISSING]`, `[TOO OLD]`, or `[BROKEN]`, the
script prints the exact `pip install` command to run — paste it into
PyCharm's terminal (still with `(.venv)` prefix) and re-run the
diagnostic.

---

## Common errors and what they actually mean

### `ModuleNotFoundError: No module named 'X'` even though you just installed it

You installed `X` into a different Python. Run `check_environment.py` and
look at the `sys.executable` line in the output — that is the Python the
*script* is running under. Then in the same PyCharm terminal where you ran
pip, run:

```bash
python -c "import sys; print(sys.executable)"
```

If those two paths differ, that's your bug. Your terminal's `pip` is
installing to a different interpreter than PyCharm is running. Fix:
re-create the venv (Step 2) and reinstall (Step 3) — and don't open a
non-PyCharm terminal for any of it.

### `pydot` installs but DAG plots fail with `ExecutableNotFound: dot`

`pydot` is the Python wrapper; it needs the **graphviz** system binary
(`dot`) on `PATH`. The Python `pip install pydot` step does not install
the binary. Fix:

- **macOS**: `brew install graphviz`
- **Ubuntu/Debian**: `sudo apt install graphviz`
- **Windows**: download from <https://graphviz.org/download/>, install,
  and add the `bin\` folder to `PATH` (then restart PyCharm so it picks
  up the new `PATH`).

The project does not strictly require `dot` to run discovery — it's only
used by `pydot` for graph rendering. The project's own plots use
`matplotlib + networkx` and work without graphviz.

### NumPy 2.x ABI errors after upgrading

Symptoms: import-time `RuntimeError`, `_ARRAY_API not found`, or seg-faults
from `causallearn` even though `numpy` and `causallearn` are both
installed. Cause: an older causal-learn build linked against NumPy 1.x.
Fix:

```bash
pip install --upgrade --force-reinstall causal-learn "numpy>=2.0"
```

The pinned `causal-learn>=0.1.4.0` in `requirements.txt` should prevent
this — if you somehow ended up with an older version, force-reinstall.

### `pip install` succeeds but lists packages as already satisfied

Pip is reporting on the system Python, not your venv. Confirm by checking
the prompt prefix `(.venv)` — if it's missing, the venv isn't activated.
Close the terminal and reopen it from **View > Tool Windows > Terminal**;
PyCharm should re-activate the venv automatically.

### Re-running pip install installs new things every time

Pip aborted on the first failure last time. The next run gets further,
hits a new failure, etc. Symptom that this happened: scroll up in the pip
output and look for `ERROR:` (not `WARNING`). The first ERROR is your
real bug. If you can't find it, run with `pip install -v -r requirements.txt`
for verbose output.

### `pip install` is slow / appears to hang on `causal-learn` or `statsmodels`

Both have large source distributions and `statsmodels` may build C
extensions on platforms without a wheel. This is normal on first install
and should not take more than ~5 minutes. If it actually hangs (no CPU
activity for minutes), `Ctrl+C` and re-run with `pip install -v` to see
where it's stuck.

---

## If nothing above helps

1. Run `check_environment.py` and copy its full output.
2. Compare the `sys.executable` it prints to the path PyCharm shows in
   **Settings > Project > Python Interpreter**. They must be identical.
3. As a last resort, blow away the venv and start over:
   ```bash
   # In PyCharm's built-in terminal:
   deactivate    # if active
   rm -rf .venv  # or: rmdir /s /q .venv  on Windows cmd
   ```
   Then redo Step 2 (create new interpreter) and Step 3 (reinstall).
