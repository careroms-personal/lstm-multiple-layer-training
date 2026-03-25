# pyconfplate — Design Principles
<!-- last human review: 2026 Mar 24 -->
<!-- last ai update: 2026 Mar 24 -->

Core design principles for AI assistants working in this codebase.
See `vibe-code-rule.yaml` for rules. See `AI-PYTHON-GUIDE.md` for implementation patterns.

---

## Processor + Executor Separation

- **Processor** owns the flow — config loading, executor ordering, result passing. No business logic.
- **Executor** owns one duty — does its work, returns a result. No knowledge of other executors.
- Executors do not call each other. Only Processor connects them.

---

## Config Design

- Config structure is **purpose-built per project** — shape reflects the project's domain and executor chain.
- Each executor's duty maps to a config section. If an executor needs settings, they live in a dedicated block.
- Common structure across projects:
  1. **Core settings** — the single target the program operates on (e.g. one DB connection, one model config)
  2. **Variable list** — what to process (e.g. list of queries, list of files)
  3. **Output block** — optional, how results are exported (consistent shape across projects)
- Config structure is fixed once defined (enforced by pydantic). Users only fill in values — never restructure.
- Use `REQUEST` as placeholder in config templates for values the user must provide.

---

## Single Responsibility

- One executor per concern. Never mix two duties in one executor.
- If an executor grows private methods for unrelated tasks, it is a signal to split.

---

## Typed Data Flow

- Data passed between executors must be a typed pydantic model — never a plain dict, tuple, or loose args.
- Result model shape should reflect what the next executor actually needs — no extra fields, no missing ones.

---

## Optional Features

- Optional behavior (e.g. export, test prediction) is controlled by config flags (`enabled: true/false`).
- Executors check their own flag and return early if disabled — Processor does not conditionally skip them.
