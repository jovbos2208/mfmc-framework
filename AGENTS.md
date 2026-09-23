# AGENTS.md

## Cost policy

Minimize total token/credit usage while preserving correctness.

Use the cheapest available model/reasoning level sufficient for the task.

LIGHT:
- file/symbol lookup
- repository navigation
- formatting
- documentation
- simple tests
- mechanical edits
- obvious bug fixes
- routine refactoring

→ Prefer the fast/cheap model with low reasoning.

HARD:
- architecture
- ambiguous debugging
- numerical algorithms
- physics/modeling
- mathematical derivations
- coupled multi-file changes
- scientific validation

→ Prefer the capable model with medium reasoning.

Escalate beyond medium only after concrete evidence that medium is
insufficient.

## Context policy

Do not scan the repository broadly.

Before reading source files:
1. use an existing repository/code graph when it can identify the
   relevant symbols or dependencies more cheaply;
2. otherwise search for the relevant symbols/paths;
3. read only the smallest necessary file sections.

Never regenerate Graphify output unless the code structure materially
changed or the existing graph is missing/stale.

Do not repeatedly read unchanged files.

Ignore generated data, large datasets, build outputs, logs, binaries,
vendor directories and lockfiles unless directly relevant.

## Tool policy

Batch independent read-only operations.
Prefer targeted searches over directory exploration.
Run the smallest relevant test first.
Do not rerun commands without a reason.

## Skills

Use Graphify for unfamiliar cross-file dependency questions or repository
architecture exploration.

Use scientific-code for physics, numerical methods, units, coordinate
systems, uncertainty propagation, simulations, or mathematical models.

Use verification after substantive code changes.

Do not invoke skills when their context/tool overhead exceeds the expected
benefit.

## Subagents

Avoid subagents for small tasks.

Use them only for genuinely independent work where parallelism or context
isolation reduces total cost.

Prefer cheap/fast models for repository exploration and routine work.
Reserve capable models with medium reasoning for difficult reasoning.

## Editing

Make minimal diffs.
Do not refactor unrelated code.
Do not reformat unrelated files.
Do not create documentation unless requested.

## Completion

Stop when the requested task is complete and sufficiently validated.

Report only:
- changes made
- validation performed
- unresolved issues