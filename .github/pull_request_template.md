## Summary

<!-- What does this PR change and why? Link the issue it closes, e.g. "Closes #12". -->

## Type of change

- [ ] `feat` — new functionality
- [ ] `fix` — bug fix
- [ ] `refactor` — no behaviour change
- [ ] `test` / `ci` / `chore` / `docs`

## ML impact

- [ ] No change to model logic or metrics (`results/metrics.json` identical on the same data)
- [ ] Metrics change on purpose — before/after table below and baseline in the README/roadmap updated

<!-- | Model | AUC test (before) | AUC test (after) | -->

## Checklist

- [ ] Branch created from an up-to-date `main` with a `type/short-name` name
- [ ] Conventional Commits, small and atomic
- [ ] `make lint` and `make test` pass locally (`make test-all` if the pipeline changed)
- [ ] Tests added or updated; coverage stays ≥ 85 %
- [ ] No data, credentials or generated artefacts committed
- [ ] Docs updated (README, `docs/ROADMAP.md`) when behaviour or commands change
