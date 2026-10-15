# ADR 0006 — Keyless CD: OIDC, one digest per commit, smoke-tested candidate revision

- **Status:** accepted
- **Phase:** 6 (cloud and CD)

## Context

Deployments go from GitHub Actions to Cloud Run in two environments, staging and production. Two
common practices were rejected here:

- long-lived JSON service-account keys stored as repository secrets, a frequent source of leaks;
- rebuilding the image per environment, which means production may not run what staging tested.

A bad revision must never receive traffic.

## Decision

- **Workload Identity Federation (OIDC).** GitHub's OIDC token is exchanged for short-lived Google
  credentials. Only jobs bound to the `staging` and `production` environments can impersonate the
  deployer service account. A separate, read-only planner identity runs `terraform plan`.
- **Build once.** Each commit is built once, tagged with its SHA, pushed with a GitHub **build provenance
  attestation**, and deployed **by digest** to staging. After a manual approval of the `production`
  environment, the same digest goes to production.
- **Candidate revision.** Each rollout creates a no-traffic `candidate` revision. `scripts/smoke_test.py`
  checks it: `/health` returns 200 with a model loaded, a prediction succeeds and an invalid request
  gets a 422. Only then is the traffic moved. `rollback.yml` moves the traffic back to an earlier
  revision.
- **Ownership split.** Terraform ignores `image`, `client` and `traffic`, so infrastructure changes and
  releases never undo each other.

## Alternatives considered

- **Service-account keys in secrets**: simpler to set up, but long-lived and leakable.
- **Cloud Build / Cloud Deploy**: more services to configure for one pipeline that Actions already runs.
- **cosign signatures**: they need key management or keyless Fulcio setup. The GitHub attestation gives
  Sigstore-backed provenance with no keys.

## Consequences

- No credentials are stored in the repository or in GitHub secrets. Access is scoped per environment.
- Until the GCP variables are set, CD runs and ends early with a "CD disabled" notice, which is how it
  was tested without credentials.
- `/health` requires a champion, so the very first deployment needs one retraining job run beforehand.
  This is documented in `infra/README.md`.
