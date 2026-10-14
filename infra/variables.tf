# ── project ────────────────────────────────────────────────────
variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = <<-EOT
    Region of every regional resource. us-central1 by default: Tier 1 Cloud Run prices and the
    Cloud Storage always-free tier (5 GB, only in us-central1, us-east1 and us-west1).
  EOT
  type        = string
  default     = "us-central1"
}

variable "repository_id" {
  description = "Artifact Registry repository created by ./bootstrap."
  type        = string
  default     = "brent"
}

variable "image_tag" {
  description = <<-EOT
    Tag of the api and dashboard images used when the services are created (e.g. the commit SHA
    of the first push). Afterwards the CD workflow deploys new images and Terraform ignores them.
  EOT
  type        = string
}

variable "environments" {
  description = "Deployment environments; each gets its own API and dashboard service."
  type        = set(string)
  default     = ["staging", "prod"]

  validation {
    condition     = alltrue([for e in var.environments : can(regex("^[a-z][a-z0-9-]{0,14}$", e))])
    error_message = "Environment names must be short lowercase DNS labels."
  }
}

variable "public" {
  description = "Allow unauthenticated access to the API and the dashboard (public demo URLs)."
  type        = bool
  default     = true
}

variable "deletion_protection" {
  description = "Protect the services, bucket contents and database from `terraform destroy`."
  type        = bool
  default     = false
}

# ── storage ────────────────────────────────────────────────────
variable "bucket_name" {
  description = "ML bucket (DVC remote, MLflow store, data, results). Default: <project>-brent-ml."
  type        = string
  default     = null
}

variable "noncurrent_versions_kept" {
  description = "Earlier versions of each object kept by the bucket lifecycle (undo a bad write)."
  type        = number
  default     = 3
}

variable "noncurrent_version_days" {
  description = "Days after which an earlier version of an object is deleted."
  type        = number
  default     = 30
}

# ── Cloud Run sizing ───────────────────────────────────────────
# One uvicorn process per instance: the measured p95 is 72 ms at ~10 req/s and an instance
# saturates at ~40 req/s (GIL). By Little's law ~3 requests are in flight at that rate, so a
# concurrency of 8 absorbs bursts without queueing behind the GIL for long; beyond that
# Cloud Run starts another instance. ~290 MiB are resident after imports (+ model, pandas,
# gcsfuse): 1 GiB leaves headroom.
variable "api" {
  description = "API service sizing."
  type = object({
    cpu           = optional(string, "1")
    memory        = optional(string, "1Gi")
    concurrency   = optional(number, 8)
    min_instances = optional(number, 0)
    max_instances = optional(number, 2)
    timeout       = optional(string, "30s")
  })
  default = {}
}

variable "dashboard" {
  description = "Dashboard (Streamlit) service sizing."
  type = object({
    cpu           = optional(string, "1")
    memory        = optional(string, "1Gi")
    concurrency   = optional(number, 20)
    max_instances = optional(number, 1)
  })
  default = {}
}

# ── MLflow ─────────────────────────────────────────────────────
variable "mlflow_backend" {
  description = <<-EOT
    Backend store of the MLflow tracking server and registry:
    - "sqlite": SQLite file in the bucket (GCS FUSE). ~0 EUR, but a single writer: the tracking
      server and the retraining job are limited to one instance, and GCS has no file locks.
    - "cloudsql": PostgreSQL on Cloud SQL (db-f1-micro). Concurrent and durable, ~9-10 USD/month.
  EOT
  type        = string
  default     = "sqlite"

  validation {
    condition     = contains(["sqlite", "cloudsql"], var.mlflow_backend)
    error_message = "mlflow_backend must be \"sqlite\" or \"cloudsql\"."
  }
}

variable "cloudsql_tier" {
  description = "Cloud SQL machine tier (mlflow_backend = \"cloudsql\")."
  type        = string
  default     = "db-f1-micro"
}

variable "mlflow_ui_members" {
  description = "Principals allowed to open the MLflow UI (e.g. user:me@example.com); it is never public."
  type        = list(string)
  default     = []
}

# ── cost control ───────────────────────────────────────────────
variable "billing_account" {
  description = "Billing account id for a budget alert (null: no budget)."
  type        = string
  default     = null
}

variable "budget" {
  description = "Monthly budget and the currency of the billing account."
  type = object({
    amount   = optional(number, 5)
    currency = optional(string, "EUR")
  })
  default = {}
}
