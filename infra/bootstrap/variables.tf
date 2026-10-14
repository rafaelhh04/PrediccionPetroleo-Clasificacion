variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Region of the state bucket and the image repository (same as the main stack)."
  type        = string
  default     = "us-central1"
}

variable "repository_id" {
  description = "Artifact Registry repository for the Docker images."
  type        = string
  default     = "brent"
}

variable "keep_image_versions" {
  description = "Most recent image versions kept per image by the cleanup policy (rollback window)."
  type        = number
  default     = 10
}

variable "services" {
  description = "Project APIs used by both stacks."
  type        = set(string)
  default = [
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ]
}
