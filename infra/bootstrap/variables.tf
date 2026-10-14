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

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to authenticate (Workload Identity Federation)."
  type        = string
  default     = "rafaelhh04/PrediccionPetroleo-Clasificacion"
}

variable "github_repository_owner_id" {
  description = "Numeric id of the repository owner: survives a rename of the account (no takeover by a new owner of the old name)."
  type        = string
  default     = "132058207"
}

variable "deploy_environments" {
  description = "GitHub environments whose jobs may impersonate the deployer service account."
  type        = set(string)
  default     = ["staging", "production"]
}
