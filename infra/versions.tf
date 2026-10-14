# Main stack: storage, service accounts, secrets and the Cloud Run services of the
# Brent direction classifier. Apply ./bootstrap first (APIs, state bucket, image repository).
terraform {
  required_version = ">= 1.10"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.7"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
  }

  # Remote state with locking (lock object in the bucket). The bucket is passed at init:
  #   terraform init -backend-config="bucket=<project>-tfstate"
  backend "gcs" {
    prefix = "brent-forecast/main"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  default_labels = {
    app        = "brent-forecast"
    managed-by = "terraform"
  }
}
