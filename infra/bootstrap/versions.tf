# Foundation of the GCP project, applied once by a human with owner rights:
# project APIs, the bucket that stores the Terraform state of ../ (the main stack)
# and the Artifact Registry repository the images are pushed to.
#
# Its own state is local (chicken-and-egg: the state bucket does not exist yet). After the
# first apply it can be moved into that bucket, see infra/README.md.
terraform {
  required_version = ">= 1.10"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.7"
    }
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
