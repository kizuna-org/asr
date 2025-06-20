# GCP Pub/Sub and IAM configuration for CI/CD pipeline
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}


provider "google" {
  project = local.project_id
  region  = local.region
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}


# Pub/Sub Topics
resource "google_pubsub_topic" "build_triggers" {
  name = local.build_topic_name

  labels = {
    environment = local.environment
    purpose     = "ci-cd-build"
  }
}

resource "google_pubsub_topic" "app_triggers" {
  name = local.app_topic_name

  labels = {
    environment = local.environment
    purpose     = "ci-cd-app"
  }
}

# Pub/Sub Subscriptions
resource "google_pubsub_subscription" "build_subscription" {
  name  = local.build_subscription_name
  topic = google_pubsub_topic.build_triggers.name

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s" # 7 days

  labels = {
    environment = local.environment
    purpose     = "ci-cd-build"
  }
}

resource "google_pubsub_subscription" "app_subscription" {
  name  = local.app_subscription_name
  topic = google_pubsub_topic.app_triggers.name

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s" # 7 days

  labels = {
    environment = local.environment
    purpose     = "ci-cd-app"
  }
}

# Service Account for GitHub Actions
resource "google_service_account" "github_actions" {
  account_id   = local.service_account_name
  display_name = "GitHub Actions CI/CD Service Account"
  description  = "Service account for GitHub Actions to publish to Pub/Sub"
}

# IAM Policy Bindings
resource "google_project_iam_member" "github_actions_pubsub_publisher" {
  project = local.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_project_iam_member" "github_actions_pubsub_subscriber" {
  project = local.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

# Service Account Key (for GitHub Actions)
resource "google_service_account_key" "github_actions_key" {
  service_account_id = google_service_account.github_actions.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Application Service Account
resource "google_service_account" "app_service_account" {
  account_id   = local.app_service_account_name
  display_name = "Application Service Account"
  description  = "Service account for application to access GCP resources"
}

# Application Service Account Key
resource "google_service_account_key" "app_service_account_key" {
  service_account_id = google_service_account.app_service_account.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Create directory for subscriber files if it doesn't exist
resource "null_resource" "create_subscriber_dir" {
  provisioner "local-exec" {
    command = "mkdir -p ${path.module}/../../whaled/app/config"
  }
}

# Save the key to the subscriber directory
resource "local_file" "app_service_account_key_file" {
  depends_on = [null_resource.create_subscriber_dir]
  content    = base64decode(google_service_account_key.app_service_account_key.private_key)
  filename   = "${path.module}/../../whaled/app/config/app-service-account-key.json"
}

# Generate .env file for subscriber
resource "local_file" "subscriber_env_file" {
  depends_on = [null_resource.create_subscriber_dir]
  content    = <<-EOT
# GCP Configuration
GCP_PROJECT_ID=${local.project_id}
APP_SUBSCRIPTION=${local.app_subscription_name}

# Path to GCP service account key file
GOOGLE_APPLICATION_CREDENTIALS=/app/config/app-service-account-key.json

# Cloudflare R2 Configuration
R2_ENDPOINT_URL=${local.r2_endpoint_url}
R2_ACCESS_KEY_ID=${local.r2_access_key_id}
R2_SECRET_ACCESS_KEY=${local.r2_secret_access_key}
R2_BUCKET_NAME=${local.r2_bucket_name}

# Hugging Face Configuration
HF_TOKEN=${local.hf_token}
EOT
  filename   = "${path.module}/../../whaled/app/config/.env"
}

# IAM Policy Bindings for Application Service Account
resource "google_project_iam_member" "app_pubsub_subscriber" {
  project = local.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.app_service_account.email}"
}

resource "google_project_iam_member" "app_storage_object_viewer" {
  project = local.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.app_service_account.email}"
}

# Cloudflare R2 Bucket
resource "cloudflare_r2_bucket" "app_bucket" {
  account_id = local.cloudflare_account_id
  name       = local.r2_bucket_name
  location   = "APAC"
}