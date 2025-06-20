# Local values for GCP Pub/Sub CI/CD infrastructure

locals {
  # Core configuration
  project_id = "chumchat"
  region     = "asia-northeast1"
  zone       = "asia-northeast1-a"

  # Environment configuration
  environment = "prod"  # Previously was a variable

  # Resource naming
  build_topic_name = "build-triggers"
  app_topic_name = "app-triggers"
  build_subscription_name = "build-triggers-sub"
  app_subscription_name = "app-triggers-sub"
  service_account_name = "github-actions-cicd"
  app_service_account_name = "app-service-account"
  
  # GitHub Actions service account name is kept for GCP resources
  # But we no longer use the GitHub provider

  # R2 Configuration
  r2_endpoint_url = "https://example.r2.cloudflarestorage.com"
  r2_access_key_id = "example_r2_access_key_id"
  r2_secret_access_key = "example_r2_secret_access_key"
  r2_bucket_name = "example-bucket"

  # Hugging Face Configuration
  hf_token = "example_hf_token"
}