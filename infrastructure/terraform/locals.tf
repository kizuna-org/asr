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

  # Cloudflare Configuration
  cloudflare_account_id = "your-cloudflare-account-id"
  cloudflare_api_token = "your-cloudflare-api-token"
  
  # R2 Configuration
  r2_endpoint_url = "https://${local.cloudflare_account_id}.r2.cloudflarestorage.com"
  r2_access_key_id = "your-r2-access-key-id"
  r2_secret_access_key = "your-r2-secret-access-key"
  r2_bucket_name = "chumchat-storage"

  # Hugging Face Configuration
  hf_token = "example_hf_token"
}