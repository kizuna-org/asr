# Variables for GCP Pub/Sub CI/CD infrastructure
# Note: project_id, region, and zone are now defined in locals block in main.tf

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
}

# Note: The following variables have been moved to locals in main.tf:
# - build_topic_name
# - app_topic_name
# - build_subscription_name
# - app_subscription_name
# - service_account_name