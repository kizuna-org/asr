# Variables for GCP Pub/Sub CI/CD infrastructure

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "build_topic_name" {
  description = "Name of the Pub/Sub topic for build triggers"
  type        = string
  default     = "build-triggers"
}

variable "app_topic_name" {
  description = "Name of the Pub/Sub topic for app triggers"
  type        = string
  default     = "app-triggers"
}

variable "build_subscription_name" {
  description = "Name of the Pub/Sub subscription for build triggers"
  type        = string
  default     = "build-triggers-sub"
}

variable "app_subscription_name" {
  description = "Name of the Pub/Sub subscription for app triggers"
  type        = string
  default     = "app-triggers-sub"
}

variable "service_account_name" {
  description = "Name of the service account for GitHub Actions"
  type        = string
  default     = "github-actions-cicd"
}