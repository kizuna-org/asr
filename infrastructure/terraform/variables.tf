variable "github_token" {
  description = "GitHub Personal Access Token for kizuna-org organization with read:packages scope for GHCR pull access"
  type        = string
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "Cloudflare API Token with R2 permissions"
  type        = string
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare Account ID for R2 storage"
  type        = string
}

variable "r2_access_key_id" {
  description = "Cloudflare R2 Access Key ID"
  type        = string
  sensitive   = true
}

variable "r2_secret_access_key" {
  description = "Cloudflare R2 Secret Access Key"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "Hugging Face API Token"
  type        = string
  sensitive   = true
}
