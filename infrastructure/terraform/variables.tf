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
