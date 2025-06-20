variable "github_token" {
  description = "GitHub Personal Access Token for kizuna-org organization with read:packages scope for GHCR pull access"
  type        = string
  sensitive   = true
}
