# terraform/main.tf

terraform {
  required_providers {
    jenkins = {
      source  = "taiidani/jenkins"
      version = "~> 0.10.2"
    }
  }
}

provider "jenkins" {
  server_url = var.jenkins_url
  username   = var.jenkins_username
  password   = var.jenkins_password
}

# --- 変数の定義 ---
variable "jenkins_url" {
  type        = string
  description = "JenkinsサーバーのURL"
}
# ... (他のJenkins関連変数は前回と同じ) ...
variable "jenkins_username" { type = string }
variable "jenkins_password" {
  type      = string
  sensitive = true
}

variable "git_repo_url" {
  type        = string
  description = "Jenkinsコンテナから見たGitリポジトリのURL"
}
variable "git_username" {
  type        = string
  description = "Gitサーバーのユーザー名"
}
variable "git_password" {
  type        = string
  description = "Gitサーバーのパスワード"
  sensitive   = true
}
variable "jenkins_job_name" {
  type    = string
  default = "docker-poc-multibranch"
}

# --- Jenkinsリソースの作成 ---

# 1. Gitサーバー用の認証情報をJenkinsに登録
resource "jenkins_credential_username" "git_credential" {
  name        = "git-server-credential-for-poc"
  username    = var.git_username
  password    = var.git_password
  description = "Credential for the local Git server (managed by Terraform)"
}

# 2. Multibranch Pipelineジョブの作成
resource "jenkins_job" "docker_multibranch_pipeline" {
  name       = var.jenkins_job_name
  template   = <<-EOT
<org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject>
  <sources>
    <data>
      <jenkins.branch.BranchSource>
        <source class="jenkins.plugins.git.GitSCMSource" plugin="git@5.2.2">
          <id>docker-git-source-from-terraform</id>
          <remote>${var.git_repo_url}</remote>
          <credentialsId>${jenkins_credential_username.git_credential.id}</credentialsId>
          <traits>
            <jenkins.plugins.git.traits.BranchDiscoveryTrait/>
          </traits>
        </source>
      </jenkins.branch.BranchSource>
    </data>
  </sources>
  <factory class="org.jenkinsci.plugins.workflow.multibranch.WorkflowBranchProjectFactory">
    <owner class="org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject" reference="../.."/>
    <scriptPath>Jenkinsfile</scriptPath>
  </factory>
</org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject>
  EOT
  depends_on = [jenkins_credential_username.git_credential]
}
