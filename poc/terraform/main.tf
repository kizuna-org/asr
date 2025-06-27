# terraform/main.tf (最終版)

terraform {
  required_providers {
    jenkins = {
      source  = "taiidani/jenkins"
      version = "0.10.2"
    }
  }
}

provider "jenkins" {
  server_url = var.jenkins_url
  username   = var.jenkins_username
  password   = var.jenkins_password
}


# --- 入力変数の定義 ---

variable "jenkins_url" {
  type        = string
  description = "JenkinsサーバーのURL (例: http://localhost:8080)"
}
variable "jenkins_username" {
  type        = string
  description = "Jenkinsの管理者ユーザー名"
}
variable "jenkins_password" {
  type        = string
  description = "JenkinsユーザーのパスワードまたはAPIトークン"
  sensitive   = true
}
variable "gitea_api_url" {
  type        = string
  description = "Jenkinsコンテナから見たGiteaのAPI URL (例: http://gitea:3000/api/v1)"
}
variable "git_username" {
  type        = string
  description = "Giteaのユーザー名 (リポジトリのオーナー)"
}
variable "git_reponame" {
  type        = string
  description = "Giteaのリポジトリ名"
}
variable "git_credential_id" {
  type        = string
  description = "Jenkinsに登録したGitea認証情報のID"
}
variable "jenkins_job_name" {
  type        = string
  description = "Jenkins上に作成するジョブの名前"
  default     = "gitea-poc-multibranch"
}


# --- Jenkinsリソースの作成 ---

resource "jenkins_job" "gitea_multibranch_pipeline" {
  name     = var.jenkins_job_name
  template = <<-EOT
<org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject>
  <properties/>
  <folderViews class="jenkins.branch.MultiBranchProjectViewHolder">
    <owner class="org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject" reference="../.."/>
  </folderViews>
  <healthMetrics>
    <com.cloudbees.hudson.plugins.folder.health.WorstChildHealthMetric>
      <nonRecursive>false</nonRecursive>
    </com.cloudbees.hudson.plugins.folder.health.WorstChildHealthMetric>
  </healthMetrics>
  <icon class="jenkins.branch.MetadataActionFolderIcon">
    <owner class="org.jenkinsci.plugins.workflow.multibranch.WorkflowMultiBranchProject" reference="../.."/>
  </icon>
  <orphanedItemStrategy class="com.cloudbees.hudson.plugins.folder.computed.DefaultOrphanedItemStrategy">
    <pruneDeadBranches>true</pruneDeadBranches>
    <daysToKeep>-1</daysToKeep>
    <numToKeep>-1</numToKeep>
  </orphanedItemStrategy>
  <triggers/>
  <sources>
    <data>
      <jenkins.branch.BranchSource>
        <source class="org.jenkinsci.plugins.github_branch_source.GitHubSCMSource">
          <id>gitea-source-from-terraform</id>
          <apiUri>${var.gitea_api_url}</apiUri>
          <credentialsId>${var.git_credential_id}</credentialsId>
          <repoOwner>${var.git_username}</repoOwner>
          <repository>${var.git_reponame}</repository>
          <traits>
            <org.jenkinsci.plugins.github__branch__source.BranchDiscoveryTrait>
              <strategyId>1</strategyId>
            </org.jenkinsci.plugins.github__branch__source.BranchDiscoveryTrait>
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
}
