# Terraform Planの修正

## 概要

`terraform plan`が失敗する問題を修正しました。

## 変更点

- `main.tf`内の`jenkins_credential_username_password`リソースを`jenkins_credential_username`に修正
- 上記に伴い、リソースを参照している箇所を修正

