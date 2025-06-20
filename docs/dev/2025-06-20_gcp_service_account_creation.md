# GCP Service Account Creation

Date: 2025-06-20

## Overview

Added Terraform configuration to create a new GCP service account for application use and generate a JSON key file that is stored in an ignored directory.

## Changes Made

1. Created a new directory `infrastructure/terraform/keys/` to store service account key files
2. Updated `.gitignore` to explicitly ignore the `keys/` directory
3. Added a new local variable `app_service_account_name` in `locals.tf`
4. Added the `local` provider to Terraform configuration
5. Created a new service account resource `google_service_account.app_service_account`
6. Generated a service account key with `google_service_account_key.app_service_account_key`
7. Used `local_file` resource to save the key to `keys/app-service-account-key.json`
8. Added IAM permissions for the service account:
   - `roles/pubsub.subscriber`
   - `roles/storage.objectViewer`

## Usage

After applying the Terraform configuration, the service account key will be available at:
```
infrastructure/terraform/keys/app-service-account-key.json
```

This file is ignored by Git and should be kept secure.

## Next Steps

1. Use the service account key in application configurations as needed
2. Consider adding more specific IAM permissions based on application requirements
