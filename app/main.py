#!/usr/bin/env python3
"""
Application Container - Main processing logic
This is a template that should be customized for specific ML/data processing tasks
"""

import json
import os
import sys
import time
from datetime import datetime
import boto3
from botocore.config import Config
from huggingface_hub import HfApi, Repository

# Add the shared directory to the path to import logger
sys.path.append('/app/shared')
from logger import StructuredLogger, Component

# --- Proxy setup ---
# If http_proxy/https_proxy is set in the environment, ensure they are in os.environ
for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    proxy_val = os.environ.get(proxy_var)
    if proxy_val:
        os.environ[proxy_var] = proxy_val

# Helper to build proxies dict for boto3
proxies = {}
if os.environ.get("http_proxy"):
    proxies["http"] = os.environ["http_proxy"]
if os.environ.get("https_proxy"):
    proxies["https"] = os.environ["https_proxy"]

class Application:
    def __init__(self):
        self.job_id = os.environ.get('JOB_ID')
        if not self.job_id:
            raise ValueError("JOB_ID environment variable is required")
        
        # Initialize structured logger
        self.logger = StructuredLogger(Component.APP, self.job_id)
        
        # Cloudflare R2 setup
        self.r2_client = boto3.client(
            's3',
            endpoint_url=os.environ.get('R2_ENDPOINT_URL'),
            aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4', proxies=proxies if proxies else None),
            region_name='auto'
        )
        self.bucket_name = os.environ.get('R2_BUCKET_NAME')
        
        # Hugging Face setup
        self.hf_token = os.environ.get('HF_TOKEN')
        self.hf_api = HfApi(token=self.hf_token) if self.hf_token else None

    def log_message(self, message, level="info", context=None, tags=None):
        """Log message using structured logger"""
        if level == "info":
            return self.logger.info(message, context=context, tags=tags)
        elif level == "error":
            return self.logger.error(message, context=context, tags=tags)
        elif level == "warn":
            return self.logger.warn(message, context=context, tags=tags)
        elif level == "debug":
            return self.logger.debug(message, context=context, tags=tags)
        else:
            return self.logger.info(message, context=context, tags=tags)

    def stream_log_to_r2(self, log_content):
        """Stream log content to R2"""
        try:
            log_key = f"{self.job_id}/app.log"
            
            # Get existing log content
            try:
                response = self.r2_client.get_object(Bucket=self.bucket_name, Key=log_key)
                existing_content = response['Body'].read().decode('utf-8')
            except:
                existing_content = ""
            
            # Append new content
            updated_content = existing_content + log_content
            
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=log_key,
                Body=updated_content,
                ContentType='text/plain'
            )
        except Exception as e:
            self.logger.error(
                "Failed to stream log to R2",
                exception=e,
                context={"operation": "stream_log_to_r2"},
                tags=["r2", "logs"]
            )

    def get_current_status(self):
        """Get current status from R2"""
        try:
            status_key = f"{self.job_id}/status.json"
            response = self.r2_client.get_object(Bucket=self.bucket_name, Key=status_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            self.logger.error(
                "Failed to get current status",
                exception=e,
                context={"operation": "get_current_status"},
                tags=["r2", "status"]
            )
            return None

    def update_status(self, status_updates):
        """Update job status in Cloudflare R2"""
        try:
            # Get current status first
            current_status = self.get_current_status()
            if not current_status:
                self.logger.warn(
                    "No existing status found for job",
                    context={"operation": "update_status"},
                    tags=["r2", "status"]
                )
                return
            
            # Update with new data
            current_status.update(status_updates)
            current_status['timestamps']['updated'] = datetime.utcnow().isoformat() + 'Z'
            
            status_key = f"{self.job_id}/status.json"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=status_key,
                Body=json.dumps(current_status, indent=2),
                ContentType='application/json'
            )
            self.logger.info(
                "Status updated successfully",
                context={"operation": "update_status"},
                tags=["r2", "status"]
            )
        except Exception as e:
            self.logger.error(
                "Failed to update status",
                exception=e,
                context={"operation": "update_status"},
                tags=["r2", "status"]
            )

    def run_task(self):
        """
        Main task execution - Override this method in your specific application
        This is a template implementation
        """
        log_entry = self.log_message("Starting task execution")
        self.stream_log_to_r2(log_entry)
        
        # Update status to task started
        self.update_status({
            'run': {
                'status': 'TaskStarted',
                'log': f'/{self.job_id}/app.log'
            }
        })
        
        try:
            # Simulate some work - replace with actual ML/data processing
            for i in range(5):
                log_entry = self.log_message(f"Processing step {i+1}/5")
                self.stream_log_to_r2(log_entry)
                time.sleep(2)  # Simulate work
            
            # Create some dummy artifacts
            artifact_path = "/tmp/model_output"
            os.makedirs(artifact_path, exist_ok=True)
            
            with open(f"{artifact_path}/README.md", "w") as f:
                f.write(f"# Model Output for Job {self.job_id}\n\n")
                f.write(f"Generated at: {datetime.utcnow().isoformat()}Z\n")
                f.write("This is a sample output from the CI/CD pipeline.\n")
            
            with open(f"{artifact_path}/results.json", "w") as f:
                json.dump({
                    "job_id": self.job_id,
                    "status": "completed",
                    "metrics": {
                        "accuracy": 0.95,
                        "loss": 0.05
                    },
                    "timestamp": datetime.utcnow().isoformat() + 'Z'
                }, f, indent=2)
            
            log_entry = self.log_message("Task processing completed, uploading artifacts")
            self.stream_log_to_r2(log_entry)
            
            # Upload to Hugging Face Hub
            artifact_url = self.upload_to_huggingface(artifact_path)
            
            # Update status to task succeeded
            self.update_status({
                'run': {
                    'status': 'TaskSucceeded',
                    'log': f'/{self.job_id}/app.log',
                    'artifactUrl': artifact_url
                }
            })
            
            log_entry = self.log_message(f"Task completed successfully. Artifacts available at: {artifact_url}")
            self.stream_log_to_r2(log_entry)
            
            return artifact_url
            
        except Exception as e:
            log_entry = self.log_message(f"Task failed with error: {str(e)}")
            self.stream_log_to_r2(log_entry)
            
            # Update status to task failed
            self.update_status({
                'run': {
                    'status': 'TaskFailed',
                    'log': f'/{self.job_id}/app.log',
                    'error': str(e)
                }
            })
            
            raise

    def upload_to_huggingface(self, artifact_path):
        """Upload artifacts to Hugging Face Hub"""
        if not self.hf_api:
            log_entry = self.log_message("No Hugging Face token provided, skipping upload")
            self.stream_log_to_r2(log_entry)
            return "No HF token provided"
        
        try:
            # Create repository name
            repo_name = f"cicd-output-{self.job_id[:8]}"
            repo_url = f"https://huggingface.co/datasets/{repo_name}"
            
            log_entry = self.log_message(f"Creating Hugging Face repository: {repo_name}")
            self.stream_log_to_r2(log_entry)
            
            # Create repository
            self.hf_api.create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                exist_ok=True
            )
            
            # Upload files
            for root, dirs, files in os.walk(artifact_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, artifact_path)
                    
                    log_entry = self.log_message(f"Uploading {relative_path}")
                    self.stream_log_to_r2(log_entry)
                    
                    self.hf_api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=relative_path,
                        repo_id=repo_name,
                        repo_type="dataset"
                    )
            
            log_entry = self.log_message(f"Upload completed: {repo_url}")
            self.stream_log_to_r2(log_entry)
            
            return repo_url
            
        except Exception as e:
            log_entry = self.log_message(f"Failed to upload to Hugging Face: {str(e)}")
            self.stream_log_to_r2(log_entry)
            return f"Upload failed: {str(e)}"

def main():
    """Main entry point"""
    logger = None
    try:
        app = Application()
        logger = app.logger
        app.run_task()
        logger.info(
            "Application completed successfully",
            context={"operation": "main"},
            tags=["application", "success"]
        )
        sys.exit(0)
    except Exception as e:
        if logger:
            logger.fatal(
                "Application failed",
                exception=e,
                context={"operation": "main"},
                tags=["application", "failure"]
            )
        else:
            # Fallback if logger not initialized
            print(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
