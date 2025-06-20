#!/usr/bin/env python3
"""
App Subscriber - Handles application execution requests from Pub/Sub
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from google.cloud import pubsub_v1
import boto3
from botocore.config import Config

# Add the shared directory to the path to import logger
sys.path.append('/app/shared')
from logger import StructuredLogger, Component

class AppSubscriber:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        self.subscription_name = os.environ.get('APP_SUBSCRIPTION', 'app-triggers-sub')
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_name
        )
        
        # Initialize structured logger
        self.logger = StructuredLogger(Component.APP_SUBSCRIBER)
        
        # Cloudflare R2 setup
        self.r2_client = boto3.client(
            's3',
            endpoint_url=os.environ.get('R2_ENDPOINT_URL'),
            aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        self.bucket_name = os.environ.get('R2_BUCKET_NAME')

    def get_current_status(self, job_id):
        """Get current status from R2"""
        try:
            status_key = f"{job_id}/status.json"
            response = self.r2_client.get_object(Bucket=self.bucket_name, Key=status_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            self.logger.error(
                "Failed to get current status",
                exception=e,
                context={"job_id": job_id, "operation": "get_status"},
                tags=["r2", "status"]
            )
            return None

    def update_status(self, job_id, status_updates):
        """Update job status in Cloudflare R2"""
        try:
            # Get current status first
            current_status = self.get_current_status(job_id)
            if not current_status:
                self.logger.warn(
                    "No existing status found for job",
                    context={"job_id": job_id, "operation": "update_status"},
                    tags=["r2", "status"]
                )
                return
            
            # Update with new data
            current_status.update(status_updates)
            current_status['timestamps']['updated'] = datetime.utcnow().isoformat() + 'Z'
            
            status_key = f"{job_id}/status.json"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=status_key,
                Body=json.dumps(current_status, indent=2),
                ContentType='application/json'
            )
            self.logger.info(
                "Status updated successfully",
                context={
                    "job_id": job_id,
                    "operation": "update_status",
                    "status_updates": status_updates
                },
                tags=["r2", "status"]
            )
        except Exception as e:
            self.logger.error(
                "Failed to update status",
                exception=e,
                context={"job_id": job_id, "operation": "update_status"},
                tags=["r2", "status"]
            )

    def upload_log(self, job_id, log_content, log_type='app'):
        """Upload log content to R2"""
        try:
            log_key = f"{job_id}/{log_type}.log"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=log_key,
                Body=log_content,
                ContentType='text/plain'
            )
            self.logger.info(
                "Log uploaded successfully",
                context={
                    "job_id": job_id,
                    "log_key": log_key,
                    "log_type": log_type,
                    "operation": "upload_log"
                },
                tags=["r2", "logs"]
            )
        except Exception as e:
            self.logger.error(
                "Failed to upload log",
                exception=e,
                context={
                    "job_id": job_id,
                    "log_type": log_type,
                    "operation": "upload_log"
                },
                tags=["r2", "logs"]
            )

    def run_application(self, job_data):
        """Pull image and run application container"""
        job_id = job_data['jobId']
        image_uri = job_data['imageUri']
        
        # Update status to running
        status_updates = {
            'overallStatus': 'Running',
            'run': {
                'status': 'Running',
                'log': f'/{job_id}/app.log'
            }
        }
        self.update_status(job_id, status_updates)

        try:
            # Pull image from GHCR
            pull_cmd = f"sudo docker pull {image_uri}"
            pull_result = subprocess.run(
                pull_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if pull_result.returncode != 0:
                raise subprocess.CalledProcessError(pull_result.returncode, pull_cmd)
            
            # Run container with environment variables for R2 access
            run_cmd = f"""sudo docker run --rm \
                -e JOB_ID={job_id} \
                -e R2_ENDPOINT_URL={os.environ.get('R2_ENDPOINT_URL')} \
                -e R2_ACCESS_KEY_ID={os.environ.get('R2_ACCESS_KEY_ID')} \
                -e R2_SECRET_ACCESS_KEY={os.environ.get('R2_SECRET_ACCESS_KEY')} \
                -e R2_BUCKET_NAME={os.environ.get('R2_BUCKET_NAME')} \
                -e HF_TOKEN={os.environ.get('HF_TOKEN')} \
                --gpus all \
                {image_uri}"""
            
            self.logger.info(
                "Starting container execution",
                context={
                    "job_id": job_id,
                    "image_uri": image_uri,
                    "operation": "run_container"
                },
                tags=["docker", "container"]
            )
            run_result = subprocess.run(
                run_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Upload app log
            app_log = f"Pull command: {pull_cmd}\n\nPull STDOUT:\n{pull_result.stdout}\n\nPull STDERR:\n{pull_result.stderr}\n\n"
            app_log += f"Run command: {run_cmd}\n\nRun STDOUT:\n{run_result.stdout}\n\nRun STDERR:\n{run_result.stderr}"
            self.upload_log(job_id, app_log, 'app')
            
            if run_result.returncode != 0:
                raise subprocess.CalledProcessError(run_result.returncode, run_cmd)
            
            # Update status to succeeded (the app container should have updated detailed status)
            status_updates = {
                'overallStatus': 'Succeeded',
                'run': {
                    'status': 'Succeeded',
                    'log': f'/{job_id}/app.log'
                }
            }
            self.update_status(job_id, status_updates)
            
            self.logger.info(
                "Application execution succeeded",
                context={
                    "job_id": job_id,
                    "image_uri": image_uri,
                    "operation": "run_application"
                },
                tags=["docker", "success"]
            )
            
        except subprocess.CalledProcessError as e:
            # Update status to failed
            status_updates = {
                'overallStatus': 'Failed',
                'run': {
                    'status': 'Failed',
                    'log': f'/{job_id}/app.log'
                }
            }
            self.update_status(job_id, status_updates)
            self.logger.error(
                "Application execution failed",
                exception=e,
                context={
                    "job_id": job_id,
                    "image_uri": image_uri,
                    "operation": "run_application"
                },
                tags=["docker", "failure"]
            )

    def callback(self, message):
        """Process incoming Pub/Sub message"""
        try:
            job_data = json.loads(message.data.decode('utf-8'))
            job_id = job_data.get('jobId', 'unknown')
            
            # Update logger with job_id for this operation
            self.logger.job_id = job_id
            
            self.logger.info(
                "Received app execution request",
                context={
                    "job_id": job_id,
                    "job_data": job_data,
                    "operation": "message_received"
                },
                tags=["pubsub", "message"]
            )
            
            self.run_application(job_data)
            message.ack()
            
        except Exception as e:
            self.logger.error(
                "Error processing message",
                exception=e,
                context={"operation": "message_processing"},
                tags=["pubsub", "error"]
            )
            message.nack()

    def start_listening(self):
        """Start listening for Pub/Sub messages"""
        self.logger.info(
            "Starting Pub/Sub subscriber",
            context={
                "subscription_path": self.subscription_path,
                "operation": "start_listening"
            },
            tags=["pubsub", "startup"]
        )
        
        flow_control = pubsub_v1.types.FlowControl(max_messages=1)
        
        streaming_pull_future = self.subscriber.pull(
            request={"subscription": self.subscription_path, "max_messages": 1000},
            callback=self.callback,
            flow_control=flow_control,
        )
        
        self.logger.info(
            "Subscriber is now listening for messages",
            context={"operation": "listening"},
            tags=["pubsub", "ready"]
        )
        
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            self.logger.info(
                "Subscriber stopped by user",
                context={"operation": "shutdown"},
                tags=["pubsub", "shutdown"]
            )

if __name__ == "__main__":
    subscriber = AppSubscriber()
    subscriber.start_listening()