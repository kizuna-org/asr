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

class AppSubscriber:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        self.subscription_name = os.environ.get('APP_SUBSCRIPTION', 'app-triggers-sub')
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_name
        )
        
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
            print(f"Failed to get current status: {e}")
            return None

    def update_status(self, job_id, status_updates):
        """Update job status in Cloudflare R2"""
        try:
            # Get current status first
            current_status = self.get_current_status(job_id)
            if not current_status:
                print(f"No existing status found for job {job_id}")
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
            print(f"Status updated for job {job_id}")
        except Exception as e:
            print(f"Failed to update status: {e}")

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
            print(f"Log uploaded: {log_key}")
        except Exception as e:
            print(f"Failed to upload log: {e}")

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
            pull_cmd = f"docker pull {image_uri}"
            pull_result = subprocess.run(
                pull_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if pull_result.returncode != 0:
                raise subprocess.CalledProcessError(pull_result.returncode, pull_cmd)
            
            # Run container with environment variables for R2 access
            run_cmd = f"""docker run --rm \
                -e JOB_ID={job_id} \
                -e R2_ENDPOINT_URL={os.environ.get('R2_ENDPOINT_URL')} \
                -e R2_ACCESS_KEY_ID={os.environ.get('R2_ACCESS_KEY_ID')} \
                -e R2_SECRET_ACCESS_KEY={os.environ.get('R2_SECRET_ACCESS_KEY')} \
                -e R2_BUCKET_NAME={os.environ.get('R2_BUCKET_NAME')} \
                -e HF_TOKEN={os.environ.get('HF_TOKEN')} \
                --gpus all \
                {image_uri}"""
            
            print(f"Running container: {image_uri}")
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
            
            print(f"Application execution succeeded for job {job_id}")
            
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
            print(f"Application execution failed for job {job_id}: {e}")

    def callback(self, message):
        """Process incoming Pub/Sub message"""
        try:
            job_data = json.loads(message.data.decode('utf-8'))
            print(f"Received app execution request: {job_data}")
            
            self.run_application(job_data)
            message.ack()
            
        except Exception as e:
            print(f"Error processing message: {e}")
            message.nack()

    def start_listening(self):
        """Start listening for Pub/Sub messages"""
        print(f"Listening for messages on {self.subscription_path}")
        
        flow_control = pubsub_v1.types.FlowControl(max_messages=1)
        
        streaming_pull_future = self.subscriber.pull(
            request={"subscription": self.subscription_path, "max_messages": 1000},
            callback=self.callback,
            flow_control=flow_control,
        )
        
        print("Listening for messages...")
        
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            print("Subscriber stopped.")

if __name__ == "__main__":
    subscriber = AppSubscriber()
    subscriber.start_listening()