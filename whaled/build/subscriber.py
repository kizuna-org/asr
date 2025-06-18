#!/usr/bin/env python3
"""
Build Subscriber - Handles container build requests from Pub/Sub
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

class BuildSubscriber:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        self.subscription_name = os.environ.get('BUILD_SUBSCRIPTION', 'build-triggers-sub')
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
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

    def update_status(self, job_id, status_data):
        """Update job status in Cloudflare R2"""
        try:
            status_key = f"{job_id}/status.json"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=status_key,
                Body=json.dumps(status_data, indent=2),
                ContentType='application/json'
            )
            print(f"Status updated for job {job_id}")
        except Exception as e:
            print(f"Failed to update status: {e}")

    def upload_log(self, job_id, log_content, log_type='build'):
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

    def build_and_push_image(self, job_data):
        """Build Docker image and push to GHCR"""
        job_id = job_data['jobId']
        repository = job_data['repository']
        
        # Update status to building
        status = {
            'jobId': job_id,
            'overallStatus': 'Building',
            'timestamps': {
                'created': datetime.utcnow().isoformat() + 'Z',
                'updated': datetime.utcnow().isoformat() + 'Z'
            },
            'build': {
                'status': 'Building',
                'log': f'/{job_id}/build.log'
            }
        }
        self.update_status(job_id, status)

        try:
            # Clone repository
            clone_cmd = f"git clone https://github.com/{repository}.git /tmp/{job_id}"
            subprocess.run(clone_cmd, shell=True, check=True)
            
            # Build Docker image
            image_tag = f"ghcr.io/{repository.lower()}:{job_id}"
            build_cmd = f"cd /tmp/{job_id} && docker build -t {image_tag} ."
            
            result = subprocess.run(
                build_cmd, 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            # Upload build log
            build_log = f"Build command: {build_cmd}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            self.upload_log(job_id, build_log, 'build')
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, build_cmd)
            
            # Push to GHCR
            push_cmd = f"docker push {image_tag}"
            push_result = subprocess.run(
                push_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if push_result.returncode != 0:
                raise subprocess.CalledProcessError(push_result.returncode, push_cmd)
            
            # Update status to build succeeded
            status['build']['status'] = 'Succeeded'
            status['build']['imageUri'] = image_tag
            status['overallStatus'] = 'BuildSucceeded'
            status['timestamps']['updated'] = datetime.utcnow().isoformat() + 'Z'
            self.update_status(job_id, status)
            
            # Trigger app execution
            self.trigger_app_execution(job_id, image_tag)
            
            print(f"Build succeeded for job {job_id}")
            
        except subprocess.CalledProcessError as e:
            # Update status to build failed
            status['build']['status'] = 'Failed'
            status['overallStatus'] = 'BuildFailed'
            status['timestamps']['updated'] = datetime.utcnow().isoformat() + 'Z'
            self.update_status(job_id, status)
            print(f"Build failed for job {job_id}: {e}")
        
        finally:
            # Cleanup
            subprocess.run(f"rm -rf /tmp/{job_id}", shell=True)

    def trigger_app_execution(self, job_id, image_uri):
        """Trigger app execution by publishing to app-triggers topic"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, 'app-triggers')
            message_data = {
                'jobId': job_id,
                'imageUri': image_uri,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            future = self.publisher.publish(
                topic_path,
                json.dumps(message_data).encode('utf-8')
            )
            future.result()  # Wait for publish to complete
            print(f"App execution triggered for job {job_id}")
            
        except Exception as e:
            print(f"Failed to trigger app execution: {e}")

    def callback(self, message):
        """Process incoming Pub/Sub message"""
        try:
            job_data = json.loads(message.data.decode('utf-8'))
            print(f"Received build request: {job_data}")
            
            self.build_and_push_image(job_data)
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
    subscriber = BuildSubscriber()
    subscriber.start_listening()