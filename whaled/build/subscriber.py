#!/usr/bin/env python3
"""
Build Subscriber - Handles container build requests from Pub/Sub
"""

import json
import os
import subprocess
from datetime import datetime
from google import pubsub_v1
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import traceback


class BuildSubscriber:
    def __init__(self):
        print("[INFO] Initializing BuildSubscriber...")
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.subscription_name = os.environ.get(
            "BUILD_SUBSCRIPTION", "build-triggers-sub"
        )
        print(f"[INFO] GCP_PROJECT_ID: {self.project_id}")
        print(f"[INFO] BUILD_SUBSCRIPTION: {self.subscription_name}")
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_name
        )
        print(f"[INFO] Subscription path: {self.subscription_path}")

        # Cloudflare R2 setup
        print("[INFO] Setting up Cloudflare R2 client...")
        self.r2_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        self.bucket_name = os.environ.get("R2_BUCKET_NAME")
        print(f"[INFO] R2_BUCKET_NAME: {self.bucket_name}")

    def update_status(self, job_id, status_data):
        """Update job status in Cloudflare R2"""
        try:
            status_key = f"{job_id}/status.json"
            print(f"[INFO] Updating status for job {job_id} at {status_key}...")
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=status_key,
                Body=json.dumps(status_data, indent=2),
                ContentType="application/json",
            )
            print(f"[INFO] Status updated for job {job_id}")
        except Exception as e:
            print(f"[ERROR] Failed to update status for job {job_id}: {e}")
            traceback.print_exc()

    def upload_log(self, job_id, log_content, log_type="build"):
        """Upload log content to R2"""
        try:
            log_key = f"{job_id}/{log_type}.log"
            print(f"[INFO] Uploading {log_type} log for job {job_id} to {log_key}...")
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=log_key,
                Body=log_content,
                ContentType="text/plain",
            )
            print(f"[INFO] Log uploaded: {log_key}")
        except Exception as e:
            print(f"[ERROR] Failed to upload log for job {job_id}: {e}")
            traceback.print_exc()

    def build_and_push_image(self, job_data):
        """Build Docker image and push to GHCR"""
        job_id = job_data["jobId"]
        repository = job_data["repository"]

        print(f"[INFO] Starting build for job {job_id}, repository: {repository}")

        # Update status to building
        status = {
            "jobId": job_id,
            "overallStatus": "Building",
            "timestamps": {
                "created": datetime.utcnow().isoformat() + "Z",
                "updated": datetime.utcnow().isoformat() + "Z",
            },
            "build": {"status": "Building", "log": f"/{job_id}/build.log"},
        }
        self.update_status(job_id, status)

        try:
            # Clone repository
            clone_cmd = f"git clone https://github.com/{repository}.git /tmp/{job_id}"
            print(f"[INFO] Cloning repository with command: {clone_cmd}")
            subprocess.run(clone_cmd, shell=True, check=True)
            print(f"[INFO] Repository cloned to /tmp/{job_id}")

            # Build Docker image
            image_tag = f"ghcr.io/{repository.lower()}:{job_id}"
            build_cmd = f"cd /tmp/{job_id} && docker build -t {image_tag} ."
            print(f"[INFO] Building Docker image with command: {build_cmd}")

            result = subprocess.run(
                build_cmd, shell=True, capture_output=True, text=True
            )

            # Upload build log
            build_log = (
                f"Build command: {build_cmd}\n\nSTDOUT:\n{result.stdout}"
                f"\n\nSTDERR:\n{result.stderr}"
            )
            self.upload_log(job_id, build_log, "build")

            print(f"[INFO] Docker build finished with return code {result.returncode}")
            if result.returncode != 0:
                print(f"[ERROR] Docker build failed for job {job_id}")
                raise subprocess.CalledProcessError(result.returncode, build_cmd)

            # Push to GHCR
            push_cmd = f"docker push {image_tag}"
            print(f"[INFO] Pushing Docker image with command: {push_cmd}")
            push_result = subprocess.run(
                push_cmd, shell=True, capture_output=True, text=True
            )
            print(
                f"[INFO] Docker push finished with return code {push_result.returncode}"
            )
            if push_result.returncode != 0:
                print(f"[ERROR] Docker push failed for job {job_id}")
                print(f"[ERROR] STDOUT: {push_result.stdout}")
                print(f"[ERROR] STDERR: {push_result.stderr}")
                raise subprocess.CalledProcessError(push_result.returncode, push_cmd)

            # Update status to build succeeded
            status["build"]["status"] = "Succeeded"
            status["build"]["imageUri"] = image_tag
            status["overallStatus"] = "BuildSucceeded"
            status["timestamps"]["updated"] = datetime.utcnow().isoformat() + "Z"
            self.update_status(job_id, status)

            # Trigger app execution
            print(
                f"[INFO] Triggering app execution for job {job_id} with image {image_tag}"
            )
            self.trigger_app_execution(job_id, image_tag)

            print(f"[INFO] Build succeeded for job {job_id}")

        except subprocess.CalledProcessError as e:
            # Update status to build failed
            print(f"[ERROR] Subprocess error during build for job {job_id}: {e}")
            print(f"[ERROR] Traceback:")
            traceback.print_exc()
            status["build"]["status"] = "Failed"
            status["overallStatus"] = "BuildFailed"
            status["timestamps"]["updated"] = datetime.utcnow().isoformat() + "Z"
            self.update_status(job_id, status)
            print(f"[ERROR] Build failed for job {job_id}: {e}")

        except Exception as e:
            print(f"[ERROR] Unexpected error during build for job {job_id}: {e}")
            traceback.print_exc()
            status["build"]["status"] = "Failed"
            status["overallStatus"] = "BuildFailed"
            status["timestamps"]["updated"] = datetime.utcnow().isoformat() + "Z"
            self.update_status(job_id, status)
            print(f"[ERROR] Build failed for job {job_id}: {e}")

        finally:
            # Cleanup
            print(f"[INFO] Cleaning up /tmp/{job_id}")
            subprocess.run(f"rm -rf /tmp/{job_id}", shell=True)

    def trigger_app_execution(self, job_id, image_uri):
        """Trigger app execution by publishing to app-triggers topic"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, "app-triggers")
            print(
                f"[INFO] Publishing app execution trigger for job {job_id} to topic {topic_path}"
            )
            message_data = {
                "jobId": job_id,
                "imageUri": image_uri,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            future = self.publisher.publish(
                topic_path, json.dumps(message_data).encode("utf-8")
            )
            future.result()  # Wait for publish to complete
            print(f"[INFO] App execution triggered for job {job_id}")

        except Exception as e:
            print(f"[ERROR] Failed to trigger app execution for job {job_id}: {e}")
            traceback.print_exc()

    def callback(self, message):
        """Process incoming Pub/Sub message"""
        try:
            print("[INFO] Received new Pub/Sub message.")
            job_data = json.loads(message.data.decode("utf-8"))
            print(f"[INFO] Received build request: {job_data}")

            self.build_and_push_image(job_data)
            print("[INFO] Acknowledging message.")
            message.ack()

        except Exception as e:
            print(f"[ERROR] Error processing message: {e}")
            traceback.print_exc()
            print("[INFO] Nacking message.")
            message.nack()

    def start_listening(self):
        """Start listening for Pub/Sub messages"""
        print(f"[INFO] Listening for messages on {self.subscription_path}")

        flow_control = pubsub_v1.types.FlowControl(max_messages=1)

        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.callback,
            flow_control=flow_control,
        )

        print("[INFO] Listening for messages...")

        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            print("[INFO] Subscriber stopped.")


if __name__ == "__main__":
    print("[INFO] Starting BuildSubscriber main...")
    subscriber = BuildSubscriber()
    subscriber.start_listening()
