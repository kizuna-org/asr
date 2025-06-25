"""
App Subscriber - Handles application execution requests from Pub/Sub
"""

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import pubsub_v1
import boto3
from botocore.config import Config

# Load environment variables from .env file
load_dotenv("/app/config/.env")

# Add the shared directory to the path to import logger
sys.path.append("/app/shared")
from logger import StructuredLogger, Component


class AppSubscriber:
    def __init__(self):
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.subscription_name = os.environ.get("APP_SUBSCRIPTION", "app-triggers-sub")

        # Check network connectivity before initializing clients
        self._check_network_connectivity()

        # Initialize PubSub client with retry settings
        self.subscriber = pubsub_v1.SubscriberClient(
            # Configure client retry settings
            client_options={
                "api_endpoint": os.environ.get(
                    "PUBSUB_EMULATOR_HOST", "pubsub.googleapis.com:443"
                )
            }
        )
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_name
        )

        # Initialize structured logger
        self.logger = StructuredLogger(Component.APP_SUBSCRIBER)

        # Cloudflare R2 setup
        self.r2_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        self.bucket_name = os.environ.get("R2_BUCKET_NAME")

    def get_current_status(self, job_id):
        """Get current status from R2"""
        try:
            status_key = f"{job_id}/status.json"
            response = self.r2_client.get_object(
                Bucket=self.bucket_name, Key=status_key
            )
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception as e:
            self.logger.error(
                f"Failed to get status for job {job_id}",
                {"error": str(e), "job_id": job_id},
            )
            return None

    def update_status(self, job_id, status_data):
        """Update status in R2"""
        try:
            status_key = f"{job_id}/status.json"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=status_key,
                Body=json.dumps(status_data),
                ContentType="application/json",
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to update status for job {job_id}",
                {"error": str(e), "job_id": job_id},
            )
            return False

    def process_message(self, message):
        """Process a message from Pub/Sub"""
        try:
            data = json.loads(message.data.decode("utf-8"))
            job_id = data.get("job_id")

            if not job_id:
                self.logger.error("Message missing job_id", {"data": data})
                message.ack()
                return

            self.logger.info(f"Processing job {job_id}", {"job_id": job_id})

            # Get current status
            status = self.get_current_status(job_id)
            if not status:
                self.logger.error(
                    f"No status found for job {job_id}", {"job_id": job_id}
                )
                message.ack()
                return

            # Update status to processing
            status["status"] = "processing"
            status["processing_started_at"] = datetime.utcnow().isoformat()
            self.update_status(job_id, status)

            # Run the job
            success = self.run_job(job_id, data)

            # Update final status
            status = self.get_current_status(job_id)
            if status:
                status["status"] = "completed" if success else "failed"
                status["completed_at"] = datetime.utcnow().isoformat()
                self.update_status(job_id, status)

            # Acknowledge the message
            message.ack()

        except Exception as e:
            self.logger.error(
                "Error processing message",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # Acknowledge the message to prevent redelivery
            message.ack()

    def run_job(self, job_id, data):
        """Run the job in a Docker container"""
        try:
            # Pull the latest app image
            self.logger.info(
                f"Pulling latest app image for job {job_id}", {"job_id": job_id}
            )
            subprocess.run(
                ["docker", "pull", "ghcr.io/kizuna-org/chumchat-app:latest"], check=True
            )

            # Run the container
            self.logger.info(f"Starting container for job {job_id}", {"job_id": job_id})

            # Prepare the docker run command
            cmd = f"""
            docker run --rm \
                -e JOB_ID={job_id} \
                -e R2_ENDPOINT_URL={os.environ.get('R2_ENDPOINT_URL')} \
                -e R2_ACCESS_KEY_ID={os.environ.get('R2_ACCESS_KEY_ID')} \
                -e R2_SECRET_ACCESS_KEY={os.environ.get('R2_SECRET_ACCESS_KEY')} \
                -e R2_BUCKET_NAME={os.environ.get('R2_BUCKET_NAME')} \
                -e HF_TOKEN={os.environ.get('HF_TOKEN')} \
                ghcr.io/kizuna-org/chumchat-app:latest
            """

            # Run the command
            result = subprocess.run(cmd, shell=True, check=False)

            if result.returncode != 0:
                self.logger.error(
                    f"Container for job {job_id} exited with non-zero status",
                    {"job_id": job_id, "return_code": result.returncode},
                )
                return False

            self.logger.info(
                f"Container for job {job_id} completed successfully", {"job_id": job_id}
            )
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Error running container for job {job_id}",
                {"job_id": job_id, "error": str(e)},
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error running job {job_id}",
                {"job_id": job_id, "error": str(e)},
            )
            return False

    def start_listening(self):
        """Start listening for messages"""
        self.logger.info(
            "Starting subscriber",
            {"project_id": self.project_id, "subscription": self.subscription_name},
        )

        def callback(message):
            self.process_message(message)

        # Add retry logic for network issues
        max_retries = 5
        retry_count = 0
        retry_delay = 10  # seconds

        while retry_count < max_retries:
            try:
                streaming_pull_future = self.subscriber.subscribe(
                    self.subscription_path, callback=callback
                )

                # Keep the main thread from exiting
                streaming_pull_future.result()
                break  # If we get here, subscription is working

            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Subscriber connection error (attempt {retry_count}/{max_retries})",
                    {"error": str(e), "traceback": traceback.format_exc()},
                )

                if retry_count >= max_retries:
                    self.logger.error("Max retries reached, giving up", {})
                    raise

                self.logger.info(f"Retrying in {retry_delay} seconds...", {})
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def _check_network_connectivity(self):
        """Check network connectivity to key services"""
        try:
            # Import socket for network connectivity checks
            import socket
            import urllib.request

            # Check DNS resolution
            socket.gethostbyname("pubsub.googleapis.com")

            # Check internet connectivity
            urllib.request.urlopen("https://www.google.com", timeout=5)

            return True
        except Exception as e:
            print(f"Network connectivity check failed: {str(e)}")
            print("Continuing anyway, will retry with exponential backoff...")
            return False


if __name__ == "__main__":
    subscriber = AppSubscriber()
    subscriber.start_listening()
