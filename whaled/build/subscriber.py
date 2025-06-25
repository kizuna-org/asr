"""
Build Subscriber - Handles container build requests from Pub/Sub
"""

print("build")

try:
    import json

    print("[IMPORT] json: OK")
except ImportError as e:
    print(f"[IMPORT] json: FAILED - {e}")

try:
    import os

    print("[IMPORT] os: OK")
except ImportError as e:
    print(f"[IMPORT] os: FAILED - {e}")

try:
    import subprocess

    print("[IMPORT] subprocess: OK")
except ImportError as e:
    print(f"[IMPORT] subprocess: FAILED - {e}")

try:
    from datetime import datetime

    print("[IMPORT] datetime: OK")
except ImportError as e:
    print(f"[IMPORT] datetime: FAILED - {e}")

try:
    from google.cloud import pubsub_v1

    print("[IMPORT] google.cloud.pubsub_v1: OK")
except ImportError as e:
    print(f"[IMPORT] google.cloud.pubsub_v1: FAILED - {e}")

try:
    import boto3

    print("[IMPORT] boto3: OK")
except ImportError as e:
    print(f"[IMPORT] boto3: FAILED - {e}")

try:
    from botocore.config import Config

    print("[IMPORT] botocore.config: OK")
except ImportError as e:
    print(f"[IMPORT] botocore.config: FAILED - {e}")

try:
    from dotenv import load_dotenv

    print("[IMPORT] dotenv: OK")
except ImportError as e:
    print(f"[IMPORT] dotenv: FAILED - {e}")

try:
    import traceback

    print("[IMPORT] traceback: OK")
except ImportError as e:
    print(f"[IMPORT] traceback: FAILED - {e}")
