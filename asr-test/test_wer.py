#!/usr/bin/env python3
"""
Word Error Rate (WER) Testing Script for ASR

This script tests the word error rate of the ASR model by:
1. Loading audio files from LJSpeech-1.1 dataset
2. Calling the backend inference API
3. Comparing the transcription with the ground truth
4. Computing and logging WER metrics
"""

import os
import sys
import csv
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import time
from datetime import datetime
import torchaudio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wer_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    Tries multiple methods: torchaudio, librosa, or ffmpeg.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or 0.0 if unable to determine
    """
    # Method 1: Try torchaudio
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        return duration
    except Exception as e:
        logger.debug(f"torchaudio failed for {audio_path}: {e}")
    
    # Method 2: Try librosa
    try:
        import librosa
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except Exception as e:
        logger.debug(f"librosa failed for {audio_path}: {e}")
    
    # Method 3: Try ffprobe
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1:noprint_filename=1', audio_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.debug(f"ffprobe failed for {audio_path}: {e}")
    
    logger.warning(f"Could not determine audio duration for {audio_path} using any method")
    return 0.0


def calculate_stt_speed(audio_duration: float, hypothesis: str, api_response: Dict) -> Dict:
    """
    Calculate speech-to-text speed metrics based on TTLT (Time To Last Token).

    Args:
        audio_duration: Duration of audio in seconds
        hypothesis: Transcribed text
        api_response: API response with timing information

    Returns:
        Dictionary with speed metrics
    """
    metrics = {}

    # Get timing information - use TTLT (Time To Last Token)
    ttlt_ms = api_response.get('ttlt_ms', 0) or api_response.get('total_time_ms', 0)
    inference_time_ms = api_response.get('inference_time_ms', 0)
    first_token_time_ms = api_response.get('first_token_time_ms', 0)

    ttlt_s = ttlt_ms / 1000.0 if ttlt_ms > 0 else 0
    inference_time_s = inference_time_ms / 1000.0 if inference_time_ms > 0 else 0
    first_token_time_s = first_token_time_ms / 1000.0 if first_token_time_ms > 0 else 0

    if audio_duration > 0:
        # Real-Time Factor (RTF): processing_time / audio_duration
        # RTF < 1 means faster than real-time
        metrics['rtf_ttlt'] = ttlt_s / audio_duration
        metrics['rtf_inference'] = inference_time_s / audio_duration
        metrics['rtf_first_token'] = first_token_time_s / audio_duration
    else:
        metrics['rtf_ttlt'] = None
        metrics['rtf_inference'] = None
        metrics['rtf_first_token'] = None

    # Words per second (based on TTLT)
    num_words = len(hypothesis.split()) if hypothesis else 0
    if ttlt_s > 0:
        metrics['words_per_second'] = num_words / ttlt_s
    else:
        metrics['words_per_second'] = None

    # Characters per second (based on TTLT)
    num_chars = len(hypothesis)
    if ttlt_s > 0:
        metrics['chars_per_second'] = num_chars / ttlt_s
    else:
        metrics['chars_per_second'] = None

    # Audio duration and processing time
    metrics['audio_duration_s'] = audio_duration
    metrics['ttlt_s'] = ttlt_s
    metrics['inference_time_s'] = inference_time_s
    metrics['first_token_time_s'] = first_token_time_s

    return metrics


def load_ljspeech_metadata(metadata_path: str) -> Dict[str, str]:
    """
    Load LJSpeech metadata file.

    Format: wav_id|transcription|normalized_transcription
    We use the transcription (second field) as ground truth.

    Args:
        metadata_path: Path to metadata.csv file

    Returns:
        Dictionary mapping audio_id -> transcription
    """
    metadata = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) >= 2:
                    audio_id = row[0].strip()
                    transcription = row[1].strip()
                    metadata[audio_id] = transcription
        logger.info(f"Loaded {len(metadata)} metadata entries from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


def compute_wer(ground_truth: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Compute Word Error Rate (WER) between ground truth and hypothesis.

    WER = (S + D + I) / N
    where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of words in reference

    Args:
        ground_truth: Reference transcription
        hypothesis: Hypothesis transcription from model

    Returns:
        Tuple of (wer_score, detailed_metrics)
    """
    # Normalize: convert to lowercase and split into words
    ref_words = ground_truth.lower().split()
    hyp_words = hypothesis.lower().split()

    # Compute edit distance (using dynamic programming)
    def edit_distance_with_ops(ref, hyp):
        """Compute edit distance and count operations"""
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        # Initialize
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )

        # Backtrack to count operations
        i, j = n, m
        substitutions = 0
        deletions = 0
        insertions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i-1][j-1] + 1 == dp[i][j]:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i-1][j] + 1 == dp[i][j]:
                deletions += 1
                i -= 1
            else:
                insertions += 1
                j -= 1

        return dp[n][m], substitutions, deletions, insertions

    distance, subs, dels, inss = edit_distance_with_ops(ref_words, hyp_words)
    num_words = len(ref_words)

    if num_words == 0:
        wer = 0.0 if distance == 0 else 1.0
    else:
        wer = distance / num_words

    metrics = {
        'wer': wer,
        'edit_distance': distance,
        'substitutions': subs,
        'deletions': dels,
        'insertions': inss,
        'num_ref_words': num_words,
        'num_hyp_words': len(hyp_words),
    }

    return wer, metrics


def call_inference_api(audio_path: str, backend_url: str, model_name: str = 'conformer') -> Dict:
    """
    Call the backend inference API with an audio file.

    Args:
        audio_path: Path to audio file
        backend_url: Backend API base URL (e.g., 'http://localhost:58081/api')
        model_name: Model to use ('conformer' or 'realtime')

    Returns:
        Dictionary with transcription and timing info
    """
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            params = {'model_name': model_name}
            response = requests.post(
                f'{backend_url}/inference',
                files=files,
                params=params,
                timeout=120
            )

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API error for {audio_path}: {response.status_code} - {response.text}")
            return {'transcription': '', 'error': response.text}

    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling API for {audio_path}")
        return {'transcription': '', 'error': 'Timeout'}
    except Exception as e:
        logger.error(f"Failed to call API for {audio_path}: {e}")
        return {'transcription': '', 'error': str(e)}


def test_wer(
    ljspeech_path: str,
    backend_url: str,
    model_name: str = 'conformer',
    num_samples: int = None,
    output_json: str = None
):
    """
    Run WER test on LJSpeech dataset.

    Args:
        ljspeech_path: Path to LJSpeech-1.1 directory
        backend_url: Backend API base URL
        model_name: Model to test
        num_samples: Number of samples to test (None = all)
        output_json: Path to save detailed results as JSON
    """
    # Validate paths
    ljspeech_path = Path(ljspeech_path)
    wavs_dir = ljspeech_path / 'wavs'
    metadata_file = ljspeech_path / 'metadata.csv'

    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        sys.exit(1)

    if not wavs_dir.exists():
        logger.error(f"Wavs directory not found: {wavs_dir}")
        sys.exit(1)

    # Load metadata
    metadata = load_ljspeech_metadata(str(metadata_file))

    # Get list of audio files to test
    audio_files = sorted([f for f in wavs_dir.glob('*.wav')])
    if num_samples:
        audio_files = random.sample(audio_files, min(num_samples, len(audio_files)))

    logger.info(f"Testing {len(audio_files)} audio files with model '{model_name}'")
    logger.info(f"Backend URL: {backend_url}")

    results = []
    wer_scores = []

    for idx, audio_file in enumerate(audio_files, 1):
        audio_id = audio_file.stem
        ground_truth = metadata.get(audio_id)

        if not ground_truth:
            logger.warning(f"No ground truth for {audio_id}, skipping")
            continue

        # Call inference API
        logger.info(f"[{idx}/{len(audio_files)}] Processing {audio_id}...")
        api_response = call_inference_api(str(audio_file), backend_url, model_name)

        if 'error' in api_response:
            logger.error(f"Failed to get transcription for {audio_id}")
            result = {
                'audio_id': audio_id,
                'ground_truth': ground_truth,
                'hypothesis': '',
                'wer': None,
                'error': api_response['error']
            }
            results.append(result)
            continue

        hypothesis = api_response.get('transcription', '')

        # Compute WER
        wer, metrics = compute_wer(ground_truth, hypothesis)
        wer_scores.append(wer)

        # Calculate STT speed metrics
        audio_duration = get_audio_duration(str(audio_file))
        speed_metrics = calculate_stt_speed(audio_duration, hypothesis, api_response)

        result = {
            'audio_id': audio_id,
            'ground_truth': ground_truth,
            'hypothesis': hypothesis,
            'wer': wer,
            'metrics': metrics,
            'speed_metrics': speed_metrics,
            'api_response': {
                'first_token_time_ms': api_response.get('first_token_time_ms'),
                'inference_time_ms': api_response.get('inference_time_ms'),
                'total_time_ms': api_response.get('total_time_ms'),
            }
        }
        results.append(result)

        # Log progress with speed metrics
        rtf_str = f"RTF: {speed_metrics['rtf_ttlt']:.3f}" if speed_metrics['rtf_ttlt'] is not None else "RTF: N/A"
        wps_str = f"WPS: {speed_metrics['words_per_second']:.2f}" if speed_metrics['words_per_second'] is not None else "WPS: N/A"
        logger.info(
            f"  WER: {wer:.4f} | "
            f"Ref words: {metrics['num_ref_words']} | "
            f"Hyp words: {metrics['num_hyp_words']} | "
            f"(S:{metrics['substitutions']}, D:{metrics['deletions']}, I:{metrics['insertions']}) | "
            f"{rtf_str} | {wps_str}"
        )

    # Compute aggregate statistics
    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        min_wer = min(wer_scores)
        max_wer = max(wer_scores)

        # Compute speed statistics
        rtf_values = [r['speed_metrics']['rtf_ttlt'] for r in results
                     if 'error' not in r and r['speed_metrics']['rtf_ttlt'] is not None]
        wps_values = [r['speed_metrics']['words_per_second'] for r in results
                     if 'error' not in r and r['speed_metrics']['words_per_second'] is not None]
        cps_values = [r['speed_metrics']['chars_per_second'] for r in results
                     if 'error' not in r and r['speed_metrics']['chars_per_second'] is not None]
        audio_durations = [r['speed_metrics']['audio_duration_s'] for r in results
                          if 'error' not in r and r['speed_metrics']['audio_duration_s'] is not None]
        ttlt_times = [r['speed_metrics']['ttlt_s'] for r in results
                      if 'error' not in r and r['speed_metrics']['ttlt_s'] is not None]

        logger.info("\n" + "="*70)
        logger.info("SUMMARY STATISTICS - WER")
        logger.info("="*70)
        logger.info(f"Total samples processed: {len(results)}")
        logger.info(f"Average WER: {avg_wer:.4f}")
        logger.info(f"Min WER: {min_wer:.4f}")
        logger.info(f"Max WER: {max_wer:.4f}")
        logger.info(f"Model: {model_name}")
        logger.info("="*70)

        logger.info("\n" + "="*70)
        logger.info("SUMMARY STATISTICS - STT SPEED (based on TTLT)")
        logger.info("="*70)
        if rtf_values:
            avg_rtf = sum(rtf_values) / len(rtf_values)
            min_rtf = min(rtf_values)
            max_rtf = max(rtf_values)
            logger.info(f"Real-Time Factor (RTF):")
            logger.info(f"  Average: {avg_rtf:.4f} (1.0 = real-time, <1.0 = faster than real-time)")
            logger.info(f"  Min: {min_rtf:.4f}")
            logger.info(f"  Max: {max_rtf:.4f}")

        if wps_values:
            avg_wps = sum(wps_values) / len(wps_values)
            logger.info(f"Words per Second (TTLT-based):")
            logger.info(f"  Average: {avg_wps:.2f} words/sec")

        if cps_values:
            avg_cps = sum(cps_values) / len(cps_values)
            logger.info(f"Characters per Second (TTLT-based):")
            logger.info(f"  Average: {avg_cps:.2f} chars/sec")

        if audio_durations and ttlt_times:
            avg_audio_duration = sum(audio_durations) / len(audio_durations)
            avg_ttlt_time = sum(ttlt_times) / len(ttlt_times)
            logger.info(f"Average Audio Duration: {avg_audio_duration*1000:.2f}ms")
            logger.info(f"Average TTLT Time: {avg_ttlt_time*1000:.2f}ms")
            if avg_audio_duration > 0:
                logger.info(f"Overall RTF (TTLT-based): {avg_ttlt_time/avg_audio_duration:.4f}")

        logger.info("="*70)

    # Save detailed results to JSON
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute speed statistics for summary
        rtf_values = [r['speed_metrics']['rtf_ttlt'] for r in results
                     if 'error' not in r and r['speed_metrics']['rtf_ttlt'] is not None]
        wps_values = [r['speed_metrics']['words_per_second'] for r in results
                     if 'error' not in r and r['speed_metrics']['words_per_second'] is not None]
        cps_values = [r['speed_metrics']['chars_per_second'] for r in results
                     if 'error' not in r and r['speed_metrics']['chars_per_second'] is not None]
        audio_durations = [r['speed_metrics']['audio_duration_s'] for r in results
                          if 'error' not in r and r['speed_metrics']['audio_duration_s'] is not None]
        ttlt_times = [r['speed_metrics']['ttlt_s'] for r in results
                      if 'error' not in r and r['speed_metrics']['ttlt_s'] is not None]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'backend_url': backend_url,
            'num_samples': len(results),
            'statistics': {
                'wer': {
                    'average_wer': avg_wer if wer_scores else None,
                    'min_wer': min_wer if wer_scores else None,
                    'max_wer': max_wer if wer_scores else None,
                },
                'speed': {
                    'metric': 'TTLT (Time To Last Token)',
                    'average_rtf': sum(rtf_values) / len(rtf_values) if rtf_values else None,
                    'min_rtf': min(rtf_values) if rtf_values else None,
                    'max_rtf': max(rtf_values) if rtf_values else None,
                    'average_words_per_second': sum(wps_values) / len(wps_values) if wps_values else None,
                    'average_chars_per_second': sum(cps_values) / len(cps_values) if cps_values else None,
                    'average_audio_duration_ms': (sum(audio_durations) / len(audio_durations) * 1000) if audio_durations else None,
                    'average_ttlt_time_ms': (sum(ttlt_times) / len(ttlt_times) * 1000) if ttlt_times else None,
                    'overall_rtf': (sum(ttlt_times) / len(ttlt_times)) / (sum(audio_durations) / len(audio_durations)) if (audio_durations and ttlt_times and len(audio_durations) > 0 and sum(audio_durations) > 0) else None,
                },
                'samples': {
                    'total_samples': len(results),
                    'successful_samples': sum(1 for r in results if 'error' not in r),
                }
            },
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test Word Error Rate (WER) on LJSpeech dataset'
    )
    parser.add_argument(
        '--ljspeech-path',
        default='../LJSpeech-1.1',
        help='Path to LJSpeech-1.1 directory (default: ./LJSpeech-1.1)'
    )
    parser.add_argument(
        '--backend-url',
        default='http://localhost:58081/api',
        help='Backend API base URL (default: http://localhost:58081/api)'
    )
    parser.add_argument(
        '--model',
        default='conformer',
        choices=['conformer', 'realtime'],
        help='Model to test (default: conformer)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to test (default: all)'
    )
    parser.add_argument(
        '--output-json',
        help='Path to save detailed results as JSON'
    )

    args = parser.parse_args()

    logger.info(f"Starting WER test with parameters:")
    logger.info(f"  LJSpeech path: {args.ljspeech_path}")
    logger.info(f"  Backend URL: {args.backend_url}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Num samples: {args.num_samples if args.num_samples else 'all'}")

    test_wer(
        ljspeech_path=args.ljspeech_path,
        backend_url=args.backend_url,
        model_name=args.model,
        num_samples=args.num_samples,
        output_json=args.output_json
    )


if __name__ == '__main__':
    main()
