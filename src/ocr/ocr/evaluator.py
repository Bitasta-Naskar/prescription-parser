"""
OCR Evaluation and Error Analysis Module.
Calculates Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein
distance, and benchmarks failure patterns on doctor handwriting.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein distance with operation counts:
    returns (distance, substitutions, deletions, insertions).
    """
    m, n = len(seq1), len(seq2)
    # dp[i][j] stores (cost, subs, dels, ins)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return int(dp[m][n])


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate:
    CER = LevenshteinDistance(ref, hyp) / len(ref)
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate:
    WER = LevenshteinDistance(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


@dataclass
class EvaluationReport:
    """Structured benchmark report comparing OCR output against ground truth."""
    cer: float
    wer: float
    reference_length: int
    hypothesis_length: int
    ground_truth: str
    ocr_prediction: str
    analysis: Dict[str, Any]


def evaluate_ocr(ground_truth: str, ocr_output: str) -> EvaluationReport:
    """
    Generate complete evaluation report comparing ground truth prescription with OCR prediction.
    """
    clean_gt = " ".join(ground_truth.strip().split())
    clean_hyp = " ".join(ocr_output.strip().split())

    cer = compute_cer(clean_gt, clean_hyp)
    wer = compute_wer(clean_gt, clean_hyp)

    # Analyze common handwriting failure modes
    gt_words = clean_gt.lower().split()
    hyp_words = clean_hyp.lower().split()

    found_words = [w for w in gt_words if w in hyp_words]
    missed_words = [w for w in gt_words if w not in hyp_words]
    word_recall = len(found_words) / len(gt_words) if gt_words else 1.0

    analysis = {
        "word_recall": round(word_recall, 4),
        "total_gt_words": len(gt_words),
        "recognized_gt_words": len(found_words),
        "missed_gt_words": missed_words[:10],
    }

    return EvaluationReport(
        cer=round(cer, 4),
        wer=round(wer, 4),
        reference_length=len(clean_gt),
        hypothesis_length=len(clean_hyp),
        ground_truth=clean_gt,
        ocr_prediction=clean_hyp,
        analysis=analysis,
    )
