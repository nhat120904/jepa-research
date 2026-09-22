"""Queries on observed RGB, with no privileged simulator state dependencies."""
import numpy as np


def chroma_features(rgb):
    """Fixed spatial color kernel for the preflight, NOT a learned SSL encoder."""
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if x.ndim < 3 or x.shape[-1] != 3:
        raise ValueError("Expected [..., height, width, RGB]")
    x = x - x.mean(axis=-1, keepdims=True)
    x = x.reshape(*x.shape[:-3], -1)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def similarity(features, anchor):
    return np.clip(np.asarray(features) @ np.asarray(anchor), 0.0, 1.0)


def ordered_score(a, b):
    """max_{i<j} min(a_i,b_j), including an empty/one-step definition of zero."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError("Paired traces must have identical [..., time] shape")
    if a.shape[-1] < 2:
        return np.zeros(a.shape[:-1], dtype=np.float32)
    prefix = np.maximum.accumulate(a[..., :-1], axis=-1)
    return np.minimum(prefix, b[..., 1:]).max(axis=-1)


def answer(features, anchor_a, anchor_b):
    a, b = similarity(features, anchor_a), similarity(features, anchor_b)
    if a.shape[-1] == 0:
        raise ValueError("Empty future segment")
    return {
        "reach_b": b.max(axis=-1),
        "reach_a": a.max(axis=-1),
        "occupancy_a": a.mean(axis=-1),
        "endpoint_b": b[..., -1],
        "a_then_b": ordered_score(a, b),
        "b_then_a": ordered_score(b, a),
    }


def observed_phase(phase, similarity_a, similarity_b, threshold):
    """Advance using ONE actual observation; cannot satisfy A and B at the same time."""
    if phase not in (0, 1, 2) or not 0 < threshold <= 1:
        raise ValueError("Invalid phase/threshold")
    if phase == 0 and similarity_a >= threshold:
        return 1
    if phase == 1 and similarity_b >= threshold:
        return 2
    return phase
