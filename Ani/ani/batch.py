"""Run a detector over many images, and rank images for review ("worst first").

A detector is either one of Ani's backends (Classic, Cellpose, your trained model…) or YOUR OWN
program, given as a command with placeholders, e.g.

    python C:/lab/my_detector.py --input {input} --output {output}

{input} is the image path, {output} the mask file Ani expects (.tif), {set} the label set name.
The program must write a label mask (0 = background, 1..N = objects) to {output}.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import io, tools
from .project import Project


def single_channel(image: np.ndarray, layout: dict, channel: int = 0) -> np.ndarray:
    """The plane a segmenter should look at: grey RGB, or one channel of a multi-channel stack."""
    if layout["rgb"]:
        return image[..., :3].mean(-1)
    if layout["channel_axis"] is not None:
        return image[min(channel, image.shape[0] - 1)]
    return image


def split_command(command: str) -> list[str]:
    return shlex.split(command, posix=os.name != "nt")


def run_external(command: str, image_path: Path, out_path: Path, label_set: str, timeout: float = 600) -> Path:
    """Run the user's own detector program for one image (no shell, so paths with spaces are safe)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    args = [a.replace("{input}", str(image_path)).replace("{output}", str(out_path)).replace("{set}", label_set)
            for a in split_command(command)]
    if not args:
        raise ValueError("The detector command is empty.")
    if args[0] in ("python", "python3", "py") and not getattr(sys, "frozen", False):
        args[0] = sys.executable                    # use Ani's Python, which has numpy etc.
        # (in the installed .exe sys.executable is Ani itself, so the system's python is used instead)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"detector exited with code {proc.returncode}: {' | '.join(msg)}")
    if not out_path.exists():
        found = io.find_mask(out_path.parent, image_path.stem)
        if found is None:
            raise RuntimeError(f"detector finished but did not write {out_path.name}")
        return found
    return out_path


def problem_score(image_1ch: np.ndarray | None, labels: np.ndarray) -> dict:
    suspects = tools.find_suspects(labels)
    missed = tools.find_missed(image_1ch, labels) if image_1ch is not None and image_1ch.shape == labels.shape else []
    n = len(tools.label_ids(labels))
    return {"suspects": len(suspects), "missed": len(missed), "objects": n,
            "score": round((len(suspects) + 2 * len(missed)) / max(n, 1) * 100, 1)}


def run_detector(project: Project, label_set: str, stems: list[str], backend=None, command: str | None = None,
                 params: dict | None = None, overwrite: bool = False, channel: int = 0):
    """Generator: detect objects in each image, write predictions/<set>/<stem>.tif, score it.

    Yields (index, total, stem, message). Stop early by simply not iterating further."""
    if (backend is None) == (not command):
        raise ValueError("Give either a backend or a command.")
    out_dir = project.predictions_dir / label_set
    total = len(stems)
    for i, stem in enumerate(stems):
        t0 = time.time()
        path = project.image_path(stem)
        if path is None:
            yield i, total, stem, "missing image – skipped"
            continue
        if not overwrite and project.prediction_path(label_set, stem) is not None:
            yield i, total, stem, "already has detector output – skipped"
            continue
        try:
            image = io.read_image(path)
            layout = io.guess_layout(image)
            plane = single_channel(image, layout, channel)
            target = out_dir / f"{stem}.tif"
            if command:
                written = run_external(command, path, target, label_set)
                mask, _ = io.read_mask(written, plane.shape)
                if written != target:
                    written.unlink()
                if tuple(mask.shape) != tuple(plane.shape):
                    raise RuntimeError(f"detector mask is {mask.shape}, image is {plane.shape}")
            else:
                mask = backend.auto_segment(plane, params or {})
            # never leave an older prediction (other formats) shadowing the new one
            for suffix in io.MASK_SUFFIXES:
                for ext in io.MASK_EXTS:
                    old = out_dir / f"{stem}{suffix}{ext}"
                    if old != target and old.exists():
                        old.unlink()
            io.write_mask(target, mask)
            score = problem_score(plane, mask)
            project.update_image(stem, "problems", label_set, score)
            yield i, total, stem, f"{score['objects']} objects, {time.time() - t0:.1f}s"
        except Exception as e:  # one bad image must not stop the batch
            yield i, total, stem, f"FAILED: {e}"


def rank_images(project: Project, label_set: str, stems: list[str], channel: int = 0):
    """Generator: compute the problem score of every image from its current labels (corrected or predicted)."""
    total = len(stems)
    for i, stem in enumerate(stems):
        path = project.image_path(stem)
        if path is None:
            continue
        try:
            image = io.read_image(path)
            layout = io.guess_layout(image)
            plane = single_channel(image, layout, channel)
            labels, _, _ = project.load_labels(label_set, stem, plane.shape)
            score = problem_score(plane, labels)
            project.update_image(stem, "problems", label_set, score)
            yield i, total, stem, f"score {score['score']}"
        except Exception as e:
            yield i, total, stem, f"FAILED: {e}"


def problem_value(project: Project, stem: str, label_set: str) -> float:
    return project.meta["images"].get(stem, {}).get("problems", {}).get(label_set, {}).get("score", -1.0)
