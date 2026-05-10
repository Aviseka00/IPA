"""Run U-Net segmentation + Cellpose-style overlay + count annotations."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve imports from IPA3/ml_infection
_IPA3 = Path(__file__).resolve().parents[4]
_ML = _IPA3 / "ml_infection"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from image_io import preprocess_clahe_lab, read_image_rgb  # noqa: E402
from instance_split import choose_labels_for_counting, labels_true_positives_filtered  # noqa: E402
from predict import InfectionSegmenter  # noqa: E402
from visualize_inference import (  # noqa: E402
    annotate_counts_on_image,
    build_observation_panel_rgb,
    overlay_cellpose_style,
)


class InferenceService:
    def __init__(self, checkpoint: Path, device=None):
        self.checkpoint = Path(checkpoint)
        self._segmenter: InfectionSegmenter | None = None
        self._device = device

    def _seg(self) -> InfectionSegmenter:
        if self._segmenter is None:
            self._segmenter = InfectionSegmenter(self.checkpoint, device=self._device)
        return self._segmenter

    def process_image_file(
        self,
        source_path: Path,
        output_path: Path,
        *,
        threshold: float | None = None,
        min_area_px: int = 40,
        border_margin_px: int = 8,
        border_touch_keep_min_area_px: int = 500,
        blend_alpha: float = 0.52,
        clahe_clip: float = 2.0,
        use_watershed: bool = False,
        watershed_peak_distance: int = 18,
        opening_ksize: int = 5,
        write_observation_panel: bool = True,
    ) -> dict:
        """
        Read image, optional CLAHE, run U-Net, then count predicted positives.

        By default (use_watershed=False) the count is the number of connected
        components in the thresholded mask (min_area filter) — this matches
        what the segmentation model was trained to predict (binary infection).
        Set use_watershed=True to further split merged regions for dense plates.
        """
        source_path = Path(source_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rgb = read_image_rgb(source_path)
        if float(clahe_clip) > 0:
            rgb = preprocess_clahe_lab(rgb, clip_limit=float(clahe_clip))

        seg = self._seg()
        bin_mask, prob, meta = seg.predict_rgb(rgb, threshold=threshold)

        artifact_circles: list[tuple[int, int, int]] = []
        if use_watershed:
            labels, n_cells = choose_labels_for_counting(
                bin_mask,
                min_area_px=min_area_px,
                use_watershed=True,
                watershed_peak_distance=watershed_peak_distance,
                opening_ksize=opening_ksize,
            )
        else:
            labels, n_cells, artifact_circles = labels_true_positives_filtered(
                bin_mask,
                min_area_px=min_area_px,
                border_margin_px=border_margin_px,
                border_touch_keep_min_area_px=border_touch_keep_min_area_px,
            )

        seg_rgb = overlay_cellpose_style(rgb, labels, alpha=blend_alpha)
        final_rgb = annotate_counts_on_image(
            seg_rgb,
            labels,
            n_cells,
            draw_cell_indices=True,
            artifact_circles=artifact_circles if artifact_circles else None,
            threshold=float(meta["threshold"]),
        )

        bgr_out = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr_out)

        observation_filename: str | None = None
        if write_observation_panel:
            obs_rgb = build_observation_panel_rgb(
                rgb,
                prob,
                labels,
                count=n_cells,
                threshold=float(meta["threshold"]),
                blend_alpha=blend_alpha,
            )
            if "_processed.png" in output_path.name:
                obs_name = output_path.name.replace("_processed.png", "_observation.png")
            else:
                obs_name = output_path.stem + "_observation.png"
            obs_path = output_path.parent / obs_name
            cv2.imwrite(str(obs_path), cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR))
            observation_filename = obs_name

        out = {
            "count": int(n_cells),
            "threshold": float(meta["threshold"]),
            "orig_shape": meta["orig_shape_hw"],
            "min_area_px": min_area_px,
            "border_margin_px": border_margin_px,
            "border_touch_keep_min_area_px": border_touch_keep_min_area_px,
            "artifacts_marked": len(artifact_circles),
            "clahe_clip": clahe_clip,
            "blend_alpha": blend_alpha,
            "use_watershed": use_watershed,
            "watershed_peak_distance": watershed_peak_distance,
            "opening_ksize": opening_ksize,
            "observation_filename": observation_filename,
            "observation_saved": observation_filename is not None,
        }
        return out
