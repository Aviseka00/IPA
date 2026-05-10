"""
Batch-test images: save detection overlay + contour visualization and report counts.

Example:
  python ml_infection/visualize_inference.py Images/QS_2834.jpg Images/test1.png
  python ml_infection/visualize_inference.py --glob "Images/QS_*.jpg"
  python ml_infection/visualize_inference.py --dir D:/IPA3/test --out-dir D:/IPA3/ml_infection/test_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from instance_split import (
    choose_labels_for_counting,
    labels_connected_components_filtered,
    labels_true_positives_filtered,
)
from predict import InfectionSegmenter
from image_io import preprocess_clahe_lab, read_image_rgb


def _distinct_rgb_colors(n: int, seed: int = 42) -> np.ndarray:
    """HSV hues spaced like Cellpose-style palettes (vivid, distinguishable)."""
    import matplotlib.colors as mcolors

    rng = np.random.RandomState(seed)
    golden = 0.618033988749895
    hues = np.array([(i * golden) % 1.0 for i in range(max(n, 1))], dtype=np.float64)
    rng.shuffle(hues)
    cols = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        rgb = mcolors.hsv_to_rgb([hues[i % len(hues)], 0.88, 0.97])
        cols[i] = rgb * 255.0
    return cols


def _label_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels where two neighbors differ in instance id (including to background)."""
    l = labels.astype(np.int32)
    e = np.zeros(l.shape, dtype=bool)
    if l.shape[1] > 1:
        d = l[:, :-1] != l[:, 1:]
        e[:, :-1] |= d
        e[:, 1:] |= d
    if l.shape[0] > 1:
        d = l[:-1, :] != l[1:, :]
        e[:-1, :] |= d
        e[1:, :] |= d
    return e


def overlay_cellpose_style(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.52,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 1,
) -> np.ndarray:
    """
    Colorful instance overlay on brightfield image (Cellpose-like).
    Each connected component gets its own hue; boundaries drawn in white.
    """
    base = rgb.astype(np.float32)
    h, w = labels.shape
    ids = sorted([int(i) for i in np.unique(labels) if i > 0])
    if not ids:
        return rgb.copy()

    palette = _distinct_rgb_colors(len(ids), seed=42)
    id_to_idx = {lid: j for j, lid in enumerate(ids)}

    fill = np.zeros((h, w, 3), dtype=np.float32)
    fg = labels > 0
    for lid in ids:
        idx = id_to_idx[lid]
        fill[labels == lid] = palette[idx]

    a = np.zeros((h, w, 1), dtype=np.float32)
    a[fg] = alpha
    blended = base * (1.0 - a) + fill * a
    out = np.clip(blended, 0, 255).astype(np.uint8)

    bd = _label_boundary_mask(labels)
    if outline_width > 0:
        thick = max(outline_width, 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick * 2 + 1, thick * 2 + 1))
        bd_u = bd.astype(np.uint8)
        bd_u = cv2.dilate(bd_u, k, iterations=1).astype(bool)
        out[bd_u] = outline_color
    return out


def fill_rounded_rectangle_bgr(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled rounded rectangle (BGR uint8 image, in-place)."""
    r = min(max(radius, 0), (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, -1)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, -1)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, -1)


def masks_panel(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pure colored instances on black (Cellpose mask RGB view)."""
    h, w = labels.shape
    ids = sorted([int(i) for i in np.unique(labels) if i > 0])
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if not ids:
        return panel
    palette = _distinct_rgb_colors(len(ids), seed=seed)
    id_to_idx = {lid: j for j, lid in enumerate(ids)}
    for lid in ids:
        idx = id_to_idx[lid]
        panel[labels == lid] = palette[idx].astype(np.uint8)
    # boundaries in white
    bd = _label_boundary_mask(labels)
    panel[bd] = (255, 255, 255)
    return panel


def annotate_counts_on_image(
    rgb: np.ndarray,
    labels: np.ndarray,
    total_count: int,
    *,
    draw_cell_indices: bool = True,
    artifact_circles: list[tuple[int, int, int]] | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Draw per-cell index numbers and a bottom-right summary card (counts / threshold).

    Expects RGB uint8 (e.g. Cellpose-style overlay output).

    ``artifact_circles``: excluded regions; drawn as red rings last (QC on main export).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if draw_cell_indices and labels.size:
        ids = sorted([int(i) for i in np.unique(labels) if i > 0])
        for num, lid in enumerate(ids, start=1):
            m = (labels == lid).astype(np.uint8)
            mu = cv2.moments(m)
            if mu["m00"] < 1e-6:
                continue
            cx = int(mu["m10"] / mu["m00"])
            cy = int(mu["m01"] / mu["m00"])
            ts = 0.45 * (w / 1536.0 + 0.4)
            cv2.putText(bgr, str(num), (cx - 6, cy + 5), font, ts, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(bgr, str(num), (cx - 6, cy + 5), font, ts, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom-right summary card (shadow + rounded panel)
    pad = max(14, min(w, h) // 80)
    card_w = min(340, int(w * 0.36))
    card_h = min(112, int(h * 0.12))
    r_corner = 14
    x2 = w - pad
    y2 = h - pad
    x1 = x2 - card_w
    y1 = y2 - card_h
    sx1, sy1, sx2, sy2 = x1 + 4, y1 + 4, x2 + 4, y2 + 4
    fill_rounded_rectangle_bgr(bgr, sx1, sy1, sx2, sy2, r_corner, (18, 18, 22))
    fill_rounded_rectangle_bgr(bgr, x1, y1, x2, y2, r_corner, (52, 58, 64))
    cv2.line(bgr, (x1 + r_corner, y1 + 2), (x2 - r_corner, y1 + 2), (140, 210, 175), 2, cv2.LINE_AA)

    num_scale = 1.35 * (w / 1536.0 + 0.35)
    cv2.putText(
        bgr,
        str(int(total_count)),
        (x1 + 18, y1 + int(card_h * 0.52)),
        cv2.FONT_HERSHEY_DUPLEX,
        num_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        "True positives",
        (x1 + 18, y1 + int(card_h * 0.78)),
        font,
        0.52 * (w / 1024.0 + 0.45),
        (200, 215, 225),
        1,
        cv2.LINE_AA,
    )
    if threshold is not None:
        cv2.putText(
            bgr,
            f"threshold {float(threshold):.2f}",
            (x1 + 18, y1 + int(card_h * 0.93)),
            font,
            0.42 * (w / 1024.0 + 0.45),
            (160, 175, 190),
            1,
            cv2.LINE_AA,
        )

    if artifact_circles:
        for cx, cy, rad in artifact_circles:
            cx = int(np.clip(cx, 0, w - 1))
            cy = int(np.clip(cy, 0, h - 1))
            r = max(2, int(rad))
            cv2.circle(bgr, (cx, cy), r, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def collect_images(images: list[Path], glob_pat: str | None, directory: Path | None, limit: int | None) -> list[Path]:
    out: list[Path] = []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in images:
        p = Path(p)
        if p.is_file():
            out.append(p.resolve())
    if glob_pat:
        from glob import glob as pyglob

        for g in pyglob(glob_pat):
            gp = Path(g)
            if gp.is_file() and gp.suffix.lower() in exts:
                out.append(gp.resolve())
    if directory:
        d = Path(directory)
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p.resolve())
    # unique preserve order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if limit is not None:
        uniq = uniq[:limit]
    return uniq


def overlay_detection(rgb: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """Semi-transparent green over predicted infected pixels + red contours (RGB)."""
    img = rgb.astype(np.float32)
    color = np.array([70.0, 255.0, 90.0], dtype=np.float32)
    m = mask_bin.astype(bool)
    blend = img.copy()
    blend[m] = (1.0 - alpha) * img[m] + alpha * color
    out = np.clip(blend, 0, 255).astype(np.uint8)
    cnt_img = (mask_bin > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 60, 60), thickness=2)
    return out


def prob_heatmap_rgb(prob: np.ndarray) -> np.ndarray:
    """Probability map as color heatmap (BGR then to RGB for consistency)."""
    p = np.clip(prob, 0, 1)
    pm = (p * 255).astype(np.uint8)
    heat = cv2.applyColorMap(pm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def heatmap_overlay_rgb(rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Blend JET probability heatmap on brightfield (same prob tensor as threshold / counting)."""
    hm = prob_heatmap_rgb(prob).astype(np.float32)
    base = rgb.astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    out = base * (1.0 - a) + hm * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_artifact_markers_rgb(rgb: np.ndarray, artifact_circles: list[tuple[int, int, int]]) -> np.ndarray:
    """Red circles on RGB image (artifacts excluded from count)."""
    if not artifact_circles:
        return rgb.copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = bgr.shape[:2]
    for cx, cy, rad in artifact_circles:
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        r = max(2, int(rad))
        cv2.circle(bgr, (cx, cy), r, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_observation_panel_rgb(
    rgb: np.ndarray,
    prob: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    threshold: float,
    blend_alpha: float = 0.52,
) -> np.ndarray:
    """
    Single figure for reporting: [probability JET | colored masks | brightfield overlay].

    No QC artifact rings; overlay column matches counted positives only.
    """
    hm = prob_heatmap_rgb(prob)
    masks = masks_panel(labels)
    vis = overlay_cellpose_style(rgb, labels, alpha=blend_alpha)
    overlay_clean = annotate_counts_on_image(
        vis,
        labels,
        count,
        draw_cell_indices=True,
        artifact_circles=None,
        threshold=float(threshold),
    )

    h, w = hm.shape[:2]
    gap = max(6, min(w, h) // 220)
    gap_bgr = np.full((h, gap, 3), (28, 28, 32), dtype=np.uint8)

    hm_b = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)
    mk_b = cv2.cvtColor(masks, cv2.COLOR_RGB2BGR)
    ov_b = cv2.cvtColor(overlay_clean, cv2.COLOR_RGB2BGR)

    row_bgr = np.hstack([hm_b, gap_bgr, mk_b, gap_bgr, ov_b])
    _, ww = row_bgr.shape[:2]
    header_h = 46
    footer_h = 52
    header_bgr = np.full((header_h, ww, 3), (52, 54, 58), dtype=np.uint8)
    footer_bgr = np.full((footer_h, ww, 3), (38, 40, 44), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.62 * (ww / 3072.0 + 0.35)
    cv2.putText(header_bgr, "Probability map", (14, 32), font, fs, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(header_bgr, "Instance masks", (w + gap + 14, 32), font, fs, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(
        header_bgr,
        "Brightfield overlay (positives only)",
        (2 * (w + gap) + 14, 32),
        font,
        fs,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )

    line = f"True positives: {int(count)}   |   model threshold {float(threshold):.2f}   |   QC artifact markers omitted in this view"
    cv2.putText(footer_bgr, line, (14, 34), font, 0.48 * (ww / 3072.0 + 0.38), (210, 218, 228), 1, cv2.LINE_AA)

    out_bgr = np.vstack([header_bgr, row_bgr, footer_bgr])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser(description="Visualize model detection and counts on test images.")
    ap.add_argument("images", nargs="*", type=Path, help="Image paths")
    ap.add_argument("--glob", dest="glob_pat", default=None, help='Glob pattern e.g. Images/QS_*.jpg')
    ap.add_argument("--dir", type=Path, default=None, help="Directory of images")
    ap.add_argument("--limit", type=int, default=None, help="Max number of images")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "unet_infected_best.pt",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "test_outputs")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument(
        "--border-margin",
        type=int,
        default=8,
        help="trained mode: width (px) of frame strip used to detect edge-touching blobs",
    )
    ap.add_argument(
        "--border-touch-keep-min-area",
        type=int,
        default=500,
        dest="border_touch_keep_min_area",
        help="trained mode: touching-frame blobs smaller than this are artifacts; larger ones count as true positives",
    )
    ap.add_argument("--save-heatmap", action="store_true", help="Also save probability heatmaps (_prob_heatmap.png)")
    ap.add_argument(
        "--heatmap-overlay",
        action="store_true",
        help="Also save heatmap blended on brightfield (_heatmap_overlay.png); uses same prob map as counting",
    )
    ap.add_argument(
        "--heatmap-overlay-alpha",
        type=float,
        default=0.42,
        help="Blend strength for --heatmap-overlay (0–1)",
    )
    ap.add_argument(
        "--simple-overlay",
        action="store_true",
        help="Also save green/red simple overlay (_detection.png) in addition to Cellpose-style",
    )
    ap.add_argument(
        "--save-mask-panel",
        action="store_true",
        help="Also save colored masks on black (_masks_only.png)",
    )
    ap.add_argument("--blend-alpha", type=float, default=0.52, help="Instance color blend strength on image")
    ap.add_argument(
        "--count-mode",
        choices=("trained", "split"),
        default="trained",
        help="trained: count connected regions of model-positive pixels (prob≥threshold, default from checkpoint); "
        "split: optional watershed splitting for touching cells (--no-watershed disables split inside that mode)",
    )
    ap.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE clip limit on L channel (0 = disable; matches web API default 2)",
    )
    ap.add_argument(
        "--no-watershed",
        action="store_true",
        help="Only for --count-mode split: disable watershed / use CC only",
    )
    ap.add_argument("--watershed-peak-distance", type=int, default=18)
    ap.add_argument("--opening-ksize", type=int, default=5)
    ap.add_argument(
        "--no-count-banner",
        action="store_true",
        help="Skip count card / index labels on overlay (artifacts still optional)",
    )
    ap.add_argument(
        "--no-observation-panel",
        dest="observation_panel",
        action="store_false",
        default=True,
        help="Do not write *_observation.png side-by-side summary figure",
    )
    args = ap.parse_args()

    paths = collect_images(list(args.images), args.glob_pat, args.dir, args.limit)
    if not paths:
        raise SystemExit("No images found. Pass paths, --glob, or --dir.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seg = InfectionSegmenter(args.checkpoint)

    use_watershed = not args.no_watershed
    summary = []
    for ip in paths:
        rgb = read_image_rgb(ip)
        if float(args.clahe_clip) > 0:
            rgb = preprocess_clahe_lab(rgb, clip_limit=float(args.clahe_clip))

        bin_mask, prob_full, meta = seg.predict_rgb(rgb, threshold=args.threshold)
        artifact_circles: list[tuple[int, int, int]] = []
        if args.count_mode == "trained":
            labels, n_cells, artifact_circles = labels_true_positives_filtered(
                bin_mask,
                min_area_px=args.min_area,
                border_margin_px=args.border_margin,
                border_touch_keep_min_area_px=args.border_touch_keep_min_area,
            )
        else:
            labels, n_cells = choose_labels_for_counting(
                bin_mask,
                min_area_px=args.min_area,
                use_watershed=use_watershed,
                watershed_peak_distance=args.watershed_peak_distance,
                opening_ksize=args.opening_ksize,
            )

        vis_cp = overlay_cellpose_style(rgb, labels, alpha=args.blend_alpha)
        if args.no_count_banner:
            final_rgb = draw_artifact_markers_rgb(vis_cp, artifact_circles) if artifact_circles else vis_cp
        else:
            final_rgb = annotate_counts_on_image(
                vis_cp,
                labels,
                n_cells,
                draw_cell_indices=True,
                artifact_circles=artifact_circles if artifact_circles else None,
                threshold=float(meta["threshold"]),
            )

        stem = ip.stem
        out_vis = args.out_dir / f"{stem}_segmented.png"
        cv2.imwrite(str(out_vis), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))

        row = {
            "image": str(ip),
            "infected_cell_count_estimate": int(n_cells),
            "count_mode": args.count_mode,
            "threshold": meta["threshold"],
            "min_area_px": args.min_area,
            "border_margin_px": args.border_margin,
            "border_touch_keep_min_area_px": args.border_touch_keep_min_area,
            "artifacts_marked": len(artifact_circles),
            "clahe_clip": float(args.clahe_clip),
            "use_watershed": use_watershed if args.count_mode == "split" else False,
            "watershed_peak_distance": args.watershed_peak_distance,
            "opening_ksize": args.opening_ksize,
            "segmented_png": str(out_vis.resolve()),
        }
        if args.simple_overlay:
            vis_simple = overlay_detection(rgb, bin_mask)
            out_simple = args.out_dir / f"{stem}_detection.png"
            cv2.imwrite(str(out_simple), cv2.cvtColor(vis_simple, cv2.COLOR_RGB2BGR))
            row["detection_png"] = str(out_simple.resolve())
        if args.save_mask_panel:
            panel = masks_panel(labels)
            if artifact_circles:
                panel = draw_artifact_markers_rgb(panel, artifact_circles)
            out_p = args.out_dir / f"{stem}_masks_only.png"
            cv2.imwrite(str(out_p), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            row["masks_only_png"] = str(out_p.resolve())
        if args.save_heatmap:
            hm = prob_heatmap_rgb(prob_full)
            if artifact_circles:
                hm = draw_artifact_markers_rgb(hm, artifact_circles)
            out_hm = args.out_dir / f"{stem}_prob_heatmap.png"
            cv2.imwrite(str(out_hm), cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))
            row["prob_heatmap_png"] = str(out_hm.resolve())
        if args.heatmap_overlay:
            ho = heatmap_overlay_rgb(rgb, prob_full, alpha=args.heatmap_overlay_alpha)
            if artifact_circles:
                ho = draw_artifact_markers_rgb(ho, artifact_circles)
            out_ho = args.out_dir / f"{stem}_heatmap_overlay.png"
            cv2.imwrite(str(out_ho), cv2.cvtColor(ho, cv2.COLOR_RGB2BGR))
            row["heatmap_overlay_png"] = str(out_ho.resolve())

        if args.observation_panel:
            obs_rgb = build_observation_panel_rgb(
                rgb,
                prob_full,
                labels,
                count=n_cells,
                threshold=float(meta["threshold"]),
                blend_alpha=args.blend_alpha,
            )
            out_obs = args.out_dir / f"{stem}_observation.png"
            cv2.imwrite(str(out_obs), cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR))
            row["observation_png"] = str(out_obs.resolve())

        summary.append(row)
        print(f"{ip.name}: count={n_cells}  ->  {out_vis.name}")

    summary_path = args.out_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {len(summary)} visualizations to {args.out_dir.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
