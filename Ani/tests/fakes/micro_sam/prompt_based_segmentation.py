import numpy as np

from . import CALLS


def segment_from_points(predictor, points, labels, image_embeddings=None, i=None, multimask_output=False,
                        return_all=False, use_best_multimask=None):
    from ani.ai.classic import ClassicBackend
    CALLS.append(("segment_from_points", {"points": np.asarray(points).tolist(), "labels": np.asarray(labels).tolist()}))
    img = image_embeddings["image"]
    empty = np.zeros(img.shape, np.uint16)
    mask, box = ClassicBackend().segment_at(img, tuple(int(v) for v in points[0]), empty)
    out = np.zeros((1,) + img.shape, bool)
    if mask is not None:
        out[0][box] = mask
    return out
