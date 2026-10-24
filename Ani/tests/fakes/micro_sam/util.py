import numpy as np

from . import CALLS


class _Predictor:
    pass


def get_sam_model(model_type="vit_l", device=None, checkpoint_path=None, return_sam=False, **kwargs):
    CALLS.append(("get_sam_model", {"model_type": model_type, "device": device}))
    return _Predictor()


def precompute_image_embeddings(predictor, input_, save_path=None, lazy_loading=False, ndim=None,
                                tile_shape=None, halo=None, verbose=True, **kwargs):
    assert input_.dtype == np.uint8
    CALLS.append(("precompute_image_embeddings", {"shape": input_.shape, "ndim": ndim}))
    return {"image": input_}


def microsam_cachedir():
    return "/tmp/fake-micro-sam-cache"
