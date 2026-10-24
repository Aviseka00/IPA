"""Fake cellpose.models with the real v3.1 signatures."""
import numpy as np

from . import CALLS


def _segment(x):
    from ani.ai.classic import ClassicBackend
    x = np.asarray(x)
    if x.ndim == 3:   # z-stack: per-slice (stitching not simulated)
        return np.stack([ClassicBackend().auto_segment(s) for s in x])
    return ClassicBackend().auto_segment(x)


class _Net:
    def save_model(self, filename):
        open(filename, "wb").write(b"fake-cellpose-weights")


class CellposeModel:
    def __init__(self, gpu=False, pretrained_model=False, model_type=None, mkldnn=True, diam_mean=30.,
                 device=None, nchan=2, pretrained_model_ortho=None, backbone="default"):
        CALLS.append(("CellposeModel", {"pretrained_model": pretrained_model, "model_type": model_type, "gpu": gpu}))
        self.net = _Net()

    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=224,
             interp=True, compute_masks=True, progress=None):
        CALLS.append(("CellposeModel.eval", {"shape": np.shape(x), "channels": channels, "diameter": diameter,
                                             "z_axis": z_axis, "stitch_threshold": stitch_threshold}))
        return _segment(x), [], []


class Cellpose:
    def __init__(self, gpu=False, model_type="cyto3", nchan=2, device=None, backbone="default"):
        CALLS.append(("Cellpose", {"model_type": model_type, "gpu": gpu}))
        self.cp = CellposeModel(gpu=gpu, model_type=model_type)

    def eval(self, x, batch_size=8, channels=[0, 0], channel_axis=None, invert=False,
             normalize=True, diameter=30., do_3D=False, **kwargs):
        masks, flows, styles = self.cp.eval(x, batch_size=batch_size, channels=channels, channel_axis=channel_axis,
                                            invert=invert, normalize=normalize, diameter=diameter, do_3D=do_3D,
                                            **kwargs)
        return masks, flows, styles, 30.0
