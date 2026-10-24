"""Fake cellpose.train.train_seg with the real v3.1 signature and save behaviour."""
from pathlib import Path

from . import CALLS


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, load_files=True, batch_size=8, learning_rate=0.005,
              n_epochs=2000, weight_decay=1e-5, momentum=0.9, SGD=False, channels=None,
              channel_axis=None, rgb=False, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=True, scale_range=None, bsize=224,
              min_train_masks=5, model_name=None):
    assert len(train_data) == len(train_labels) and len(train_data) > 0
    assert all(d.ndim == 2 for d in train_data), "fake expects 2-D grey images"
    CALLS.append(("train_seg", {"n_train": len(train_data), "n_test": len(test_data or []), "epochs": n_epochs,
                                "channels": channels, "lr": learning_rate, "SGD": SGD}))
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)       # real cellpose: no parents=True
    net.save_model(filename)
    return filename, [1.0, 0.5], [1.1, 0.6]
