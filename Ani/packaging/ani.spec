# PyInstaller spec for Ani.  Build:  pyinstaller packaging/ani.spec --noconfirm
# Produces dist/Ani/ (a folder app: starts faster and trips antivirus less than a one-file exe).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []
for pkg in ("napari", "napari_builtins", "napari_resources", "napari_console", "napari_plugin_engine", "vispy", "app_model", "magicgui", "superqt", "psygnal", "npe2",
            "in_n_out", "pint", "skimage", "imageio", "tifffile", "lazy_loader", "pydantic_compat"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h
for dist in ("napari", "napari-builtins", "npe2", "imageio", "app-model", "magicgui", "scikit-image", "tifffile",
             "numpy", "pydantic", "ani"):
    try:
        datas += copy_metadata(dist)          # plugin entry points & version lookups need dist-info
    except Exception:
        pass
hiddenimports += ["ani", "ani.simple_ui", "ani.review", "ani.features", "ani.safety", "ani.seg", "ani.segment",
                  "ani.spots", "ani.machine", "ani.cache", "ani.lanes", "ani.report", "ani.bigimage",
                  "ani.collab", "ani.density", "ani.density_ui", "ani.io",
                  "ani.ai", "ani.ai.classic", "ani.ai.cellpose_backend", "ani.ai.sam_backend",
                  "ani.ai.yolo_backend", "qtpy", "PyQt6", "PyQt6.QtOpenGL", "PyQt6.QtOpenGLWidgets", "PyQt6.QtSvg"]

a = Analysis([str(ROOT / "packaging" / "ani_launcher.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports,
             excludes=["torch", "tensorflow", "cellpose", "ultralytics", "micro_sam", "PyQt5", "PySide2",
                       "PySide6", "tkinter", "matplotlib", "IPython", "jupyter", "notebook"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Ani", console=False,
          icon=str(ROOT / "packaging" / "ani.ico") if (ROOT / "packaging" / "ani.ico").exists() else None)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Ani-console", console=True)  # for --selftest
coll = COLLECT(exe, cli, a.binaries, a.datas, name="Ani")
