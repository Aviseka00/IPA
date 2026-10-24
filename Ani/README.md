# Ani – image annotation

**Ani** is a desktop app for outlining objects in any image, giving each one a class, and saving that work for
training. It runs offline on an ordinary CPU. No GPU is required.

Open an image, trace or brush the objects, right-click to set the class, then save seg.npy, TIFF, PNG, COCO, or YOLO.
The same outlines can be exported as a feature table. `python -m ani --full` opens the older project workspace.

---

## Install (one time, needs internet once)

**Windows: easiest.** Run `Ani-Setup-x.y.z.exe` (see *Building the Windows installer*). It creates Start-menu entries
for Ani, the demo, and a self-test.

**Windows: from source**
1. Install **Python 3.11 or 3.12** from <https://www.python.org/downloads/>. Tick **"Add python.exe to PATH"**.
2. Double-click **`setup_windows.bat`** (about 5–10 minutes).
3. Double-click **`run_demo.bat`** to practise, or **`run_ani.bat`** for your own work.
4. *Optional AI:* double-click **`install_ai.bat`**. It adds **Cellpose, micro-sam and YOLO** (CPU versions,
   about 2–3 GB, needs internet once). It also downloads their models and checks they work, so Ani runs offline
   afterwards. micro-sam needs Python 3.11 or newer.
5. *Check everything works:* double-click **`selftest_windows.bat`**.

**Linux / macOS**
```bash
./setup_linux_mac.sh
./run_ani.sh --demo          # practise
./install_ai.sh              # optional: Cellpose + micro-sam + YOLO (CPU)
.venv/bin/python -m ani --selftest
```
On Ubuntu, if the window does not open, run `sudo apt install libxcb-cursor0`.


---

## The four tabs

### 1 · Annotate: fix the detector's mistakes
| Mistake | Tool | Key |
|---|---|---|
| Missed a cell | **AI click**: click inside it and the cell is outlined for you | Shift-A |
|  | **Outline**: click around it, double-click to close · or **New cell** + paint | Shift-O / Shift-N |
| False detection or debris | **Delete**, then click it | Shift-D |
| Two cells detected as one | **Split**, then click it | Shift-X |
| One cell cut in pieces | **Merge**: click the piece to keep, then the other piece | Shift-M |
| Wrong boundary | **Paint / Erase / Fill** | Shift-P / E / F |
| Wrong type | **Set class**, then click it | Shift-C |

- **🔍 Find problems** (Shift-R) rings everything suspicious and jumps through them (Shift-K / Shift-J). It finds:
  - missed cells
  - merged cells
  - cells cut in half
  - debris and broken objects
- **Undo / Redo** (Ctrl+Z / Ctrl+Shift+Z) covers everything.
- **Label sets** handle structures that overlap: *cells*, *nuclei*, *nucleoli*, *mitochondria*… Classes go inside a set, e.g. *dividing*, *apoptotic*, *stained*.
- **Auto-save** saves when you move to another image. A **backup every minute** survives power cuts.
  Both happen in the background, so you never wait for the disk.

### 2 · AI: let a detector do the first pass
| Detector | Needs | Whole image | AI click | Speed on CPU |
|---|---|---|---|---|
| **Classic** (built in) | nothing | ✔ | ✔ | about 3 s for 2048×2048 |
| **Cellpose cyto3 / nuclei** | `install_ai.bat` | ✔ | ✔ | tens of seconds per image |
| **Your trained Cellpose / YOLO model** | the model file in `models/` (YOLO: `install_ai.bat`) | ✔ | ✔ (Cellpose) | as above |
| **micro-sam** (vit_t_lm) | `install_ai.bat` (Python 3.11+) | – | ✔ | about 10 s per image once, then each click is fast |

- **Auto-label (replace / add missing)** works on the current image.
- **Run detector on all images** runs in the background, so you can keep working.
- **Use my own detector program:** Ani runs your program once per image, e.g.
  `python C:/lab/detect.py --input {input} --output {output}`.
  Your program writes a label mask (0 = background, 1..N = cells) to `{output}`.
- **Rank all images (worst first)** sorts the list so the images with the most suspected mistakes come first.
- **Objects are:** set *brighter than background* (fluorescence) or *darker* (brightfield). Leave it on *auto* and Ani learns this from cells that are already labelled.

### 3 · Train: measure, improve, export
1. **Evaluate**: compare a detector with your corrections on images marked Done. It reports precision, recall,
   F1 and IoU, and keeps a history table so you can see each round improve.
2. **Improve**:
   - **Tune Classic**: the built-in detector learns its settings from your corrections in about a minute on CPU. On crowded cells it went from F1 0.96 to 0.99 in testing.
   - **Fine-tune Cellpose**: re-trains Cellpose on your corrections (slow on CPU).
   - **Free GPU**: tick *Add Google Colab notebooks* when exporting. Upload the dataset to Colab, run the notebook, and put the model you download into `models/cellpose/` or `models/yolo/`. Ani lists it as a detector.
3. **Export dataset** writes a train/validation split in three formats:
   - masks (Cellpose, StarDist, U-Net)
   - COCO (Detectron2, MMDetection)
   - YOLO-seg (Ultralytics)

   Images are saved as 8-bit PNG for COCO and YOLO.

### 4 · Settings
See which AI packages are installed, and view the shortcuts.

---

## A project is just a folder
```
my_project/
  images/              your microscopy images (.tif .tiff .png .jpg .bmp)
  predictions/         detector output for the first label set, SAME file name as the image
  predictions/<set>/   detector output for other sets (Ani's detectors write here too)
  labels/<set>/        your corrected masks
  models/cellpose/     your trained Cellpose models    models/yolo/   your YOLO-seg models (.pt)
  history/  .recovery/ older versions (last 5) and power-cut backups
  exports/             training datasets (+ Colab notebooks)
  project.json         label sets, classes, image status, statistics, evaluation history
```
Detector output can be a label mask (`img01.tif`, `img01_mask.png`, `img01_cp_masks.png`), a NumPy array (`.npy`),
a colour mask, or YOLO-seg (`.txt`). Cellpose and StarDist outputs work directly.

---

## Building the Windows installer
- **On GitHub (recommended):** push this folder to a GitHub repository. The included workflow
  (`.github/workflows/ci.yml`):
  1. tests Ani on Windows, Linux and macOS
  2. runs real Cellpose
  3. builds and checks the app
  4. uploads **Ani-Setup-x.y.z.exe**
- **On a Windows PC:** run `build_windows_installer.bat`. It needs [Inno Setup 6](https://jrsoftware.org/isinfo.php) for
  the installer. Without Inno Setup you still get `dist\Ani\Ani.exe`.

The installed app is about 450 MB on disk (200 MB download) and needs no Python. Cellpose is not bundled.

## Development and QA
```bash
pip install -e ".[dev]"
ruff check ani tests
pytest -q --cov=ani                         # Linux headless: xvfb-run -a pytest -q
ANI_INTEGRATION=1 pytest -m integration     # real Cellpose (downloads its model)
python -m ani --selftest                    # quick health check on any PC
python -m ani --download-models             # fetch AI model weights once (for offline use)
```

## Credits and license
Ideas inspired by [digitalsreeni-image-annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator),
[DeepCell Label](https://github.com/vanvalenlab/deepcell-label), [CellSAM](https://github.com/vanvalenlab/cellsam) and
[QuickAnnotator](https://github.com/choosehappy/QuickAnnotator). Ani's code is written independently.
Built on [napari](https://napari.org), [scikit-image](https://scikit-image.org) and [SciPy](https://scipy.org).
Optional: [Cellpose](https://github.com/MouseLand/cellpose), [micro-sam](https://github.com/computational-cell-analytics/micro-sam), [Ultralytics](https://github.com/ultralytics/ultralytics).

**MIT license:** free for everyone, including students, colleges, labs and companies.
