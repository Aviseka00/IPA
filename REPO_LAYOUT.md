# IPA monorepo layout

This repository is **one product**: IPA assay + Ani labeling companion.

```
IPA/
  Ani/                 Ani — pixel annotation (run alone OR with IPA)
  run_ani.bat          Start Ani only (independent)
  run_ipa.bat          Start IPA (+ Ani autostart)
  upgrade_ani.bat      Re-setup Ani after you drop in a newer Ani release
  web_app/             IPA web platform (FastAPI + React)
  ipa4/ ipa6/ …        Detection models & training
  datasets/            Including retrain_from_review/ from human+Ani labels
```

## Run Ani independently

```bat
run_ani.bat
```

or:

```bat
cd Ani
run_ani.bat
run_demo.bat
```

Ani does not need the IPA web server. It keeps its own `.venv` under `Ani\`.

## Run IPA assay

```bat
run_ipa.bat
```

IPA **always loads Ani from `Ani\` in this repo**. Whatever version is in that folder is what the assay uses.

## Upgrade Ani (IPA picks it up automatically)

1. Close Ani.
2. Update files inside `Ani\` with the new Ani release (overwrite source; keep or recreate `.venv`).
3. Run `upgrade_ani.bat` (runs `Ani\setup_windows.bat`).
4. Start IPA again — no IPA code change needed. Path stays `IPA\Ani`.

IPA reads Ani’s version from `Ani\pyproject.toml` and launches `Ani\.venv\Scripts\python.exe -m ani`, so upgrades are live on the next start.

## First-time setup after clone

1. IPA: see [web_app/README.md](web_app/README.md)
2. Ani once: `cd Ani` → `setup_windows.bat`
3. Daily use: `run_ipa.bat` or `run_ani.bat`

## How they connect in the assay

1. IPA detects infected cells  
2. Review → **Open in Ani** (or use always-on Ani)  
3. Keep confirmed detections; add FN; remove FP  
4. **Import Ani labels** → ground truth + `datasets/retrain_from_review/`

## Do not commit

- `Ani/.venv/`, `web_app/backend/.venv/`, `node_modules/`
- `.env`, `web_app/storage/`, large weights / personal datasets
