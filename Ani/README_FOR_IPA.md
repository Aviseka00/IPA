# Ani inside the IPA monorepo

This folder is a **full, independent Ani install**. You can:

| Goal | Action |
|------|--------|
| Use Ani alone | Double-click `run_ani.bat` (or `..\run_ani.bat` from repo root) |
| Use with IPA assay | IPA starts Ani from **this folder** automatically |
| Upgrade Ani | Replace/update files here, then `..\upgrade_ani.bat` |

IPA does not copy Ani into the web app. It always runs:

`Ani\.venv\Scripts\python.exe -m ani …`

So any future Ani upgrade in this directory is what the assay uses next time Ani launches.
