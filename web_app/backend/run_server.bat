@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt -q
pip install tifffile imagecodecs scikit-image scipy -q
python -c "import tifffile, imagecodecs; from skimage.morphology import remove_small_objects; print('deps OK')"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
