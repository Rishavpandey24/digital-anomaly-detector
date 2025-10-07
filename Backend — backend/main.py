nd/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
import shutil
import uuid
from detectors import DetectorRouter
from model_store import ModelStore


app = FastAPI(title='Universal Anomaly Detection System (UADS)')


app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BASE_DIR.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


# one ModelStore instance (persistence)
store = ModelStore(MODELS_DIR)
router = DetectorRouter(store)


@app.post('/upload')
async def upload_file(file: UploadFile = File(...), contamination: float = Form(0.01)):
# save file
uid = uuid.uuid4().hex
dest = UPLOAD_DIR / f"{uid}_{file.filename}"
with open(dest, 'wb') as f:
shutil.copyfileobj(file.file, f)
# route to detector
try:
result = router.handle_file(str(dest), contamination=contamination)
return JSONResponse({'ok': True, 'result': result})
except Exception as e:
return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)


@app.post('/stream_sensor')
async def stream_sensor(rows: str = Form(...), contamination: float = Form(0.01)):
# rows is CSV-like text streamed from client
import pandas as pd
from io import StringIO
df = pd.read_csv(StringIO(rows))
result = router.handle_sensor_dataframe(df, contamination=contamination)
return JSONResponse({'ok': True, 'result': result})


@app.get('/models')
def list_models():
return JSONResponse({'models': store.list_models()})


@app.post('/retrain')
async def retrain_model(model_name: str = Form(...)):
try:
store.retrain_model(model_name)
return JSONResponse({'ok': True, 'message': 'retrain triggered'})
except Exception as e:
return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)


@app.get('/download_report/{fname}')
def download_report(fname: str):
p = BASE_DIR.parent / 'outputs' / fname
if p.exists():
return FileResponse(str(p), media_type='text/csv', filename=fname)
return JSONResponse({'ok': False, 'error': 'not found'}, status_code=404)


if __name__ == '__main__':
uvicorn.run(app, host='0.0.0.0', port=8000)
