import os, subprocess, tempfile, uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="Clip backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_model = None

def model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise HTTPException(500, p.stderr[-500:])

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), start: float = Form(0), end: float = Form(30), language: str = Form("")):
    with tempfile.TemporaryDirectory() as d:
        src, wav = os.path.join(d, "in"), os.path.join(d, "a.wav")
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        run(["ffmpeg", "-y", "-ss", str(start), "-t", str(max(end - start, 1)), "-i", src,
             "-vn", "-ac", "1", "-ar", "16000", wav])
        segs, _ = model().transcribe(wav, language=language or None, vad_filter=True)
        return {"segments": [{"start": round(s.start + start, 2), "end": round(s.end + start, 2), "text": s.text.strip()} for s in segs]}

@app.post("/download")
def download(url: str = Form(...)):
    from fastapi.responses import FileResponse
    d = tempfile.mkdtemp()
    out = os.path.join(d, f"{uuid.uuid4().hex}.mp4")
    run(["yt-dlp", "-f", "mp4[height<=720]/best[height<=720]", "-o", out, url])
    return FileResponse(out, media_type="video/mp4")
