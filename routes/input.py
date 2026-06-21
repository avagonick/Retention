import json
import os
import uuid
from pathlib import Path

import aiofiles
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter()


@router.websocket("/ws/transcribe")
async def transcribe(websocket: WebSocket):
    await websocket.accept()

    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    dg = deepgram.listen.asynclive.v("1")

    async def on_transcript(self, result, **kwargs):
        text = result.channel.alternatives[0].transcript
        if text:
            await websocket.send_json({
                "type": "transcript",
                "text": text,
                "is_final": result.is_final,
            })

    async def on_error(self, error, **kwargs):
        await websocket.send_json({"type": "error", "message": str(error)})

    dg.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-2",
        language="en-US",
        punctuate=True,
        interim_results=True,
        endpointing=300,
    )

    started = await dg.start(options)
    if not started:
        await websocket.send_json({"type": "error", "message": "Could not connect to Deepgram"})
        await websocket.close()
        return

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
            await dg.send(audio_chunk)
    except WebSocketDisconnect:
        pass
    finally:
        await dg.finish()


@router.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    suffix = Path(video.filename).suffix or ".mp4"
    save_path = UPLOAD_DIR / f"{video_id}{suffix}"

    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            await f.write(chunk)

    return JSONResponse({
        "video_id": video_id,
        "filename": video.filename,
        "path": str(save_path),
    })


@router.post("/process")
async def process(data: dict):
    video_id = data.get("video_id")
    transcript = data.get("transcript", "").strip()

    if not video_id or not transcript:
        return JSONResponse({"error": "video_id and transcript are required"}, status_code=400)

    from agents import run_loop
    from generate import generate

    source_video_path = str(UPLOAD_DIR / f"{video_id}.mp4")

    result = await run_loop(
        session_id=video_id,
        question=transcript,
        source_video_path=source_video_path,
        generate_fn=generate,
    )

    return JSONResponse({
        "video_id": video_id,
        "question": transcript,
        **result,
    })
