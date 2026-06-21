import asyncio
from contextlib import asynccontextmanager

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from routes.input import router as input_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Band AI agents in background when server boots
    from band_agents import start_agents
    asyncio.create_task(start_agents(), name="band-agents")
    yield


app = FastAPI(title="Retention", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(input_router)


@app.get("/")
async def root():
    async with aiofiles.open("static/index.html") as f:
        return HTMLResponse(await f.read())
