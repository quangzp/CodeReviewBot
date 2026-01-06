import sys
import os

# 1. Get the path to the current folder ('api')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent folder ('project_root')
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent folder to the system path
sys.path.append(parent_dir)

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from src_bot.bot import bot_instance
from src_bot.service import bot_service_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bot_instance.initialize() 
        bot_service_instance.initialize()
    except Exception as e:
        print(f"Lỗi khởi tạo Bot: {e}")
    yield
    bot_instance.close()
    bot_service_instance.close()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def receive_webhook(payload: dict, background_tasks: BackgroundTasks):
    if payload.get('action') != 'opened':
        return {"status": "ignored"}
    background_tasks.add_task(
        bot_service_instance.process_pr_review, 
        repo_name=payload['repository']['full_name'], 
        pr_number=payload['pull_request']['number']
    )
    return {"status": "ok"}