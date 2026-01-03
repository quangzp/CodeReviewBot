from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from src_bot.bot import bot_instance
from src_bot.service import process_pr_review

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bot_instance.initialize() 
    except Exception as e:
        print(f"Lỗi khởi tạo Bot: {e}")
    yield
    bot_instance.close()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def receive_webhook(payload: dict, background_tasks: BackgroundTasks):
   
    background_tasks.add_task(
        process_pr_review, 
        repo_name=payload['repo_full_name'], 
        pr_number=payload['pr_number'],
        bot=bot_instance 
    )
    return {"status": "ok"}