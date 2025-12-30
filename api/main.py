from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/code-bot")
async def receive_webhook(payload: dict):
    # 1. Lấy đối tượng pull_request (chứa 90% thông tin bạn cần)
    pr = payload.get("pull_request", {})
    
    # 2. Tạo một dictionary mới chỉ chứa thông tin quan trọng
    # Dùng .get() để tránh lỗi nếu dữ liệu bị thiếu
    clean_data = {
        "action": payload.get("action"),               # opened, closed, merged...
        "pr_number": pr.get("number"),                 # Số thứ tự PR (vd: #12)
        "title": pr.get("title"),                      # Tiêu đề PR
        "author": pr.get("user", {}).get("login"),     # Người tạo
        "url": pr.get("html_url"),                     # Link để bấm vào xem
        "body": pr.get("body"),                        # Nội dung mô tả (Description)
        "from_branch": pr.get("head", {}).get("ref"),  # Nhánh nguồn (vd: dev)
        "to_branch": pr.get("base", {}).get("ref"),    # Nhánh đích (vd: main)
        "changed_files": pr.get("changed_files"),      # Số file bị thay đổi
        "additions": pr.get("additions"),              # Số dòng code thêm vào
        "deletions": pr.get("deletions")               # Số dòng code bị xóa
    }

    print("\n" + "="*30)
    print(clean_data)