import os
from github import Github
from bot import bot_instance as langgraph_bot

def get_github_client():
    """Tạo GitHub Client (Dùng Token hoặc App ID)"""
    token = os.getenv("GITHUB_TOKEN")
    return Github(token)

def process_pr_review(repo_name: str, pr_number: int):
    print(f"--- STARTING REVIEW FOR PR: {repo_name}#{pr_number} ---")
    
    try:
        
        g = get_github_client()
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        
        # Lấy file diff (chỉ lấy các file code, bỏ qua ảnh/doc nếu cần)
        # Lưu ý: diff_url chứa raw diff
        import requests
        diff_content = requests.get(pr.diff_url).text
        
        if not diff_content:
            print("Diff is empty.")
            return

        bot_input = {"pr_diff": diff_content}
        result = langgraph_bot.invoke(bot_input)
        
        review_body = result.get("final_review", "Bot failed to generate review.")

        pr.create_issue_comment(review_body)
        
        print(f"--- FINISHED REVIEW FOR {repo_name}#{pr_number} ---")

    except Exception as e:
        print(f"ERROR reviewing PR {repo_name}#{pr_number}: {str(e)}")
        # Có thể gửi thông báo lỗi về Slack/Discord ở đây