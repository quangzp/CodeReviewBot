from github import Github, Auth,PullRequest
from unidiff import PatchSet
from src_bot.bot import bot_instance as langgraph_bot
from src_bot.config.config import configs
import requests
class BotService:
    def __init__(self):
        self.github_token = configs.GITHUB_TOKEN
    
    def initialize(self):
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is not set in the configuration.")
        self.github_auth = Auth.Token(self.github_token)
        self.github_client = Github(auth=self.github_auth)
        print("GitHub client initialized.")
    
    def close(self):
        if self.github_client is not None:
            self.github_client.close()
            print("GitHub client closed.")


    def process_pr_review(self, repo_name: str, pr_number: int):
        print(f"--- STARTING REVIEW FOR PR: {repo_name}#{pr_number} ---")
    
        try:
            g = self.github_client
            repo = g.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
        
            files = pr.get_files()
            code_files = [f for f in files if f.filename.endswith(('.py', '.js', '.java', '.cpp', '.ts', '.go', '.rb'))]

            diff_response = requests.get(pr.url, headers={"Authorization": f"token {self.github_token}", "Accept": "application/vnd.github.v3.diff"})
            diff_content = diff_response.text
            # print(diff_content)
            patch_set = PatchSet(diff_content)

            for f in code_files:
                bot_input = {"pr_diff": f"{f.filename}\n{f.patch}"}
                result = langgraph_bot.invoke(bot_input)
                review_body = result.get("final_review")
                if review_body:
                    self.post_comment_on_line(pr, patch_set, f.filename, review_body)
            
            print(f"--- FINISHED REVIEW FOR {repo_name}#{pr_number} ---")

        except Exception as e:
            print(f"ERROR reviewing PR {repo_name}#{pr_number}: {str(e)}")
    
    def post_comment_on_line(self,pr : PullRequest, patch_set:PatchSet,file_path:str, comment_body:str, side="RIGHT"):
        target_file = None
        for patched_file in patch_set:
            if patched_file.path == file_path:
                target_file = patched_file
                break
                
        if not target_file:
            print(f"File {file_path} không tìm thấy trong PR này.")
            return
        
        first_hunk = target_file[0] 
        if not first_hunk:
            return
        valid_line = None
        for line in first_hunk:
            if not line.is_removed:
                valid_line = line.target_line_no
                break
        commits = list(pr.get_commits())
        last_commit = commits[-1] if commits else None
        if not last_commit:
            print(f"No commits found for PR {pr.number}")
            return
        if valid_line:
            try:
                pr.create_review_comment(
                    body=comment_body, 
                    commit=last_commit, 
                    path=file_path, 
                    side=side,
                    line=valid_line
                )
            except Exception as e:
                print(f"  -> Error: {e}")
    
    def get_pr(self, repo_name: str, pr_number: int):
        """Get a PullRequest object for the given repository and PR number."""
        repo = self.github_client.get_repo(repo_name)
        return repo.get_pull(pr_number)
    
    def get_pr_files(self, pr: PullRequest):
        """Get all files changed in a Pull Request."""
        return pr.get_files()

bot_service_instance = BotService()