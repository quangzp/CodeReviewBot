from turtle import position
from github import Github, Auth, PullRequest

from github import Github
from numpy import diff
from unidiff import PatchSet
import requests

def post_comment_on_line(pr : PullRequest, patch_set:PatchSet,file_path:str, comment_body:str, side="RIGHT"):
   
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
    last_commit = pr.get_commits()[pr.commits - 1]
    if valid_line:
        try:
            pr.create_review_comment(
                body=comment_body, 
                commit=last_commit, 
                path=f.filename, 
                side=side,
                line=valid_line
            )
        except Exception as e:
            print(f"  -> Error: {e}")
        
auth = Auth.Token('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')
g = Github(auth=auth)
repo = g.get_repo("quangzp/Rabiloo-Hamic-BE")
pr = repo.get_pull(2)
files = pr.get_files()
code_files = [f for f in files if f.filename.endswith(('.py', '.js', '.java', '.cpp', '.ts', '.go', '.rb'))]
diff_response = requests.get(pr.url, headers={"Authorization": f"token {auth.token}", "Accept": "application/vnd.github.v3.diff"})
diff_content = diff_response.text
for f in code_files:
    # print(f.filename)
    print(f.patch)
    # post_comment_on_line(pr, diff_content, f.filename, "Test comment từ bot", side="RIGHT",)
    print("-----"*20)