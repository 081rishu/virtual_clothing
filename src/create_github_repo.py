import os
from github import Github

access_token = os.getenv("GITHUB_TOKEN")
if not access_token:
    raise ValueError('GITHUB_TOKEN environment variable not set')

print(f"Using access_token : {access_token}")
g = Github(access_token)

user = g.get_user()
repo_name = "virtual_clothing"
repo_description = "A computer vision model for virtual clothing and accessories using given prompts"

repo = user.create_repo(repo_name, description=repo_description, private=False)
print(f"Respository '{repo_name}' created successfully")