"""Download all github commits, adapted from: https://pypi.org/project/GitHubCommitsDownloader/"""
import os
import sys
import time
import json
import shutil
import pathlib
import traceback
import zipfile
if __name__ == '__main__':
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR.parent))

import requests

from utils.development import suppress

class Colors:
    RED = "\033[1;31m"
    YELLOW = "\033[1;33m"
    GREEN = "\033[1;32m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    NONE = "\033[0m"

def log(mess, color, *options):
    print(f"{color}{mess}{Colors.NONE}", *options)

def mkdir(path):
    path = pathlib.Path(SCRIPT_DIR / path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class GitHubCommitsDownloader:
    def __init__(self, user, repo, path, branch='master'):
        self.user = user
        self.repo = repo
        self.path = path
        self.branch = branch
        self.dates = {}

    def parse(self):
        args = self.parse_args(self.user, self.repo, self.branch)
        self.parse_branch(*args)

    def request(self, url: str) -> dict | list[dict]:
        req = requests.get(url)

        if int(req.headers.get("X-Ratelimit-Remaining")) == 0:
            time.sleep(int(req.headers.get('X-Ratelimit-Reset')))
            return self.request(url)

        return json.loads(req.content)

    def parse_args(self, _user: str, _repo: str, _branch: str):
        user, repo, branch = _user, _repo, _branch
        if not _user:
            raise Exception("User is empty use -u or --user ...")
        
        userGitHub = f"https://api.github.com/users/{user}"
        resp = self.request(userGitHub)
        if resp.get('message') == "Not Found":
            raise Exception("User Not Found")
        user = _user

        if not _repo:
            raise Exception("Reposetory is empty use -r or --repo ...")

        resp = self.request(userGitHub + "/repos")
        repos = []
        for repo in resp:
            repos.append(repo.get('name'))
        if not _repo in repos:
            raise Exception(f"User's have not Reposetory with named {_repo}")
        repo = _repo
        
        if not _branch:
            raise Exception("Branch is empty use -b or --branch ...")

        resp = self.request(f"https://api.github.com/repos/{user}/{_repo}/branches")
        branches = []

        for b in [*resp, {"name": "all"}]:
            branches.append(b.get('name'))

        for b in _branch.split(","):
            if not b.strip() in branches:
                raise Exception(f"Branch({b}) is not exists in this Reposetory")
        branch = _branch
        return user, repo, branch

    def parse_commits(self, branch, user, repo, commits):
        commits = self.request(commits)
        commitIndex = 0

        for commit in commits:
            sha = commit.get('sha')
            commit = f"https://github.com/{user}/{repo}/archive/{sha}.zip"

            p = f"{self.path}/{branch} - {commitIndex} - {sha[:6]}"
            _path = mkdir(p)
            downloader = Downloader(commit, _path)
            downloader.download()
            downloader.unzip()
            commitIndex += 1

            self.dates[p] = commit['commit']['commiter']['date']
            

    def parse_branch(self, user:str, repo:str, branch:str):
        log(f"Hi. I'm Parsing {user}'s {repo}'s {branch} branch", Colors.PURPLE)
        if branch == "all":
            branch_urls = f"https://api.github.com/repos/{user}/{repo}/branches"
            branches = self.request(branch_urls)
            for branch in branches:
                branch = branch.get('name')
                log(f"Started Parse of {branch} branch", Colors.CYAN)
                self.parse_commits(branch, user, repo, f"https://api.github.com/repos/{user}/{repo}/commits?sha={branch}")

        else:
            for b in branch.split(","):
                log(f"Started Parse of {b} branch", Colors.CYAN)
                b = b.strip()
                self.parse_commits(b, user, repo, f"https://api.github.com/repos/{user}/{repo}/commits?sha={b}")
                
class Downloader:
    def __init__(self, url, path: pathlib.Path):
        self.url = url
        self.path = path

    def download(self):
        try:
            with suppress(Exception):
                os.remove(self.path)

            with open(self.path / "archive.zip", "wb") as f:
                log("Downloading from:\n  " + self.url, Colors.GREEN)
                response = requests.get(self.url, stream=True)
                total_length = response.headers.get('content-length')
                if total_length is None:
                    f.write(response.content)
                    return

                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=16384):
                    dl += len(data)
                    f.write(data)

        except KeyboardInterrupt:
            print(str(self.path) + " is not Downloaded Fully\nWe Delete the file")
            os.remove(self.path)
            sys.exit(0)
        except Exception:
            traceback.print_exc()
            print("Error Found")
            os.remove(self.path)

    def unzip(self):
        try:
            log("\nUnzipping:\n  " + str(self.path) + "\n", Colors.YELLOW)
            with zipfile.ZipFile(self.path / "archive.zip", 'r') as zip_ref:
                zip_ref.extractall(self.path)
            os.remove(self.path / "archive.zip")
        except Exception:
            traceback.print_exc()
            if os.path.isdir(self.path) is not False:
                shutil.rmtree(self.path)


if __name__ == "__main__":
    gcd = GitHubCommitsDownloader(user='odota', repo='dotaconstants', path='dotaconstants', branch='master')
    gcd.parse()