import subprocess
import argparse
import csv
import json
import os
import requests
from collections import Counter

def get_login_by_sha(sha, repo, token, cache):
    """通过 Commit SHA 获取真实的 GitHub Login ID (带缓存)"""
    if sha in cache:
        return cache[sha]

    url = f"https://api.github.com/repos/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            author_obj = data.get("author")
            if author_obj:
                login = author_obj.get("login")
                cache[sha] = login
                return login
    except Exception as e:
        print(f"SHA查询异常({sha}): {e}")
    return None

def load_ignore_users(file_path):
    """从外部 JSON 加载屏蔽名单"""
    if not os.path.exists(file_path):
        return set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {str(u).strip().lower() for u in json.load(f)}
    except:
        return set()

def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--token", required=True)
    parser.add_argument("-r", "--repo", required=True)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--ignore", default="ignore_users.json")
    parser.add_argument("--output", default="commit_stats.csv")
    args = parser.parse_args()

    ignore_set = load_ignore_users(args.ignore)
    
    # 获取本地 Git 仓库的 Commit SHA 列表
    cmd = ["git", "log", "--pretty=%H"]
    if args.since: cmd.append(f"--since={args.since}")
    if args.until: cmd.append(f"--until={args.until}")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return

    shas = [s.strip() for s in result.stdout.split('\n') if s.strip()]
    login_counts = Counter()
    sha_to_login_cache = {}

    print(f"检测到 {len(shas)} 个提交，正在追溯归属...")

    for sha in shas:
        login = get_login_by_sha(sha, args.repo, args.token, sha_to_login_cache)
        if login and login.lower() not in ignore_set:
            login_counts[login] += 1

    # 导出
    sorted_stats = sorted(login_counts.items(), key=lambda x: x[1], reverse=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GitHub_Login", "Commits"])
        writer.writerows(sorted_stats)
    print(f"分析完成，导出至 {args.output}")

if __name__ == "__main__":
    run_analysis()