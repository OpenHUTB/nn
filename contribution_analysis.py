import subprocess
import argparse
import csv
import json
import os
from collections import Counter
from typing import Optional

import requests


def get_login_by_sha(sha: str, repo: str, token: str,
                    cache: dict[str, Optional[str]]) -> Optional[str]:
    """Get GitHub login ID by commit SHA with caching.

    Args:
        sha: Commit SHA hash.
        repo: GitHub repository in format 'owner/repo'.
        token: GitHub API token.
        cache: Dictionary for caching SHA to login mappings.

    Returns:
        GitHub login ID or None if not found.
    """
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
    except requests.RequestException as e:
        print(f"SHA查询异常({sha}): {e}")
    return None


def load_ignore_users(file_path: str) -> set[str]:
    """Load ignore users list from JSON file.

    Args:
        file_path: Path to JSON file containing ignore list.

    Returns:
        Set of lowercase usernames to ignore.
    """
    if not os.path.exists(file_path):
        return set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {str(u).strip().lower() for u in json.load(f)}
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载屏蔽名单失败: {e}")
        return set()


def run_analysis() -> None:
    """Run contribution analysis for a GitHub repository."""
    parser = argparse.ArgumentParser(description="GitHub contribution analysis")
    parser.add_argument("-t", "--token", required=True, help="GitHub API token")
    parser.add_argument("-r", "--repo", required=True, help="GitHub repository (owner/repo)")
    parser.add_argument("--since", help="Start date for analysis")
    parser.add_argument("--until", help="End date for analysis")
    parser.add_argument("--ignore", default="ignore_users.json", help="Ignore users JSON file")
    parser.add_argument("--output", default="commit_stats.csv", help="Output CSV file")
    args = parser.parse_args()

    ignore_set = load_ignore_users(args.ignore)

    # Get commit SHA list from local Git repository
    cmd = ["git", "log", "--pretty=%H"]
    if args.since:
        cmd.append(f"--since={args.since}")
    if args.until:
        cmd.append(f"--until={args.until}")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("获取Git日志失败")
        return

    shas = [s.strip() for s in result.stdout.split('\n') if s.strip()]
    login_counts: Counter[str] = Counter()
    sha_to_login_cache: dict[str, Optional[str]] = {}

    print(f"检测到 {len(shas)} 个提交，正在追溯归属...")

    for sha in shas:
        login = get_login_by_sha(sha, args.repo, args.token, sha_to_login_cache)
        if login and login.lower() not in ignore_set:
            login_counts[login] += 1

    # Export results
    sorted_stats = sorted(login_counts.items(), key=lambda x: x[1], reverse=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GitHub_Login", "Commits"])
        writer.writerows(sorted_stats)
    print(f"分析完成，导出至 {args.output}")


if __name__ == "__main__":
    run_analysis()