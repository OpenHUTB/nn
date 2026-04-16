<!-- Markdown 内嵌交互面板：输入 GitHub ID 查询 Issue 总数 -->
<div style="margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 8px;">
  <h4>GitHub Issue 数量查询</h4>
  <input
    type="text"
    id="githubId"
    placeholder="请输入 GitHub ID（例如 SEMHAQ）"
    style="padding: 8px; width: 280px; margin-right: 10px;"
  />
  <button
    onclick="searchIssues()"
    style="padding: 8px 16px; background: #0969da; color: white; border: none; border-radius: 4px; cursor: pointer;"
  >
    查询数量
  </button>
  <div style="margin-top: 12px; font-size: 16px;">
    查询结果：<span id="result" style="font-weight: bold; color: #0969da;">-</span>
  </div>
</div>

<script>
function searchIssues() {
  // 获取输入的 GitHub ID
  const userId = document.getElementById('githubId').value.trim();
  const resultEl = document.getElementById('result');
  
  if (!userId) {
    resultEl.textContent = '请输入有效的 GitHub ID';
    return;
  }

  // 加载中提示
  resultEl.textContent = '查询中...';

  // 调用 GitHub API（固定仓库：OpenHUTB/nn）
  const apiUrl = `https://api.github.com/search/issues?q=is:issue+involves:${userId}+repo:OpenHUTB/nn`;

  fetch(apiUrl)
    .then(res => res.json())
    .then(data => {
      // 直接取出 total_count（你要的核心数字）
      resultEl.textContent = data.total_count || 0;
    })
    .catch(err => {
      resultEl.textContent = '查询失败';
      console.error(err);
    });
}
</script>