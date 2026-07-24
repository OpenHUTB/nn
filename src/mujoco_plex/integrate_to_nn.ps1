# Merge mujoco_plex project into nn repo
# Run command: powershell -ExecutionPolicy Bypass -File integrate_to_nn.ps1
$ErrorActionPreference = "Stop"

# Get current script folder
$scriptDir = $PSScriptRoot
# 你的真实 nn 仓库根目录：脚本往上两级 D:\nn\nn\
$nnRepoRoot = Split-Path (Split-Path $scriptDir -Parent) -Parent
# nn_bundle 放在脚本同目录 D:\nn\nn\src\mujoco_plex\nn_bundle
$bundleDir = Join-Path $scriptDir "nn_bundle"

# Check nn repo git folder
$gitDir = Join-Path $nnRepoRoot ".git"
if (-not (Test-Path $gitDir))
{
    Write-Host "[ERROR] nn git repo not found at $nnRepoRoot" -ForegroundColor Red
    Write-Host "Run clone cmd first: git clone --depth 1 https://github.com/Liu-Yirun/nn.git `"$nnRepoRoot`""
    exit 1
}

# Check bundle source folder
if (-not (Test-Path $bundleDir))
{
    Write-Host "[ERROR] nn_bundle folder missing at $bundleDir" -ForegroundColor Red
    Write-Host "Make sure nn_bundle is placed same directory as this script"
    exit 1
}

# Copy source code
$srcSource = Join-Path $bundleDir "src\mujoco_plex"
$srcDest = Join-Path $nnRepoRoot "src\mujoco_plex"
Copy-Item -Path $srcSource -Destination $srcDest -Recurse -Force

# Copy docs
$docSource = Join-Path $bundleDir "docs\mujoco_plex"
$docDest = Join-Path $nnRepoRoot "docs\mujoco_plex"
Copy-Item -Path $docSource -Destination $docDest -Recurse -Force
Write-Host "[OK] Copied src/mujoco_plex & docs/mujoco_plex" -ForegroundColor Green

# Update mkdocs.yml nav
$mkdocsFile = Join-Path $nnRepoRoot "mkdocs.yml"
$mkdocsRaw = Get-Content -Path $mkdocsFile -Raw -Encoding UTF8
$newNavItem = "- ANYmal C Quadruped Robot Simulation: 'mujoco_plex/README.md'"
$matchCarlaLine = "(- CARLA Multi-sensor Autonomous Driving Platform: 'carla_multisensor_platform/carla_multisensor_platform.md')"

if ($mkdocsRaw -notmatch "mujoco_plex")
{
    $mkdocsRaw = $mkdocsRaw -replace $matchCarlaLine, "`$1`n$newNavItem"
    Set-Content -Path $mkdocsFile -Value $mkdocsRaw -Encoding UTF8 -NoNewline
    Write-Host "[OK] Updated mkdocs.yml navigation" -ForegroundColor Green
}
else
{
    Write-Host "[SKIP] mujoco_plex already exists in mkdocs.yml" -ForegroundColor Cyan
}

# Update docs/index.md homepage link
$indexFile = Join-Path $nnRepoRoot "docs\index.md"
$indexRaw = Get-Content -Path $indexFile -Raw -Encoding UTF8
$newIndexLink = "- [__ANYmal C Quadruped Robot Simulation__](./mujoco_plex/README.md) - MuJoCo based ANYmal C quadruped robot simulation & leg swing control"
$matchAntLine = "(- \[__Robot Simulation(MuJoCo)__\]\(ant_robot/robot_simulation.md\))"

if ($indexRaw -notmatch "mujoco_plex")
{
    $indexRaw = $indexRaw -replace $matchAntLine, "`$1`n`n$newIndexLink"
    Set-Content -Path $indexFile -Value $indexRaw -Encoding UTF8 -NoNewline
    Write-Host "[OK] Updated docs/index.md project list" -ForegroundColor Green
}
else
{
    Write-Host "[SKIP] mujoco_plex link already exists in index.md" -ForegroundColor Cyan
}

Write-Host "`n==================== Task Complete ====================" -ForegroundColor Green
Write-Host "Next git commands:"
Write-Host "  cd `"$nnRepoRoot`""
Write-Host "  git add src/mujoco_plex docs/mujoco_plex mkdocs.yml docs/index.md"
Write-Host "  git commit -m 'Add MuJoCo ANYmal C quadruped robot simulation project'"
Write-Host "  git push origin main"