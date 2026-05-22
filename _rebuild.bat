@echo off
cd /d D:\github\nn

git checkout dev-yss >nul 2>&1

:: pr8
git push origin --delete pr8-section5 2>nul
git branch -D pr8-section5 2>nul
git checkout upstream/main >nul 2>&1
git checkout -b pr8-section5 >nul 2>&1
git checkout c1c6847b -- docs/carla_traffic_sign_recognition\ src\carla_traffic_sign_recognition\
git checkout fee829c0 -- docs\index.md
git add -A >nul 2>&1
git commit -m "up8" >nul 2>&1
git push --force origin pr8-section5
echo pr8 done

:: pr9
git checkout dev-yss >nul 2>&1
git push origin --delete pr9-section6 2>nul
git branch -D pr9-section6 2>nul
git checkout upstream/main >nul 2>&1
git checkout -b pr9-section6 >nul 2>&1
git checkout 542ec975 -- docs/carla_traffic_sign_recognition\ src\carla_traffic_sign_recognition\
git checkout fee829c0 -- docs\index.md
git add -A >nul 2>&1
git commit -m "up9" >nul 2>&1
git push --force origin pr9-section6
echo pr9 done

:: pr10
git checkout dev-yss >nul 2>&1
git push origin --delete pr10-section7 2>nul
git branch -D pr10-section7 2>nul
git checkout upstream/main >nul 2>&1
git checkout -b pr10-section7 >nul 2>&1
git checkout 3b855943 -- docs/carla_traffic_sign_recognition\ src\carla_traffic_sign_recognition\
git checkout fee829c0 -- docs\index.md
git add -A >nul 2>&1
git commit -m "up10" >nul 2>&1
git push --force origin pr10-section7
echo pr10 done

:: pr11
git checkout dev-yss >nul 2>&1
git push origin --delete pr11-tweak1 2>nul
git branch -D pr11-tweak1 2>nul
git checkout upstream/main >nul 2>&1
git checkout -b pr11-tweak1 >nul 2>&1
git checkout 3b2dac63 -- docs/carla_traffic_sign_recognition\ src\carla_traffic_sign_recognition\
git checkout fee829c0 -- docs\index.md
git add -A >nul 2>&1
git commit -m "up11" >nul 2>&1
git push --force origin pr11-tweak1
echo pr11 done

:: pr12
git checkout dev-yss >nul 2>&1
git push origin --delete pr12-tweak2 2>nul
git branch -D pr12-tweak2 2>nul
git checkout upstream/main >nul 2>&1
git checkout -b pr12-tweak2 >nul 2>&1
git checkout 05c5594b -- docs/carla_traffic_sign_recognition\ src\carla_traffic_sign_recognition\
git checkout fee829c0 -- docs\index.md
git add -A >nul 2>&1
git commit -m "up12" >nul 2>&1
git push --force origin pr12-tweak2
echo pr12 done

echo ALL DONE
