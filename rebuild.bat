@echo off  
cd /d D:\github\nn  
git checkout dev-yss >nul 2>&1  
git push origin --delete pr8-section5 2>nul  
git branch -D pr8-section5 2>nul  
git checkout upstream/main >nul 2>&1  
git checkout -b pr8-section5 >nul 2>&1  
git checkout c1c6847b -- docs/carla_traffic_sign_recognition/ src/carla_traffic_sign_recognition/ 2>nul  
git checkout fee829c0 -- docs/index.md 2>nul  
git add -A >nul 2>&1  
git commit -m \"up8\" >nul 2>&1  
git push --force origin pr8-section5  
echo \"pr8 done\" 
