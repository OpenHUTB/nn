#!/usr/bin/env python3
"""
fix_all_carla_files.py - 一次性修复CARLA项目的所有文件
"""

import os
import re
import sys
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份文件"""
    backup_path = file_path + '.backup'
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"  ⚠️  备份失败: {e}")
        return None

def fix_tracker_py(file_path):
    """修复tracker.py的所有问题"""
    print(f"\n🔧 修复 tracker.py...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 修复重复的 import queue
        if content.count("import queue") > 1:
            lines = content.split('\n')
            cleaned = []
            queue_count = 0
            for line in lines:
                if line.strip() == "import queue":
                    queue_count += 1
                    if queue_count == 1:
                        cleaned.append(line)
                else:
                    cleaned.append(line)
            content = '\n'.join(cleaned)
            fixes_applied.append("移除重复的import queue")
        
        # 2. 减少过多的注释分隔线
        lines = content.split('\n')
        cleaned = []
        separator_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('# =') and '=======' in stripped:
                separator_count += 1
                if separator_count <= 10:  # 保留最多10个分隔线
                    cleaned.append(line)
            else:
                cleaned.append(line)
        content = '\n'.join(cleaned)
        
        # 3. 减少连续空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(fixes_applied)}")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_config_yaml(file_path):
    """修复config.yaml的所有问题"""
    print(f"\n🔧 修复 config.yaml...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 修复Python表达式
        if 'if torch.cuda.is_available()' in content:
            content = content.replace(
                'device: "cuda" if torch.cuda.is_available() else "cpu"',
                '# 注意：需要根据实际情况设置device为"cuda"或"cpu"\ndevice: "cuda"  # 或 "cpu"'
            )
            fixes_applied.append("修复Python表达式")
        
        # 2. 注释未使用的配置
        lines = content.split('\n')
        cleaned = []
        
        unused_configs = [
            'auto_adjust_detection:',
            'smooth_alpha:',
            'yolo_iou:',
            'yolo_quantize:',
            'track_line_width:',
            'track_alpha:',
            'record_format:',
            'record_fps:',
            'pcd_view_size:'
        ]
        
        for line in lines:
            stripped = line.strip()
            is_unused = False
            
            for unused in unused_configs:
                if unused in stripped and not stripped.startswith('#'):
                    is_unused = True
                    break
            
            if is_unused:
                cleaned.append(f'# {line}  # 未使用或已优化')
                fixes_applied.append(f"注释{stripped.split(':')[0]}")
            else:
                cleaned.append(line)
        
        content = '\n'.join(cleaned)
        
        # 3. 减少空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(set(fixes_applied))[:50]}...")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_utils_py(file_path):
    """修复utils.py的所有问题"""
    print(f"\n🔧 修复 utils.py...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 简化日志配置
        simple_logger = '''# 配置日志
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
'''
        
        # 查找复杂日志配置
        old_logger_pattern = r'try:\s*from loguru import logger[\s\S]*?logger = SimpleLogger\(\)'
        if re.search(old_logger_pattern, content, re.MULTILINE):
            content = re.sub(old_logger_pattern, simple_logger, content, flags=re.MULTILINE)
            fixes_applied.append("简化日志配置")
        
        # 2. 移除YAML警告（如果配置已简化）
        if 'YAML_AVAILABLE = False' in content and 'logger.warning("PyYAML未安装"' in content:
            lines = content.split('\n')
            cleaned = []
            for line in lines:
                if 'logger.warning("PyYAML未安装"' in line:
                    cleaned.append(line.replace('warning', 'debug'))
                else:
                    cleaned.append(line)
            content = '\n'.join(cleaned)
        
        # 3. 减少注释分隔线
        content = re.sub(r'# ={20,}[\s\S]*?={20,}', '', content)
        
        # 4. 减少连续空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(fixes_applied)}")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_main_py(file_path):
    """修复main.py的所有问题"""
    print(f"\n🔧 修复 main.py...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 移除过多的注释分隔线
        lines = content.split('\n')
        cleaned = []
        last_was_sep = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('# =') and '=======' in stripped:
                if not last_was_sep:
                    cleaned.append(line)
                    last_was_sep = True
                else:
                    fixes_applied.append("移除冗余分隔线")
            else:
                cleaned.append(line)
                last_was_sep = False
        
        content = '\n'.join(cleaned)
        
        # 2. 修复可能的重复导入
        # 检查重复的 from utils import
        import_pattern = r'from utils import'
        if len(re.findall(import_pattern, content)) > 1:
            lines = content.split('\n')
            cleaned = []
            utils_imported = False
            for line in lines:
                if 'from utils import' in line:
                    if not utils_imported:
                        cleaned.append(line)
                        utils_imported = True
                    else:
                        fixes_applied.append("移除重复的utils导入")
                else:
                    cleaned.append(line)
            content = '\n'.join(cleaned)
        
        # 3. 减少连续空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(set(fixes_applied))}")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_sensors_py(file_path):
    """修复sensors.py的所有问题"""
    print(f"\n🔧 修复 sensors.py...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 检查并移除重复导入
        lines = content.split('\n')
        cleaned = []
        imports_seen = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped not in imports_seen:
                    imports_seen.add(stripped)
                    cleaned.append(line)
                else:
                    fixes_applied.append(f"移除重复导入: {stripped.split()[1] if ' ' in stripped else stripped}")
            else:
                cleaned.append(line)
        
        content = '\n'.join(cleaned)
        
        # 2. 移除过多的注释分隔线
        content = re.sub(r'# ={20,}[\s\S]*?={20,}', '', content)
        
        # 3. 减少连续空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(fixes_applied)}")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_requirements_txt(file_path):
    """修复requirements.txt的所有问题"""
    print(f"\n🔧 修复 requirements.txt...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixes_applied = []
        
        # 1. 移除重复依赖
        lines = content.split('\n')
        cleaned = []
        deps_seen = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # 提取包名
                dep_name = stripped.split('>=')[0].split('==')[0].split('<')[0].strip()
                if dep_name not in deps_seen:
                    deps_seen.add(dep_name)
                    cleaned.append(line)
                else:
                    fixes_applied.append(f"移除重复: {dep_name}")
            else:
                cleaned.append(line)
        
        content = '\n'.join(cleaned)
        
        # 2. 排序依赖（可选）
        lines = content.split('\n')
        core_deps = []
        optional_deps = []
        comments = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith('#'):
                comments.append(line)
            elif stripped.startswith('# 可选'):
                optional_deps.append(line)
            else:
                core_deps.append(line)
        
        # 重新组合
        sorted_content = []
        if comments:
            sorted_content.extend(comments)
            sorted_content.append('')
        if core_deps:
            sorted_content.extend(sorted(core_deps))
            sorted_content.append('')
        if optional_deps:
            sorted_content.extend(sorted(optional_deps))
        
        content = '\n'.join(sorted_content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if fixes_applied:
            print(f"  ✅ 修复完成: {', '.join(fixes_applied)}")
        else:
            print(f"  ℹ️  无需修复")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def fix_readme_md(file_path):
    """优化README.md（可选）"""
    print(f"\n🔧 优化 README.md...")
    
    backup = backup_file(file_path)
    if backup:
        print(f"  📦 已备份到: {os.path.basename(backup)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 可选：移除过多的空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ 优化完成")
        return True
        
    except Exception as e:
        print(f"  ❌ 优化失败: {e}")
        return False

def analyze_file(file_path):
    """分析文件的问题"""
    filename = os.path.basename(file_path)
    print(f"\n📊 分析 {filename}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        
        # 检查文件大小
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > 50:
            issues.append(f"文件较大 ({size_kb:.1f}KB)")
        
        # 检查特定问题
        if filename == "tracker.py":
            if content.count("import queue") > 1:
                issues.append("重复的import queue语句")
        
        elif filename == "config.yaml":
            if 'if torch.cuda.is_available()' in content:
                issues.append("包含Python表达式")
        
        elif filename.endswith('.py'):
            # 检查过多的分隔线
            sep_count = len(re.findall(r'# ={20,}', content))
            if sep_count > 10:
                issues.append(f"过多的注释分隔线 ({sep_count}个)")
            
            # 检查空行比例
            lines = content.split('\n')
            empty_lines = sum(1 for line in lines if not line.strip())
            if len(lines) > 0 and empty_lines / len(lines) > 0.25:
                issues.append(f"空行过多 ({empty_lines}/{len(lines)} 行)")
        
        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
            return issues
        else:
            print(f"  ✅ 无发现问题")
            return []
            
    except Exception as e:
        print(f"  ❌ 分析失败: {e}")
        return ["分析失败"]

def main():
    """主函数"""
    print("="*70)
    print("🚀 CARLA项目完整修复工具")
    print("="*70)
    
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 定义要处理的文件
    target_files = {
        "tracker.py": fix_tracker_py,
        "config.yaml": fix_config_yaml,
        "utils.py": fix_utils_py,
        "main.py": fix_main_py,
        "sensors.py": fix_sensors_py,
        "requirements.txt": fix_requirements_txt,
        "README.md": fix_readme_md,
    }
    
    # 检查哪些文件存在
    existing_files = {}
    for filename, fix_func in target_files.items():
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            existing_files[filename] = (file_path, fix_func)
            print(f"✅ 找到 {filename}")
        else:
            print(f"❌ 未找到 {filename}")
    
    if not existing_files:
        print("\n❌ 未找到任何CARLA项目文件！")
        print("请确保在正确的目录运行本脚本")
        return
    
    print(f"\n📋 找到 {len(existing_files)} 个文件需要处理")
    
    # 询问用户
    print("\n请选择操作:")
    print("1. 先分析所有文件的问题")
    print("2. 直接修复所有文件")
    print("3. 退出")
    
    try:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            # 分析模式
            print("\n" + "="*70)
            print("📊 文件分析报告")
            print("="*70)
            
            all_issues = {}
            for filename, (file_path, _) in existing_files.items():
                issues = analyze_file(file_path)
                if issues:
                    all_issues[filename] = issues
            
            if all_issues:
                print(f"\n⚠️  发现 {len(all_issues)} 个文件有问题:")
                for filename, issues in all_issues.items():
                    print(f"\n  {filename}:")
                    for issue in issues:
                        print(f"    • {issue}")
                
                fix_choice = input("\n是否要修复这些问题？(y/N): ").strip().lower()
                if fix_choice == 'y':
                    choice = '2'  # 进入修复模式
                else:
                    print("退出")
                    return
            else:
                print("\n✅ 所有文件都正常，无需修复")
                return
        
        if choice == '2':
            # 修复模式
            print("\n" + "="*70)
            print("🔧 开始修复所有文件")
            print("="*70)
            
            results = []
            for filename, (file_path, fix_func) in existing_files.items():
                success = fix_func(file_path)
                results.append((filename, success))
            
            # 显示结果
            print("\n" + "="*70)
            print("📋 修复完成报告")
            print("="*70)
            
            successful = sum(1 for _, success in results if success)
            total = len(results)
            
            print(f"✅ 成功修复: {successful}/{total} 个文件")
            print("\n详细结果:")
            for filename, success in results:
                status = "✅ 成功" if success else "❌ 失败"
                print(f"  {filename:20} {status}")
            
            print(f"\n✨ 所有原文件已备份为 .backup 文件")
            print("💡 如需恢复原文件:")
            print("  1. 删除修复后的文件")
            print("  2. 将 .backup 文件重命名回去（去掉.backup后缀）")
        
        elif choice == '3':
            print("退出")
            return
            
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n操作取消")

if __name__ == "__main__":
    main()