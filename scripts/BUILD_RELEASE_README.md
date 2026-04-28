# GorgeChase Release Packaging

这个脚本用于生成GorgeChase项目的release版本。

## 📦 包含内容

Release版本包含以下结构：

```
code/
├── conf/              # 配置目录
├── agent_ppo/         # PPO代理实现
├── .vscode/           # VS Code配置
├── kaiwu.json         # Kaiwu配置文件
└── train_test.py      # 训练测试脚本
```

## 🚀 使用方法

### 方法1: Python脚本（推荐，跨平台）

#### 默认模式（目录形式）
```bash
# 使用默认输出目录 (release/)
python scripts/build_release.py

# 自定义输出目录
python scripts/build_release.py --output /path/to/output

# 自定义归档名称
python scripts/build_release.py --output ./releases --name my-release
```

#### ZIP模式（创建压缩包）
```bash
# 创建zip格式的发布包
python scripts/build_release.py --zip

# 自定义输出目录和名称
python scripts/build_release.py --output ./releases --name my-release --zip
python scripts/build_release.py -o ./releases -n my-release -z
```

### 方法2: PowerShell脚本（Windows）

#### 默认模式（目录形式）
```powershell
# 使用默认输出目录
.\scripts\build_release.ps1

# 自定义输出目录
.\scripts\build_release.ps1 -OutputDir "D:\releases"

# 自定义归档名称
.\scripts\build_release.ps1 -OutputDir ".\releases" -ArchiveName "my-release"
```

#### ZIP模式（创建压缩包）
```powershell
# 创建zip格式的发布包
.\scripts\build_release.ps1 -UseZip

# 自定义输出目录和名称
.\scripts\build_release.ps1 -OutputDir "D:\releases" -ArchiveName "my-release" -UseZip
```

## 📋 输出说明

### 目录模式（默认）
脚本会生成以下目录到输出目录：

- `gorgechase-release-20260426_172106/` - 带时间戳的release版本目录
  - `code/`
    - `conf/`
    - `agent_ppo/`
    - `.vscode/`
    - `kaiwu.json`
    - `train_test.py`
- `latest/` - 指向最新release的快捷目录

示例输出：
```
✅ Release directory created successfully!
   Directory: P:\Repos\GorgeChase\release\gorgechase-release-20260426_172106
   Size: 3.39 MB

✓ Created latest directory reference

📦 Release Info:
   Name: gorgechase-release-20260426_172106
   Location: P:\Repos\GorgeChase\release\gorgechase-release-20260426_172106
   Latest: P:\Repos\GorgeChase\release\latest
```

### ZIP模式
脚本会生成以下文件到输出目录：

- `gorgechase-release-20260426_172106.zip` - 带时间戳的release压缩包
- `latest.zip` - 指向最新release的快捷链接

## 🔍 验证内容

### 目录模式
直接查看目录结构：
```bash
cd release/latest/code/
ls -la
# 应该看到: conf, agent_ppo, .vscode, kaiwu.json, train_test.py
```

### ZIP模式
提取archive后检查内容：
```bash
unzip release/latest.zip
cd code/
ls -la
# 应该看到: conf, agent_ppo, .vscode, kaiwu.json, train_test.py
```

## ⚙️ 脚本特性

- ✓ 两种模式可选：目录模式（默认）和ZIP模式
- ✓ 自动创建输出目录
- ✓ 增量打包（每次生成新的时间戳版本）
- ✓ 自动维护latest快捷引用
- ✓ 自动清理临时文件
- ✓ 提供详细的执行日志
- ✓ 错误处理和警告提示
- ✓ 显示包大小信息
- ✓ 跨平台支持（Python版本）

