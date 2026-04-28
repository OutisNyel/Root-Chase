# 宿主机离线监控（不依赖容器监控服务）

脚本路径：`train_monitor/offline_monitor.py`

目标：
- 离线查看训练日志健康状态
- 提取 learner / aisrv 关键训练指标
- 导出结构化 `summary.json`，便于 Codex 直接读取

## 1. 汇总查看

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log summary
```

## 2. 导出给 Codex 读的 JSON

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log --json-out P:\Repos\GorgeChase.d\train\log\summary.json summary
```

## 3. 实时刷新（近似监控面板）

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log watch --interval 5
```

## 4. 快速查日志

只看 learner 最新 80 行：

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log tail --module learner --lines 80
```

只看 ERROR：

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log tail --level ERROR --lines 200
```

按关键词筛选：

```powershell
python train_monitor\offline_monitor.py --log-dir P:\Repos\GorgeChase.d\train\log tail --grep "save model" --lines 100
```

## 说明

- 脚本是纯 Python 标准库实现，不需要额外安装依赖。
- `summary` 输出里包含：
  - 各模块日志量与级别统计
  - learner 最新 `global_step/train_count/loss/checkpoint`
  - aisrv `GAMEOVER` 汇总（失败率、平均 reward、最新一条）
  - 最近 WARNING/ERROR 列表
