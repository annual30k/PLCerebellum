# PatrolLink 单兵边缘智能小脑 Docker 模拟系统

该目录用于模拟“单兵边缘智能小脑服务器”的量产运行环境。它不是完整 AI 算法镜像，而是一个贴近目标硬件和系统裁剪策略的可运行原型，用于验证设备状态、视频接入、车牌/人脸候选、报告生成、审计日志和系统加固策略。

## 模拟配置

| 项目 | 配置 |
|---|---|
| 目标硬件 | NVIDIA Jetson Orin NX 16GB |
| AI 算力 | 按 157 TOPS 设备画像模拟 |
| 容器 CPU 限制 | 8 核 |
| 容器内存限制 | 16GB |
| 存储画像 | 1TB NVMe |
| 电池画像 | 60Wh |
| 主力大模型 | Qwen3.5-4B-INT4 |
| 备用模型 | Qwen3.5-2B-INT4 |
| 批处理模型 | Qwen3.5-9B-INT4 |
| 默认上下文 | 16K tokens |
| 最大上下文 | 32K tokens |

## Linux 裁剪与加固

当前镜像采用 `python:3.12-slim`，只保留运行 API 服务、健康检查和视频处理模拟所需组件。Compose 中启用了以下量产安全策略：

| 策略 | 状态 |
|---|---|
| 非 root 用户运行 | 已启用 |
| 只读根文件系统 | 已启用 |
| `/tmp` 和 `/run` tmpfs | 已启用 |
| Linux capabilities | 全部丢弃 |
| no-new-privileges | 已启用 |
| SSH | 未安装、未开放 |
| 桌面环境 | 未安装 |
| USB 自动挂载 | 未启用 |
| 包管理器运行时安装 | 不作为业务入口暴露 |

## 启动

```bash
cd /Users/qiuqiquan/Desktop/SmartHeadsetSystem/PatrolLink/PLCerebellum
docker compose up --build -d
```

查看状态：

```bash
docker compose ps
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/api/v1/device/status
```

## 接口示例

模拟媒体接入：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/media/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"bodycam-rtsp-01","media_type":"stream","duration_seconds":60,"note":"巡逻视频流"}'
```

模拟车牌识别：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/plate \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"frame-20260514-001","camera_id":"bodycam-01"}'
```

模拟人脸候选提示：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/face \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"frame-20260514-002","camera_id":"bodycam-01"}'
```

模拟 Qwen3.5-4B 报告生成：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/llm/report \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","report_type":"daily","operator_note":"今日重点巡逻商业街区域"}'
```

查看事件和审计：

```bash
curl http://127.0.0.1:8088/api/v1/events
curl http://127.0.0.1:8088/api/v1/audit
```

## 替换真实模型

真实部署时，将量化后的 Qwen3.5-4B 模型挂载到：

```text
/opt/cerebellum/models
```

然后把 `app.services.generate_report` 替换为 llama.cpp、Ollama、TensorRT-LLM 或国产 NPU SDK 的实际推理调用。当前版本保留同名模型、上下文限制和任务调度策略，方便后续替换真实推理后端。

