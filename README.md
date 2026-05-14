# PatrolLink 单兵边缘智能小脑 Docker 模拟系统

该目录用于模拟“单兵边缘智能小脑服务器”的量产运行环境。它不是完整 AI 算法镜像，而是一个贴近目标硬件和系统裁剪策略的可运行原型，用于验证设备状态、视频接入、车牌/人脸候选、报告生成、审计日志和系统加固策略。

当前版本已经接入真实 `Qwen3.5-4B Q4_K_M GGUF` 本地模型，通过 llama.cpp server 提供 OpenAI 兼容接口。车牌、人脸和视频分析仍是模拟服务，后续需要替换成真实视觉模型。

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
| 包管理器入口 | `apt` / `apt-get` / `dpkg` 常用入口已从运行镜像移除；真正量产仍应配合镜像签名和只读系统分区 |
| 端口暴露 | 默认仅绑定 `127.0.0.1:8088`，不直接暴露到局域网 |
| 事件与审计 | 内存中保留最近 1000 条，审计文件超过约 5MB 自动轮转 |

## 启动

```bash
cd /Users/qiuqiquan/Desktop/SmartHeadsetSystem/PatrolLink/PLCerebellum
docker compose up --build -d
```

首次启动会拉取 llama.cpp server 镜像，并由 llama.cpp 从 Hugging Face 加载 `jc-builds/Qwen3.5-4B-Q4_K_M-GGUF:Q4_K_M`。模型文件约 2.6GB，实际占用会落在 `models/.cache` 下。

当前桌面 Docker 环境使用 CPU 推理，速度明显慢于目标 Jetson Orin NX GPU/CUDA 环境。真实设备上应换成 Jetson/CUDA 版 llama.cpp 或 TensorRT-LLM 后端。

查看状态：

```bash
docker compose ps
curl http://127.0.0.1:8088/health
curl http://127.0.0.1:8088/api/v1/device/status
curl http://127.0.0.1:8089/health
```

查看日志：

```bash
docker compose logs -f cerebellum
```

当前模拟系统没有开启业务认证，端口默认只绑定本机。若要开放给手机或局域网设备访问，应增加认证、证书、反向代理访问控制或专用隔离网络。

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

调用 Qwen3.5-4B 报告生成：

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

真实部署时，将量化后的 Qwen3.5-4B 模型或 llama.cpp 下载缓存挂载到：

```text
/opt/cerebellum/models
```

当前版本已经通过 llama.cpp server 接入真实 Qwen3.5-4B Q4_K_M GGUF。若 llama.cpp 服务不可用，`report-service` 会自动回落到模拟文本，并在审计日志中记录 `llm.report.fallback`。

可用以下接口确认模型是否加载成功：

```bash
curl http://127.0.0.1:8089/v1/models
```

报告接口返回中的 `backend` 为 `llama.cpp` 时，表示已经使用真实本地 Qwen3.5-4B 模型；如果为 `simulated-fallback`，表示模型服务不可用或请求超时，系统回落到了模拟报告。
