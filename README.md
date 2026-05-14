# PatrolLink 单兵边缘智能小脑 Docker 模拟系统

该目录用于模拟“单兵边缘智能小脑服务器”的量产运行环境。它不是完整 AI 算法镜像，而是一个贴近目标硬件和系统裁剪策略的可运行原型，用于验证设备状态、视频接入、车牌/人脸候选、报告生成、审计日志和系统加固策略。

当前版本已经接入真实 `Qwen3.5-4B Q4_K_M GGUF` 本地模型，通过 llama.cpp server 提供 OpenAI 兼容接口。车牌识别和人脸检测/特征提取也已经接入真实开源算法；视频流接入已经支持 RTSP/HTTP/本地视频文件的后台抽帧分析。视频摘要、ASR、目标检测、证据保护和后台同步已经具备 MVP API，其中 ASR 默认使用本地 `sherpa-onnx + SenseVoice int8`，并支持外部 ASR 服务和模拟回退，目标检测支持可选 Ultralytics YOLO 并提供模拟回退。

车牌识别已经接入真实 `HyperLPR3`，用于中文车牌检测与识别。人脸识别已经接入 OpenCV Zoo 的 `YuNet + SFace`，用于人脸检测、特征提取和本地特征库候选比对。

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
| 车牌识别 | HyperLPR3 |
| 人脸检测 | OpenCV Zoo YuNet |
| 人脸特征 | OpenCV Zoo SFace |
| 视频流接入 | RTSP/HTTP/本地视频文件 |
| 默认抽帧分析 | 1 FPS，最高 5 FPS |
| 默认并发流 | 2 路 |
| 视频摘要 | 结构化时间线摘要，支持可选 Qwen 润色 |
| 语音转写 | 本地 sherpa-onnx SenseVoice int8 默认，外部 ASR 服务可选 |
| 目标检测 | Ultralytics YOLO 可选扩展，未安装/未配置时模拟回退 |
| 证据保护 | SHA-256 清单，默认 Fernet 加密存储 |
| 后台同步 | 本地任务队列，支持 HTTP POST 推送 |

## 视觉算法来源

| 能力 | 项目 | 说明 |
|---|---|---|
| 中文车牌识别 | [HyperLPR3](https://github.com/szad670401/HyperLPR) | 面向中文车牌的开源识别方案，当前小脑用作车牌检测和 OCR |
| 人脸检测 | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 轻量人脸检测模型，适合边缘部署 |
| 人脸特征 | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | 人脸特征提取模型，用于本地库候选比对 |

说明：InsightFace 识别效果也很强，但其预训练模型授权对商业/警务产品化存在额外合规风险，所以当前默认没有采用 InsightFace。后续如果拿到授权，可以替换为 InsightFace/SCRFD/ArcFace 方案。

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

可选 API Key 鉴权：

```bash
export CEREBELLUM_API_KEY='change-this-key'
docker compose up --build -d cerebellum
curl http://127.0.0.1:8088/api/v1/device/status -H 'X-API-Key: change-this-key'
```

设置 `CEREBELLUM_API_KEY` 后，所有 `/api/v1/*` 接口都必须带 `X-API-Key`。`/health` 保持无鉴权，方便本机健康检查。

## 接口示例

模拟媒体接入：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/media/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"bodycam-rtsp-01","media_type":"stream","duration_seconds":60,"note":"巡逻视频流"}'
```

功能识别/指令分类：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/functions/recognize \
  -H 'Content-Type: application/json' \
  -d '{"text":"帮我分析 sample-test 的视频摘要并生成日报","media_type":"video","context":{"mission_id":"mission-20260514-001","stream_id":"sample-test"}}'
```

该接口用于手机端或语音文本的近端任务路由，默认采用离线关键词规则，返回推荐功能、接口路径、置信度、缺失参数和建议请求载荷。它不依赖大模型，避免实时巡逻阶段因为本地 LLM 慢速推理阻塞操作。

注册实时视频流或本地视频文件抽帧分析：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/streams \
  -H 'Content-Type: application/json' \
  -d '{"stream_id":"bodycam-01-main","source_uri":"rtsp://192.168.50.10/live/main","camera_id":"bodycam-01","sample_fps":1,"analyze_plate":true,"analyze_face":true}'
```

使用本地样本视频测试时，把视频放入 `samples` 目录，然后传文件名：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/streams \
  -H 'Content-Type: application/json' \
  -d '{"stream_id":"sample-test","source_uri":"patrol.mp4","camera_id":"bodycam-01","sample_fps":1,"max_analyzed_frames":5,"analyze_object":true}'
```

查询和停止视频流：

```bash
curl http://127.0.0.1:8088/api/v1/streams
curl http://127.0.0.1:8088/api/v1/streams/sample-test
curl -X POST http://127.0.0.1:8088/api/v1/streams/sample-test/stop
```

说明：当前流分析采用后台线程读取视频，按 `sample_fps` 抽样保存帧，再复用车牌识别和人脸识别接口。它适合验证单兵单路视频的边缘分析链路；真实设备上应把 RTSP 解码切到硬件编解码，并将抽帧、检测、特征比对拆成独立队列。

生成视频结构化摘要：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/video/summary \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","stream_id":"sample-test","operator_note":"测试视频摘要","event_limit":100}'
```

默认返回结构化时间线摘要，速度快。需要调用本地 Qwen3.5-4B 润色时传 `use_llm:true`：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/video/summary \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","stream_id":"sample-test","use_llm":true,"max_tokens":600}'
```

模拟车牌识别：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/plate \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"frame-20260514-001","camera_id":"bodycam-01"}'
```

真实车牌识别需要把图片放到 `samples` 目录，并传入 `image_uri`：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/plate \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"plate-real-001","camera_id":"bodycam-01","image_uri":"your-plate-image.jpg"}'
```

返回中的 `backend` 为 `hyperlpr3` 时，表示已走真实车牌识别算法。

模拟人脸候选提示：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/face \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"frame-20260514-002","camera_id":"bodycam-01"}'
```

真实人脸检测和特征提取：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/face \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"face-real-001","camera_id":"bodycam-01","image_uri":"your-face-image.jpg"}'
```

返回中的 `backend` 为 `opencv-zoo-yunet+sface` 时，表示已走真实人脸检测和特征提取。

本地人脸库登记：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/face/enroll \
  -H 'Content-Type: application/json' \
  -d '{"person_id":"person-0001","display_name":"测试人员","image_uri":"your-face-image.jpg"}'
```

查看和同步 PLBackend 下发的人脸库：

```bash
curl http://127.0.0.1:8088/api/v1/face/library/status

curl -X POST http://127.0.0.1:8088/api/v1/face/library/apply \
  -H 'Content-Type: application/json' \
  -d '{"version":"face-lib-demo-001","source":"PLBackend","full_snapshot":true,"persons":[{"person_id":"CP-001","display_name":"李某某","risk_level":"HIGH","status":"ENABLED","embedding":[0.1,0.2,0.3]}]}'

curl -X POST http://127.0.0.1:8088/api/v1/face/library/sync \
  -H 'Content-Type: application/json' \
  -d '{"backend_url":"http://127.0.0.1:8080","device_id":"PL-CB-SIM-0001","force":true}'
```

人脸库同步采用小脑主动拉取模式。PLBackend 返回版本包，小脑校验并后台构建本地特征库，构建完成后原子切换；失败时保留上一版可用库。版本包中如果带 `embedding`、`image_base64`、小脑可访问的本地图片或 `image_url`，会立即进入比对库；`image_url` 支持绝对地址，也支持在配置 `CEREBELLUM_BACKEND_BASE_URL` 后使用相对地址，带 `image_sha256` 时会校验照片完整性。无法下载或无法提取特征的人员会记录为 pending，等待下一轮同步重试或人工处理。

配置 `CEREBELLUM_BACKEND_BASE_URL` 后，小脑启动会自动同步人脸库，并按 `CEREBELLUM_FACE_LIBRARY_SYNC_INTERVAL_SECONDS` 周期重试，默认 300 秒。实时流命中采用多帧确认，默认 `CEREBELLUM_FACE_MATCH_CONFIRM_FRAMES=3`、`CEREBELLUM_FACE_MATCH_WINDOW_SECONDS=8`，满足条件后产生 `stream_face_alert` 候选告警事件，仍需人工确认。

如果配置了 `CEREBELLUM_BACKEND_TOKEN`，小脑同步人脸库、下载后端照片和上报 `stream_face_alert` 时都会携带 `Authorization: Bearer ...`。多帧确认告警会回传到 `POST /api/v1/cerebellum/face-alerts`，由 PLBackend 生成布控预警并推送到 Web。

目标检测：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/analyze/object \
  -H 'Content-Type: application/json' \
  -d '{"frame_id":"object-real-001","camera_id":"bodycam-01","image_uri":"your-scene-image.jpg","target_classes":["person","car","motorcycle"]}'
```

返回中的 `backend` 为 `ultralytics-yolo` 时，表示已走 YOLO；为 `simulated-fallback` 时表示真实模型不可用，系统回落为确定性模拟候选。当前默认镜像不安装 `ultralytics`，避免把完整 PyTorch/CUDA 依赖拉进 slim 原型镜像；Jetson 工程镜像应单独安装匹配硬件的 YOLO/TensorRT 运行时。

语音转写：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/asr/transcribe \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","audio_uri":"patrol-audio.wav","operator_note":"现场口述记录"}'
```

ASR 默认策略为：本地 `sherpa-onnx + SenseVoice int8` 优先，外部 ASR 服务其次，最后模拟回退。

本地 SenseVoice int8 方式：

```bash
./tools/asr/setup_sherpa_onnx_sensevoice.sh
export CEREBELLUM_ASR_BACKEND=sherpa-onnx-sensevoice
export CEREBELLUM_ASR_LOCAL_BINARY=/opt/cerebellum/bin/sherpa-onnx-offline
export CEREBELLUM_ASR_MODEL_PATH=/opt/cerebellum/models/asr/sensevoice-int8/model.int8.onnx
export CEREBELLUM_ASR_TOKENS_PATH=/opt/cerebellum/models/asr/sensevoice-int8/tokens.txt
export CEREBELLUM_ASR_THREADS=4
docker compose up --build -d cerebellum
```

当前默认 compose 会把 `tools/asr/bin` 挂载到 `/opt/cerebellum/bin`，把 `models` 挂载到 `/opt/cerebellum/models`。接口会先用 `ffmpeg` 把输入音视频转成 16kHz 单声道 WAV，再调用本地 ASR。

如需接入外部真实 ASR 服务，可设置 `CEREBELLUM_ASR_BASE_URL`，接口会按 OpenAI 兼容 `/audio/transcriptions` 上传音频文件。

登记证据文件：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/evidence \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","file_uri":"patrol-test.mp4","evidence_type":"video","note":"巡逻样本视频"}'
curl http://127.0.0.1:8088/api/v1/evidence
```

默认会计算源文件 SHA-256，并把证据副本加密写入数据卷。首次加密会在数据卷中生成本机证据密钥；量产设备应改为 TPM 或安全芯片托管密钥。

创建和执行同步任务：

```bash
curl -X POST http://127.0.0.1:8088/api/v1/sync/tasks \
  -H 'Content-Type: application/json' \
  -d '{"mission_id":"mission-20260514-001","event_limit":100}'
curl http://127.0.0.1:8088/api/v1/sync/tasks
curl -X POST http://127.0.0.1:8088/api/v1/sync/tasks/sync-your-task-id/run
```

未配置 `destination_url` 或 `CEREBELLUM_SYNC_DESTINATION_URL` 时，任务会停留在 `offline_waiting`，用于弱网缓存验证。

查看证书/双向 TLS 准备状态：

```bash
curl http://127.0.0.1:8088/api/v1/security/certificates
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
