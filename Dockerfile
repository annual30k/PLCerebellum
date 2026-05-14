FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    CEREBELLUM_CONFIG=/etc/cerebellum/device.yaml \
    CEREBELLUM_DATA_DIR=/var/lib/cerebellum \
    CEREBELLUM_LOG_DIR=/var/log/cerebellum \
    CEREBELLUM_MODEL_DIR=/opt/cerebellum/models

WORKDIR /opt/cerebellum

# 精简量产镜像只保留运行期必要组件：
# - tini: 正确处理容器内 PID 1 信号
# - curl: 健康检查
# - ffmpeg: 模拟视频抽帧/转码运行依赖，真实硬件版可替换为硬件编解码运行时
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl ffmpeg ca-certificates \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && groupadd --system cerebellum \
    && useradd --system --gid cerebellum --home-dir /nonexistent --shell /usr/sbin/nologin cerebellum \
    && mkdir -p /etc/cerebellum /var/lib/cerebellum /var/log/cerebellum /opt/cerebellum/models \
    && chown -R cerebellum:cerebellum /var/lib/cerebellum /var/log/cerebellum /opt/cerebellum/models \
    && rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/apt-cache /usr/bin/dpkg /usr/bin/dpkg-query

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY app ./app
COPY config/device.yaml /etc/cerebellum/device.yaml

USER cerebellum

EXPOSE 8088

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8088/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "1"]
