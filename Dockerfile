# 第一阶段：构建阶段
FROM python:3.13-slim as builder

WORKDIR /app

# 仅安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件
COPY requirements.txt .

# 安装依赖到虚拟环境
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# 第二阶段：运行时阶段
FROM python:3.9-slim

WORKDIR /app

# 仅从构建阶段复制必要内容
COPY --from=builder /opt/venv /opt/venv

# 只复制项目必要文件
COPY barcounter/api/ ./barcounter/api/
COPY barcounter/utils/ ./barcounter/utils/
COPY barcounter/models/ ./barcounter/models/
COPY barcounter/checkpoints/ ./barcounter/checkpoints/

# 设置环境变量
ENV PATH="/opt/venv/bin:$PATH"

# 清理apt缓存
RUN apt-get update && apt-get install -y \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["python", "main.py"]