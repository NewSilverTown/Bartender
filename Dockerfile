# 第一阶段：构建阶段 - 安装依赖
FROM python:3.9-slim AS builder

WORKDIR /app

# 1. 修复APT源配置问题
RUN echo "deb http://mirrors.aliyun.com/debian/ bullseye main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bullseye-security main" >> /etc/apt/sources.list

# 2. 安装系统依赖（先更新源）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 配置PIP清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 分层安装PyTorch CPU版本
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

# 第二阶段：运行时阶段 - 最小镜像
FROM python:3.9-slim

WORKDIR /app

# 4. 仅复制必要内容
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 5. 精准复制代码
COPY barcounter/ ./barcounter/

# 6. 最小运行时依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 7. 设置Python环境变量
ENV PYTHONPATH=/app

EXPOSE 8000

# 8. 修正启动命令（使用完整模块路径）
CMD ["python", "-m", "barcounter.api.main"]