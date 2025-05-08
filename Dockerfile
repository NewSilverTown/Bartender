# 使用官方Python基础镜像（可根据需求调整版本）
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（根据你的实际需求调整）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件（建议先复制requirements.txt利用Docker缓存）
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目（通过.dockerignore过滤不需要的文件）
COPY . .

# 暴露API端口（根据你的实际端口修改）
EXPOSE 5000

# 设置启动命令（根据你的入口文件调整）
CMD ["python", "./app/main.py"]