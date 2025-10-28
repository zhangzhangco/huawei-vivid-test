# HDR色调映射专利可视化工具 - 部署指南

## 目录
1. [系统要求](#系统要求)
2. [环境配置](#环境配置)
3. [安装部署](#安装部署)
4. [配置选项](#配置选项)
5. [运行模式](#运行模式)
6. [监控维护](#监控维护)
7. [故障排除](#故障排除)

## 系统要求

### 最低配置
- **操作系统**: Windows 10+ / macOS 10.14+ / Linux (Ubuntu 18.04+)
- **Python**: 3.8或更高版本
- **内存**: 4GB RAM
- **存储**: 1GB可用空间
- **网络**: HTTP/HTTPS访问能力

### 推荐配置
- **操作系统**: Windows 11 / macOS 12+ / Linux (Ubuntu 20.04+)
- **Python**: 3.9或更高版本
- **内存**: 8GB RAM或更多
- **存储**: 2GB可用空间（包含示例数据）
- **CPU**: 支持AVX指令集
- **网络**: 千兆网络连接

### 性能优化配置
- **内存**: 16GB RAM或更多
- **CPU**: 多核处理器（8核心+）
- **存储**: SSD固态硬盘
- **GPU**: 支持CUDA的NVIDIA显卡（可选）

## 环境配置

### Python环境设置

#### 使用Conda（推荐）
```bash
# 创建虚拟环境
conda create -n hdr-tone-mapping python=3.9
conda activate hdr-tone-mapping

# 安装基础依赖
conda install numpy scipy matplotlib
conda install -c conda-forge gradio
```

#### 使用pip
```bash
# 创建虚拟环境
python -m venv hdr-tone-mapping-env

# 激活环境
# Windows:
hdr-tone-mapping-env\Scripts\activate
# macOS/Linux:
source hdr-tone-mapping-env/bin/activate

# 升级pip
python -m pip install --upgrade pip
```

### 依赖包安装

#### 核心依赖
```bash
pip install -r requirements.txt
```

#### requirements.txt内容
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
gradio>=3.0.0
Pillow>=8.3.0
opencv-python>=4.5.0
imageio>=2.9.0
imageio-ffmpeg>=0.4.0
psutil>=5.8.0
dataclasses-json>=0.5.0
```

#### 可选依赖（性能优化）
```bash
# GPU加速支持
pip install cupy-cuda11x  # 根据CUDA版本选择

# 高性能数值计算
pip install numba>=0.56.0

# 高级图像格式支持
pip install OpenEXR>=1.3.0
pip install imageio[ffmpeg]
```

## 安装部署

### 1. 获取源代码

#### 从Git仓库克隆
```bash
git clone <repository-url>
cd hdr-tone-mapping-gradio
```

#### 从发布包安装
```bash
# 下载发布包
wget <release-package-url>
unzip hdr-tone-mapping-gradio-v1.0.zip
cd hdr-tone-mapping-gradio-v1.0
```

### 2. 验证安装

#### 运行测试套件
```bash
# 运行核心功能测试
python -m pytest tests/test_core.py -v

# 运行集成测试
python tests/test_integration_final.py

# 运行完整验证测试
python tests/run_validation_tests.py --test-type all
```

#### 验证依赖
```bash
# 检查Python版本
python --version

# 检查关键依赖
python -c "import numpy, scipy, matplotlib, gradio; print('所有依赖正常')"

# 检查可选依赖
python -c "
try:
    import cupy
    print('GPU加速: 可用')
except ImportError:
    print('GPU加速: 不可用')

try:
    import numba
    print('Numba加速: 可用')
except ImportError:
    print('Numba加速: 不可用')
"
```

### 3. 初始化配置

#### 创建配置目录
```bash
mkdir -p .kiro_state
mkdir -p logs
mkdir -p exports
mkdir -p temp
```

#### 设置权限（Linux/macOS）
```bash
chmod 755 src/
chmod 644 src/*.py
chmod 755 tests/
chmod 644 tests/*.py
```

## 配置选项

### 环境变量配置

创建 `.env` 文件：
```bash
# 应用配置
APP_HOST=0.0.0.0
APP_PORT=7860
APP_DEBUG=false

# 性能配置
MAX_IMAGE_SIZE=4194304  # 4MP
AUTO_DOWNSAMPLE=true
ENABLE_GPU_ACCELERATION=auto

# 存储配置
STATE_DIR=.kiro_state
LOG_DIR=logs
EXPORT_DIR=exports
TEMP_DIR=temp

# 安全配置
ENABLE_FILE_UPLOAD=true
MAX_UPLOAD_SIZE=100MB
ALLOWED_EXTENSIONS=.exr,.hdr,.png,.jpg,.jpeg,.tiff

# 日志配置
LOG_LEVEL=INFO
LOG_ROTATION=daily
LOG_RETENTION=30
```

### 应用配置文件

创建 `config/app_config.json`：
```json
{
  "application": {
    "name": "HDR色调映射专利可视化工具",
    "version": "1.0.0",
    "debug": false
  },
  "server": {
    "host": "0.0.0.0",
    "port": 7860,
    "share": false,
    "auth": null
  },
  "processing": {
    "max_image_size": 4194304,
    "auto_downsample": true,
    "downsample_threshold": 2097152,
    "curve_samples": 512,
    "enable_gpu": "auto"
  },
  "storage": {
    "state_dir": ".kiro_state",
    "auto_save": true,
    "save_interval": 30,
    "max_history": 100
  },
  "export": {
    "default_lut_samples": 1024,
    "csv_precision": 6,
    "include_metadata": true
  },
  "ui": {
    "theme": "soft",
    "show_advanced": false,
    "enable_shortcuts": true
  }
}
```

### 日志配置

创建 `config/logging_config.json`：
```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    },
    "detailed": {
      "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "standard",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "logs/app.log",
      "maxBytes": 10485760,
      "backupCount": 5
    }
  },
  "loggers": {
    "core": {
      "level": "DEBUG",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "gradio": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console", "file"]
  }
}
```

## 运行模式

### 1. 开发模式

```bash
# 启动开发服务器
python src/gradio_app.py

# 或使用调试模式
python -m pdb src/gradio_app.py
```

特点：
- 自动重载代码变更
- 详细错误信息
- 调试工具可用
- 本地访问限制

### 2. 生产模式

#### 直接运行
```bash
# 设置生产环境变量
export APP_DEBUG=false
export LOG_LEVEL=WARNING

# 启动应用
python src/gradio_app.py
```

#### 使用Gunicorn（推荐）
```bash
# 安装Gunicorn
pip install gunicorn

# 创建WSGI入口文件 wsgi.py
cat > wsgi.py << 'EOF'
from src.gradio_app import create_app
app = create_app()
EOF

# 启动Gunicorn服务器
gunicorn --bind 0.0.0.0:7860 --workers 4 --timeout 300 wsgi:app
```

#### 使用Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p .kiro_state logs exports temp

# 设置权限
RUN chmod -R 755 src/ tests/

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "src/gradio_app.py"]
```

构建和运行Docker容器：
```bash
# 构建镜像
docker build -t hdr-tone-mapping:latest .

# 运行容器
docker run -d \
  --name hdr-tone-mapping \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  hdr-tone-mapping:latest
```

### 3. 集群部署

#### 使用Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  hdr-tone-mapping:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./exports:/app/exports
      - ./logs:/app/logs
    environment:
      - APP_DEBUG=false
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - hdr-tone-mapping
    restart: unless-stopped
```

启动集群：
```bash
docker-compose up -d
```

## 监控维护

### 1. 健康检查

创建健康检查脚本 `scripts/health_check.py`：
```python
#!/usr/bin/env python3
import requests
import sys
import json

def check_health():
    try:
        # 检查应用响应
        response = requests.get('http://localhost:7860', timeout=10)
        if response.status_code != 200:
            print(f"应用响应异常: {response.status_code}")
            return False
            
        # 检查API端点
        api_response = requests.post(
            'http://localhost:7860/api/predict',
            json={"data": [2.0, 0.5]},
            timeout=30
        )
        
        if api_response.status_code != 200:
            print(f"API响应异常: {api_response.status_code}")
            return False
            
        print("健康检查通过")
        return True
        
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
```

### 2. 性能监控

创建监控脚本 `scripts/monitor.py`：
```python
#!/usr/bin/env python3
import psutil
import time
import json
import logging

def monitor_system():
    """监控系统资源使用"""
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_gb = memory.used / (1024**3)
    
    # 磁盘使用
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    disk_free_gb = disk.free / (1024**3)
    
    # 网络IO
    net_io = psutil.net_io_counters()
    
    metrics = {
        'timestamp': time.time(),
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_gb': round(memory_used_gb, 2),
        'disk_percent': disk_percent,
        'disk_free_gb': round(disk_free_gb, 2),
        'network_bytes_sent': net_io.bytes_sent,
        'network_bytes_recv': net_io.bytes_recv
    }
    
    return metrics

def main():
    logging.basicConfig(
        filename='logs/monitor.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    while True:
        try:
            metrics = monitor_system()
            logging.info(json.dumps(metrics))
            
            # 检查告警条件
            if metrics['cpu_percent'] > 80:
                logging.warning(f"CPU使用率过高: {metrics['cpu_percent']}%")
                
            if metrics['memory_percent'] > 85:
                logging.warning(f"内存使用率过高: {metrics['memory_percent']}%")
                
            if metrics['disk_percent'] > 90:
                logging.warning(f"磁盘使用率过高: {metrics['disk_percent']}%")
                
        except Exception as e:
            logging.error(f"监控异常: {e}")
            
        time.sleep(60)  # 每分钟监控一次

if __name__ == "__main__":
    main()
```

### 3. 日志轮转

创建日志轮转配置 `/etc/logrotate.d/hdr-tone-mapping`：
```
/path/to/hdr-tone-mapping/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 user group
    postrotate
        # 重启应用以重新打开日志文件
        systemctl reload hdr-tone-mapping || true
    endscript
}
```

### 4. 自动备份

创建备份脚本 `scripts/backup.sh`：
```bash
#!/bin/bash

BACKUP_DIR="/backup/hdr-tone-mapping"
APP_DIR="/path/to/hdr-tone-mapping"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份状态文件
tar -czf "$BACKUP_DIR/states_$DATE.tar.gz" -C "$APP_DIR" .kiro_state/

# 备份配置文件
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" -C "$APP_DIR" config/

# 备份日志文件
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" -C "$APP_DIR" logs/

# 清理旧备份（保留30天）
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "备份完成: $DATE"
```

## 故障排除

### 常见问题

#### 1. 应用启动失败

**症状**: 应用无法启动或立即退出

**排查步骤**:
```bash
# 检查Python版本
python --version

# 检查依赖安装
pip list | grep -E "(numpy|scipy|matplotlib|gradio)"

# 检查端口占用
netstat -tulpn | grep 7860

# 查看详细错误信息
python src/gradio_app.py --debug
```

**常见原因**:
- Python版本不兼容
- 依赖包缺失或版本不匹配
- 端口被占用
- 权限不足

#### 2. 内存不足

**症状**: 处理大图像时应用崩溃或响应缓慢

**解决方案**:
```bash
# 启用自动降采样
export AUTO_DOWNSAMPLE=true
export MAX_IMAGE_SIZE=2097152  # 2MP

# 增加虚拟内存
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. 性能问题

**症状**: 曲线更新或图像处理缓慢

**优化措施**:
```bash
# 安装性能优化包
pip install numba cupy-cuda11x

# 调整配置
export CURVE_SAMPLES=256  # 降低采样密度
export ENABLE_GPU_ACCELERATION=true

# 检查CPU使用
top -p $(pgrep -f gradio_app.py)
```

#### 4. 网络访问问题

**症状**: 无法通过浏览器访问应用

**排查步骤**:
```bash
# 检查应用是否运行
ps aux | grep gradio_app.py

# 检查端口监听
netstat -tulpn | grep 7860

# 检查防火墙设置
sudo ufw status
sudo iptables -L

# 测试本地访问
curl http://localhost:7860
```

### 日志分析

#### 查看应用日志
```bash
# 实时查看日志
tail -f logs/app.log

# 搜索错误信息
grep -i error logs/app.log

# 分析性能日志
grep -i "processing time" logs/app.log | tail -20
```

#### 系统日志
```bash
# 查看系统日志
journalctl -u hdr-tone-mapping -f

# 查看内存相关错误
dmesg | grep -i "out of memory"

# 查看磁盘空间
df -h
```

### 性能调优

#### CPU优化
```bash
# 设置CPU亲和性
taskset -c 0-3 python src/gradio_app.py

# 调整进程优先级
nice -n -10 python src/gradio_app.py
```

#### 内存优化
```bash
# 设置内存限制
ulimit -v 8388608  # 8GB虚拟内存限制

# 启用内存映射
export MMAP_ENABLED=true
```

#### 网络优化
```bash
# 调整TCP参数
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

---

**版本**: 1.0  
**更新日期**: 2025-10-27  
**维护者**: HDR Tone Mapping Team