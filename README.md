# OmniTalker

一个能听懂、看懂、理解并做出对应动作的全能数字人。

## 功能概览

- 基于 Emage 音频驱动模型生成 3D 数字人动作与表情。
- 支持 Qwen3-Omni 多模态 API（摄像头 + 麦克风 + 文本）驱动对话。
- 提供 Gradio 交互界面：实时采集输入并生成 3D 数字人视频。

## 环境准备

1. 安装依赖（示例）：

```bash
pip install -r requirements.txt
```

2. 配置 Qwen3-Omni API（必要）：

```bash
export QWEN_OMNI_API_KEY="你的APIKey"
export QWEN_OMNI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_OMNI_MODEL="qwen3-omni"
```

3. 确保模型资源已准备好：

- Emage 模型：`/workspace/data/models/emage_audio`
- SMPLX 模型：`./emage_evaltools/smplx_models/`

## 使用方式

### 1) 启动 Gradio 端到端数字人

```bash
python test_emage_audio.py --gradio --host 0.0.0.0 --port 7860
```

打开浏览器访问 `http://localhost:7860`，允许摄像头和麦克风权限。

### 2) 批处理音频（原始模式）

```bash
python test_emage_audio.py --audio_folder ./examples/audio --save_folder ./examples/motion --visualization
```

## GitHub 上传流程（命令行）

1. 初始化并提交：

```bash
git init

git add .

git commit -m "Initial commit"
```

2. 关联远程仓库并推送：

```bash
git remote add origin https://github.com/<your_org>/<your_repo>.git

git branch -M main

git push -u origin main
```

> 如果你已经有仓库，只需要 `git remote add` 和 `git push` 即可。

## 常见问题

- **Gradio 无法获取摄像头/麦克风**：确保浏览器已授权权限，并使用 HTTPS 或 `localhost`。
- **渲染失败**：确认 `emage_evaltools/smplx_models/` 存在且模型文件完整。
- **API 失败**：检查 `QWEN_OMNI_API_KEY` 是否正确设置。
