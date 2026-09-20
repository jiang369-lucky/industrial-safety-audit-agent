# 基于动态路由与 SFT-VLM 的工业安全智能审核系统

这是放在 `agent` 目录下的独立实习项目版代码。它不会修改你原来的论文实验代码，而是把已有 YOLO、Qwen-LoRA、视频审核经验包装成一个更适合 Agent 岗位展示的工程：视频/图片输入后，系统完成视觉检测、动态路由、VLM 复核、规范检索和结构化审核报告生成。

## 项目定位

原始论文工程更偏实验验证：场景 MoE、焊接/切割检测、Qwen 复核、消融实验和论文图表。

本目录的项目更偏工程实践：把上述能力封装成可调用的 Agent 工作流，重点体现：

- **动态路由**：高置信样本直接生成审核结论，中置信样本调用 VLM，低置信或跳变样本进入多帧时序校验，无法判断时进入人工兜底。
- **多工具编排**：YOLO 检测工具、SFT-VLM 复核工具、RAG 规范检索工具、结构化报告工具均以 JSON Schema 描述，可用于 MCP 化接入。
- **RAG 合规审核**：通过 BM25 检索安全规范片段，把视觉结果转化为有依据的审核意见。
- **可演示服务**：提供 FastAPI 接口、命令行入口和可选 MCP Server。

## 和你简历描述的对应关系

建议简历中把模型写得更稳一点：

> Qwen-VL 系列视觉语言模型（支持 Qwen2.5-VL-3B / Qwen3-VL 本地 LoRA 适配）

原因是你原始工程里当前可见资源主要是 `Qwen3-VL-4B-Instruct` 和对应 LoRA。如果你后续真的重新用 Qwen2.5-VL-3B 做了 SFT/LoRA，再把简历固定写成 Qwen2.5-VL-3B 会更严谨。

## 目录结构

```text
agent/
  README.md
  requirements-agent.txt
  .env.example
  pyproject.toml
  src/industrial_audit_agent/
    api.py                 FastAPI 服务入口
    cli.py                 命令行演示入口
    config.py              路径、阈值、模型后端配置
    workflow.py            LangGraph/异步备用工作流
    schemas.py             Pydantic 数据结构与报告格式
    report.py              JSON 与 Markdown 审核报告生成
    mcp_server.py          可选 MCP Server 入口
    tools/
      vision_detection.py  YOLO/模拟视觉检测工具
      vlm_review.py        Qwen-VL/模拟 VLM 复核工具
      rag_retriever.py     BM25/可扩展 Milvus 规范检索工具
      temporal.py          多帧时序校验工具
      mcp_tools.py         MCP 风格工具清单与 JSON Schema
    resources/
      safety_policy.md     示例安全规范库
      report_schema.json   结构化报告 JSON Schema
  tests/
    test_routing.py
```

## 环境配置

推荐 Python 3.11。你之前的 4060 + CUDA 环境可以继续用，但这个 Agent 项目也支持 mock 模式，因此不加载大模型也能跑通流程。

在 cmd 中执行：

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
py -3.11 -m venv agent\agent_env
agent\agent_env\Scripts\activate
python -m pip install --upgrade pip
pip install -r agent\requirements-agent.txt
pip install -e agent
```

如果你只是想先演示 API，不想装大模型相关包，可以先装精简依赖：

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
agent\agent_env\Scripts\activate
pip install fastapi uvicorn pydantic numpy opencv-python
pip install -e agent
```

## 运行命令

### 1. 命令行审核单张图片

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
agent\agent_env\Scripts\activate
python -m industrial_audit_agent.cli image --image "industrial_cctv_training\dataset\dataset_cot\weld_samples\dht004.jpg" --scene welding
```

运行后会在终端输出结构化 JSON 和 Markdown 风格审核意见。若没有加载真实 YOLO，会使用可复现的模拟检测结果，方便你先讲清 Agent 流程。

### 2. 命令行审核视频抽帧

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
agent\agent_env\Scripts\activate
python -m industrial_audit_agent.cli video --video "dual_moe_enhanced\test_video\clean\welding\RWL_VID_00007.mp4" --scene welding --stride 80 --max-frames 6
```

系统会把抽帧临时保存到 `agent\runtime\frames`，逐帧进入 Agent 审核流程，并输出视频级风险摘要。

### 3. 启动 FastAPI 服务

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
agent\agent_env\Scripts\activate
uvicorn industrial_audit_agent.api:app --host 127.0.0.1 --port 8008 --reload
```

浏览器打开：

```text
http://127.0.0.1:8008/docs
```

### 4. 调用 API

cmd 示例：

```bat
curl -X POST "http://127.0.0.1:8008/audit/image" -H "Content-Type: application/json" -d "{\"image_path\":\"industrial_cctv_training\\dataset\\dataset_cot\\weld_samples\\dht004.jpg\",\"scene_hint\":\"welding\"}"
```

### 5. 可选 MCP Server

安装 `mcp` 后可运行：

```bat
cd /d "D:\yolo\Spark-Gated MoE Framework"
agent\agent_env\Scripts\activate
python -m industrial_audit_agent.mcp_server
```

## 真实模型接入

默认 `AGENT_DETECTOR_BACKEND=auto`，当 YOLO 权重存在并安装了 `ultralytics` 时会尝试加载真实模型；否则自动回退到 mock 模式。

默认 `AGENT_VLM_BACKEND=mock`，不会加载 Qwen。若要接入本地 Qwen-VL：

```bat
set AGENT_VLM_BACKEND=qwen
set AGENT_VLM_BASE_MODEL_PATH=D:\yolo\Spark-Gated MoE Framework\model\Qwen2.5-VL-3B-Instruct
set AGENT_VLM_LORA_PATH=D:\yolo\Spark-Gated MoE Framework\LlamaFactory-main\saves\qwen3-vl-weld-3epoch
```

如果你实际使用的是 Qwen3-VL，就把 `AGENT_VLM_BASE_MODEL_PATH` 指向 Qwen3 的本地路径，并在简历中写成 Qwen-VL 系列或 Qwen3-VL，避免被面试官追问时口径不一致。

## 面试讲法

你可以这样介绍：

> 我把原来的工业安全视觉检测系统改造成了一个 Agent 化审核系统。YOLO 负责低成本感知，LangGraph 负责任务编排和动态路由，VLM 只在不确定样本上复核，RAG 用于检索安全规范并给出审核依据，最后通过 JSON Schema 输出结构化报告。这样既保留实时检测能力，又能把大模型用于更有价值的模糊场景。

这个项目的重点不是“所有模型都自己训练到 SOTA”，而是展示你能把视觉模型、大模型、工具调用、RAG 和工程接口组织成一个可落地系统。
