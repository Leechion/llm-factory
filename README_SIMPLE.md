# LLaMA-Factory 精简版

这是 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的简化版本，专注于 **SFT（指令微调）** 和 **DPO（偏好对齐）**，适合新手快速上手。

精简版移除了 80% 的代码（v1 架构、PPO/RM/PT/KTO、vLLM/SGLang、Ray 分布式、多模态等高级功能），只保留最核心的微调流程。

## 快速开始（3 步）

```bash
# 1. 安装
pip install -e .

# 2. 训练（用内置 demo 数据微调 Qwen3-4B）
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 3. 聊天测试
llamafactory-cli chat saves/qwen3-4b/lora/sft
```

或者一键运行：

```bash
bash quickstart.sh
```

## 目录结构

```
├── src/llamafactory/     # 源代码
│   ├── cli.py            # 命令行入口
│   ├── launcher.py       # 启动器
│   ├── data/             # 数据处理（模板、加载器、格式转换）
│   ├── model/            # 模型加载和补丁
│   ├── train/            # 训练器（仅 sft/ 和 dpo/）
│   ├── chat/             # 推理（仅 HuggingFace 引擎）
│   ├── webui/            # LLaMA Board 可视化界面
│   ├── api/              # OpenAI 风格 API 服务
│   └── hparams/          # 训练参数定义
├── data/                 # 内置 demo 数据集
├── examples/             # 训练配置示例
│   ├── train_lora/       # LoRA 微调配置
│   ├── accelerate/       # 加速配置
│   └── deepspeed/        # DeepSpeed 配置
├── quickstart.sh         # 一键启动脚本
└── README_SIMPLE.md      # 本文件
```

## 训练你的数据

1. 准备 JSON 数据文件放在 `data/` 目录下：

```json
[
  {"instruction": "你是谁？", "output": "我是 LLaMA-Factory 训练的助手。"},
  {"instruction": "1+1=？", "output": "1+1=2"}
]
```

2. 在 `data/dataset_info.json` 末尾添加注册信息：

```json
"my_data": {"file_name": "my_data.json"}
```

3. 修改训练配置中的 `dataset` 和 `template` 即可。

## 支持的训练方式

| 方式 | 说明 | 显存需求 |
|------|------|----------|
| LoRA SFT | 低秩适应指令微调（推荐） | 8GB+ |
| Full SFT | 全参数指令微调 | 16GB+ |
| LoRA DPO | LoRA 偏好对齐 | 12GB+ |

## WebUI

```bash
llamafactory-cli webui
```

浏览器打开 http://localhost:7860，在界面上配置和启动训练。

## 更多文档

- [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/)
- [入门教程](https://zhuanlan.zhihu.com/p/695287607)
- [B站视频教程](https://www.bilibili.com/video/BV1djgRzxEts/)

## 完整版

需要更多功能（PPO、KTO、预训练、vLLM 推理、多模态等）请使用完整版：
https://github.com/hiyouga/LLaMA-Factory
