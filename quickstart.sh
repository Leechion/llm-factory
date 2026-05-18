#!/bin/bash
# LLaMA-Factory 精简版 快速启动脚本
# 使用 Qwen3-4B + LoRA SFT 在 alpaca demo 数据上微调

set -e

echo "========================================="
echo " LLaMA-Factory 精简版 快速上手"
echo "========================================="

# Step 1: 安装依赖
echo ""
echo "[1/3] 安装依赖..."
pip install -e . 2>/dev/null || echo "依赖可能已安装，跳过..."

# Step 2: 开始训练
echo ""
echo "[2/3] 开始 LoRA SFT 训练..."
echo "  模型: Qwen/Qwen3-4B-Instruct-2507"
echo "  数据: identity + alpaca_en_demo"
echo "  方法: LoRA (rank=8)"
echo ""

llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# Step 3: 训练完成
echo ""
echo "[3/3] 训练完成!"
echo "  模型保存在: saves/qwen3-4b/lora/sft"
echo ""
echo "运行对话测试:"
echo "  llamafactory-cli chat saves/qwen3-4b/lora/sft"
echo ""
echo "启动 WebUI:"
echo "  llamafactory-cli webui"
