# LLaMA-Factory Simple 技术文档

> 版本: 精简版 (SFT + DPO only) | 基于 LLaMA-Factory | Apache 2.0

---

## 目录

1. [架构总览](#1-架构总览)
2. [入口点 Entry Points](#2-入口点-entry-points)
3. [配置系统 HParams](#3-配置系统-hparams)
4. [数据管线 Data Pipeline](#4-数据管线-data-pipeline)
5. [模型加载 Model Loading](#5-模型加载-model-loading)
6. [训练系统 Training](#6-训练系统-training)
7. [推理引擎 Chat](#7-推理引擎-chat)
8. [WebUI 可视化](#8-webui-可视化)
9. [API 服务](#9-api-服务)
10. [评估系统 Evaluation](#10-评估系统-evaluation)
11. [工具模块 Extras](#11-工具模块-extras)
12. [如何修改：完整指南](#12-如何修改完整指南)

---

## 1. 架构总览

### 调用流程

```
CLI 命令 (llamafactory-cli)
  └─ cli.py:main()
       └─ launcher.py:launch()
            ├─ train ──┬─ 多GPU: torchrun 子进程 → tuner.py:run_exp()
            │          └─ 单GPU: tuner.py:run_exp()
            │                └─ sft/workflow.py:run_sft()  或  dpo/workflow.py:run_dpo()
            ├─ chat  ──→ chat_model.py:run_chat()
            ├─ api   ──→ api/app.py:run_api()
            ├─ export──→ tuner.py:export_model()
            ├─ webui ──→ webui/interface.py:run_web_ui()
            └─ webchat → webui/interface.py:run_web_demo()
```

### 模块依赖

```
hparams ← extras
data ← hparams + extras
model ← hparams + extras
train ← data + model + hparams + extras
api/chat/webui/eval ← train
```

---

## 2. 入口点 Entry Points

### 2.1 `src/llamafactory/cli.py` — CLI 入口

| 函数 | 作用 |
|------|------|
| `main()` | PyPI 安装后的命令行入口，调用 `launcher.launch()`。`if __name__ == "__main__"` 时启用 `multiprocessing.freeze_support()` 以兼容 Windows/Mac 打包。 |

**如何修改**: 如果需要添加新的子命令（如 `llamafactory-cli benchmark`），修改 `launcher.py` 而非此文件。

### 2.2 `src/llamafactory/launcher.py` — 命令调度器

| 函数 | 作用 |
|------|------|
| `launch()` | **核心调度函数**。根据 `sys.argv[1]` 分发到不同子命令：`train`、`chat`、`api`、`export`、`webchat`、`webui`、`env`、`version`、`help`。对于 `train` 命令，自动检测多GPU环境并通过 `torchrun` 启动分布式训练（支持弹性训练 elastic launch）。 |
| `USAGE` | 帮助字符串，显示所有可用命令。 |

**子命令路由**:
| 命令 | 调用 | 说明 |
|------|------|------|
| `train` | `tuner.run_exp()` | 多GPU时用 torchrun 包装本文件自身 |
| `chat` | `chat_model.run_chat()` | CLI 对话 |
| `api` | `app.run_api()` | OpenAI 风格 API |
| `export` | `tuner.export_model()` | 合并 LoRA 导出模型 |
| `webchat` | `interface.run_web_demo()` | 网页对话 |
| `webui` | `interface.run_web_ui()` | LLaMA Board |

**如何修改**: 添加新子命令时，在 `launch()` 中添加 `elif command == "xxx":` 分支，并更新 `USAGE` 字符串。

### 2.3 `src/train.py` — Training Worker

| 函数 | 作用 |
|------|------|
| `main()` | `torchrun` 启动的子进程入口，直接调用 `run_exp()` |
| `_mp_fn(index)` | TPU 的 xla_spawn 兼容入口 |

### 2.4 `src/webui.py` — WebUI 入口

| 函数 | 作用 |
|------|------|
| `main()` | 创建 Gradio Web UI，设置代理，启动 `0.0.0.0:7860` 的 Web 服务 |

### 2.5 `src/api.py` — API 入口

| 函数 | 作用 |
|------|------|
| `main()` | 创建 ChatModel 实例，构建 FastAPI app，启动 `0.0.0.0:8000` 的 HTTP 服务 |

---

## 3. 配置系统 HParams

### 3.1 `hparams/parser.py` — 参数解析器（核心配置门面）

| 函数 | 作用 |
|------|------|
| `read_args(args)` | 从命令行或 YAML/JSON 配置文件读取参数。如果 `sys.argv[1]` 以 `.yaml`/`.yml`/`.json` 结尾，用 OmegaConf 解析配置文件，并允许 CLI 追加覆盖参数。 |
| `_parse_args(parser, args)` | 使用 HuggingFace `HfArgumentParser` 将参数解析为 dataclass 元组 |
| `_parse_train_args(args)` | 解析训练所需的 5 个 dataclass: Model + Data + Training + Finetuning + Generating |
| `_parse_infer_args(args)` | 解析推理所需的 4 个 dataclass: Model + Data + Finetuning + Generating |
| `_parse_eval_args(args)` | 解析评估所需的 4 个 dataclass |
| `get_train_args(args)` | **训练参数获取入口**。解析参数 → 设置日志 → 验证参数合法性 → 后处理（如 ddp_find_unused_parameters 在 LoRA 下自动设为 False）→ 设置 dtype/device_map/model_max_length |
| `get_infer_args(args)` | **推理参数获取入口**。解析参数 → 验证 → 后处理 device_map |
| `get_eval_args(args)` | **评估参数获取入口**。解析参数 → 验证 |
| `_verify_model_args(...)` | 验证模型参数一致性（如 adapter 只适用于 LoRA，OFT可与量化配合等） |
| `_check_extra_dependencies(...)` | 检查可选依赖（如 deepspeed、unsloth、liger-kernel 等） |
| `_verify_trackio_args(...)` | 验证 Trackio 日志记录参数（project 必须指定等） |
| `_set_env_vars()` | 设置环境变量（NPU JIT 编译、VLLM spawn 方式等） |
| `_set_transformers_logging()` | 根据 `LLAMAFACTORY_VERBOSITY` 环境变量设置 transformers 日志级别 |

**如何修改**:
- 添加新的配置字段：在对应的 `*_args.py` dataclass 中添加 `field()`
- 添加新的参数验证：在 `get_train_args()` 的验证区域添加逻辑
- 添加新的训练阶段：在 `_TRAIN_ARGS` 元组中添加新的 args dataclass

### 3.2 `hparams/model_args.py` — 模型参数

| 类 | 作用 |
|------|
| `BaseModelArguments` | 模型基础参数：`model_name_or_path`（模型路径/HF ID）、`adapter_name_or_path`（已训练的 adapter）、`cache_dir`、`resize_vocab`（调整词表大小）、`add_tokens`/`add_special_tokens`（添加新 token）、`flash_attn`、`rope_scaling`、`use_unsloth` 等 |
| `QuantizationArguments` | 量化参数：`quantization_bit`、`quantization_method`（bitsandbytes/awq/gptq 等） |
| `ProcessorArguments` | 多模态处理参数：`image_max_pixels`、`video_fps`、`audio_sampling_rate` 等 |
| `ExportArguments` | 模型导出参数：`export_dir`、`export_size`（分片大小 GB）、`export_hub_model_id` |
| `VllmArguments` | vLLM 引擎参数（精简版基本不启用） |
| `SGLangArguments` | SGLang 引擎参数（精简版基本不启用） |
| `KTransformersArguments` | KTransformers AMX MoE 训练参数：`use_kt`、`kt_weight_path`、`get_kt_config_dict()`、`apply_kt_config()` |
| `ModelArguments` | **聚合 dataclass**：继承以上所有类。`__post_init__()` 调用各父类的初始化。`copyfrom()` 用于创建副本并覆盖字段。`to_dict()` 序列化为字典（token 字段打码）。 |

### 3.3 `hparams/data_args.py` — 数据参数

| 函数/类 | 作用 |
|------|------|
| `DataArguments.__post_init__()` | 将逗号分隔的字符串解析为列表（如 `dataset: "a,b"` → `["a", "b"]`），验证 `val_size` 与 `eval_dataset` 互斥 |
| `DataArguments.to_dict()` | 序列化为字典 |

**关键字段**:
- `template`: 模型提示模板名称（如 `qwen3_nothink`）
- `dataset` / `eval_dataset`: 数据集名称列表
- `cutoff_len`: token 截断长度（默认 2048）
- `train_on_prompt`: 是否在 prompt 上计算 loss（默认只计算 response）
- `packing` / `neat_packing`: 序列打包
- `streaming`: 流式加载大数据集
- `val_size`: 从训练集切分的验证集比例

### 3.4 `hparams/finetuning_args.py` — 微调参数

| 类 | 作用 |
|------|------|
| `FreezeArguments` | 部分参数冻结：`freeze_trainable_layers`（可训练的层数）、`freeze_trainable_modules`（可训练的模块名） |
| `LoraArguments` | LoRA 参数：`lora_rank`(默认8)、`lora_alpha`、`lora_target`(目标模块，默认 all)、`use_dora`、`use_rslora` |
| `RLHFArguments` | DPO 对齐参数：`pref_beta`(0.1)、`pref_loss`(sigmoid/ipo/orpo/simpo)、`ref_model`（参考模型路径） |
| `FinetuningArguments` | **聚合类**：继承以上所有。`stage`(sft/dpo)、`finetuning_type`(lora/oft/freeze/full)、`pure_bf16`、`compute_accuracy`、`plot_loss` |
| `FinetuningArguments.__post_init__()` | 计算 `lora_alpha` 默认值（2 * lora_rank），设置 `use_ref_model` 标记（DPO非orpo/simpo时需要），验证各参数组合合法性 |

### 3.5 `hparams/training_args.py` — 训练参数

| 类 | 作用 |
|------|------|
| `TrainingArguments` | 继承 `transformers.Seq2SeqTrainingArguments`，添加 `overwrite_output_dir`（已废弃，向后兼容）。所有标准训练参数（batch_size、learning_rate、epochs、deepspeed、fp16/bf16等）均由父类提供。 |

### 3.6 `hparams/generating_args.py` — 生成参数

| 函数/类 | 作用 |
|------|------|
| `GeneratingArguments` | 解码参数：`do_sample`(默认True)、`temperature`(0.95)、`top_p`(0.7)、`top_k`(50)、`num_beams`(1)、`max_new_tokens`(1024)、`repetition_penalty`(1.0) |
| `GeneratingArguments.to_dict()` | 转换为字典。如果 `max_new_tokens > 0` 则移除 `max_length`，反之亦然。`obey_generation_config=True` 时只保留 `GenerationConfig` 支持的参数。 |

### 3.7 `hparams/evaluation_args.py` — 评估参数

| 类 | 作用 |
|------|------|
| `EvaluationArguments` | 评估参数：`task`（任务名）、`batch_size`、`n_shot`(默认5)、`lang`(en/zh)、`save_dir` |

---

## 4. 数据管线 Data Pipeline

### 4.1 `data/loader.py` — 数据加载器

这是数据管线的**总调度函数**。

| 函数 | 作用 |
|------|------|
| `get_dataset(template, model_args, data_args, training_args, stage, tokenizer, processor)` | **数据加载主函数**。流程：检查是否有缓存的 tokenized 数据（`tokenized_path`）→ 调用 `_get_merged_dataset()` 加载原始数据 → 调用 `split_dataset()` 切分训练/验证集 → 调用 `_get_preprocessed_dataset()` 进行 tokenization → 返回 `DatasetModule`（包含 train/eval dataset） |
| `_load_single_dataset(dataset_attr, ...)` | 加载单个数据集。支持 5 种数据源：`hf_hub`（HF Hub 直接加载）、`ms_hub`（ModelScope）、`om_hub`（OpenMind）、`script`（本地脚本）、`file`（本地文件 JSON/JSONL/Parquet 等）。自动识别文件类型。支持 `num_samples` 采样和 `max_samples` 截断。 |
| `_get_merged_dataset(dataset_names, ...)` | 遍历数据集名称列表，对每个数据集调用 `_load_single_dataset()` 和 `align_dataset()`，然后用 `merge_dataset()` 合并。 |
| `_get_dataset_processor(data_args, stage, template, tokenizer, processor)` | 根据阶段返回对应的处理器：`sft` → `SupervisedDatasetProcessor` 或 `PackedSupervisedDatasetProcessor`（启用packing时），`rm` → `PairwiseDatasetProcessor`。 |
| `_get_preprocessed_dataset(dataset, ...)` | 对数据集进行 tokenization 预处理。使用 `dataset.map()` 调用 `DatasetProcessor.preprocess_dataset()`，支持多进程并行、缓存。 |

**如何修改**:
- 添加新的数据源类型：在 `_load_single_dataset()` 中添加 `elif dataset_attr.load_from == "xxx":` 分支
- 自定义预处理逻辑：修改对应的 `DatasetProcessor` 类

### 4.2 `data/parser.py` — 数据集注册表解析器

| 函数/类 | 作用 |
|------|------|
| `DatasetAttr` | 数据类，描述单个数据集的属性：`load_from`（数据源类型）、`formatting`（alpaca/sharegpt/openai）、`ranking`（是否偏好数据）、字段映射（`prompt`/`query`/`response`/`chosen`/`rejected` 等） |
| `get_dataset_list(dataset_names, dataset_dir)` | **数据集注册入口**。读取 `data/dataset_info.json`，根据每个数据集名称找到对应的配置（文件路径或 Hub URL），构建 `DatasetAttr` 列表。支持从远程 Hub 拉取 `dataset_info.json`（`dataset_dir = "REMOTE:org/repo"`）。 |

**如何修改**:
- 注册新数据集：在 `data/dataset_info.json` 中添加条目：
```json
"my_dataset": {"file_name": "my_data.json"}
```
- DPO 数据需指定 `"ranking": true, "formatting": "sharegpt"` 及 `chosen`/`rejected` 列名
- 从 HuggingFace 加载：`"hf_hub_url": "username/dataset"`

### 4.3 `data/converter.py` — 格式对齐器

将不同格式的原始数据统一转换为内部标准格式。

| 函数/类 | 作用 |
|------|------|
| `DatasetConverter` | 抽象基类，提供 `_find_medias()` 方法（拼接媒体文件的本地路径） |
| `AlpacaDatasetConverter.__call__()` | 将 Alpaca 格式 `{instruction, input, output, history}` 转换为标准格式 `{_prompt, _response, _system, _tools, _images, _videos, _audios}` |
| `SharegptDatasetConverter.__call__()` | 将 ShareGPT 格式（带 role/content tag）的对话转为标准格式。验证角色交替是否为 user/assistant，跳过异常数据 |
| `OpenAIDatasetConverter.__call__()` | 将 OpenAI 消息格式转换为标准格式。支持 tool_calls 提取、多个 tool_response 合并 |
| `align_dataset(dataset, dataset_attr, data_args, training_args)` | 对数据集 `map()` 调用对应的 converter |
| `register_dataset_converter(name, converter)` | 注册自定义格式转换器 |
| `get_dataset_converter(name, ...)` | 获取格式转换器实例 |

**如何修改**:
- 添加自定义数据格式：继承 `DatasetConverter`，实现 `__call__()`，然后调用 `register_dataset_converter("my_format", MyConverter)`。`_prompt` 是对话历史（奇数个消息，最后一条是用户），`_response` 是助手回复列表。

### 4.4 `data/formatter.py` — 消息格式化器（模板的一部分）

将 `Template` 定义的 slot 列表与实际内容组合成 token 前的字符串。

| 类 | 作用 |
|------|------|
| `Formatter` | 抽象基类，定义 `apply(**kwargs) -> SLOTS` 和 `extract(content)`。`slots` 是模板的字符串/集合/字典列表 |
| `EmptyFormatter.apply()` | 直接返回 slots（无需替换占位符），用于固定文本如 `<\|im_start\|>system` |
| `StringFormatter.apply()` | 将 slots 中的 `{{name}}` 占位符替换为 kwargs 中的实际值，如 `{{content}}` → 实际消息内容 |
| `FunctionFormatter.apply()` | 继承 StringFormatter，额外处理 function call。用正则匹配 `thought_words` 和 `tool_call_words`，解析 JSON 中的函数调用，用 `tool_utils` 格式化 |
| `ToolFormatter.apply()` | 解析 tool 描述的 JSON，用 `tool_utils.tool_formatter()` 格式化工具列表 |

### 4.5 `data/template.py` — 提示模板系统

这是整个项目中**最核心和复杂的模块之一**，定义了每种模型的对话格式。

| 函数/类 | 作用 |
|------|------|
| `Template` | 数据类，包含模型的所有格式信息：`format_user`/`format_assistant`/`format_system`/`format_function`/`format_observation`/`format_tools`/`format_prefix`（各 role 的 Formatter）、`default_system`（默认系统提示）、`stop_words`（停止词列表）、`thought_words`/`tool_call_words`（thinking/tool_call 包裹词）、`efficient_eos`（是否使用高效 EOS 编码）、`mm_plugin`（多模态插件） |
| `Template.encode_oneturn(tokenizer, messages, system, tools)` | **单轮编码**：将消息列表编码为一对 token ids `(prompt_ids, response_ids)`。调用内部 `_encode()` 方法，将除最后一条外的全部 token 合并为 prompt。 |
| `Template.encode_multiturn(tokenizer, messages, system, tools)` | **多轮编码**：返回列表 `[(prompt_ids, response_ids), ...]`，每对代表一轮对话 |
| `Template.get_stop_token_ids(tokenizer)` | 获取所有停止 token（EOS + stop_words 的 token ids） |
| `Template.add_thought(content)` / `remove_thought(content)` | 添加/移除 thinking 标记 |
| `Template.get_ollama_modelfile(tokenizer)` | 生成 Ollama Modelfile（用于导出为 Ollama 可加载格式） |
| `TEMPLATES` | **全局模板注册表**（`OrderedDict`），key 为模板名（如 `qwen3_nothink`），value 为 `Template` 实例 |
| `get_template_and_fix_tokenizer(tokenizer, data_args)` | 根据 `data_args.template` 查找模板，修正 tokenizer（添加停止词、设置 pad_token 等）。若 `default_system` 被设置，覆盖模板默认值。 |
| `register_template(name, template)` | 注册自定义模板到 `TEMPLATES` |

**如何修改**:
- 添加新模型模板：在 `TEMPLATES` 字典中添加条目，需要定义各 role 的 Formatter slot 列表。参考已有模板的定义格式。
- 修改系统提示词：设置 `data_args.default_system`

### 4.6 `data/collator.py` — 批处理整理器

| 函数/类 | 作用 |
|------|------|
| `MultiModalDataCollatorForSeq2Seq.__call__()` | 继承 `DataCollatorForSeq2Seq`，支持多模态（图片/视频/音频）。处理 dummy image/audio（避免 ZeRO3/FSDP 挂起）、调用 mm_plugin 生成多模态输入、计算 MRoPE position_ids |
| `SFTDataCollatorWith4DAttentionMask.__call__()` | SFT 用数据整理器。支持 block_diag_attn（4D 注意力掩码，防止跨序列注意力）和 neat_packing（FA2 下的序列打包） |
| `PairwiseDataCollatorWithPadding.__call__()` | DPO 用数据整理器。将每个样本拆分为 `chosen` 和 `rejected` 两部分，复制为 2n 个样本，前半是 chosen 后半是 rejected |
| `prepare_4d_attention_mask()` | 将 2D 注意力掩码扩展为 4D（用于打包训练时的序列隔离） |

### 4.7 `data/data_utils.py` — 数据工具函数

| 函数/类 | 作用 |
|------|------|
| `Role` | 枚举：`USER`、`ASSISTANT`、`SYSTEM`、`FUNCTION`、`OBSERVATION` |
| `DatasetModule` | TypedDict：`{"train_dataset": ..., "eval_dataset": ...}` |
| `merge_dataset(all_datasets, data_args, seed)` | 合并多个数据集。支持 concat 和 interleave 策略（undersampling/oversampling/sampling without replacement） |
| `split_dataset(dataset, eval_dataset, data_args, seed)` | 切分训练/验证集。若 `eval_dataset` 已指定则直接使用；若 `val_size > 0` 则从训练集切分；否则全部用于训练 |
| `get_dataset_module(dataset)` | 将 DatasetDict 转换为 DatasetModule 格式 |
| `read_cloud_json(cloud_path)` | 读取 S3/GCS 云存储中的 JSON/JSONL 文件 |
| `setup_fs(path)` | 根据路径协议（s3://, gs://）创建 fsspec 文件系统对象 |

### 4.8 `data/processor/` — 数据处理器

| 文件 | 类 | 作用 |
|------|------|------|
| `supervised.py` | `SupervisedDatasetProcessor` | SFT 监督微调数据处理。`preprocess_dataset()` 将原始消息 encode 为 input_ids/labels，`print_data_example()` 打印 tokenize 后示例 |
| `packed_supervised.py` | `PackedSupervisedDatasetProcessor` | 打包版 SFT。将多个短序列拼接为一个长序列，通过调整 attention_mask 的 sequence_id 实现序列隔离 |
| `pairwise.py` | `PairwiseDatasetProcessor` | DPO 偏好对数据处理。同时编码 chosen 和 rejected，分别生成 `chosen_input_ids`/`chosen_labels` 和 `rejected_input_ids`/`rejected_labels` |
| `feedback.py` | `FeedbackDatasetProcessor` | KTO 反馈数据处理（精简版保留但基本不用） |
| `pretrain.py` | `PretrainDatasetProcessor` | 预训练数据处理 |
| `unsupervised.py` | `UnsupervisedDatasetProcessor` | 无监督数据处理 |

---

## 5. 模型加载 Model Loading

### 5.1 `model/loader.py` — 模型加载器

| 函数 | 作用 |
|------|------|
| `load_tokenizer(model_args)` | **加载 tokenizer 和 processor**。从 `model_name_or_path` 加载 AutoTokenizer（fast/slow 自动回退）、AutoProcessor（多模态），并调用 `patch_tokenizer()`/`patch_processor()` 修正。返回 `TokenizerModule`。 |
| `load_config(model_args)` | 加载模型配置 `AutoConfig` |
| `load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)` | **模型加载主函数**。流程：加载 config → 打补丁 → 应用 liger kernel → 选择 AutoModel 类型（CausalLM/ImageTextToText/Seq2SeqLM/TextToWaveform）→ `from_pretrained()` → 模型补丁 → `init_adapter()`（LoRA/Freeze/Full）→ 可选 valuehead 加载 → 参数统计 |
| `_get_init_kwargs(model_args)` | 构建加载模型/tokenizer 的初始化参数（cache_dir、revision、token等） |

**如何修改**:
- 支持从本地路径加载：在 YAML 中设置 `model_name_or_path: /path/to/model`
- 添加新的模型类型支持：在 `load_model()` 中添加 AutoModel 匹配逻辑

### 5.2 `model/adapter.py` — 适配器初始化

| 函数 | 作用 |
|------|------|
| `init_adapter(config, model, model_args, finetuning_args, is_trainable)` | **适配器总入口**。根据 `finetuning_type` 分发到不同策略。决定是否将可训练参数 cast 到 float32（QLoRA 和 ZeRO3 下的特殊情况处理） |
| `_setup_full_tuning(model, finetuning_args, is_trainable, cast)` | **全参数微调**：冻结 forbidden modules（vision_tower、multi_modal_projector、language_model 按配置冻结），其余参数设为可训练 |
| `_setup_freeze_tuning(model, finetuning_args, is_trainable, cast)` | **部分参数微调**：根据 `freeze_trainable_layers`（正数=最后n层，负数=前n层）和 `freeze_trainable_modules` 精确控制哪些参数可训练 |
| `_setup_lora_tuning(config, model, model_args, finetuning_args, is_trainable, cast)` | **LoRA/OFT 微调**：处理 adapter 合并/恢复、查找 target_modules、创建 LoRA/OFT config、调用 `get_peft_model()`。支持 PiSSA 初始化、LoRA+、DoRA、RSLoRA |

**如何修改**:
- 添加新的微调方法：在 `init_adapter()` 中添加 `elif finetuning_args.finetuning_type == "xxx"`，并实现对应的 `_setup_xxx_tuning()` 函数
- 自定义 LoRA 目标模块：在 YAML 设置 `lora_target: "q_proj,v_proj"`

### 5.3 `model/patcher.py` — 模型修补器

| 函数 | 作用 |
|------|------|
| `patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)` | **配置修补**。设置 attention 实现、RoPE 缩放、量化配置、MoE 配置、视觉模型配置、KV cache。兼容各种模型架构特殊性（qwen、minicpmo、internvl 等） |
| `patch_tokenizer(tokenizer, model_args)` | **分词器修补**。添加自定义 tokens（`add_tokens`/`add_special_tokens`），扩展 model_max_length |
| `patch_processor(processor, tokenizer, model_args)` | **处理器修补**。将 processor 与 tokenizer 关联，设置图像/视频/音频的处理参数 |
| `patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)` | **模型修补**。调整 generation_config、应用 `resize_embedding_layer()`、准备训练（梯度检查点）、autocast projector dtype、注册模型标签 |
| `patch_valuehead_model(model)` | 为 valuehead 模型（RLHF 用）补齐缺失方法（`tie_weights`、`get_input_embeddings` 等） |
| `patch_qwen3_5_forward(model)` | Qwen3.5 特定修补：Monkey-patch decoder forward 和 GDN forward 以支持 cu_seqlens 输入（packing 训练必需） |

### 5.4 `model/model_utils/` — 模型工具子模块

| 文件 | 主要函数 | 作用 |
|------|------|------|
| `attention.py` | `configure_attn_implementation()`, `print_attn_implementation()` | 根据 `flash_attn` 配置选择 sdpa/flash_attention_2/eager |
| `quantization.py` | `configure_quantization()` | 配置 BitsAndBytes/GPTQ/AWQ 量化 |
| `rope.py` | `configure_rope()` | 配置 RoPE 缩放策略（linear/dynamic/yarn 等） |
| `moe.py` | `configure_moe()`, `add_z3_leaf_module()` | 配置 MoE 参数（aux_loss_coef），DeepSpeed ZeRO3 下注册 leaf module |
| `misc.py` | `find_all_linear_modules()`, `find_expanded_modules()`, `register_autoclass()` | 查找所有线性模块（LoRA 目标）、查找扩展模块（Llama Pro）、注册 AutoConfig |
| `embedding.py` | `resize_embedding_layer()` | 调整嵌入层大小（添加新 token 后重初始化） |
| `checkpointing.py` | `prepare_model_for_training()` | 启用梯度检查点 |
| `kv_cache.py` | `configure_kv_cache()` | 配置 KV cache |
| `visual.py` | `configure_visual_model()`, `autocast_projector_dtype()`, `patch_target_modules()` | 视觉模型配置、投影层数据类型、LoRA target 过滤（排除视觉模块） |
| `liger_kernel.py` | `apply_liger_kernel()` | 应用 Liger Kernel（替换 transformer 中的某些操作以加速） |
| `longlora.py` | `configure_longlora()` | 配置 LongLoRA 的 shift short attention |
| `unsloth.py` | `load_unsloth_pretrained_model()`, `get_unsloth_peft_model()` | Unsloth 加速库集成 |
| `valuehead.py` | `load_valuehead_params()`, `prepare_valuehead_model()` | RLHF valuehead 模型加载和准备 |
| `packing.py` | 序列打包相关工具 | `neat_packing` 的底层实现 |
| `mod.py` | `convert_pretrained_model_to_mod()`, `load_mod_pretrained_model()` | Mixture-of-Depths 模型转换 |

---

## 6. 训练系统 Training

### 6.1 `train/tuner.py` — 训练调度器

| 函数 | 作用 |
|------|------|
| `run_exp(args, callbacks)` | **训练入口**。读取参数 → 构建回调列表（LogCallback + PissaConvertCallback + EarlyStoppingCallback + ReporterCallback）→ 根据 stage 分发到 `run_sft()` 或 `run_dpo()` |
| `export_model(args)` | **模型导出入口**。加载 tokenizer + template + model → 合并 adapter → 设置 dtype → `save_pretrained()` → 保存 tokenizer/processor → 生成 Ollama Modelfile |
| `_training_function(config)` | 内部函数，解析参数 → 创建回调 → 分发训练 → 清理分布式进程组 |

**如何修改**:
- 添加新的训练阶段：在 `run_exp()` 中添加 `elif finetuning_args.stage == "xxx"` 分发
- 自定义导出逻辑：修改 `export_model()` 中的保存参数

### 6.2 `train/sft/workflow.py` — SFT 训练流程

| 函数 | 作用 |
|------|------|
| `run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)` | **SFT 完整工作流**。流程：加载 tokenizer → 获取模板 → 加载数据集 → 加载模型 → 创建 SFTDataCollator → 配置评估指标（ComputeSimilarity 用于 predict_with_generate，ComputeAccuracy 用于 compute_accuracy）→ 创建 CustomSeq2SeqTrainer → 训练 → 保存模型 → 评估 → 预测 → 绘制 loss 曲线 → 创建 model card |

### 6.3 `train/dpo/workflow.py` — DPO 训练流程

| 函数 | 作用 |
|------|------|
| `run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)` | **DPO 完整工作流**。流程：加载 tokenizer → 获取模板 → 加载偏好数据集（stage="rm"）→ 加载模型 → 创建参考模型（如需要）→ 创建 PairwiseDataCollator → 创建 CustomDPOTrainer → 训练 → 评估 → 绘制 loss 曲线 → 创建 model card |

### 6.4 `train/sft/trainer.py` — SFT Trainer

`CustomSeq2SeqTrainer` 继承 `Seq2SeqTrainer`，重写 `compute_loss()` 和 `prediction_step()`，支持 ASFT/DFT/EAFT 辅助损失。

### 6.5 `train/dpo/trainer.py` — DPO Trainer

`CustomDPOTrainer` 继承 `DPOTrainer`，支持 sigmoid/hinge/ipo/orpo/simpo 等 DPO 变体损失。

### 6.6 `train/callbacks.py` — 训练回调

| 类 | 作用 |
|------|------|
| `LogCallback` | **日志回调**。记录训练进度（耗时、剩余时间、loss、lr、epoch）、token 吞吐量、VRAM 使用。支持 WebUI 的中止信号（SIGABRT）。通过 ThreadPoolExecutor 异步写入 trainer_log.jsonl |
| `LogCallback._timing()` | 计算已用时间和剩余时间 |
| `LogCallback._write_log()` | 将日志行追加写入 JSONL 文件 |
| `LogCallback.on_log()` | 在每个日志步骤收集指标并写入文件 |
| `PissaConvertCallback` | PiSSA 适配器转换回调。训练开始时保存初始 PiSSA adapter，训练结束时转换为正常 LoRA adapter |
| `ReporterCallback` | **外部报告回调**。将训练配置上报到 wandb 和 trackio（实验追踪工具） |
| `FixValueHeadModelCallback` | 修复 valuehead checkpoint 的回调 |
| `SaveProcessorCallback` | 保存 processor（多模态用）的回调 |

### 6.7 `train/trainer_utils.py` — Trainer 工具

| 函数 | 作用 |
|------|------|
| `create_ref_model(model_args, finetuning_args)` | 为 DPO 创建参考模型（加载原始模型或 ref_model 路径），冻结参数 |
| `create_modelcard_and_push(trainer, ...)` | 创建模型卡片并推送到 HuggingFace Hub |

### 6.8 `train/test_utils.py` — 测试用工具

| 函数 | 作用 |
|------|------|
| `load_inference_model()` | 测试时快速加载推理模型 |

---

## 7. 推理引擎 Chat

### 7.1 `chat/chat_model.py` — Chat 模型包装器

| 类/函数 | 作用 |
|------|------|
| `ChatModel.__init__(args)` | 创建 HuggingfaceEngine 实例，启动独立的 asyncio 事件循环线程 |
| `ChatModel.chat(messages, ...)` | **同步对话**：通过 `run_coroutine_threadsafe` 将异步调用桥接到同步 |
| `ChatModel.achat(messages, ...)` | **异步对话**：直接 await engine.chat() |
| `ChatModel.stream_chat(messages, ...)` | **同步流式对话**：逐 token yield 返回 |
| `ChatModel.astream_chat(messages, ...)` | **异步流式对话**：async for 逐 token |
| `ChatModel.get_scores(batch_input)` | 同步批量打分 |
| `run_chat()` | **CLI 聊天入口**：创建 ChatModel → 循环读取用户输入 → 流式输出回复。支持 `exit` 退出和 `clear` 清空历史 |

**如何修改**:
- 自定义聊天行为：修改 `ChatModel` 的 system/tools 默认值
- 添加新的推理后端：在 `ChatModel.__init__()` 中根据 `model_args.infer_backend` 切换 engine

### 7.2 `chat/hf_engine.py` — HuggingFace 推理引擎

`HuggingfaceEngine` 继承 `BaseEngine`：

| 方法 | 作用 |
|------|------|
| `__init__()` | 加载 model + tokenizer + template，配置生成参数 |
| `chat()` | 使用 `model.generate()` 生成回复 |
| `stream_chat()` | 使用 `TextIteratorStreamer` 实现流式生成 |
| `get_scores()` | 用 reward model 对输入打分 |

### 7.3 `chat/base_engine.py` — 推理引擎基类

| 类 | 作用 |
|------|------|
| `BaseEngine` | 抽象基类，定义 `chat()`、`stream_chat()`、`get_scores()` 接口 |
| `Response` | 响应数据类：`response_text`、`response_length`、`prompt_length`、`finish_reason` |

---

## 8. WebUI 可视化

### 8.1 `webui/interface.py` — WebUI 界面

| 函数 | 作用 |
|------|------|
| `create_ui()` | 创建 Gradio Blocks 界面，组合多个 Tab（Train/Evaluate/Chat/Export） |
| `run_web_ui()` | 启动 LLaMA Board |
| `run_web_demo()` | 启动独立聊天界面 |

### 8.2 `webui/` 组件

| 文件 | 作用 |
|------|------|
| `runner.py` | 训练/评估/推理/导出任务的执行器。每个 Runner 方法将 UI 输入转换为参数字典，调用对应的 backend 函数 |
| `manager.py` | 数据/模型管理器。管理 dataset_info、模板列表、模型选择 |
| `components/train.py` | 训练 Tab 的 UI 组件 |
| `components/infer.py` | 推理 Tab 的 UI 组件 |
| `components/eval.py` | 评估 Tab 的 UI 组件 |
| `components/export.py` | 导出 Tab 的 UI 组件 |
| `components/chatbot.py` | 聊天 UI 组件 |
| `components/data.py` | 数据集预览组件 |
| `components/top.py` | 顶部导航栏 |
| `components/footer.py` | 底部状态栏 |
| `control.py` | 环境控制（GPU 数、加速后端等） |
| `engine.py` | 多进程引擎管理 |
| `css.py` | 自定义 CSS 样式 |
| `locales.py` | 多语言支持（中/英/俄/日/德/法/西/葡/韩） |
| `common.py` | 公共工具函数 |

---

## 9. API 服务

### 9.1 `api/app.py`

| 函数 | 作用 |
|------|------|
| `create_app(chat_model)` | 创建 FastAPI 应用，注册路由 |
| `run_api()` | 从命令行参数创建 ChatModel 并启动服务 |
| `/v1/chat/completions` | OpenAI 兼容的 chat completions API，支持流式和非流式 |
| `/v1/models` | 列出可用模型 |

### 9.2 `api/chat.py`

| 函数 | 作用 |
|------|------|
| 处理 API 请求，调用 `ChatModel` 生成回复，处理流式/非流式 SSE 响应 |
| `process_chat()` | 核心处理逻辑 |
| `process_stream_chat()` | 流式处理逻辑 |

### 9.3 `api/protocol.py`

| 类 | 作用 |
|------|------|
| Pydantic 模型，定义 API 请求/响应的数据结构（`ChatCompletionRequest`、`ChatCompletionResponse`、`Message` 等） |

### 9.4 `api/common.py`

| 函数 | 作用 |
|------|------|
| 公共工具函数：错误码定义、CORS 中间件 |

---

## 10. 评估系统 Evaluation

### 10.1 `eval/evaluator.py`

| 函数 | 作用 |
|------|------|
| `Evaluator` 类 | 标准 benchmark 评估（MMLU/C-Eval 等）。使用 `lm-evaluation-harness` 风格的模板，支持 few-shot 评估 |

### 10.2 `eval/template.py`

| 函数 | 作用 |
|------|------|
| `get_eval_template()` | 根据 task name 获取评估模板（包含 few-shot exemplars 和 prompt 格式） |

---

## 11. 工具模块 Extras

### 11.1 `extras/constants.py`

定义全局常量：

| 常量 | 说明 |
|------|------|
| `IGNORE_INDEX = -100` | loss 计算中忽略的 label token ID |
| `CHECKPOINT_NAMES` | checkpoint 文件名集合 |
| `DATA_CONFIG = "dataset_info.json"` | 数据集注册表文件名 |
| `TRAINER_LOG = "trainer_log.jsonl"` | 训练日志文件名 |
| `FILEEXT2TYPE` | 文件扩展名到 HuggingFace Dataset 数据类型的映射 |
| `EngineName` | 引擎名枚举（HF） |
| `QuantizationMethod` | 量化方法枚举（BNB/HQQ/FP8 等） |
| `RopeScaling` | RoPE 缩放策略枚举 |
| `AttentionFunction` | Attention 实现枚举（AUTO/SDPA/FA2） |
| `MCA_SUPPORTED_MODELS` | 支持的 Megatron 模型列表 |
| `MROPE_MODELS` | 使用 MRoPE 位置编码的模型列表 |

### 11.2 `extras/misc.py`

| 函数 | 作用 |
|------|------|
| `get_current_device()` | 获取当前设备（cuda/mps/npu/xpu/cpu） |
| `get_device_count()` | 获取当前设备的数量 |
| `get_device_name()` | 获取当前设备名称 |
| `infer_optim_dtype(model_dtype)` | 推断最优计算数据类型（bf16 > fp16 > fp32） |
| `count_parameters(model)` | 统计可训练参数和总参数量 |
| `find_available_port()` | 查找可用端口（分布式训练用） |
| `torch_gc()` | 清理 GPU 内存缓存 |
| `check_dependencies()` | 检查核心依赖版本 |
| `check_version(package, mandatory)` | 检查特定包的版本 |
| `is_env_enabled(key, default)` | 检查环境变量是否被设置为 `1`/`true`/`yes` |
| `use_modelscope()` | 是否使用 ModelScope 下载模型 |
| `try_download_model_from_other_hub()` | 尝试从备选 Hub 下载模型 |
| `calculate_tps(dataset, metrics, stage)` | 计算训练吞吐量（tokens/秒） |
| `get_peak_memory()` | 获取 GPU VRAM 使用峰值 |

### 11.3 `extras/logging.py`

自定义日志系统：支持 rank 感知日志（只在 rank 0 输出）、日志级别过滤、文件输出。

| 类 | 作用 |
|------|------|
| `LoggerHandler` | WebUI 模式下将日志重定向到文件 |
| `get_logger(name)` | 获取配置好的 logger |
| `logger.info_rank0()` / `logger.warning_rank0()` | 只在 rank 0（主进程）输出日志 |

### 11.4 `extras/env.py`

| 函数 | 作用 |
|------|------|
| `print_env()` | 打印完整环境信息（版本、依赖、GPU 状态等） |
| `VERSION` | 当前版本号 |

### 11.5 `extras/packages.py`

| 函数 | 作用 |
|------|------|
| `is_package_available(name)` | 检查 Python 包是否可用 |
| `is_transformers_version_greater_than(version)` | 比较 transformers 版本 |
| `is_torch_version_greater_than(version)` | 比较 PyTorch 版本 |

### 11.6 `extras/ploting.py`

| 函数 | 作用 |
|------|------|
| `plot_loss(output_dir, keys)` | 从 trainer_state.json 读取训练历史，用 matplotlib 绘制 loss/eval_loss/accuracy 等曲线图 |

---

## 12. 如何修改：完整指南

### 12.1 修改训练配置（最快上手）

1. 复制 `examples/train_lora/qwen3_lora_sft.yaml` 并修改：
   - `model_name_or_path`: 模型路径或 HuggingFace ID
   - `dataset`: 你的数据集名称
   - `template`: 对应你的模型的模板名
   - `output_dir`: 输出目录
   - `learning_rate` / `num_train_epochs` / `per_device_train_batch_size`: 超参数

2. 运行：`llamafactory-cli train your_config.yaml`

### 12.2 添加自定义数据集

**Step 1**: 准备数据文件（JSON/JSONL），放到 `data/` 目录

**SFT 数据（Alpaca 格式）**:
```json
[
  {"instruction": "问题", "input": "", "output": "回答"},
  {"instruction": "...", "input": "...", "output": "..."}
]
```

**SFT 数据（ShareGPT 格式）**:
```json
[
  {"conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]}
]
```

**DPO 数据**:
```json
[
  {
    "conversations": [{"from": "human", "value": "问题"}],
    "chosen": {"from": "gpt", "value": "好的回答"},
    "rejected": {"from": "gpt", "value": "差的回答"}
  }
]
```

**Step 2**: 在 `data/dataset_info.json` 中注册：
```json
{
  "my_sft_data": {"file_name": "my_sft_data.json"},
  "my_dpo_data": {
    "file_name": "my_dpo_data.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
  }
}
```

**Step 3**: 在训练配置中设置 `dataset: my_sft_data`

### 12.3 添加新模型的模板

**Step 1**: 在 `src/llamafactory/data/template.py` 的 `TEMPLATES` 字典中添加条目：

```python
TEMPLATES["my_model"] = Template(
    format_user=StringFormatter(slots=["<|user|>\n{{content}}\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|end|>\n"], tool_format="default"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}\n<|assistant|>\n"]),
    format_tools=ToolFormatter(slots=["{{content}}\n"], tool_format="default"),
    format_prefix=EmptyFormatter(slots=["<|begin|>"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|end|>"],
    thought_words=("", ""),
    tool_call_words=("", ""),
    efficient_eos=False,
    replace_eos=False,
    replace_jinja_template=False,
    mm_plugin=get_mm_plugin(name="base"),
)
```

**Step 2**: 在训练配置中设置 `template: my_model`

### 12.4 添加新的训练方法（如自定义 Loss）

**Step 1**: 在 `src/llamafactory/train/` 下创建新目录（如 `train/my_method/`）

**Step 2**: 实现 `workflow.py`：
```python
def run_my_method(model_args, data_args, training_args, finetuning_args, callbacks):
    # 加载 tokenizer、数据、模型
    # 创建 trainer
    # 训练循环
    pass
```

**Step 3**: 在 `hparams/finetuning_args.py` 中将 `stage` 的类型添加 `"my_method"`

**Step 4**: 在 `tuner.py` 的 `_training_function()` 中添加分发：
```python
elif finetuning_args.stage == "my_method":
    run_my_method(model_args, data_args, training_args, finetuning_args, callbacks)
```

### 12.5 添加新的数据格式

**Step 1**: 在 `src/llamafactory/data/converter.py` 中创建新的 Converter 类：
```python
@dataclass
class MyFormatConverter(DatasetConverter):
    def __call__(self, example):
        return {
            "_prompt": [{"role": "user", "content": example["question"]}],
            "_response": [{"role": "assistant", "content": example["answer"]}],
            "_system": "",
            "_tools": "",
            "_images": None,
            "_videos": None,
            "_audios": None,
        }

register_dataset_converter("my_format", MyFormatConverter)
```

**Step 2**: 在 `dataset_info.json` 中指定 `"formatting": "my_format"`

### 12.6 添加新的配置字段

**Step 1**: 在对应的 `hparams/*_args.py` dataclass 中添加 `field()`：
```python
my_new_param: float = field(
    default=0.5,
    metadata={"help": "My new parameter description."}
)
```

**Step 2**: 在 YAML 配置中使用：
```yaml
my_new_param: 0.8
```

### 12.7 导出模型用于 Ollama

训练完成后，使用以下命令导出：

```bash
llamafactory-cli export exports/my_model.yaml
```

其中 `exports/my_model.yaml`：
```yaml
model_name_or_path: saves/qwen3-4b/lora/sft
template: qwen3_nothink
export_dir: ./exported_model
export_size: 5
```

导出成功后，在 `export_dir/` 下会生成 `Modelfile`，可用 Ollama 加载：
```bash
ollama create my-model -f ./exported_model/Modelfile
```

### 12.8 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `ValueError: Undefined dataset xxx` | 数据集未在 `dataset_info.json` 注册 | 添加注册条目 |
| `ValueError: Please provide model_name_or_path` | 未指定模型 | 在 YAML 中设置 `model_name_or_path` |
| `OSError: Failed to load tokenizer` | 模型路径错误或网络问题 | 检查路径或设置 `use_modelscope: true` |
| `torch.cuda.OutOfMemoryError` | 显存不足 | 减少 `cutoff_len`/`per_device_train_batch_size`，启用 QLoRA |
| `distributed training` 报错 | 单GPU未通过 `llamafactory-cli` 启动 | 使用 `llamafactory-cli train` 而非 `python train.py` |

### 12.9 目录约定

| 目录 | 作用 |
|------|------|
| `data/` | 内置 demo 数据集 + `dataset_info.json` |
| `examples/` | 训练配置 YAML 示例 |
| `src/llamafactory/` | 源代码根目录 |
| `saves/` | 训练输出（模型 checkpoint） |
| `tests/` | 单元测试 |
| `requirements/` | 额外依赖定义文件 |

### 12.10 从 HuggingFace 下载模型的替代方案

如果 HuggingFace 访问不稳定，可在训练配置中添加：

```yaml
use_modelscope: true  # 使用 ModelScope（国内源）
```

或使用环境变量：

```bash
export USE_MODELSCOPE_HUB=1
export HF_ENDPOINT=https://hf-mirror.com
```
