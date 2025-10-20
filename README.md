# 🧠 Fine-Tuning Large Language Models (LLMs) Efficiently with Unsloth + LoRA

> A practical, lightweight pipeline for fine-tuning large language models using [Unsloth](https://unsloth.ai) and [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685).  
> Created and maintained by [Firas Tlili](https://medium.com/@firastlili).

---
![thumbnail](thunb.png)

## 🚀 Project Overview

This repository demonstrates how to **fine-tune state-of-the-art language models** such as **Llama 3**, **Mistral**, or **Phi-3** efficiently — even on **modest hardware**.

By combining **Unsloth’s GPU-optimized training engine** with **parameter-efficient tuning (LoRA)** and optional **4-bit quantization**, you can adapt large LLMs to your custom datasets **up to 2× faster** and with **70% less memory usage** than traditional fine-tuning.


## 💡 Motivation

Fine-tuning large language models from scratch is expensive — requiring high-end GPUs and huge memory.  
This project bridges that gap by using **parameter-efficient fine-tuning (PEFT)** with **LoRA** and **Unsloth**, allowing you to:

- Customize powerful open-weight models for your domain (legal, medical, industrial, etc.)
- Train and deploy efficiently on a **single consumer GPU**
- Preserve base-model generalization while injecting new knowledge

---

## ✨ Key Features

| Feature | Description |
|:--|:--|
| 🪶 **Unsloth Integration** | Optimized training backend (fast kernels, mixed precision, memory offloading). |
| 🧩 **LoRA (Low-Rank Adaptation)** | Fine-tune only a small set of weights for efficiency. |
| 🧠 **Quantization Support** | 4-bit or 8-bit loading for reduced VRAM footprint. |
| 📊 **Supervised Fine-Tuning (SFT)** | Instruction-response or conversational data. |
| 🧪 **Evaluation & Merging** | Merge LoRA adapters into the base model for deployment. |
| 🖥️ **Colab/Local Ready** | Compatible with Google Colab or local GPUs. |

---

## 🏗️ Architecture
![Architecture](arch.png)

## 🧠 Tips & Best Practices

- 🔹 **LoRA Rank:** Use ranks **8–32** for small datasets; increase for complex or large-scale tasks.  
- 🔹 **Data Quality:** Always verify dataset formatting — avoid trailing spaces, inconsistent keys, or newline issues.  
- 🔹 **Learning Rate:** Start with **2e-4** and fine-tune based on validation loss trends.  
- 🔹 **Memory Optimization:** Enable **gradient checkpointing** to save GPU memory.  
- 🔹 **Evaluation:** Assess both **quantitative metrics** (BLEU, ROUGE, accuracy) and **qualitative results** (response quality).  
- 🔹 **Deployment:** Merge adapters before serving models — set `merge_lora_weights=True`.  
- 🔹 **Backup:** Regularly back up checkpoints to prevent loss during long runs.

## 🙏 Acknowledgments

- 🪶 [**Unsloth**](https://unsloth.ai) — for the efficient fine-tuning framework.  
- 🤗 [**Hugging Face Transformers**](https://huggingface.co/transformers) — model APIs and tokenizer tools.  
- 🔧 [**PEFT**](https://github.com/huggingface/peft) — implementation of LoRA and adapter methods.  
- ✍️ Original guide by [**Firas Tlili**](https://medium.com/@firastlili/fine-tuning-large-language-models-llms-efficiently-with-unsloth-lora-54b6e10fbfcb).
