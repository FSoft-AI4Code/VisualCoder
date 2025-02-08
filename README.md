<div align="center">

# [NAACL'25] VISUALCODER: Guiding Large Language Models in Code Execution with Fine-grained Multimodal Chain-of-Thought Reasoning  
[![arXiv](https://img.shields.io/badge/arXiv-2410.23402-b31b1b.svg)](https://arxiv.org/abs/2410.23402)  

</div>

## 🚀 Introduction

We introduce **VISUALCODER**, a **novel framework** that enhances **code execution reasoning** by integrating **multimodal Chain-of-Thought (CoT) prompting** with **visual Control Flow Graphs (CFGs)**. While **Large Language Models (LLMs)** are highly effective at analyzing **static code**, they struggle with **dynamic execution reasoning**, leading to errors in **program behavior prediction, fault localization, and program repair**. To address these challenges, **VISUALCODER** aligns **code snippets with their corresponding CFGs**, providing LLMs with a **structured understanding of execution flows**. Unlike prior methods that rely on **text-based CFG descriptions**, **VISUALCODER** leverages **visual CFG representations** and a **Reference Mechanism** to establish a direct connection between **code structure and execution dynamics**.


<div align="center">
  <img src="img/VisualCoder.png" width="80%">
<p><b>Figure 1: VISUALCODER – CFG + CoT + Reference for more accurate code execution understanding.</b></p>
</div>


## ✨ Key Features

✔ **Multimodal CoT Reasoning** – Combines **source code** and **CFG images** for enhanced program analysis.  
✔ **Reference Mechanism** – Explicitly **links code lines to corresponding CFG nodes**, improving reasoning accuracy.  
✔ **Error Detection & Fault Localization** – Prevents hallucinations by grounding reasoning in actual execution flows.  
✔ **Program Repair Assistance** – Helps LLMs understand execution errors and suggest fixes.  

## 📜 Paper  
📄 **NAACL 2025**: [VISUALCODER: Guiding Large Language Models in Code Execution with Fine-grained Multimodal Chain-of-Thought Reasoning](https://arxiv.org/abs/2410.23402)  

## ⚙️ Installation

To set up the environment and install the necessary dependencies, run:

```sh
./setup.sh


## **Usage**


- **Fault Localization with Close-Source LLM**:

  ```bash
  python fault_localization_close_source_LLM.py --session <session_number> --setting <setting_name> --close_model <model_type>
  ```

  - `<session_number>`: Numerical identifier for the session (e.g., `1`, `2`, etc.)
  - `<setting_name>`: One of the following: `buggy` (plain code), `buggy_CoT` (plain code +CoT), `buggy_cfg_CoT` (plain code +CFG +CoT), `NeXT` (NeXT), `VisualCoder` (VisualCoder), `Multimodal_CoT` (2-stage Multimodal_CoT), `Multimodal-CoT_VisualCoder` (Multimodal_CoT + VisualCoder)
  - `<model_type>`: Specify either `claude` or `gpt` for the close-source model
  - `claude_api_key`: API key for Claude
  - `openai_api_key`: API key for OpenAI
  - `azure_endpoint`: Azure endpoint for OpenAI
  - `deployment_name`: Deployment name for OpenAI
  - `version`: Version of the model

- **Fault Localization with InternVL**:

  ```bash
  python fault_localization_InternVL.py --session <session_number> --setting <setting_name>
  ```
  - `<session_number>`: Numerical identifier for the session (e.g., `1`, `2`, etc.)
  - `<setting_name>`: One of the following: `buggy` (plain code), `buggy_CoT` (plain code +CoT), `buggy_cfg_CoT` (plain code +CFG +CoT), `NeXT` (NeXT), `VisualCoder` (VisualCoder), `Multimodal_CoT` (2-stage Multimodal_CoT), `Multimodal-CoT_VisualCoder` (Multimodal_CoT + VisualCoder)

- **Program Repair with Close-Source LLM**:

  ```bash
  python program_repair_close_source_LLM.py --session <session_number> --setting <setting_name> --close_model <model_type>
  ```

- **Program Repair with InternVL**:

  ```bash
  python program_repair_InternVL.py --session <session_number> --setting <setting_name>
  ```

- **Get Attention Scores**:

  This command calculates attention scores for a given session, available only for `get_attention_score.py`:

  ```bash
  python get_attention_score.py --session <session_number> --prompt_mode <prompt_type> --setting <setting_name>
  ```

  - `<prompt_type>`: Specify the prompt type (e.g., `zeroshot`)

### Example Command

Here’s an example for running **Get Attention Scores**:

```bash
python get_attention_score.py --session 1 --prompt_mode zeroshot --setting VisualCoder
```

The command can be adapted to any other script by modifying the script name and options.

## Configuration

The configuration parameters like `session`, `setting`, and `close_model` (for close-source models) are consistent across all commands. Customize these parameters based on your task and model requirements:

- **Session**: Numerical identifier for the session
- **Setting**: One of the following: `buggy`, `cfg`, `buggy_CoT`, `buggy_cfg_CoT`, `cfg_CoT`, `VisualCoder`, `Multimodal_CoT`, `Multimodal-CoT_VisualCoder`
- **Close Model**: Only applicable for close-source LLM scripts (`claude` or `gpt`)
