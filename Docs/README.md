# Daily Paper Reading List

> Curated daily paper reading list, organized by topic with 📄 links to PDFs and `→` `←` cross-references.

---

## Table of Contents

- [LLM Technology Landscape](#llm-technology-landscape)
- [1. Transformer Foundations](#1-transformer-foundations)
- [2. Foundation Models & Technical Reports](#2-foundation-models-and-technical-reports)
- [3. Scaling & Training Theory](#3-scaling-and-training-theory)
- [4. Distributed Training & Systems](#4-distributed-training-and-systems)
- [5. Quantization & Inference Optimization](#5-quantization-and-inference-optimization)
- [6. Post-Training & Alignment](#6-post-training-and-alignment)
- [7. Fine-Tuning & Parameter-Efficient Methods](#7-fine-tuning-and-parameter-efficient-methods)
- [8. Novel Architectures](#8-novel-architectures)
- [9. Mixture of Experts (MoE)](#9-mixture-of-experts-moe)
- [10. RAG & Retrieval](#10-rag-and-retrieval)
- [11. Reasoning](#11-reasoning)
- [12. AI Agents](#12-ai-agents)
- [13. Memory Systems](#13-memory-systems)
- [14. Data Engineering](#14-data-engineering)
- [15. Evaluation & Hallucination](#15-evaluation-and-hallucination)
- [16. Safety, Trust & Ethics](#16-safety-trust-and-ethics)
- [17. Scientific Discovery & Auto-Research](#17-scientific-discovery-and-auto-research)
- [18. Multimodal & Vision-Language](#18-multimodal-and-vision-language)
- [19. MCP & Tools](#19-mcp-model-context-protocol-and-tools)
- [20. World Models & Simulation](#20-world-models-and-simulation)
- [21. Latent Reasoning & Interpretability](#21-latent-reasoning-and-interpretability)
- [22. Computer Vision & Media](#22-computer-vision-and-media)
- [23. Software Engineering & Code](#23-software-engineering-and-code)
- [24. Industry & Applications](#24-industry-and-applications)
- [25. Theory & Foundations](#25-theory-and-foundations)
- [26. Domain-Specific LLMs](#26-domain-specific-llms)
- [27. Miscellaneous](#27-miscellaneous-and-other)

---

## <a id="llm-technology-landscape"></a> LLM Technology Landscape & Evolution (2017–2026)

> Rooted in the Transformer (2017), this landscape organizes the LLM technology stack into two pillars: **Core Model Artifacts** (left) and **Infrastructure & Application Ecosystem** (right). Each sub-line follows chronological order (`──>` = leads to). Nodes are drawn from the paper collection below.

```
================================================================================
           【 FOUNDATION: Transformer (Self-Attention Birth, 2017) 】
================================================================================
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 Ⅰ. CORE MODEL ARTIFACTS                               Ⅱ. INFRASTRUCTURE &
    (Algorithms & Architectures)                         APPLICATION ECOSYSTEM
                                                        (Systems & Engineering)
================================================================================
```

### Ⅰ. CORE MODEL ARTIFACTS

```
 ├─ 1. Architecture Innovations
 │   ├─ Attention Mechanisms   : MHA ──> MQA ──> GQA
 │   ├─ Efficient Attention     : MLA (Low-Rank Joint Comp.) ──> NSA ──> DSA
 │   ├─ Mixture of Experts (MoE): GShard ──> Switch Trans. ──> GLaM ──> Mixtral
 │   │                           └──> DeepSeekMoE
 │   ├─ Long Context PE        : RoPE ──> ALiBi ──> PI ──> YaRN ──> LongRoPE
 │   ├─ Context Management     : StreamingLLM ──> InfLLM
 │   └─ Non-Transformer / SSM  : Linear Trans. ──> Performer ──> RWKV ──> RetNet
 │                               └──> Mamba ──> xLSTM ──> Titans
 │
 ├─ 2. Tokenization & Lexicon Engineering
 │   ├─ Algorithmic Theory     : BPE ──> WordPiece ──> SentencePiece ──> BBPE
 │   └─ Tokenizer Engineering  : HF Tokenizers ──> Fast Tokenizers ──> tiktoken
 │
 ├─ 3. Data Engineering Paradigm
 │   ├─ Corpus Construction    : Web Scrapers ──> Common Crawl Processing
 │   ├─ Data Preprocessing     : Deduplication (MinHash/LSH) ──> Quality Filtering
 │   ├─ Data Mixture & Curri.  : Mixing Laws ──> Stage-based Curriculum Learning
 │   └─ Data Generation        : Synthetic Data ──> Self-play Data Extraction
 │
 ├─ 4. Pre-training & Model Families
 │   ├─ Foundations            : Kaplan Scaling ──> Chinchilla Scaling Laws
 │   ├─ Encoder Lineage        : BERT ──> RoBERTa ──> DeBERTa
 │   └─ Decoder Lineage
 │       ├─ Proprietary
 │       │   ├─ GPT-1 ──> GPT-4 ──> GPT-4o ──> GPT-5 ──> GPT-5.5 (Apr 2026)
 │       │   ├─ Grok-1 ──> Grok 4 ──> Grok 4.3 (Apr 2026)
 │       │   ├─ Claude 1 ──> Opus 4.6 ──> Opus 4.8 ──> Fable 5 (Jun 2026)
 │       │   └─ Gemini 1.0 ──> Gemini 2.5 ──> Gemini 3 ──> Gemini 3.5
 │       └─ Open-Weight
 │           ├─ Llama 1 ──> Llama 2 ──> Llama 3 ──> Llama 4
 │           ├─ Falcon ──> Mistral ──> Yi
 │           ├─ Qwen 2.5 ──> Qwen 3 ──> Qwen 3.5 ──> Qwen 3.6
 │           ├─ DeepSeek-V3 ──> DeepSeek-V4 (Apr 2026)
 │           ├─ Kimi K2 ──> K2.5 ──> K2.7 Code (Jun 2026)
 │           └─ GLM-5 ──> GLM-5.1 ──> GLM-5.2 (Jun 2026)
 │
 ├─ 5. Post-training, Alignment & Policy Scaling
 │   ├─ Classic Alignment Flow : SFT ──> Reward Modeling ──> RLHF (PPO)
 │   └─ Direct Preference Opt  : DPO ──> IPO ──> ORPO ──> KTO
 │
 ├─ 6. Reasoning & Test-Time Compute (System 2 Scaling)
 │   ├─ Prompt-driven Reason.  : CoT ──> Self-Consistency ──> Least-to-Most ──> ToT
 │   └─ RL-scaled CoT & Math   : STaR ──> Quiet-STaR ──> PRM (Process Reward) ──> GRPO
 │                               └──> RLVR ──> OpenAI o1 ──> DeepSeek-R1
 │                               └──> o3 ──> o4-mini (Apr 2025)
 │
 └─ 7. Parameter-Efficient Fine-Tuning (PEFT)
     └─ PEFT Evolution         : Adapters ──> Prefix ──> Prompt ──> LoRA
                                 └──> QLoRA ──> AdaLoRA ──> DoRA
```

### Ⅱ. INFRASTRUCTURE & APPLICATION ECOSYSTEM

```
 ├─ 8. Distributed Training & Systems Infrastructure
 │   ├─ Parallelism Strategies : PP (GPipe) ──> TP (Megatron) ──> ZeRO ──> FSDP
 │   │                           └──> 3D/4D Parallel
 │   ├─ Distributed Attention  : Sequence Parallel (SP) ──> Context Parallel (CP)
 │   │                           └──> RingAttention Topology
 │   ├─ Topologies & Scheduling: Megatron-LM ──> DeepSpeed ──> DualPipe Overlap
 │   └─ Kernel Optimization    : Triton ──> FlashAttention-1/2/3 ──> CUDA Graph
 │
 ├─ 9. LLM Serving & Runtime Systems
 │   ├─ Hardware Precision     : BF16 ──> FP8 ──> MXFP4 (Microscaling Formats)
 │   ├─ Weight Quantization    : GPTQ ──> SmoothQuant ──> AWQ ──> INT4 ──> BitNet
 │   ├─ Runtime Execution      : Speculative Decoding ──> Continuous Batching
 │   │                           └──> PagedAttention
 │   └─ Serving & Runtime Sys  : Ray ──> llama.cpp ──> vLLM ──> TensorRT-LLM
 │                               └──> LMDeploy ──> SGLang
 │
 ├─ 10. Retrieval-Augmented Generation (RAG)
 │   └─ Paradigm Shift         : Naive ──> Advanced ──> GraphRAG ──> Self-RAG
 │                               └──> Agentic RAG
 │
 ├─ 11. Autonomous Agents & Memory Ecosystem
 │   ├─ Agentic Control Flows  : ReAct ──> AutoGPT ──> Tool Use ──> Function Calling
 │   │                           └──> Code-as-Action ──> LangGraph
 │   │                           └──> Claude Code ──> Codex CLI ──> SWE-agent
 │   ├─ Protocol Layer         : MCP (Model Context Protocol) ──> A2A
 │   └─ Agentic Memory Track   : MemoryBank ──> Generative Agents ──> MemGPT
 │                               └──> LongMem ──> MemOS
 │
 ├─ 12. Comprehensive Evaluation & Benchmarking Track
 │   ├─ Static Capability      : MMLU ──> BIG-bench ──> HELM ──> MMLU-Pro
 │   ├─ Dynamic Interaction    : LMSYS Chatbot Arena ──> LLM-as-Judge
 │   ├─ Process-based Reasoning: Process Reward Benchmarks
 │   └─ Safety Evaluation      : Red-teaming ──> Jailbreak ──> Alignment Tests
 │
 └─ 13. Multimodal Frontiers & Boundary Expansion
     ├─ Cross-Modal Perception : CLIP ──> Flamingo ──> BLIP-2 ──> Kosmos-1
     │                           └──> LLaVA ──> Qwen-VL ──> GPT-4o ──> Gemini 3
     │                           └──> Qwen3.5-Omni (omnimodal, 2026)
     ├─ Video Generation        : Sora ──> Veo 3
     └─ Boundary Expansion     : Embodied AI ──> World Models ──> Computer Use
                                 └──> AI for Science
                                       └─ [Track: Agentic Software Eng.]
                                             └──> Devin / Claude Code / OpenHands / Codex CLI
```
> Each of the 13 modules maps to the paper sections below. Nodes in every sub-line can be cross-referenced by name against their corresponding PDFs in `papers/`.  

**Cross-cutting Trade-offs** — these four tensions drive the evolution across both pillars above:

```
 T1  Memory Wall               T2  Communication Overlap
  └─ §1 MHA→GQA→MLA              └─ §8 TP/PP/DualPipe
  └─ §9 PagedAttention           └─ §8 CP/SP → RingAttention
                                  └─ §8 Megatron → DeepSpeed
 T3  Precision Co-design        T4  System 1 → System 2
  └─ §9 INT4→AWQ→MXFP4           └─ §4 Kaplan → Chinchilla
  └─ §9 BF16→FP8→MXFP4           └─ §6 CoT → o1 → R1
```

> Each trade-off is unpacked in [TRADEOFFS.md](TRADEOFFS.md) — from mechanism to open research questions.

---

## 1. Transformer Foundations

### 1.1 Classic Architecture
- [ ] **Attention Is All You Need** Vaswani et al. 2017 — The origin.
  📄 [papers/Attention Is All You Need.pdf](papers/Attention-Is-All-You-Need.pdf)
  - → BERT GPT-1 GPT-2
  - → FlashAttention Section 4.1
- [ ] **BERT: Pre-training of Deep Bidirectional Transformers** Devlin et al. 2018
  📄 [papers/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf](papers/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding.pdf)
- [ ] **ALBERT: A Lite BERT** Lan et al. 2019
  📄 [papers/ALBERT A Lite BERT for Self-supervised Learning of Language Representations.pdf](papers/ALBERT-A-Lite-BERT-for-Self-supervised-Learning-of-Language-Representations.pdf)
- [ ] **GPT-1: Improving Language Understanding by Generative Pre-Training** Radford et al. 2018
  📄 [papers/GPT1 Improving Language Understanding by Generative Pre-Training.pdf](papers/GPT1-Improving-Language-Understanding-by-Generative-Pre-Training.pdf)
- [ ] **GPT-2: Language Models are Unsupervised Multitask Learners** Radford et al. 2019
  📄 [papers/GPT2 language_models_are_unsupervised_multitask_learners.pdf](papers/GPT2-language-models-are-unsupervised-multitask-learners.pdf)
- [ ] **GPT-3: Language Models are Few-Shot Learners** Brown et al. 2020
  📄 [papers/GPT3 Language Models are Few-Shot Learners.pdf](papers/GPT3-Language-Models-are-Few-Shot-Learners.pdf)
- [ ] **GPT-4 Technical Report** OpenAI 2023
  📄 [papers/GPT-4 Technical Report.pdf](papers/GPT-4-Technical-Report.pdf)
- [ ] **GPT-4o System Card** OpenAI 2024
  📄 [papers/GPT-4o System Card.pdf](papers/GPT-4o-System-Card.pdf)
- [ ] **Transformers and LLM** course slides
  📄 [papers/Transformers-and-LLM.pdf](papers/Transformers-and-LLM.pdf)

### 1.2 Attention Deep-Dive
- [ ] **Understanding Self-attention Mechanism via Dynamical System Perspective**
  📄 [papers/Understanding-Self-attention-Mechanism-via-Dynamical-System-Perspective.pdf](papers/Understanding-Self-attention-Mechanism-via-Dynamical-System-Perspective.pdf)
- [ ] **Attention Residuals**
  📄 [papers/Attention-Residuals.pdf](papers/Attention-Residuals.pdf)
- [ ] **Gated Attention for Large Language Models: Non-linearity Sparsity and Attention-Sink-Free**
  📄 [papers/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free.pdf](papers/Gated-Attention-for-Large-Language-Models-Non-linearity-Sparsity-and-Attention-Sink-Free.pdf)
- [ ] **Mixture-of-Depths Attention**
  📄 [papers/Mixture-of-Depths-Attention.pdf](papers/Mixture-of-Depths-Attention.pdf)

### 1.3 Sequence and Context Models
- [ ] **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention**
  📄 [papers/Native-Sparse-Attention-Hardware-Aligned-and-Natively-Trainable-Sparse-Attention.pdf](papers/Native-Sparse-Attention-Hardware-Aligned-and-Natively-Trainable-Sparse-Attention.pdf)
- [ ] **The Topological Trouble With Transformers**
  📄 [papers/The-Topological-Trouble-With-Transformers.pdf](papers/The-Topological-Trouble-With-Transformers.pdf)
- [ ] **Recursive Language Models**
  📄 [papers/Recursive-Language-Models.pdf](papers/Recursive-Language-Models.pdf)
- [ ] **Parcae: Scaling Laws For Stable Looped Language Models**
  📄 [papers/Parcae-Scaling-Laws-For-Stable-Looped-Language-Models.pdf](papers/Parcae-Scaling-Laws-For-Stable-Looped-Language-Models.pdf)

### 1.4 Hybrid and Emerging Architectures
- [ ] **Hybrid Architectures for Language Models: Systematic Analysis and Design Insights**
  📄 [papers/Hybrid-Architectures-for-Language-Models-Systematic-Analysis-and-Design-Insights.pdf](papers/Hybrid-Architectures-for-Language-Models-Systematic-Analysis-and-Design-Insights.pdf)
  - → Mamba Section 8.1
  - → RWKV Section 8.1
- [ ] **Speed Always Wins: A Survey on Efficient Architectures for Large Language Models**
  📄 [papers/Speed-Always-Wins-A-Survey-on-Efficient-Architectures-for-Large-Language-Models.pdf](papers/Speed-Always-Wins-A-Survey-on-Efficient-Architectures-for-Large-Language-Models.pdf)
- [ ] **Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures**
  📄 [papers/Unlocking-the-Potential-of-Generative-AI-through-Neuro-Symbolic-Architectures-Benefits-and-Limitations.pdf](papers/Unlocking-the-Potential-of-Generative-AI-through-Neuro-Symbolic-Architectures-Benefits-and-Limitations.pdf)

---

## 2. Foundation Models and Technical Reports

### 2.1 OpenAI
- [ ] **GPT-4 Technical Report** → Section 1.1
- [ ] **GPT-4o System Card** → Section 1.1
- [ ] **InstructGPT** → RLHF Section 6
  📄 [papers/InstructGPT Training language models to follow instructions with human feedback.pdf](papers/InstructGPT-Training-language-models-to-follow-instructions-with-human-feedback.pdf)

### 2.2 Meta Llama Family
- [ ] **LLaMA: Open and Efficient Foundation Language Models** Touvron et al. 2023
  📄 [papers/LLaMA  Open and Efficient Foundation Language Models.pdf](papers/LLaMA-Open-and-Efficient-Foundation-Language-Models.pdf)
- [ ] **Llama 2: Open Foundation and Fine-Tuned Chat Models**
  📄 [papers/Llama 2  Open Foundation and Fine-Tuned Chat Models.pdf](papers/Llama-2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf)
- [ ] **THE UNIQUENESS OF LLAMA3-70B SERIES**
  📄 [papers/THE UNIQUENESS OF LLAMA3-70B SERIES WITH.pdf](papers/THE-UNIQUENESS-OF-LLAMA3-70B-SERIES-WITH.pdf)

### 2.3 DeepSeek
- [ ] **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism**
  📄 [papers/DeepSeek LLM Scaling Open-Source Language Models with Longtermism.pdf](papers/DeepSeek-LLM-Scaling-Open-Source-Language-Models-with-Longtermism.pdf)
- [ ] **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
  📄 [papers/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.pdf](papers/DeepSeek-R1-Incentivizing-Reasoning-Capability-in-LLMs-via-Reinforcement-Learning.pdf)
  - ← GRPO Section 6.3
  - → DeepSeek-V3.2
- [ ] **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models**
  📄 [papers/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models.pdf](papers/DeepSeek-V3.2-Pushing-the-Frontier-of-Open-Large-Language-Models.pdf)
- [ ] **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**
  📄 [papers/DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.pdf](papers/DeepSeek-V4-Towards-Highly-Efficient-Million-Token-Context-Intelligence.pdf)
- [ ] **A Review of DeepSeek Models: Key Innovative Techniques**
  📄 [papers/A Review of DeepSeek Models Key Innovative Techniques.pdf](papers/A-Review-of-DeepSeek-Models-Key-Innovative-Techniques.pdf)

### 2.4 Surveys and Landscape
- [ ] **Large Language Models: A Survey**
  📄 [papers/Large-Language-Models-A-Survey.pdf](papers/Large-Language-Models-A-Survey.pdf)
- [ ] **Foundations of Large Language Models**
  📄 [papers/Foundations-of-Large-Language-Models.pdf](papers/Foundations-of-Large-Language-Models.pdf)
- [ ] **A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge, Privacy and Trustworthiness**
  📄 [papers/A-Survey-on-Collaborating-Small-and-Large-Language-Models-for-Performance-Cost-effectiveness-Cloud-edge-Privacy-and-Trustworthiness.pdf](papers/A-Survey-on-Collaborating-Small-and-Large-Language-Models-for-Performance-Cost-effectiveness-Cloud-edge-Privacy-and-Trustworthiness.pdf)

### 2.5 Other Frontier Models
- [ ] **Mistral 7B**
  📄 [papers/Mistral 7B.pdf](papers/Mistral-7B.pdf)
- [ ] **Baichuan 2: Open Large-scale Language Models**
  📄 [papers/Baichuan 2: Open Large-scale Language Models.pdf](papers/Baichuan-2-Open-Large-scale-Language-Models.pdf)
- [ ] **Gemini 2.5: Pushing the Frontier with Advanced Reasoning**
  📄 [papers/Gemini 2.5 Pushing the Frontier with Advanced Reasoning Multimodality Long Context and Next Generation Agentic Capabilities.pdf](papers/Gemini-2.5-Pushing-the-Frontier-with-Advanced-Reasoning-Multimodality-Long-Context-and-Next-Generation-Agentic-Capabilities.pdf)
- [ ] **Qwen3-Coder-Next Technical Report**
  📄 [papers/Qwen3-Coder-Next-Technical-Report.pdf](papers/Qwen3-Coder-Next-Technical-Report.pdf)
- [ ] **Qwen3.5-Omni Technical Report**
  📄 [papers/Qwen3.5-Omni-Technical-Report.pdf](papers/Qwen3.5-Omni-Technical-Report.pdf)
- [ ] **Qwen-Image Technical Report**
  📄 [papers/Qwen-Image-Technical-Report.pdf](papers/Qwen-Image-Technical-Report.pdf)
- [ ] **ERNIE 5.0 Technical Report**
  📄 [papers/ERNIE-5.0-Technical-Report.pdf](papers/ERNIE-5.0-Technical-Report.pdf)
- [ ] **GLM-4.5 Agentic Reasoning and Coding ARC Foundation Models**
  📄 [papers/GLM-4.5-Agentic-Reasoning-and-Coding-ARC-Foundation-Models.pdf](papers/GLM-4.5-Agentic-Reasoning-and-Coding-ARC-Foundation-Models.pdf)
- [ ] **GLM-5: from Vibe Coding to Agentic Engineering**
  📄 [papers/GLM-5: from Vibe Coding to Agentic Engineering.pdf](papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering.pdf)
- [ ] **GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents**
  📄 [papers/GLM-5V-Turbo-Toward-a-Native-Foundation-Model-for-Multimodal-Agents.pdf](papers/GLM-5V-Turbo-Toward-a-Native-Foundation-Model-for-Multimodal-Agents.pdf)
- [ ] **LongCat-Flash-Thinking-2601 Technical Report**
  📄 [papers/LongCat-Flash-Thinking-2601-Technical-Report.pdf](papers/LongCat-Flash-Thinking-2601-Technical-Report.pdf)
- [ ] **Intern-S1: A Scientific Multimodal Foundation Model**
  📄 [papers/Intern-S1: A Scientific Multimodal Foundation Model.pdf](papers/Intern-S1-A-Scientific-Multimodal-Foundation-Model.pdf)

### 2.5 Survey of Models
- [ ] **A Survey of Large Language Models**
  📄 [papers/A-Survey-of-Large-Language-Models.pdf](papers/A-Survey-of-Large-Language-Models.pdf)
- [ ] **Large Language Models: A Survey**
  📄 [papers/Agentic-Reasoning-for-Large-Language-Models.pdf](papers/Agentic-Reasoning-for-Large-Language-Models.pdf)
- [ ] **Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond**
  📄 [papers/Harnessing-the-Power-of-LLMs-in-Practice-A-Survey-on-ChatGPT-and-Beyond.pdf](papers/Harnessing-the-Power-of-LLMs-in-Practice-A-Survey-on-ChatGPT-and-Beyond.pdf)
- [ ] **A Survey of Small Language Models**
  📄 [papers/A-Survey-of-Small-Language-Models.pdf](papers/A-Survey-of-Small-Language-Models.pdf)
- [ ] **Foundations of Large Language Models**
  📄 [papers/Foundations of Large Language Models.pdf](papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf)
- [ ] **LLMs4All: A Systematic Review of Large Language Models Across Academic Disciplines**
  📄 [papers/LLMs4All-A-Systematic-Review-of-Large-Language-Models-Across-Academic-Disciplines.pdf](papers/LLMs4All-A-Systematic-Review-of-Large-Language-Models-Across-Academic-Disciplines.pdf)

---

## 3. Scaling and Training Theory

### 3.1 Scaling Laws
- [ ] **Scaling Laws for Neural Language Models** Kaplan et al. 2020
  📄 [papers/Scaling-Laws-for-Neural-Language-Models.pdf](papers/Scaling-Laws-for-Neural-Language-Models.pdf)
  - → Chinchilla
- [ ] **Model Merging Scaling Laws in Large Language Models**
  📄 [papers/Model-Merging-Scaling-Laws-in-Large-Language-Models.pdf](papers/Model-Merging-Scaling-Laws-in-Large-Language-Models.pdf)

### 3.2 Optimization and Learning Dynamics
- [ ] **How Learning Rates Regulate Catastrophic Overtraining.pdf**
  📄 [papers/How-Learning-Rates-Regulate-Catastrophic-Overtraining.pdf](papers/How-Learning-Rates-Regulate-Catastrophic-Overtraining.pdf)
- [ ] **Adam's Law: Textual Frequency Law on Large Language Models**
  📄 [papers/Adam's-Law-Textual-Frequency-Law-on-Large-Language-Models.pdf](papers/Adam's-Law-Textual-Frequency-Law-on-Large-Language-Models.pdf)
- [ ] **Transformers Learn Faster with Semantic Focus**
  📄 [papers/Transformers-Learn-Faster-with-Semantic-Focus.pdf](papers/Transformers-Learn-Faster-with-Semantic-Focus.pdf)
- [ ] **The Geometry of Consolidation**
  📄 [papers/The-Geometry-of-Consolidation.pdf](papers/The-Geometry-of-Consolidation.pdf)
- [ ] **Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training**
  📄 [papers/Why-Diffusion-Models-Don’t-Memorize-The-Role-of-Implicit-Dynamical-Regularization-in-Training.pdf](papers/Why-Diffusion-Models-Don’t-Memorize-The-Role-of-Implicit-Dynamical-Regularization-in-Training.pdf)
- [ ] **Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning**
  📄 [papers/Logical-Phase-Transitions-Understanding-Collapse-in-LLM-Logical-Reasoning.pdf](papers/Logical-Phase-Transitions-Understanding-Collapse-in-LLM-Logical-Reasoning.pdf)

### 3.3 Training Efficiency
- [ ] **Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation**
  📄 [papers/EveryActivation-Boosted-Scaling-General-Reasoner-to-1-Trillion-Open-Language-Foundation.pdf](papers/EveryActivation-Boosted-Scaling-General-Reasoner-to-1-Trillion-Open-Language-Foundation.pdf)
- [ ] **Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca**
  📄 [papers/Efficient-and-Effective-Text-Encoding-for-Chinese-LLaMA-and-Alpaca.pdf](papers/Efficient-and-Effective-Text-Encoding-for-Chinese-LLaMA-and-Alpaca.pdf)
- [ ] **Training-Inference Consistent Segmented Execution for Long-Context LLMs**
  📄 [papers/Training-Inference-Consistent-Segmented-Execution-for-Long-Context-LLMs.pdf](papers/Training-Inference-Consistent-Segmented-Execution-for-Long-Context-LLMs.pdf)

---

## 4. Distributed Training and Systems

### 4.1 GPU Kernels and Acceleration
- [ ] **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** Dao et al. 2022
  📄 [papers/FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf](papers/FlashAttention-Fast-and-Memory-Efficient-Exact-Attention-with-IO-Awareness.pdf)
  - → FlashAttention-2 FlashAttention-3
- [ ] **FlashAttention alt version**
  📄 [papers/FlashAttention Fast and Memory-Efficient Exact Attention.pdf](papers/FlashAttention-Fast-and-Memory-Efficient-Exact-Attention-with-IO-Awareness.pdf)
- [ ] **AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization**
  📄 [papers/AccelOpt A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization.pdf](papers/AccelOpt-A-Self-Improving-LLM-Agentic-System-for-AI-Accelerator-Kernel-Optimization.pdf)

### 4.2 Distributed Training
- [ ] **Distributed-Training-I.pdf** course slides
  📄 [papers/Distributed-Training-I.pdf](papers/Distributed-Training-I.pdf)
- [ ] **Distributed-Training-II.pdf** course slides
  📄 [papers/Distributed-Training-II.pdf](papers/Distributed-Training-II.pdf)
- [ ] **LLM Post-training_Part1.pdf**
  📄 [papers/LLM-Post-training_Part1.pdf](papers/LLM-Post-training_Part1.pdf)
- [ ] **LLM Post-training_Part2.pdf**
  📄 [papers/LLM-Post-training_Part2.pdf](papers/LLM-Post-training_Part2.pdf)

### 4.3 On-Device and Edge
- [ ] **On-Device-Training-And-Transfer-Learning.pdf**
  📄 [papers/On-Device-Training-And-Transfer-Learning.pdf](papers/On-Device-Training-And-Transfer-Learning.pdf)

---

## 5. Quantization and Inference Optimization

### 5.1 Quantization Techniques
- [ ] **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
  📄 [papers/AWQ Activation-aware Weight Quantization for LLM Compression and Acceleration.pdf](papers/AWQ-Activation-aware-Weight-Quantization-for-LLM-Compression-and-Acceleration.pdf)
- [ ] **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**
  📄 [papers/SmoothQuant Accurate and Efficient Post-Training Quantization for Large Language Models.pdf](papers/SmoothQuant-Accurate-and-Efficient-Post-Training-Quantization-for-Large-Language-Models.pdf)
- [ ] **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models**
  📄 [papers/EfficientQAT Efficient Quantization-Aware Training for Large Language Models.pdf](papers/EfficientQAT-Efficient-Quantization-Aware-Training-for-Large-Language-Models.pdf)
- [ ] **BitNet a4.8: 4-bit Activations for 1-bit LLMs**
  📄 [papers/BitNet a4.8 4-bit Activations for 1-bit LLMs.pdf](papers/BitNet-a4.8-4-bit-Activations-for-1-bit-LLMs.pdf)
- [ ] **A Comprehensive Study on Quantization Techniques for Large Language Models**
  📄 [papers/A Comprehensive Study on Quantization Techniques for Large Language Models.pdf](papers/A-Comprehensive-Study-on-Quantization-Techniques-for-Large-Language-Models.pdf)
- [ ] **Quantization-I.pdf** course slides
  📄 [papers/Quantization-I.pdf](papers/Quantization-I.pdf)
- [ ] **Quantization-II.pdf** course slides
  📄 [papers/Quantization-II.pdf](papers/Quantization-II.pdf)
- [ ] **FlattenQuant: Breaking Through the Inference Compute-bound for LLMs with Per-tensor Quantization**
  📄 [papers/FlattenQuant-Breaking-Through-the-Inference-Compute-bound-for-Large-Language-Models-with-Per-tensor-Quantization.pdf](papers/FlattenQuant-Breaking-Through-the-Inference-Compute-bound-for-Large-Language-Models-with-Per-tensor-Quantization.pdf)
- [ ] **FrameQuant: Flexible Low-Bit Quantization for Transformers**
  📄 [papers/FrameQuant-Flexible-Low-Bit-Quantization-for-Transformers.pdf](papers/FrameQuant-Flexible-Low-Bit-Quantization-for-Transformers.pdf)
- [ ] **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
  📄 [papers/TurboQuant-Online-Vector-Quantization-with-Near-optimal-Distortion-Rate.pdf](papers/TurboQuant-Online-Vector-Quantization-with-Near-optimal-Distortion-Rate.pdf)
- [ ] **LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference**
  📄 [papers/LUT-GEMM-Quantized-Matrix-Multiplication-based-on-LUTs-for-Efficient-Inference-in-Large-Scale-Generative-Language-Models.pdf](papers/LUT-GEMM-Quantized-Matrix-Multiplication-based-on-LUTs-for-Efficient-Inference-in-Large-Scale-Generative-Language-Models.pdf)
- [ ] **INT vs FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats**
  📄 [papers/INT-vs-FP-A-Comprehensive-Study-of-Fine-Grained-Low-bit-Quantization-Formats.pdf](papers/INT-vs-FP-A-Comprehensive-Study-of-Fine-Grained-Low-bit-Quantization-Formats.pdf)
- [ ] **Optimal Quantization for Matrix Multiplication**
  📄 [papers/Optimal-Quantization-for-Matrix-Multiplication.pdf](papers/Optimal-Quantization-for-Matrix-Multiplication.pdf)
- [ ] **IGQ-ViT: Instance-Aware Group Quantization for Vision Transformers**
  📄 [papers/Instance-Aware-Group-Quantization-for-Vision-Transformers.pdf](papers/Instance-Aware-Group-Quantization-for-Vision-Transformers.pdf)

### 5.2 Inference Systems
- [ ] **LLM-Deployment.pdf**
  📄 [papers/LLM-Deployment.pdf](papers/LLM-Deployment.pdf)
- [ ] **LLMInference Enhanced by External Knowledge: A Survey**
  📄 [papers/LLMInference-Enhanced-by-External-Knowledge-A-Survey.pdf](papers/LLMInference-Enhanced-by-External-Knowledge-A-Survey.pdf)
- [ ] **Tutorial Proposal: Efficient Inference for Large Language Models**
  📄 [papers/Tutorial-Proposal-Efficient-Inference-for-Large-Language-Models.pdf](papers/Tutorial-Proposal-Efficient-Inference-for-Large-Language-Models.pdf)
- [ ] **Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference**
  📄 [papers/Seed-Diffusion-A-Large-Scale-Diffusion-Language-Model-with-High-Speed-Inference.pdf](papers/Seed-Diffusion-A-Large-Scale-Diffusion-Language-Model-with-High-Speed-Inference.pdf)

### 5.3 Knowledge Distillation
- [ ] **Knowledge-Distillation.pdf**
  📄 [papers/Knowledge-Distillation.pdf](papers/Knowledge-Distillation.pdf)
- [ ] **Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends Challenges and Future Directions**
  📄 [papers/Knowledge-Distillation-and-Dataset-Distillation-of-Large-Language-Models-Emerging-Trends-Challenges-and-Future-Directions.pdf](papers/Knowledge-Distillation-and-Dataset-Distillation-of-Large-Language-Models-Emerging-Trends-Challenges-and-Future-Directions.pdf)
- [ ] **A Study on Hidden Layer Distillation for Large Language Model Pre-Training**
  📄 [papers/A-Study-on-Hidden-Layer-Distillation-for-Large-Language-Model-Pre-Training.pdf](papers/A-Study-on-Hidden-Layer-Distillation-for-Large-Language-Model-Pre-Training.pdf)
- [ ] **Self-Distillation Enables Continual Learning**
  📄 [papers/Self-Distillation-Enables-Continual-Learning.pdf](papers/Self-Distillation-Enables-Continual-Learning.pdf)
- [ ] **Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision**
  📄 [papers/Self-Distillation-Zero-Self-Revision-Turns-Binary-Rewards-into-Dense-Supervision.pdf](papers/Self-Distillation-Zero-Self-Revision-Turns-Binary-Rewards-into-Dense-Supervision.pdf)

### 5.4 Long Context
- [ ] **Long-Context LLM.pdf**
  📄 [papers/Long-Context-LLM.pdf](papers/Long-Context-LLM.pdf)
- [ ] **Long Context RAG Performance of Large Language Models**
  📄 [papers/Long-Context-RAG-Performance-of-Large-Language.pdf](papers/Long-Context-RAG-Performance-of-Large-Language.pdf)

---

## 6. Post-Training and Alignment

### 6.1 RLHF Fundamentals
- [ ] **Deep Reinforcement Learning from Human Preferences** Christiano et al. 2017
  📄 [papers/RLHF-Deep-Reinforcement-Learning-from-Human-Preferences.pdf](papers/RLHF-Deep-Reinforcement-Learning-from-Human-Preferences.pdf)
- [ ] **Learning to Summarize from Human Feedback** Stiennon et al. 2020
  📄 [papers/RLHF-Learning-to-summarize-from-human-feedback.pdf](papers/RLHF-Learning-to-summarize-from-human-feedback.pdf)
- [ ] **InstructGPT: Training language models to follow instructions with human feedback** Ouyang et al. 2022
  📄 [papers/InstructGPT Training language models to follow instructions with human feedback.pdf](papers/InstructGPT-Training-language-models-to-follow-instructions-with-human-feedback.pdf)
- [ ] **A Survey of Reinforcement Learning from Human Feedback**
  📄 [papers/A Survey of Reinforcement Learning from Human Feedback.pdf](papers/A-Survey-of-Reinforcement-Learning-from-Human-Feedback.pdf)

### 6.2 DPO and Variants
- [ ] **A Survey of Direct Preference Optimization**
  📄 [papers/A Survey of Direct Preference Optimization.pdf](papers/A-Survey-of-Direct-Preference-Optimization.pdf)
- [ ] **A Semantically-Aware Kernel-Enhanced and DivergenceRich Paradigm for DPO**
  📄 [papers/A Semantically-Aware Kernel-Enhanced and DivergenceRich Paradigm for Direct Preference Optimization.pdf](papers/A-Semantically-Aware-Kernel-Enhanced-and-DivergenceRich-Paradigm-for-Direct-Preference-Optimization.pdf)
- [ ] **It Takes Two: Your GRPO Is Secretly DPO**
  📄 [papers/It Takes Two Your GRPO Is Secretly DPO.pdf](papers/It-Takes-Two-Your-GRPO-Is-Secretly-DPO.pdf)
  - ← GRPO Section 6.3 DPO Section 6.2
- [ ] **DEPO: Dual Efficiency Preference Optimization for LLM Agents**
  📄 [papers/DEPO-Dual-Efficiency-Preference-Optimization-for-LLM-Agents.pdf](papers/DEPO-Dual-Efficiency-Preference-Optimization-for-LLM-Agents.pdf)
- [ ] **VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training**
  📄 [papers/VESPO-Variational-Sequence-Level-Soft-Policy-Optimization-for-Stable-Off-Policy-LLM-Training.pdf](papers/VESPO-Variational-Sequence-Level-Soft-Policy-Optimization-for-Stable-Off-Policy-LLM-Training.pdf)

### 6.3 GRPO and Reasoning RL
- [ ] **Proximal Policy Optimization Algorithms** Schulman et al. 2017
  📄 [papers/Proximal Policy Optimization Algorithms.pdf](papers/Proximal-Policy-Optimization-Algorithms.pdf)
- [ ] **Guided Exploration with Proximal Policy Optimization using a Single Demonstration**
  📄 [papers/Guided-Exploration-with-Proximal-Policy-Optimization-using-a-Single.pdf](papers/Guided-Exploration-with-Proximal-Policy-Optimization-using-a-Single.pdf)
- [ ] **Learning to Hint for Reinforcement Learning**
  📄 [papers/Learning-to-Hint-for-Reinforcement-Learning.pdf](papers/Learning-to-Hint-for-Reinforcement-Learning.pdf)
- [ ] **1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities**
  📄 [papers/1000-Layer-Networks-for-Self-Supervised-RL-Scaling-Depth-Can-Enable-New-Goal-Reaching-Capabilities.pdf](papers/1000-Layer-Networks-for-Self-Supervised-RL-Scaling-Depth-Can-Enable-New-Goal-Reaching-Capabilities.pdf)
  - → DeepSeek-R1 GRPO variants
- [ ] **DeepSeek-R1** — GRPO in production → Section 2.3
  📄 [papers/DeepSeek-R1.pdf](papers/DeepSeek-R1.pdf)
- [ ] **FlowRL: Matching Reward Distributions for LLM Reasoning**
  📄 [papers/FlowRL Matching Reward Distributions for LLM Reasoning.pdf](papers/FlowRL-Matching-Reward-Distributions-for-LLM-Reasoning.pdf)
- [ ] **Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing**
  📄 [papers/Sharing is Caring Efficient LM Post-Training with Collective RL Experience Sharing.pdf](papers/Sharing-is-Caring-Efficient-LM-Post-Training-with-Collective-RL-Experience-Sharing.pdf)

### 6.4 Reward and Feedback
- [ ] **Reinforcement Learning with Human Feedback: Learning Dynamic Choices via Pessimism**
  📄 [papers/Reinforcement-Learning-with-Human-Feedback-Learning-Dynamic-Choices-via-Pessimism.pdf](papers/Reinforcement-Learning-with-Human-Feedback-Learning-Dynamic-Choices-via-Pessimism.pdf)
- [ ] **Reinforcement Pre-Training**
  📄 [papers/Reinforcement-Pre-Training.pdf](papers/Reinforcement-Pre-Training.pdf)
- [ ] **Efficiently Aligning Language Models with Online Natural Language Feedback**
  📄 [papers/Efficiently-Aligning-Language-Models-with-Online-Natural-Language-Feedback.pdf](papers/Efficiently-Aligning-Language-Models-with-Online-Natural-Language-Feedback.pdf)
- [ ] **Asking Clarifying Questions for Preference Elicitation With Large Language Models**
  📄 [papers/Asking-Clarifying-Questions-for-Preference-Elicitation-With-Large-Language-Models.pdf](papers/Asking-Clarifying-Questions-for-Preference-Elicitation-With-Large-Language-Models.pdf)
- [ ] **Generalized Distributional Alignment Games for Unbiased Answer-Level Fine-Tuning**
  📄 [papers/Generalized-Distributional-Alignment-Games-for-Unbiased-Answer-Level-Fine-Tuning.pdf](papers/Generalized-Distributional-Alignment-Games-for-Unbiased-Answer-Level-Fine-Tuning.pdf)
- [ ] **References Improve LLM Alignment in Non-Verifiable Domains**
  📄 [papers/References-Improve-LLM-Alignment-in-Non-Verifiable-Domains.pdf](papers/References-Improve-LLM-Alignment-in-Non-Verifiable-Domains.pdf)

### 6.5 Alignment and Safety
- [ ] **Language Models Resist Alignment: Evidence From Data Compression**
  📄 [papers/Language Models Resist Alignment: Evidence From Data Compression.pdf](papers/Language-Models-Resist-Alignment-Evidence-From-Data-Compression.pdf)
- [ ] **Model Spec Midtraining: Improving How Alignment Training Generalizes**
  📄 [papers/Model-Spec-Midtraining-Improving-How-Alignment-Training-Generalizes.pdf](papers/Model-Spec-Midtraining-Improving-How-Alignment-Training-Generalizes.pdf)
- [ ] **How does Alignment Enhance LLMs Multilingual Capabilities: A Language Neurons Perspective**
  📄 [papers/How-does-Alignment-Enhance-LLMs-Multilingual-Capabilities-A-Language-Neurons-Perspective.pdf](papers/How-does-Alignment-Enhance-LLMs-Multilingual-Capabilities-A-Language-Neurons-Perspective.pdf)
- [ ] **Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger**
  📄 [papers/Weak-Driven-Learning-How-Weak-Agents-make-Strong-Agents-Stronger.pdf](papers/Weak-Driven-Learning-How-Weak-Agents-make-Strong-Agents-Stronger.pdf)

---

## 7. Fine-Tuning and Parameter-Efficient Methods

### 7.1 LoRA and PEFT
- [ ] **LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS** Hu et al. 2021
  📄 [papers/LORA LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS.pdf](papers/LORA-LOW-RANK-ADAPTATION-OF-LARGE-LANGUAGE-MODELS.pdf)
  - → QLoRA AdaLoRA DoRA SingLoRA
- [ ] **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**
  📄 [papers/AdaLoRA Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.pdf](papers/AdaLoRA-Adaptive-Budget-Allocation-for-Parameter-Efficient-Fine-Tuning.pdf)
- [ ] **SingLoRA: Low Rank Adaptation Using a Single Matrix**
  📄 [papers/SingLoRA Low Rank Adaptation Using a Single Matrix.pdf](papers/SingLoRA-Low-Rank-Adaptation-Using-a-Single-Matrix.pdf)
- [ ] **Text to LoRA: Instant Transformer Adaption**
  📄 [papers/Text-to-LoRA-Instant-Transformer-Adaption.pdf](papers/Text-to-LoRA-Instant-Transformer-Adaption.pdf)
- [ ] **Low-Rank Adaptation for Foundation Models: A Comprehensive Review**
  📄 [papers/Low-Rank Adaptation for Foundation Models A Comprehensive Review.pdf](papers/Low-Rank-Adaptation-for-Foundation-Models-A-Comprehensive-Review.pdf)
- [ ] **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**
  📄 [papers/Parameter-Efficient Fine-Tuning for Large Models A Comprehensive Survey.pdf](papers/Parameter-Efficient-Fine-Tuning-for-Large-Models-A-Comprehensive-Survey.pdf)
- [ ] **Parameter-Efficient Transfer Learning for NLP**
  📄 [papers/Parameter-Efficient-Transfer-Learning-for-NLP.pdf](papers/Parameter-Efficient-Transfer-Learning-for-NLP.pdf)

### 7.2 SFT and Instruction Tuning
- [ ] **Self-Instruct: Aligning Language Models with Self-Generated Instructions**
  📄 [papers/Self-Instruct-Aligning-Language-Models-with-Self-Generated-Instructions.pdf](papers/Self-Instruct-Aligning-Language-Models-with-Self-Generated-Instructions.pdf)
- [ ] **Instruction Tuning for Large Language Models: A Survey**
  📄 [papers/Instruction-Tuning-for-Large-Language-Models-A-Survey.pdf](papers/Instruction-Tuning-for-Large-Language-Models-A-Survey.pdf)
- [ ] **Prefix-Tuning: Optimizing Continuous Prompts for Generation**
  📄 [papers/Prefix-Tuning-Optimizing-Continuous-Prompts-for-Generation.pdf](papers/Prefix-Tuning-Optimizing-Continuous-Prompts-for-Generation.pdf)
- [ ] **Fine-tuning and Utilization Methods of Domain-specific LLMs**
  📄 [papers/Fine-tuning-and-Utilization-Methods-of-Domain-specific-LLMs.pdf](papers/Fine-tuning-and-Utilization-Methods-of-Domain-specific-LLMs.pdf)
- [ ] **The Ultimate Guide to Fine-Tuning LLMs**
  📄 [papers/The-Ultimate-Guide-to-Fine-Tuning-LLMs-from-Basics-to-Breakthroughs-An-Exhaustive-Review-of-Technologies-Research-Best-Practices-Applied-Research-Challenges-and-Opportunities.pdf](papers/The-Ultimate-Guide-to-Fine-Tuning-LLMs-from-Basics-to-Breakthroughs-An-Exhaustive-Review-of-Technologies-Research-Best-Practices-Applied-Research-Challenges-and-Opportunities.pdf)
- [ ] **Rethinking Generalization in Reasoning SFT**
  📄 [papers/Rethinking-Generalization-in-Reasoning-SFT-A-Conditional-Analysis-on-Optimization-Data-and-Model-Capability.pdf](papers/Rethinking-Generalization-in-Reasoning-SFT-A-Conditional-Analysis-on-Optimization-Data-and-Model-Capability.pdf)
- [ ] **On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification**
  📄 [papers/On-the-Generalization-of-SFT-A-Reinforcement-Learning-Perspective-with-Reward-Rectification.pdf](papers/On-the-Generalization-of-SFT-A-Reinforcement-Learning-Perspective-with-Reward-Rectification.pdf)
- [ ] **RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response**
  📄 [papers/RobustFT-Robust-Supervised-Fine-tuning-for-Large-Language-Models-under-Noisy-Response.pdf](papers/RobustFT-Robust-Supervised-Fine-tuning-for-Large-Language-Models-under-Noisy-Response.pdf)
- [ ] **On the Limits of LLM Adaptability: Impact of Model-Internalized Priors**
  📄 [papers/On-the-Limits-of-LLM-Adaptability-Impact-of-Model-Internalized-Priors-on-Annotation-Task-Performance.pdf](papers/On-the-Limits-of-LLM-Adaptability-Impact-of-Model-Internalized-Priors-on-Annotation-Task-Performance.pdf)
- [ ] **You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model**
  📄 [papers/You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Model.pdf](papers/You-Only-Fine-tune-Once-Many-Shot-In-Context-Fine-Tuning-for-Large-Language-Model.pdf)
- [ ] **DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models**
  📄 [papers/DataFlex-A-Unified-Framework-for-Data-Centric-Dynamic-Training-of-Large-Language-Models.pdf](papers/DataFlex-A-Unified-Framework-for-Data-Centric-Dynamic-Training-of-Large-Language-Models.pdf)

### 7.3 Continual Learning and Evolution
- [ ] **How Do Large Language Models Learn Concepts During Continual Pre-Training**
  📄 [papers/How-Do-Large-Language-Models-Learn-Concepts-During-Continual-Pre-Training.pdf](papers/How-Do-Large-Language-Models-Learn-Concepts-During-Continual-Pre-Training.pdf)
- [ ] **Evolving Language Models without Labels: Majority Drives Selection Novelty Promotes Variation**
  📄 [papers/Evolving Language Models without Labels: Majority Drives Selection Novelty Promotes Variation.pdf](papers/Evolving-Language-Models-without-Labels-Majority-Drives-Selection-Novelty-Promotes-Variation.pdf)

---

## 8. Novel Architectures

### 8.1 State Space Models and Alternatives
- [ ] **ELF: Embedded Language Flows**
  📄 [papers/ELF-Embedded-Language-Flows.pdf](papers/ELF-Embedded-Language-Flows.pdf)
- [ ] **A Powerful xLSTM-based Method for Anomaly Detection**
  📄 [papers/A-Powerful-xLSTM-based-Method-for-Anomaly-Detection.pdf](papers/A-Powerful-xLSTM-based-Method-for-Anomaly-Detection.pdf)
- [ ] **Titans: Learning to Memorize at Test Time**
  📄 [papers/Titans-Learning-to-Memorize-at-Test-Time.pdf](papers/Titans-Learning-to-Memorize-at-Test-Time.pdf)
  - → Agent Memory Section 15

### 8.2 Diffusion and Generation
- [ ] **Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference**
  📄 [papers/Seed-Diffusion-A-Large-Scale-Diffusion-Language-Model-with-High-Speed-Inference.pdf](papers/Seed-Diffusion-A-Large-Scale-Diffusion-Language-Model-with-High-Speed-Inference.pdf)
- [ ] **Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations**
  📄 [papers/Generation-Space-Size-Understanding-and-Calibrating-Open-Endedness-of-LLM-Generations.pdf](papers/Generation-Space-Size-Understanding-and-Calibrating-Open-Endedness-of-LLM-Generations.pdf)

---

## 9. Mixture of Experts MoE

- [ ] **A Comprehensive Survey of Mixture-of-Experts: Algorithms Theory and Applications**
  📄 [papers/A-Comprehensive-Survey-of-Mixture-of-Experts-Algorithms-Theory-and-Applications.pdf](papers/A-Comprehensive-Survey-of-Mixture-of-Experts-Algorithms-Theory-and-Applications.pdf)
- [ ] **LIBMoE: A Library for Comprehensive Benchmarking Mixture of Experts in Large Language Models**
  📄 [papers/LIBMoE-A-Library-for-comprehensive-benchmarking-Mixture-of-Experts-in-Large-Language-Models.pdf](papers/LIBMoE-A-Library-for-comprehensive-benchmarking-Mixture-of-Experts-in-Large-Language-Models.pdf)
- [ ] **Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts**
  📄 [papers/Geometric-and-Stochastic-Analysis-of-Discontinuities-in-Sparse-Mixture-of-Experts.pdf](papers/Geometric-and-Stochastic-Analysis-of-Discontinuities-in-Sparse-Mixture-of-Experts.pdf)
- [ ] **MoE in DeepSeek-V3** → **[DeepSeek-V3.2](papers/DeepSeek-V3.2-Pushing-the-Frontier-of-Open-Large-Language-Models.pdf)** (Section 2.3)
- [ ] **Multi-expert Prompting Improves Reliability Safety and Usefulness of Large Language Models**
  📄 [papers/Multi-expert-Prompting-Improves-Reliability-Safety-and-Usefulness-of-Large-Language-Models.pdf](papers/Multi-expert-Prompting-Improves-Reliability-Safety-and-Usefulness-of-Large-Language-Models.pdf)

---

## 10. RAG and Retrieval

### 10.1 RAG Foundations
- [ ] **Self-RAG: Learning to Retrieve Generate and Critique through Self-Reflection**
  📄 [papers/Self-RAG - Learning to Retrieve  Generate  and Critique through Self-Reflection.pdf](papers/Self-RAG-Learning-to-Retrieve-Generate-and-Critique-through-Self-Reflection.pdf)
- [ ] **Corrective Retrieval Augmented Generation**
  📄 [papers/Corrective Retrieval Augmented Generation.pdf](papers/Corrective-Retrieval-Augmented-Generation.pdf)
- [ ] **ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems**
  📄 [papers/ChunkRAG Novel LLM-Chunk Filtering Method for RAG Systems.pdf](papers/ChunkRAG-Novel-LLM-Chunk-Filtering-Method-for-RAG-Systems.pdf)
- [ ] **AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline**
  📄 [papers/AutoRAG-Automated-Framework-for-optimization-of-Retrieval-Augmented-Generation-Pipeline.pdf](papers/AutoRAG-Automated-Framework-for-optimization-of-Retrieval-Augmented-Generation-Pipeline.pdf)
- [ ] **REFRAG: Rethinking RAG-based Decoding**
  📄 [papers/REFRAG-Rethinking-RAG-based-Decoding.pdf](papers/REFRAG-Rethinking-RAG-based-Decoding.pdf)
- [ ] **RAG-Fusion: a New Take on Retrieval Augmented Generation**
  📄 [papers/RAG-Fusion-a-New-Take-on-Retrieval-Augmented-Generation.pdf](papers/RAG-Fusion-a-New-Take-on-Retrieval-Augmented-Generation.pdf)
- [ ] **RAG from scratch — Overview**
  📄 [papers/RAG-from-scratch--Overview.pdf](papers/RAG-from-scratch--Overview.pdf)

### 10.2 Retrieval and Search
- [ ] **Query Rewriting for Retrieval-Augmented Large Language Models**
  📄 [papers/Query-Rewriting-for-Retrieval-Augmented-Large-Language-Models.pdf](papers/Query-Rewriting-for-Retrieval-Augmented-Large-Language-Models.pdf)
- [ ] **Query Expansion by Prompting Large Language Models**
  📄 [papers/Query-Expansion-by-Prompting-Large-Language-Models.pdf](papers/Query-Expansion-by-Prompting-Large-Language-Models.pdf)
- [ ] **Precise Zero-Shot Dense Retrieval without Relevance Labels**
  📄 [papers/Precise-Zero-Shot-Dense-Retrieval-without-Relevance-Labels.pdf](papers/Precise-Zero-Shot-Dense-Retrieval-without-Relevance-Labels.pdf)
- [ ] **A Survey of Long-Document Retrieval in the PLM and LLM Era**
  📄 [papers/A-Survey-of-Long-Document-Retrieval-in-the-PLM-and-LLM-Era.pdf](papers/A-Survey-of-Long-Document-Retrieval-in-the-PLM-and-LLM-Era.pdf)
- [ ] **ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability**
  📄 [papers/ReasonRank-Empowering-Passage-Ranking-with-Strong-Reasoning-Ability.pdf](papers/ReasonRank-Empowering-Passage-Ranking-with-Strong-Reasoning-Ability.pdf)

### 10.3 Advanced RAG
- [ ] **Synergizing RAG and Reasoning: A Systematic Review**
  📄 [papers/Synergizing-RAG-and-Reasoning-A-Systematic-Review.pdf](papers/Synergizing-RAG-and-Reasoning-A-Systematic-Review.pdf)
- [ ] **Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs**
  📄 [papers/Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs.pdf](papers/Towards-Agentic-RAG-with-Deep-Reasoning-A-Survey-of-RAG-Reasoning-Systems-in-LLMs.pdf)
- [ ] **Long Context RAG Performance of Large Language Models**
  📄 [papers/Long-Context-RAG-Performance-of-Large-Language.pdf](papers/Long-Context-RAG-Performance-of-Large-Language.pdf)
- [ ] **Retrieval Augmented Conversational Recommendation with Reinforcement Learning**
  📄 [papers/Retrieval-Augmented-Conversational-Recommendation-with-Reinforcement-Learning.pdf](papers/Retrieval-Augmented-Conversational-Recommendation-with-Reinforcement-Learning.pdf)
- [ ] **Rationale-Guided Retrieval Augmented Generation for Medical Question Answering**
  📄 [papers/Rationale-Guided-Retrieval-Augmented-Generation-for-Medical-Question-Answering.pdf](papers/Rationale-Guided-Retrieval-Augmented-Generation-for-Medical-Question-Answering.pdf)
- [ ] **Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation**
  📄 [papers/Mixture-of-Experts-Knowledge-Graph-Retrieval-Augmented-Generation-for-Multi-Agent-LLM-based-Recommendation.pdf](papers/Mixture-of-Experts-Knowledge-Graph-Retrieval-Augmented-Generation-for-Multi-Agent-LLM-based-Recommendation.pdf)

---

## 11. Reasoning
- [ ] **Aligned Orthogonal or In conflict When can we safely optimize Chain of Thought**
  📄 [papers/Aligned-Orthogonal-or-In-conflict-When-can-we-safely-optimize-Chain-of-Thought.pdf](papers/Aligned-Orthogonal-or-In-conflict-When-can-we-safely-optimize-Chain-of-Thought.pdf)
- [ ] **Chain of Thought Monitorability A New and Fragile Opportunity for AI Safety**
  📄 [papers/Chain-of-Thought-Monitorability-A-New-and-Fragile-Opportunity-for-AI-Safety.pdf](papers/Chain-of-Thought-Monitorability-A-New-and-Fragile-Opportunity-for-AI-Safety.pdf)
- [ ] **Focused Chain of Thought Efficient LLM Reasoning via Structured Input Information**
  📄 [papers/Focused-Chain-of-Thought-Efficient-LLM-Reasoning-via-Structured-Input-Information.pdf](papers/Focused-Chain-of-Thought-Efficient-LLM-Reasoning-via-Structured-Input-Information.pdf)
- [ ] **Likelihood Based Reward Designs for General LLM Reasoning**
  📄 [papers/Likelihood-Based-Reward-Designs-for-General-LLM-Reasoning.pdf](papers/Likelihood-Based-Reward-Designs-for-General-LLM-Reasoning.pdf)
- [ ] **Perception Reason Think and Plan A Survey on Large Multimodal Reasoning Models**
  📄 [papers/Perception-Reason-Think-and-Plan-A-Survey-on-Large-Multimodal-Reasoning-Models.pdf](papers/Perception-Reason-Think-and-Plan-A-Survey-on-Large-Multimodal-Reasoning-Models.pdf)
- [ ] **Procedural Knowledge at Scale Improves Reasoning**
  📄 [papers/Procedural-Knowledge-at-Scale-Improves-Reasoning.pdf](papers/Procedural-Knowledge-at-Scale-Improves-Reasoning.pdf)
- [ ] **REFRAG Rethinking RAG based Decoding**
  📄 [papers/REFRAG-Rethinking-RAG-based-Decoding.pdf](papers/REFRAG-Rethinking-RAG-based-Decoding.pdf)
- [ ] **Reverse Engineered Reasoning for Open Ended Generation**
  📄 [papers/Reverse-Engineered-Reasoning-for-Open-Ended-Generation.pdf](papers/Reverse-Engineered-Reasoning-for-Open-Ended-Generation.pdf)

### 11.1 Chain-of-Thought and Prompting
- [ ] **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** Wei et al. 2022
  📄 [papers/Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.pdf](papers/Chain-of-Thought-Prompting-Elicits-Reasoning-in-Large-Language-Models.pdf)
  - → Everything below
- [ ] **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**
  📄 [papers/Take a Step Back- Evoking Reasoning via Abstraction in Large Language Models.pdf](papers/Take-a-Step-Back-Evoking-Reasoning-via-Abstraction-in-Large-Language-Models.pdf)
- [ ] **Is Chain-of-Thought Reasoning of LLMs a Mirage**
  📄 [papers/Is Chain-of-Thought Reasoning of LLMs a Mirage A Data Distribution Lens.pdf](papers/Is-Chain-of-Thought-Reasoning-of-LLMs-a-Mirage-A-Data-Distribution-Lens.pdf)
- [ ] **Thoughts without Thinking: Reconsidering the Explanatory Value of Chain-of-Thought Reasoning**
  📄 [papers/Thoughts-without-Thinking-Reconsidering-the-Explanatory-Value-of-Chain-of-Thought-Reasoning-in-LLMs-through-Agentic-Pipelines.pdf](papers/Thoughts-without-Thinking-Reconsidering-the-Explanatory-Value-of-Chain-of-Thought-Reasoning-in-LLMs-through-Agentic-Pipelines.pdf)
- [ ] **Aligned Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought**
  📄 [papers/Aligned-Orthogonal-or-In-conflict-When-can-we-safely-optimize-Chain-of-Thought.pdf](papers/Aligned-Orthogonal-or-In-conflict-When-can-we-safely-optimize-Chain-of-Thought.pdf)
- [ ] **Chain-of-Thought Monitorability: A New and Fragile Opportunity for AI Safety**
  📄 [papers/Chain-of-Thought-Monitorability-A-New-and-Fragile-Opportunity-for-AI-Safety.pdf](papers/Chain-of-Thought-Monitorability-A-New-and-Fragile-Opportunity-for-AI-Safety.pdf)
- [ ] **Focused Chain-of-Thought: Efficient LLM Reasoning via Structured Input Information**
  📄 [papers/Focused-Chain-of-Thought-Efficient-LLM-Reasoning-via-Structured-Input-Information.pdf](papers/Focused-Chain-of-Thought-Efficient-LLM-Reasoning-via-Structured-Input-Information.pdf)

### 11.2 Advanced Reasoning
- [ ] **Implicit Reasoning in Large Language Models: A Comprehensive Survey**
  📄 [papers/Implicit-Reasoning-in-Large-Language-Models-A-Comprehensive-Survey.pdf](papers/Implicit-Reasoning-in-Large-Language-Models-A-Comprehensive-Survey.pdf)
- [ ] **A Survey on Latent Reasoning**
  📄 [papers/A-Survey-on-Latent-Reasoning.pdf](papers/A-Survey-on-Latent-Reasoning.pdf)
- [ ] **Efficient Reasoning with Balanced Thinking**
  📄 [papers/Efficient-Reasoning-with-Balanced-Thinking.pdf](papers/Efficient-Reasoning-with-Balanced-Thinking.pdf)
- [ ] **LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling**
  📄 [papers/LLMs-Improving-LLMs-Agentic-Discovery-for-Test-Time-Scaling.pdf](papers/LLMs-Improving-LLMs-Agentic-Discovery-for-Test-Time-Scaling.pdf)
- [ ] **ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning**
  📄 [papers/ThinkMorph-Emergent-Properties-in-Multimodal-Interleaved-Chain-of-Thought-Reasoning.pdf](papers/ThinkMorph-Emergent-Properties-in-Multimodal-Interleaved-Chain-of-Thought-Reasoning.pdf)
- [ ] **Thinkless: LLM Learns When to Think**
  📄 [papers/Thinkless-LLM-Learns-When-to-Think.pdf](papers/Thinkless-LLM-Learns-When-to-Think.pdf)
- [ ] **Thinking vs Doing: Agents that Reason by Scaling Test Time Interaction**
  📄 [papers/Thinking-vs-Doing-Agents-that-Reason-by-Scaling-Test-Time-Interaction.pdf](papers/Thinking-vs-Doing-Agents-that-Reason-by-Scaling-Test-Time-Interaction.pdf)
- [ ] **Does Your Reasoning Model Implicitly Know When to Stop Thinking**
  📄 [papers/Does-Your-Reasoning-Model-Implicitly-Know-When-to-Stop-Thinking.pdf](papers/Does-Your-Reasoning-Model-Implicitly-Know-When-to-Stop-Thinking.pdf)
- [ ] **R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth**
  📄 [papers/R-Horizon-How-Far-Can-Your-Large-Reasoning-Model-Really-Go-in-Breadth-and-Depth.pdf](papers/R-Horizon-How-Far-Can-Your-Large-Reasoning-Model-Really-Go-in-Breadth-and-Depth.pdf)

### 11.3 Reasoning Models LRMs
- [ ] **Towards a Mechanistic Understanding of Large Reasoning Models: A Survey of Training Inference and Failures**
  📄 [papers/Towards-a-Mechanistic-Understanding-of-Large-Reasoning-Models-A-Survey-of-Training-Inference-and-Failures.pdf](papers/Towards-a-Mechanistic-Understanding-of-Large-Reasoning-Models-A-Survey-of-Training-Inference-and-Failures.pdf)
- [ ] **Are Your Reasoning Models Reasoning or Guessing: A Mechanistic Analysis of Hierarchical Reasoning Models**
  📄 [papers/Are-Your-Reasoning-Models-Reasoning-or-Guessing-A-Mechanistic-Analysis-of-Hierarchical-Reasoning-Models.pdf](papers/Are-Your-Reasoning-Models-Reasoning-or-Guessing-A-Mechanistic-Analysis-of-Hierarchical-Reasoning-Models.pdf)
- [ ] **DO EXPLANATIONS GENERALIZE ACROSS LARGE REASONING MODELS**
  📄 [papers/DO-EXPLANATIONS-GENERALIZE-ACROSS-LARGE-REA-SONING-MODELS.pdf](papers/DO-EXPLANATIONS-GENERALIZE-ACROSS-LARGE-REA-SONING-MODELS.pdf)
- [ ] **Parallel-R1: Towards Parallel Thinking via Reinforcement Learning**
  📄 [papers/Parallel-R1: Towards Parallel Thinking via Reinforcement Learning.pdf](papers/Parallel-R1-Towards-Parallel-Thinking-via-Reinforcement-Learning.pdf)
- [ ] **Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic**
  📄 [papers/Reasoning-Vectors-Transferring-Chain-of-Thought-Capabilities-via-Task-Arithmetic.pdf](papers/Reasoning-Vectors-Transferring-Chain-of-Thought-Capabilities-via-Task-Arithmetic.pdf)

### 11.4 Planning and Search
- [ ] **Make Planning Research Rigorous Again**
  📄 [papers/Make-Planning-Research-Rigorous-Again.pdf](papers/Make-Planning-Research-Rigorous-Again.pdf)
- [ ] **Programming over Thinking: Efficient and Robust Multi-Constraint Planning**
  📄 [papers/Programming-over-Thinking-Efficient-and-Robust-Multi-Constraint-Planning.pdf](papers/Programming-over-Thinking-Efficient-and-Robust-Multi-Constraint-Planning.pdf)
- [ ] **PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems**
  📄 [papers/PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems.pdf](papers/PlanBench-XL-Evaluating-Long-Horizon-Planning-of-LLM-Tool-Use-Agents-in-Large-Scale-Tool-Ecosystems.pdf)
- [ ] **DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search**
  📄 [papers/DeepSearch-Overcome-the-Bottleneck-of-Reinforcement-Learning-with-Verifiable-Rewards-via-Monte-Carlo-Tree-Search.pdf](papers/DeepSearch-Overcome-the-Bottleneck-of-Reinforcement-Learning-with-Verifiable-Rewards-via-Monte-Carlo-Tree-Search.pdf)

---

## 12. AI Agents
- [ ] **A Comprehensive Survey on Agent Skills Taxonomy Techniques and Applications**
  📄 [papers/A-Comprehensive-Survey-on-Agent-Skills-Taxonomy-Techniques-and-Applications.pdf](papers/A-Comprehensive-Survey-on-Agent-Skills-Taxonomy-Techniques-and-Applications.pdf)
- [ ] **A Comprehensive Survey on Reinforcement Learning based Agentic Search Foundations Roles Optimizations Evaluations and Application**
  📄 [papers/A-Comprehensive-Survey-on-Reinforcement-Learning-based-Agentic-Search-Foundations-Roles-Optimizations-Evaluations-and-Application.pdf](papers/A-Comprehensive-Survey-on-Reinforcement-Learning-based-Agentic-Search-Foundations-Roles-Optimizations-Evaluations-and-Application.pdf)
- [ ] **A Survey on the Security of Long Term Memory in LLM Agents Toward Mnemonic Sovereignty**
  📄 [papers/A-Survey-on-the-Security-of-Long-Term-Memory-in-LLM-Agents-Toward-Mnemonic-Sovereignty.pdf](papers/A-Survey-on-the-Security-of-Long-Term-Memory-in-LLM-Agents-Toward-Mnemonic-Sovereignty.pdf)
- [ ] **AI Agents v Agentic AI A Conceptual Taxonomy Applications and Challenges**
  📄 [papers/AI-Agents-v-Agentic-AI-A-Conceptual-Taxonomy-Applications-and-Challenges.pdf](papers/AI-Agents-v-Agentic-AI-A-Conceptual-Taxonomy-Applications-and-Challenges.pdf)
- [ ] **Agentic AI A Comprehensive Survey of Architectures Applications and Future Directions**
  📄 [papers/Agentic-AI-A-Comprehensive-Survey-of-Architectures-Applications-and-Future-Directions.pdf](papers/Agentic-AI-A-Comprehensive-Survey-of-Architectures-Applications-and-Future-Directions.pdf)
- [ ] **ByteRover Agent Native Memory Through LLM Curated Hierarchical Context**
  📄 [papers/ByteRover-Agent-Native-Memory-Through-LLM-Curated-Hierarchical-Context.pdf](papers/ByteRover-Agent-Native-Memory-Through-LLM-Curated-Hierarchical-Context.pdf)
- [ ] **Calibrate Then Act Cost Aware Exploration in LLM Agents**
  📄 [papers/Calibrate-Then-Act-Cost-Aware-Exploration-in-LLM-Agents.pdf](papers/Calibrate-Then-Act-Cost-Aware-Exploration-in-LLM-Agents.pdf)
- [ ] **DEPO Dual Efficiency Preference Optimization for LLM Agents**
  📄 [papers/DEPO-Dual-Efficiency-Preference-Optimization-for-LLM-Agents.pdf](papers/DEPO-Dual-Efficiency-Preference-Optimization-for-LLM-Agents.pdf)
- [ ] **From Skill Text to Skill Structure The Scheduling Structural Logical Representation for Agent Skills**
  📄 [papers/From-Skill-Text-to-Skill-Structure-The-Scheduling-Structural-Logical-Representation-for-Agent-Skills.pdf](papers/From-Skill-Text-to-Skill-Structure-The-Scheduling-Structural-Logical-Representation-for-Agent-Skills.pdf)
- [ ] **General Agentic Memory Via Deep Research**
  📄 [papers/General-Agentic-Memory-Via-Deep-Research.pdf](papers/General-Agentic-Memory-Via-Deep-Research.pdf)
- [ ] **Hierarchical Policy Gradient Reinforcement Learning for Multi Agent Shepherding Control of Non Cohesive Targets**
  📄 [papers/Hierarchical-Policy-Gradient-Reinforcement-Learning-for-Multi-Agent-Shepherding-Control-of-Non-Cohesive-Targets.pdf](papers/Hierarchical-Policy-Gradient-Reinforcement-Learning-for-Multi-Agent-Shepherding-Control-of-Non-Cohesive-Targets.pdf)
- [ ] **LLM Agent as Data Analyst A Survey**
  📄 [papers/LLM-Agent-as-Data-Analyst-A-Survey.pdf](papers/LLM-Agent-as-Data-Analyst-A-Survey.pdf)
- [ ] **MIRIX Multi Agent Memory System for LLM Based Agents**
  📄 [papers/MIRIX-Multi-Agent-Memory-System-for-LLM-Based-Agents.pdf](papers/MIRIX-Multi-Agent-Memory-System-for-LLM-Based-Agents.pdf)
- [ ] **MemSlides A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi turn Local Revision**
  📄 [papers/MemSlides-A-Hierarchical-Memory-Driven-Agent-Framework-for-Personalized-Slide-Generation-with-Multi-turn-Local-Revision.pdf](papers/MemSlides-A-Hierarchical-Memory-Driven-Agent-Framework-for-Personalized-Slide-Generation-with-Multi-turn-Local-Revision.pdf)
- [ ] **O Mem Omni Memory System for Personalized Long Horizon Self Evolving Agents**
  📄 [papers/O-Mem-Omni-Memory-System-for-Personalized-Long-Horizon-Self-Evolving-Agents.pdf](papers/O-Mem-Omni-Memory-System-for-Personalized-Long-Horizon-Self-Evolving-Agents.pdf)
- [ ] **SWE agent Agent Computer Interfaces Enable**
  📄 [papers/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering.pdf](papers/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering.pdf)
- [ ] **Scaling Test Time Compute for Agentic Coding**
  📄 [papers/Scaling-Test-Time-Compute-for-Agentic-Coding.pdf](papers/Scaling-Test-Time-Compute-for-Agentic-Coding.pdf)
- [ ] **Thesis Optimizing Agentic Workflows **
  📄 [papers/Thesis-Optimizing-Agentic-Workflows-.pdf](papers/Thesis-Optimizing-Agentic-Workflows-.pdf)
- [ ] **Thoughts without Thinking Reconsidering the Explanatory Value of Chain of Thought Reasoning in LLMs through Agentic Pipelines**
  📄 [papers/Thoughts-without-Thinking-Reconsidering-the-Explanatory-Value-of-Chain-of-Thought-Reasoning-in-LLMs-through-Agentic-Pipelines.pdf](papers/Thoughts-without-Thinking-Reconsidering-the-Explanatory-Value-of-Chain-of-Thought-Reasoning-in-LLMs-through-Agentic-Pipelines.pdf)
- [ ] **Towards Edge General Intelligence Knowledge Distillation for Mobile Agentic AI**
  📄 [papers/Towards-Edge-General-Intelligence-Knowledge-Distillation-for-Mobile-Agentic-AI.pdf](papers/Towards-Edge-General-Intelligence-Knowledge-Distillation-for-Mobile-Agentic-AI.pdf)
- [ ] **Towards a Science of Scaling Agent System**
  📄 [papers/Towards-a-Science-of-Scaling-Agent-System.pdf](papers/Towards-a-Science-of-Scaling-Agent-System.pdf)
- [ ] **Why Do Multi-Agent LLM Systems Fail**
  📄 [papers/WhyDoMulti-Agent-LLM-Systems-Fail.pdf](papers/WhyDoMulti-Agent-LLM-Systems-Fail.pdf)
- [ ] **rStar2 Agent Agentic Reasoning Technical Report**
  📄 [papers/rStar2-Agent-Agentic-Reasoning-Technical-Report.pdf](papers/rStar2-Agent-Agentic-Reasoning-Technical-Report.pdf)

### 12.1 Agent Foundations
- [ ] **Introduction to Agents.pdf**
  📄 [papers/Introduction-to-Agents.pdf](papers/Introduction-to-Agents.pdf)
- [ ] **a-practical-guide-to-building-agents.pdf**
  📄 [papers/a-practical-guide-to-building-agents.pdf](papers/a-practical-guide-to-building-agents.pdf)
- [ ] **Agentic AI: A Comprehensive Survey of Architectures Applications and Future Directions**
  📄 [papers/Agentic AI: A Comprehensive Survey of Architectures Applications and Future Directions.pdf](papers/A-Comprehensive-Survey-of-Self-Evolving-AI-Agents-A-New-Paradigm-Bridging-Foundation-Models-and-Lifelong-Agentic-Systems.pdf)
- [ ] **AI Agent Systems: Architectures Applications and Evaluation**
  📄 [papers/AI-Agent-Systems-Architectures-Applications-and-Evaluation.pdf](papers/AI-Agent-Systems-Architectures-Applications-and-Evaluation.pdf)
- [ ] **AI Agents Evolution Architecture and Real-World Applications**
  📄 [papers/AI-Agents-Evolution-Architecture-and-Real-World-Applications.pdf](papers/AI-Agents-Evolution-Architecture-and-Real-World-Applications.pdf)
- [ ] **AI Agents v Agentic AI: A Conceptual Taxonomy Applications and Challenges**
  📄 [papers/AI Agents v Agentic AI: A Conceptual Taxonomy Applications and Challenges.pdf](papers/A-Comprehensive-Survey-of-Self-Evolving-AI-Agents-A-New-Paradigm-Bridging-Foundation-Models-and-Lifelong-Agentic-Systems.pdf)
- [ ] **AI Agents as Universal Task Solvers**
  📄 [papers/AI-Agents-as-Universal-Task-Solvers.pdf](papers/AI-Agents-as-Universal-Task-Solvers.pdf)
- [ ] **AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION**
  📄 [papers/AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION.pdf](papers/AGENT-AI-SURVEYING-THE-HORIZONS-OF-MULTIMODAL-INTERACTION.pdf)

### 12.2 Agent Architectures
- [ ] **Cognitive Architectures for Language Agents**
  📄 [papers/Cognitive-Architectures-for-Language-Agents.pdf](papers/Cognitive-Architectures-for-Language-Agents.pdf)
- [ ] **Foundation-Model-Based Agents in Industrial Automation**
  📄 [papers/Foundation-Model-Based-Agents-in-Industrial-Automation-Purposes-Capabilities-and-Open-Challenges.pdf](papers/Foundation-Model-Based-Agents-in-Industrial-Automation-Purposes-Capabilities-and-Open-Challenges.pdf)
- [ ] **Infrastructure for AI Agents**
  📄 [papers/Infrastructure-for-AI-Agents.pdf](papers/Infrastructure-for-AI-Agents.pdf)
- [ ] **Agent System Operations: Categorization Challenges and Future Directions**
  📄 [papers/Agent-System-Operations-Categorization-Challenges-and-Future-Directions.pdf](papers/Agent-System-Operations-Categorization-Challenges-and-Future-Directions.pdf)
- [ ] **A Survey on AgentOps: Categorization Challenges and Future Directions**
  📄 [papers/A-Survey-on-AgentOps-Categorization-Challenges-and-Future-Directions.pdf](papers/A-Survey-on-AgentOps-Categorization-Challenges-and-Future-Directions.pdf)
- [ ] **Agent Harness Engineering: A Survey**
  📄 [papers/Agent-Harness-Engineering-A-Survey.pdf](papers/Agent-Harness-Engineering-A-Survey.pdf)
- [ ] **Meta-Harness: End-to-End Optimization of Model Harnesses**
  📄 [papers/Meta-Harness-End-to-End-Optimization-of-Model-Harnesses.pdf](papers/Meta-Harness-End-to-End-Optimization-of-Model-Harnesses.pdf)
- [ ] **Code as Agent Harness**
  📄 [papers/Code-as-Agent-Harness.pdf](papers/Code-as-Agent-Harness.pdf)
- [ ] **Natural-Language Agent Harnesses**
  📄 [papers/Natural-Language-Agent-Harnesses.pdf](papers/Natural-Language-Agent-Harnesses.pdf)

### 12.3 Multi-Agent Systems
- [ ] **Collaborative AI Agents in the Era of Large Language Models**
  📄 [papers/Collaborative-AI-Agents-in-the-Era-of-Large-Language-Models.pdf](papers/Collaborative-AI-Agents-in-the-Era-of-Large-Language-Models.pdf)
- [ ] **Beyond Individual Intelligence: Surveying Collaboration Failure Attribution and Self-Evolution in LLM-based Multi-Agent Systems**
  📄 [papers/Beyond Individual Intelligence: Surveying Collaboration Failure Attribution and Self-Evolution in LLM-based Multi-Agent Systems.pdf](papers/Beyond-Individual-Intelligence-Surveying-Collaboration-Failure-Attribution-and-Self-Evolution-in-LLM-based-Multi-Agent-Systems.pdf)
- [ ] **Why Do Multi-Agent LLM Systems Fail**
  📄 [papers/WhyDoMulti-Agent-LLM-Systems-Fail.pdf](papers/WhyDoMulti-Agent-LLM-Systems-Fail.pdf)
- [ ] **Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets**
  📄 [papers/Single-Agent-LLMs-Outperform-Multi-Agent-Systems-on-Multi-Hop-Reasoning-Under-Equal-Thinking-Token-Budgets.pdf](papers/Single-Agent-LLMs-Outperform-Multi-Agent-Systems-on-Multi-Hop-Reasoning-Under-Equal-Thinking-Token-Budgets.pdf)
- [ ] **Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review**
  📄 [papers/Generative-Multi-Agent-Collaboration-in-Embodied-AI-A-Systematic-Review.pdf](papers/Generative-Multi-Agent-Collaboration-in-Embodied-AI-A-Systematic-Review.pdf)
- [ ] **Recursive Multi-Agent Systems**
  📄 [papers/Recursive-Multi-Agent-Systems.pdf](papers/Recursive-Multi-Agent-Systems.pdf)
- [ ] **WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning**
  📄 [papers/WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning.pdf](papers/WideSeek-R1-Exploring-Width-Scaling-for-Broad-Information-Seeking-via-Multi-Agent-Reinforcement-Learning.pdf)
- [ ] **Adaptive Multi-Agent Response Refinement in Conversational Systems**
  📄 [papers/Adaptive-Multi-Agent-Response-Refinement-in-Conversational-Systems.pdf](papers/Adaptive-Multi-Agent-Response-Refinement-in-Conversational-Systems.pdf)
- [ ] **MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems**
  📄 [papers/MASPO--Joint-Prompt-Optimization-for-LLM-based-Multi-Agent-Systems.pdf](papers/MASPO--Joint-Prompt-Optimization-for-LLM-based-Multi-Agent-Systems.pdf)
- [ ] **MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning**
  📄 [papers/MARL-GPT-Foundation-Model-for-Multi-Agent-Reinforcement-Learning.pdf](papers/MARL-GPT-Foundation-Model-for-Multi-Agent-Reinforcement-Learning.pdf)

### 12.4 Self-Evolving Agents
- [ ] **A Comprehensive Survey of Self-Evolving AI Agents**
  📄 [papers/A-Comprehensive-Survey-of-Self-Evolving-AI-Agents-A-New-Paradigm-Bridging-Foundation-Models-and-Lifelong-Agentic-Systems.pdf](papers/A-Comprehensive-Survey-of-Self-Evolving-AI-Agents-A-New-Paradigm-Bridging-Foundation-Models-and-Lifelong-Agentic-Systems.pdf)
- [ ] **A SURVEY OF SELF-EVOLVING AGENTS ON PATH TO ARTIFICIAL SUPER INTELLIGENCE**
  📄 [papers/A-SURVEY-OF-SELF-EVOLVING-AGENTS-ON-PATH-TO-ARTIFICIAL-SUPER-INTELLIGENCE.pdf](papers/A-SURVEY-OF-SELF-EVOLVING-AGENTS-ON-PATH-TO-ARTIFICIAL-SUPER-INTELLIGENCE.pdf)
- [ ] **Self-Improving LLM Agents at Test-Time**
  📄 [papers/Self-Improving-LLM-Agents-at-Test-Time.pdf](papers/Self-Improving-LLM-Agents-at-Test-Time.pdf)
- [ ] **Toward Scalable and Self-Improving Large Language Model Agents**
  📄 [papers/Toward-Scalable-and-Self-Improving-Large-Language-Model-Agents.pdf](papers/Toward-Scalable-and-Self-Improving-Large-Language-Model-Agents.pdf)
- [ ] **SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning**
  📄 [papers/SiriuS-Self-improving-Multi-agent-Systems-via-Bootstrapped-Reasoning.pdf](papers/SiriuS-Self-improving-Multi-agent-Systems-via-Bootstrapped-Reasoning.pdf)
- [ ] **MetaClaw: Just Talk - An Agent That Meta-Learns and Evolves in the Wild**
  📄 [papers/MetaClaw-Just-Talk-An-Agent-That-Meta-Learns-and-Evolves-in-the-Wild.pdf](papers/MetaClaw-Just-Talk-An-Agent-That-Meta-Learns-and-Evolves-in-the-Wild.pdf)
- [ ] **Self-Evolving Multi-Agent Systems via Decentralized Memory**
  📄 [papers/Self-Evolving-Multi-Agent-Systems-via-Decentralized-Memory.pdf](papers/Self-Evolving-Multi-Agent-Systems-via-Decentralized-Memory.pdf)
- [ ] **SkillOS: Learning Skill Curation for Self-Evolving Agents**
  📄 [papers/SkillOS-Learning-Skill-Curation-for-Self-Evolving-Agents.pdf](papers/SkillOS-Learning-Skill-Curation-for-Self-Evolving-Agents.pdf)
- [ ] **SkillNet: Create Evaluate and Connect AI Skills**
  📄 [papers/SkillNet-Create-Evaluate-and-Connect-AI-Skills.pdf](papers/SkillNet-Create-Evaluate-and-Connect-AI-Skills.pdf)

### 12.5 Agentic AI and Autonomous Systems
- [ ] **From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review**
  📄 [papers/From-LLM-Reasoning-to-Autonomous-AI-Agents-A-Comprehensive-Review.pdf](papers/From-LLM-Reasoning-to-Autonomous-AI-Agents-A-Comprehensive-Review.pdf)
- [ ] **From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery**
  📄 [papers/From-Automation-to-Autonomy-ASurvey-on-Large-Language-Models-in-Scientific-Discovery.pdf](papers/From-Automation-to-Autonomy-ASurvey-on-Large-Language-Models-in-Scientific-Discovery.pdf)
- [ ] **Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI**
  📄 [papers/Beyond-Pipelines-A-Survey-of-the-Paradigm-Shift-toward-Model-Native-Agentic-AI.pdf](papers/Beyond-Pipelines-A-Survey-of-the-Paradigm-Shift-toward-Model-Native-Agentic-AI.pdf)
- [ ] **Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary Collaborative and Safe Systems**
  📄 [papers/ADVANCES-AND-CHALLENGES-IN-FOUNDATION-AGENTS-FROM-BRAIN-INSPIRED-INTELLIGENCE-TO-EVOLUTIONARY-COLLABORATIVE-AND-SAFE-SYSTEMS.pdf](papers/ADVANCES-AND-CHALLENGES-IN-FOUNDATION-AGENTS-FROM-BRAIN-INSPIRED-INTELLIGENCE-TO-EVOLUTIONARY-COLLABORATIVE-AND-SAFE-SYSTEMS.pdf)
- [ ] **Robust Agents in Open-Ended Worlds**
  📄 [papers/Robust-Agents-in-Open-Ended-Worlds.pdf](papers/Robust-Agents-in-Open-Ended-Worlds.pdf)
- [ ] **Agentic World Modeling: Foundations Capabilities Laws and Beyond**
  📄 [papers/Agentic-World-Modeling-Foundations-Capabilities-Laws-and-Beyond.pdf](papers/Agentic-World-Modeling-Foundations-Capabilities-Laws-and-Beyond.pdf)
- [ ] **Agentic Context Engineering: Evolving Contexts for Self-Improving**
  📄 [papers/Agentic-Context-Engineering-Evolving-Contextsfor-Self-Improving.pdf](papers/Agentic-Context-Engineering-Evolving-Contextsfor-Self-Improving.pdf)

### 12.6 Agent Training and RL
- [ ] **The Landscape of Agentic Reinforcement Learning for LLMs: A Survey**
  📄 [papers/The-Landscape-of-Agentic-Reinforcement-Learning-for-LLMs-A-Survey.pdf](papers/The-Landscape-of-Agentic-Reinforcement-Learning-for-LLMs-A-Survey.pdf)
- [ ] **Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle**
  📄 [papers/Reinforcement-Learning-Meets-Large-Language-Models-A-Survey-of-Advancements-and-Applications-Across-the-LLM-Lifecycle.pdf](papers/Reinforcement-Learning-Meets-Large-Language-Models-A-Survey-of-Advancements-and-Applications-Across-the-LLM-Lifecycle.pdf)
- [ ] **Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces**
  📄 [papers/Reinforcement-Learning-for-LLM-based-Multi-Agent-Systems-through-Orchestration-Traces.pdf](papers/Reinforcement-Learning-for-LLM-based-Multi-Agent-Systems-through-Orchestration-Traces.pdf)
- [ ] **Agent Learning via Early Experience**
  📄 [papers/Agent-Learning-via-Early-Experience.pdf](papers/Agent-Learning-via-Early-Experience.pdf)
- [ ] **From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory**
  📄 [papers/From-Player-to-Master-Enhancing-Test-Time-Learning-of-LLM-Agents-via-Reinforcement-Learning-over-Memory.pdf](papers/From-Player-to-Master-Enhancing-Test-Time-Learning-of-LLM-Agents-via-Reinforcement-Learning-over-Memory.pdf)
- [ ] **Bridging Offline and Online Reinforcement Learning for LLMs**
  📄 [papers/Bridging-Offline-and-Online-Reinforcement-Learning-for-LLMs.pdf](papers/Bridging-Offline-and-Online-Reinforcement-Learning-for-LLMs.pdf)
- [ ] **Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents**
  📄 [papers/Calibrate-Then-Act-Cost-Aware-Exploration-in-LLM-Agents.pdf](papers/Calibrate-Then-Act-Cost-Aware-Exploration-in-LLM-Agents.pdf)
- [ ] **Tree Search for LLM Agent Reinforcement Learning**
  📄 [papers/Tree-Search-for-LLM-Agent-Reinforcement-Learning.pdf](papers/Tree-Search-for-LLM-Agent-Reinforcement-Learning.pdf)
- [ ] **OpenClaw-RL: Train Any Agent Simply by Talking**
  📄 [papers/OpenClaw-RL: Train Any Agent Simply by Talking.pdf](papers/OpenClaw-RL-Train-Any-Agent-Simply-by-Talking.pdf)

### 12.7 Agent Evaluation and Benchmarking
- [ ] **Agents' Last Exam**
  📄 [papers/Agents'-Last-Exam.pdf](papers/Agents'-Last-Exam.pdf)
- [ ] **CodeClash: Benchmarking Goal-Oriented Software Engineering**
  📄 [papers/CodeClash-Benchmarking-Goal-Oriented-Software-Engineering.pdf](papers/CodeClash-Benchmarking-Goal-Oriented-Software-Engineering.pdf)
- [ ] **Can Agent Conquer Web: Exploring the Frontiers of ChatGPT Atlas Agent in Web Games**
  📄 [papers/Can-Agent-Conquer-Web-Exploring-the-Frontiers-of-ChatGPT-Atlas-Agent-in-Web-Games.pdf](papers/Can-Agent-Conquer-Web-Exploring-the-Frontiers-of-ChatGPT-Atlas-Agent-in-Web-Games.pdf)
- [ ] **Evolutionary Perspectives on the Evaluation of LLM-Based AI Agents: A Comprehensive Survey**
  📄 [papers/Evolutionary-Perspectives-on-the-Evaluation-of-LLM-Based-AI-Agents-A-Comprehensive-Survey.pdf](papers/Evolutionary-Perspectives-on-the-Evaluation-of-LLM-Based-AI-Agents-A-Comprehensive-Survey.pdf)
- [ ] **The Tool Illusion: Rethinking Tool Use in Web Agents**
  📄 [papers/The-Tool-Illusion-Rethinking-Tool-Use-in-Web-Agents.pdf](papers/The-Tool-Illusion-Rethinking-Tool-Use-in-Web-Agents.pdf)

### 12.8 Code and SWE Agents
- [ ] **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**
  📄 [papers/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering.pdf](papers/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering.pdf)
- [ ] **Agentic Code Reasoning**
  📄 [papers/Agentic-Code-Reasoning.pdf](papers/Agentic-Code-Reasoning.pdf)
- [ ] **Agentic AI in the Software Development Lifecycle**
  📄 [papers/Agentic-AI-in-the-Software-Development-Lifecycle-Architecture-Empirical-Evidence-and-the-Reshaping-of-Software-Engineering.pdf](papers/Agentic-AI-in-the-Software-Development-Lifecycle-Architecture-Empirical-Evidence-and-the-Reshaping-of-Software-Engineering.pdf)
- [ ] **From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence**
  📄 [papers/From-Code-Foundation-Models-to-Agents-and-Applications-A-Practical-Guide-to-Code-Intelligence.pdf](papers/From-Code-Foundation-Models-to-Agents-and-Applications-A-Practical-Guide-to-Code-Intelligence.pdf)
- [ ] **From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence**
  📄 [papers/From-CodeFoundation-Models-to-Agents-and-Applications-A-Comprehensive-Survey-and-Practical-Guide-to-Code-Intelligence.pdf](papers/From-CodeFoundation-Models-to-Agents-and-Applications-A-Comprehensive-Survey-and-Practical-Guide-to-Code-Intelligence.pdf)
- [ ] **VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation**
  📄 [papers/VeriGuard-Enhancing-LLM-Agent-Safety-via-Verified-Code-Generation.pdf](papers/VeriGuard-Enhancing-LLM-Agent-Safety-via-Verified-Code-Generation.pdf)
- [ ] **AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions**
  📄 [papers/AutoKaggle-A-Multi-Agent-Framework-for-Autonomous-Data-Science-Competitions.pdf](papers/AutoKaggle-A-Multi-Agent-Framework-for-Autonomous-Data-Science-Competitions.pdf)
- [ ] **A.S.E: A Repository-Level Benchmark for Evaluating Security in AI-Generated Code**
  📄 [papers/A.S.E-A-Repository-Level-Benchmark-for-Evaluating-Security-in-AI-Generated-Code.pdf](papers/A.S.E-A-Repository-Level-Benchmark-for-Evaluating-Security-in-AI-Generated-Code.pdf)
- [ ] **Advances and Frontiers of LLM-based Issue Resolution in Software Engineering: A Comprehensive Survey**
  📄 [papers/Advances-and-Frontiers-of-LLM-based-Issue-Resolution-in-Software-Engineering-A-Comprehensive-Survey.pdf](papers/Advances-and-Frontiers-of-LLM-based-Issue-Resolution-in-Software-Engineering-A-Comprehensive-Survey.pdf)

### 12.9 Agent Skills and Tools
- [ ] **A Comprehensive Survey on Agent Skills: Taxonomy Techniques and Applications**
  📄 [papers/A Comprehensive Survey on Agent Skills: Taxonomy Techniques and Applications.pdf](papers/A-Comprehensive-Survey-of-Mixture-of-Experts-Algorithms-Theory-and-Applications.pdf)
- [ ] **AutoTool: Efficient Tool Selection for Large Language Model Agents**
  📄 [papers/AutoTool-Efficient-Tool-Selection-for-Large-Language-Model-Agents.pdf](papers/AutoTool-Efficient-Tool-Selection-for-Large-Language-Model-Agents.pdf)
- [ ] **From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills**
  📄 [papers/From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills.pdf](papers/From-Context-to-Skills-Can-Language-Models-Learn-from-Context-Skillfully.pdf)
- [ ] **Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills**
  📄 [papers/Trace2Skill-Distill-Trajectory-Local-Lessons-into-Transferable-Agent-Skills.pdf](papers/Trace2Skill-Distill-Trajectory-Local-Lessons-into-Transferable-Agent-Skills.pdf)
- [ ] **Distilling LLM Agent into Small Models with Retrieval and Code Tools**
  📄 [papers/Distilling-LLM-Agent-into-Small-Models-with-Retrieval-and-Code-Tools.pdf](papers/Distilling-LLM-Agent-into-Small-Models-with-Retrieval-and-Code-Tools.pdf)

### 12.10 Agent-Specific Topics
- [ ] **Agentic AI for Payment Fraud: CASE Framework**
  📄 [papers/Agentic AI for Payment Fraud: CASE Framework.pdf](papers/CASE-An-Agentic-AI-Framework-for-Enhancing-Scam-Intelligence-in-Digital-Payments.pdf)
- [ ] **Comparing AI Agents to Cybersecurity Professionals in Real-World Penetration Testing**
  📄 [papers/Comparing-AI-Agents-to-Cybersecurity-Professionals-in-Real-World-Penetration-Testing.pdf](papers/Comparing-AI-Agents-to-Cybersecurity-Professionals-in-Real-World-Penetration-Testing.pdf)
- [ ] **Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1**
  📄 [papers/Human-Agent-Collaborative-Paper-to-Page-Crafting-for-Under-$0.1.pdf](papers/Human-Agent-Collaborative-Paper-to-Page-Crafting-for-Under-$0.1.pdf)
- [ ] **SymptomAI: Towards a Conversational AI Agent for Everyday Symptom Assessment**
  📄 [papers/SymptomAI-Towards-a-Conversational-AI-Agent-for-Everyday-Symptom-Assessment.pdf](papers/SymptomAI-Towards-a-Conversational-AI-Agent-for-Everyday-Symptom-Assessment.pdf)
- [ ] **LLM-Agent-as-Data-Analyst: A Survey**
  📄 [papers/LLM-Agent-as-Data-Analyst: A Survey.pdf](papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf)
- [ ] **DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle**
  📄 [papers/DAComp-Benchmarking-Data-Agents-across-the-Full-Data-Intelligence-Lifecycle.pdf](papers/DAComp-Benchmarking-Data-Agents-across-the-Full-Data-Intelligence-Lifecycle.pdf)
- [ ] **InfoAgent: Advancing Autonomous Information-Seeking Agents**
  📄 [papers/InfoAgent-Advancing-Autonomous-Information-Seeking-Agents.pdf](papers/InfoAgent-Advancing-Autonomous-Information-Seeking-Agents.pdf)
- [ ] **LLM-based Agents Suffer from Hallucinations**
  📄 [papers/LLM-based-Agents-Suffer-from-Hallucinations-A-Survey-of-Taxonomy-Methods-and-Directions.pdf](papers/LLM-based-Agents-Suffer-from-Hallucinations-A-Survey-of-Taxonomy-Methods-and-Directions.pdf)
- [ ] **Multi-User Large Language Model Agents**
  📄 [papers/Multi-User-Large-Language-Model-Agents.pdf](papers/Multi-User-Large-Language-Model-Agents.pdf)
- [ ] **LLMs Corrupt Your Documents When You Delegate**
  📄 [papers/LLMs-Corrupt-Your-Documents-When-You-Delegate.pdf](papers/LLMs-Corrupt-Your-Documents-When-You-Delegate.pdf)
- [ ] **Did You Forget What I Asked: Prospective Memory Failures in Large Language Models**
  📄 [papers/Did You Forget What I Asked: Prospective Memory Failures in Large Language Models.pdf](papers/Did-You-Forget-What-I-Asked-Prospective-Memory-Failures-in-Large-Language-Models.pdf)
- [ ] **Large Language Model Agent: A Survey on Methodology Applications and Challenges**
  📄 [papers/Large-Language-Model-Agent-A-Survey-on-Methodology-Applications-and-Challenges.pdf](papers/Large-Language-Model-Agent-A-Survey-on-Methodology-Applications-and-Challenges.pdf)

---

## 13. Memory Systems

### 13.1 Agent Memory
- [ ] **Are We Ready For An Agent-Native Memory System**
  📄 [papers/Are We Ready For An Agent-Native Memory System.pdf](papers/Are-We-Ready-For-An-Agent-Native-Memory-System.pdf)
- [ ] **MemOS: A Memory OS for AI System**
  📄 [papers/MemOS A Memory OS for AI System.pdf](papers/MemOS-A-Memory-OS-for-AI-System.pdf)
- [ ] **ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context**
  📄 [papers/ByteRover-Agent-Native-Memory-Through-LLM-Curated-Hierarchical-Context.pdf](papers/ByteRover-Agent-Native-Memory-Through-LLM-Curated-Hierarchical-Context.pdf)
- [ ] **MIRIX: Multi-Agent Memory System for LLM-Based Agents**
  📄 [papers/MIRIX-Multi-Agent-Memory-System-for-LLM-Based-Agents.pdf](papers/MIRIX-Multi-Agent-Memory-System-for-LLM-Based-Agents.pdf)
- [ ] **MemSlides: A Hierarchical Memory-Driven Agent Framework for Personalized Slide Generation**
  📄 [papers/MemSlides-A-Hierarchical-Memory-Driven-Agent-Framework-for-Personalized-Slide-Generation-with-Multi-turn-Local-Revision.pdf](papers/MemSlides-A-Hierarchical-Memory-Driven-Agent-Framework-for-Personalized-Slide-Generation-with-Multi-turn-Local-Revision.pdf)
- [ ] **O-Mem: Omni-Memory System for Personalized Long-Horizon Self-Evolving Agents**
  📄 [papers/O-Mem-Omni-Memory-System-for-Personalized-Long-Horizon-Self-Evolving-Agents.pdf](papers/O-Mem-Omni-Memory-System-for-Personalized-Long-Horizon-Self-Evolving-Agents.pdf)
- [ ] **General Agentic Memory Via Deep Research**
  📄 [papers/General-Agentic-Memory-Via-Deep-Research.pdf](papers/General-Agentic-Memory-Via-Deep-Research.pdf)
- [ ] **A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty**
  📄 [papers/A-Survey-on-the-Security-of-Long-Term-Memory-in-LLM-Agents-Toward-Mnemonic-Sovereignty.pdf](papers/A-Survey-on-the-Security-of-Long-Term-Memory-in-LLM-Agents-Toward-Mnemonic-Sovereignty.pdf)

### 13.2 Memory Mechanisms
- [ ] **Simulating Human Memory with Language Models**
  📄 [papers/Simulating-Human-Memory-with-Language-Models.pdf](papers/Simulating-Human-Memory-with-Language-Models.pdf)
- [ ] **Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping**
  📄 [papers/Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping.pdf](papers/Memory-as-a-Markov-Matrix-Sample-Efficient-Knowledge-Expansion-via-Token-to-Dictionary-Mapping.pdf)
- [ ] **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models**
  📄 [papers/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.pdf](papers/Conditional-Memory-via-Scalable-Lookup-A-New-Axis-of-Sparsity-for-Large-Language-Models.pdf)
- [ ] **Titans: Learning to Memorize at Test Time**
  📄 [papers/Titans-Learning-to-Memorize-at-Test-Time.pdf](papers/Titans-Learning-to-Memorize-at-Test-Time.pdf)
  - → Agent Memory Section 15

---

## 14. Data Engineering

### 14.1 Data Selection and Quality
- [ ] **OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration**
  📄 [papers/OPUS-Towards-Efficient-and-Principled-Data-Selection-in-Large-Language-Model-Pre-training-in-Every-Iteration.pdf](papers/OPUS-Towards-Efficient-and-Principled-Data-Selection-in-Large-Language-Model-Pre-training-in-Every-Iteration.pdf)
- [ ] **LLMSurgeon: Diagnosing Data Mixture of Large Language Models**
  📄 [papers/LLMSurgeon-Diagnosing-Data-Mixture-of-Large-Language-Models.pdf](papers/LLMSurgeon-Diagnosing-Data-Mixture-of-Large-Language-Models.pdf)

### 14.2 Data Pipeline
- [ ] **Training Data Efficiency in Multimodal Process Reward Models**
  📄 [papers/Training-Data-Efficiency-in-Multimodal-Process-Reward-Models.pdf](papers/Training-Data-Efficiency-in-Multimodal-Process-Reward-Models.pdf)

### 14.3 Prompt Engineering
- [ ] **A Survey of Context Engineering for Large Language Models**
  📄 [papers/A-Survey-of-Context-Engineering-for-Large-Language-Models.pdf](papers/A-Survey-of-Context-Engineering-for-Large-Language-Models.pdf)

---

## 15. Evaluation and Hallucination

### 15.1 Evaluation Frameworks
- [ ] **A Survey on Evaluation of Large Language Models**
  📄 [papers/A-Survey-on-Evaluation-of-Large-Language-Models.pdf](papers/A-Survey-on-Evaluation-of-Large-Language-Models.pdf)
- [ ] **A Survey on LLM-as-a-Judge**
  📄 [papers/A-Survey-on-LLM-as-a-Judge.pdf](papers/A-Survey-on-LLM-as-a-Judge.pdf)
- [ ] **Toward Generalizable Evaluation in the LLM Era: A Survey Beyond Benchmarks**
  📄 [papers/Toward-Generalizable-Evaluation-in-the-LLM-Era-A-Survey-Beyond-Benchmarks.pdf](papers/Toward-Generalizable-Evaluation-in-the-LLM-Era-A-Survey-Beyond-Benchmarks.pdf)
- [ ] **LLMCBench: Benchmarking Large Language Model**
  📄 [papers/LLMCBench-Benchmarking-Large-Language-Model.pdf](papers/LLMCBench-Benchmarking-Large-Language-Model.pdf)
- [ ] **REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge**
  📄 [papers/REAL-Regression-Aware-Reinforcement-Learning-for-LLM-as-a-Judge.pdf](papers/REAL-Regression-Aware-Reinforcement-Learning-for-LLM-as-a-Judge.pdf)
- [ ] **Uncertainty Estimation with Small and Large Models**
  📄 [papers/Uncertainty-Estimation-with-Small-and-Large-Models-Sharma-2023.pdf](papers/Uncertainty-Estimation-with-Small-and-Large-Models-Sharma-2023.pdf)

### 15.2 Hallucination
- [ ] **Hallucination-Free: Assessing the Reliability of Leading AI**
  📄 [papers/Hallucination-Free-Assessing-the-Reliability-of-Leading-AI-.pdf](papers/Hallucination-Free-Assessing-the-Reliability-of-Leading-AI-.pdf)
- [ ] **Learning to Reason for Hallucination Span Detection**
  📄 [papers/Learning-to-Reason-for-Hallucination-Span-Detection.pdf](papers/Learning-to-Reason-for-Hallucination-Span-Detection.pdf)
- [ ] **ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models**
  📄 [papers/ODE-Open-Set-Evaluation-of-Hallucinations-in-Multimodal-Large-Language-Models.pdf](papers/ODE-Open-Set-Evaluation-of-Hallucinations-in-Multimodal-Large-Language-Models.pdf)

### 15.3 Benchmarks and Metrics
- [ ] **DatBench: Discriminative Faithful and Efficient VLM Evaluations**
  📄 [papers/DatBench-Discriminative-Faithful-and-Efficient-VLM-Evaluations.pdf](papers/DatBench-Discriminative-Faithful-and-Efficient-VLM-Evaluations.pdf)
- [ ] **ReportBench: Evaluating Deep Research Agents via Academic Survey Tasks**
  📄 [papers/ReportBench-Evaluating-Deep-Research-Agents-via-Academic-Survey-Tasks.pdf](papers/ReportBench-Evaluating-Deep-Research-Agents-via-Academic-Survey-Tasks.pdf)
- [ ] **Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity**
  📄 [papers/Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity.pdf](papers/Incompressible-Knowledge-Probes-Estimating-Black-Box-LLM-Parameter-Counts-via-Factual-Capacity.pdf)

### 15.4 Reliability and Uncertainty
- [ ] **The Illusion of Certainty: Uncertainty quantification for LLMs fails under ambiguity**
  📄 [papers/The-Illusion-of-Certainty-Uncertainty-quantification-for-LLMs-fails-under-ambiguity.pdf](papers/The-Illusion-of-Certainty-Uncertainty-quantification-for-LLMs-fails-under-ambiguity.pdf)
- [ ] **The Illusion of Stochasticity in LLMs**
  📄 [papers/The-Illusion-of-Stochasticity-in-LLMs.pdf](papers/The-Illusion-of-Stochasticity-in-LLMs.pdf)
- [ ] **Uneven Evolution of Cognition Across Generations of Generative AI Models**
  📄 [papers/Uneven-Evolution-of-Cognition-Across-Generations-of-Generative-AI-Models.pdf](papers/Uneven-Evolution-of-Cognition-Across-Generations-of-Generative-AI-Models.pdf)

---

## 16. Safety Trust and Ethics

### 16.1 AI Safety
- [ ] **Trustworthy AI: Ensuring Reliability and Accountability from Models to Agents**
  📄 [papers/Trustworthy AI: Ensuring Reliability and Accountability from Models to Agents.pdf](papers/Trustworthy-AI-Ensuring-Reliability-and-Accountability-from-Models-to-Agents.pdf)
- [ ] **Towards trustworthy agentic AI: a comprehensive survey of safety robustness privacy and system security**
  📄 [papers/Towards trustworthy agentic AI: a comprehensive survey of safety robustness privacy and system security.pdf](papers/Towards-trustworthy-agentic-AI-a-comprehensive-survey-of-safety-robustness-privacy-and-system-security.pdf)
- [ ] **Distributional AGI Safety**
  📄 [papers/Distributional-AGI-Safety.pdf](papers/Distributional-AGI-Safety.pdf)
- [ ] **Safety in Embodied AI: A Survey of Risks Attacks and Defenses**
  📄 [papers/Safety in Embodied AI: A Survey of Risks Attacks and Defenses.pdf](papers/Safety-in-Embodied-AI-A-Survey-of-Risks-Attacks-and-Defenses.pdf)
- [ ] **A Survey on Agentic Security: Applications Threats and Defenses**
  📄 [papers/A Survey on Agentic Security: Applications Threats and Defenses.pdf](papers/A-Comprehensive-Survey-of-Self-Evolving-AI-Agents-A-New-Paradigm-Bridging-Foundation-Models-and-Lifelong-Agentic-Systems.pdf)
- [ ] **Evaluating Language Models for Harmful Manipulation**
  📄 [papers/Evaluating-Language-Models-for-Harmful-Manipulation.pdf](papers/Evaluating-Language-Models-for-Harmful-Manipulation.pdf)

### 16.2 Policy and Compliance
- [ ] **Do LLMs Follow Their Own Rules: A Reflexive Audit of Self-Stated Safety Policies**
  📄 [papers/Do LLMs Follow Their Own Rules: A Reflexive Audit of Self-Stated Safety Policies.pdf](papers/Do-LLMs-Follow-Their-Own-Rules-A-Reflexive-Audit-of-Self-Stated-Safety-Policies.pdf)
- [ ] **Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs**
  📄 [papers/Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs.pdf](papers/Fairness-through-Difference-Awareness-Measuring-Desired-Group-Discrimination-in-LLMs.pdf)

### 16.3 Adversarial and Backdoors
- [ ] **Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs**
  📄 [papers/Weird-Generalization-and-Inductive-Backdoors-New-Ways-to-Corrupt-LLMs.pdf](papers/Weird-Generalization-and-Inductive-Backdoors-New-Ways-to-Corrupt-LLMs.pdf)

### 16.4 Scientific Integrity
- [ ] **A Quantitative Approach to Estimating Bias Favouritism and Distortion in Scientific Journalism**
  📄 [papers/A-Quantitative-Approach-to-Estimating-Bias-Favouritism-and-Distortion-in-Scientific-Journalism.pdf](papers/A-Quantitative-Approach-to-Estimating-Bias-Favouritism-and-Distortion-in-Scientific-Journalism.pdf)

---

## 17. Scientific Discovery and Auto-Research

### 17.1 Auto-Research Systems
- [ ] **AI for Auto-Research: Roadmap and User Guide**
  📄 [papers/AI-for-Auto-Research-Roadmap-and-User-Guide.pdf](papers/AI-for-Auto-Research-Roadmap-and-User-Guide.pdf)
- [ ] **AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery**
  📄 [papers/AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery.pdf](papers/AutoResearch-AI-Towards-AI-Powered-Research-Automation-for-Scientific-Discovery.pdf)
- [ ] **Accelerating Scientific Discovery with Autonomous Goal-evolving Agents**
  📄 [papers/Accelerating-Scientific-Discovery-with-Autonomous-Goal-evolving-Agents.pdf](papers/Accelerating-Scientific-Discovery-with-Autonomous-Goal-evolving-Agents.pdf)
- [ ] **Deep Research: A Systematic Survey**
  📄 [papers/Deep-Research-A-Systematic-Survey.pdf](papers/Deep-Research-A-Systematic-Survey.pdf)
- [ ] **Robin: A multi-agent system for automating scientific discovery**
  📄 [papers/Robin-A-multi-agent-system-for-automating-scientific-discovery.pdf](papers/Robin-A-multi-agent-system-for-automating-scientific-discovery.pdf)
- [ ] **Claw AI Lab: An Autonomous Multi-Agent Research Team**
  📄 [papers/Claw-AI-Lab-An-Autonomous-Multi-Agent-Research-Team.pdf](papers/Claw-AI-Lab-An-Autonomous-Multi-Agent-Research-Team.pdf)
- [ ] **Cognitive Kernel-Pro: A Framework for Deep Research Agents and Agent Foundation Models Training**
  📄 [papers/Cognitive-Kernel-Pro-A-Framework-for-Deep-Research-Agents-and-Agent-Foundation-Models-Training.pdf](papers/Cognitive-Kernel-Pro-A-Framework-for-Deep-Research-Agents-and-Agent-Foundation-Models-Training.pdf)
- [ ] **Step-DeepResearch Technical Report**
  📄 [papers/Step-DeepResearch-Technical-Report.pdf](papers/Step-DeepResearch-Technical-Report.pdf)
- [ ] **Reinforcement Learning Foundations for Deep Research Systems: A Survey**
  📄 [papers/Reinforcement-Learning-Foundations-for-Deep-Research-Systems-A-Survey.pdf](papers/Reinforcement-Learning-Foundations-for-Deep-Research-Systems-A-Survey.pdf)

### 17.2 Scientific LLMs
- [ ] **A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers**
  📄 [papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf](papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf)
- [ ] **Early science acceleration experiments with GPT-5**
  📄 [papers/Early-science-acceleration-experiments-with-GPT-5.pdf](papers/Early-science-acceleration-experiments-with-GPT-5.pdf)
- [ ] **An AI system to help scientists write expert-level empirical software**
  📄 [papers/An-AI-system-to-help-scientists-write-expert-level-empirical-software.pdf](papers/An-AI-system-to-help-scientists-write-expert-level-empirical-software.pdf)
- [ ] **Agents of Discovery**
  📄 [papers/Agents-of-Discovery.pdf](papers/Agents-of-Discovery.pdf)
- [ ] **AI Can Learn Scientific Taste**
  📄 [papers/AI-Can-Learn-Scientific-Taste.pdf](papers/AI-Can-Learn-Scientific-Taste.pdf)
- [ ] **Glia: A Human-Inspired AI for Automated Systems Design and Optimization**
  📄 [papers/Glia-A-Human-Inspired-AI-for-Automated-Systems-Design-and-Optimization.pdf](papers/Glia-A-Human-Inspired-AI-for-Automated-Systems-Design-and-Optimization.pdf)
- [ ] **From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery**
  📄 [papers/From-Automation-to-Autonomy-ASurvey-on-Large-Language-Models-in-Scientific-Discovery.pdf](papers/From-Automation-to-Autonomy-ASurvey-on-Large-Language-Models-in-Scientific-Discovery.pdf)
- [ ] **Intern-S1: A Scientific Multimodal Foundation Model**
  📄 [papers/Intern-S1: A Scientific Multimodal Foundation Model.pdf](papers/Intern-S1-A-Scientific-Multimodal-Foundation-Model.pdf)

---

## 18. Multimodal and Vision-Language

### 18.1 Multimodal Foundations
- [ ] **Multimodal Large Language Models for Medicine: A Comprehensive Survey**
  📄 [papers/Multimodal-Large-Language-Models-for-Medicine-A-Comprehensive-Survey.pdf](papers/Multimodal-Large-Language-Models-for-Medicine-A-Comprehensive-Survey.pdf)
- [ ] **Qwen-Image Technical Report**
  📄 [papers/Qwen-Image-Technical-Report.pdf](papers/Qwen-Image-Technical-Report.pdf)
- [ ] **Qwen3.5-Omni Technical Report**
  📄 [papers/Qwen3.5-Omni-Technical-Report.pdf](papers/Qwen3.5-Omni-Technical-Report.pdf)

### 18.2 Self-Improvement
- [ ] **Self-Improvement in Multimodal Large Language Models: A Survey**
  📄 [papers/Self-Improvement-in-Multimodal-Large-Language-Models-A-Survey.pdf](papers/Self-Improvement-in-Multimodal-Large-Language-Models-A-Survey.pdf)

### 18.3 GUI Agents
- [ ] **UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning**
  📄 [papers/UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning.pdf](papers/UI-TARS-2-Technical-Report-Advancing-GUI-Agent-with-Multi-Turn-Reinforcement-Learning.pdf)

---

## 19. MCP Model Context Protocol and Tools

- [ ] **AutoTool: Efficient Tool Selection for Large Language Model Agents**
  📄 [papers/AutoTool-Efficient-Tool-Selection-for-Large-Language-Model-Agents.pdf](papers/AutoTool-Efficient-Tool-Selection-for-Large-Language-Model-Agents.pdf)
  - → Agent Skills Section 12.9

---

## 20. World Models and Simulation

- [ ] **World Models: A Comprehensive Survey of Architectures Methodologies Reasoning Paradigms and Applications**
  📄 [papers/World-Models-A-Comprehensive-Survey-of-Architectures-Methodologies-Reasoning-Paradigms-and-Applications.pdf](papers/World-Models-A-Comprehensive-Survey-of-Architectures-Methodologies-Reasoning-Paradigms-and-Applications.pdf)
- [ ] **Agent World Model Infinity: Synthetic Environments for Agentic Reinforcement Learning**
  📄 [papers/Agent-World-Model-Infinity-Synthetic-Environments-for-Agentic-Reinforcement-Learning.pdf](papers/Agent-World-Model-Infinity-Synthetic-Environments-for-Agentic-Reinforcement-Learning.pdf)
- [ ] **Agent-World Scaling: Real-World Environment Synthesis for Evolving General Agent Intelligence**
  📄 [papers/Agent-World-Scaling-Real-World-Environment-Synthesis-for-Evolving-General-Agent-Intelligence.pdf](papers/Agent-World-Scaling-Real-World-Environment-Synthesis-for-Evolving-General-Agent-Intelligence.pdf)
- [ ] **Looped World Models**
  📄 [papers/Looped-World-Models.pdf](papers/Looped-World-Models.pdf)
- [ ] **Qwen-AgentWorld: Language World Models for General Agents**
  📄 [papers/Qwen-AgentWorld-Language-World-Models-for-General-Agents.pdf](papers/Qwen-AgentWorld-Language-World-Models-for-General-Agents.pdf)
- [ ] **Scaling Environments for LLM Agents in the Era of Learning from Interaction: A Survey**
  📄 [papers/Scaling-Environments-for-LLM-Agents-in-the-Era-of-Learning-from-Interaction-A-Survey.pdf](papers/Scaling-Environments-for-LLM-Agents-in-the-Era-of-Learning-from-Interaction-A-Survey.pdf)
- [ ] **The Latent Space: Foundation Evolution Mechanism Ability and Outlook**
  📄 [papers/The-Latent-Space-Foundation-Evolution-Mechanism-Ability-and-Outlook.pdf](papers/The-Latent-Space-Foundation-Evolution-Mechanism-Ability-and-Outlook.pdf)
- [ ] **LLMs and generative agent-based models for complex systems research**
  📄 [papers/LLMs-and-generative-agent-based-models-for-complex-systems-research.pdf](papers/LLMs-and-generative-agent-based-models-for-complex-systems-research.pdf)

---

## 21. Latent Reasoning and Interpretability

### 21.1 Latent Reasoning
- [ ] **A Survey on Latent Reasoning**
  📄 [papers/A-Survey-on-Latent-Reasoning.pdf](papers/A-Survey-on-Latent-Reasoning.pdf)
- [ ] **Implicit Reasoning in Large Language Models: A Comprehensive Survey**
  📄 [papers/Implicit-Reasoning-in-Large-Language-Models-A-Comprehensive-Survey.pdf](papers/Implicit-Reasoning-in-Large-Language-Models-A-Comprehensive-Survey.pdf)
- [ ] **Large Language Models Explore by Latent Distilling**
  📄 [papers/Large-Language-Models-Explore-by-Latent-Distilling.pdf](papers/Large-Language-Models-Explore-by-Latent-Distilling.pdf)

### 21.2 Interpretability and Mechanistic Understanding
- [ ] **Because we have LLMs we Can and Should Pursue Agentic Interpretability**
  📄 [papers/Because-we-have-LLMs-we-Can-and-Should-Pursue-Agentic-Interpretability.pdf](papers/Because-we-have-LLMs-we-Can-and-Should-Pursue-Agentic-Interpretability.pdf)
- [ ] **How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals**
  📄 [papers/How-LLMs-Detect-and-Correct-Their-Own-Errors-The-Role-of-Internal-Confidence-Signals.pdf](papers/How-LLMs-Detect-and-Correct-Their-Own-Errors-The-Role-of-Internal-Confidence-Signals.pdf)
- [ ] **How Transformers Learn to Plan via Multi-Token Prediction**
  📄 [papers/How-Transformers-Learn-to-Plan-via-Multi-Token-Prediction.pdf](papers/How-Transformers-Learn-to-Plan-via-Multi-Token-Prediction.pdf)
- [ ] **Autoregressive Language Models are Secretly Energy-Based Models**
  📄 [papers/Autoregressive-Language-Models-are-Secretly-Energy-Based-Models-Insights-into-the-Lookahead-Capabilities-of-Next-Token-Prediction.pdf](papers/Autoregressive-Language-Models-are-Secretly-Energy-Based-Models-Insights-into-the-Lookahead-Capabilities-of-Next-Token-Prediction.pdf)

### 21.3 Neuro-Science Connections
- [ ] **Remapping and navigation of an embedding space via error minimization: a fundamental organizational principle of cognition**
  📄 [papers/Remapping-and-navigation-of-an-embedding-space-via-error-minimization-a-fundamental-organizational-principle-of-cognition-in-natural-and-artificial-systems.pdf](papers/Remapping-and-navigation-of-an-embedding-space-via-error-minimization-a-fundamental-organizational-principle-of-cognition-in-natural-and-artificial-systems.pdf)
- [ ] **Shape of Thought: When Distribution Matters More than Correctness in Reasoning Tasks**
  📄 [papers/Shape-of-Thought-When-Distribution-Matters-More-than-Correctness-in-Reasoning-Tasks.pdf](papers/Shape-of-Thought-When-Distribution-Matters-More-than-Correctness-in-Reasoning-Tasks.pdf)

---

## 22. Computer Vision and Media

### 22.1 Computer Vision
- [ ] **Instance-Aware Group Quantization for Vision Transformers**
  📄 [papers/Instance-Aware-Group-Quantization-for-Vision-Transformers.pdf](papers/Instance-Aware-Group-Quantization-for-Vision-Transformers.pdf)

### 22.2 Image Generation
- [ ] **Post-training Quantization for Text-to-Image Diffusion Models**
  📄 [papers/Post-training-Quantization-for-Text-to-Image-Diffusion-Models-with-Progressive-Calibration-and-Activation-Relaxing.pdf](papers/Post-training-Quantization-for-Text-to-Image-Diffusion-Models-with-Progressive-Calibration-and-Activation-Relaxing.pdf)

### 22.3 Robotics and Embodied
- [ ] **Safety in Embodied AI: A Survey of Risks Attacks and Defenses**
  📄 [papers/Safety in Embodied AI: A Survey of Risks Attacks and Defenses.pdf](papers/Safety-in-Embodied-AI-A-Survey-of-Risks-Attacks-and-Defenses.pdf)
- [ ] **Brain-Inspired Planning for Better Generalization in Reinforcement Learning**
  📄 [papers/Brain-Inspired-Planning-for-Better-Generalization-in-Reinforcement-Learning.pdf](papers/Brain-Inspired-Planning-for-Better-Generalization-in-Reinforcement-Learning.pdf)

---

## 23. Software Engineering and Code

- [ ] **SWE-agent** → **[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](papers/SWE-agent-Agent-Computer-Interfaces-Enable-Automated-Software-Engineering.pdf)** (Section 12.8)
- [ ] **CodeClash: Benchmarking Goal-Oriented Software Engineering** → Section 12.8
- [ ] **From Code Foundation Models to Agents and Applications**
  📄 [papers/From-Code-Foundation-Models-to-Agents-and-Applications-A-Practical-Guide-to-Code-Intelligence.pdf](papers/From-Code-Foundation-Models-to-Agents-and-Applications-A-Practical-Guide-to-Code-Intelligence.pdf)
- [ ] **Advances and Frontiers of LLM-based Issue Resolution in Software Engineering**
  📄 [papers/Advances-and-Frontiers-of-LLM-based-Issue-Resolution-in-Software-Engineering-A-Comprehensive-Survey.pdf](papers/Advances-and-Frontiers-of-LLM-based-Issue-Resolution-in-Software-Engineering-A-Comprehensive-Survey.pdf)
- [ ] **VibeTensor: System Software for Deep Learning Fully Generated by AI Agents**
  📄 [papers/VibeTensor-System-Software-for-Deep-Learning-Fully-Generated-by-AI-Agents.pdf](papers/VibeTensor-System-Software-for-Deep-Learning-Fully-Generated-by-AI-Agents.pdf)
- [ ] **ASurvey on Agentic Security**
  📄 [papers/ASurvey-on-Agentic-Security-Applications-Threats-and-Defenses.pdf](papers/ASurvey-on-Agentic-Security-Applications-Threats-and-Defenses.pdf)

---

## 24. Industry and Applications

### 24.1 Industry Reports
- [ ] **Startup technical guide AI agents.pdf**
  📄 [papers/Startup-technical-guide-AI-agents.pdf](papers/Startup-technical-guide-AI-agents.pdf)
- [ ] **Kaggle Chronicles: 15 Years of Competitions Community and Data Science Innovation**
  📄 [papers/Kaggle-Chronicles-15-Years-of-Competitions-Community-and-Data-Science-Innovation.pdf](papers/Kaggle-Chronicles-15-Years-of-Competitions-Community-and-Data-Science-Innovation.pdf)

### 24.2 Recommender Systems
- [ ] **How Can Recommender Systems Benefit from Large Language Models: A Survey**
  📄 [papers/How-Can-Recommender-Systems-Benefit-from-Large-Language-Models-A-Survey.pdf](papers/How-Can-Recommender-Systems-Benefit-from-Large-Language-Models-A-Survey.pdf)
- [ ] **Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations**
  📄 [papers/Meta-Lattice-Model-Space-Redesign-for-Cost-Effective-Industry-Scale-Ads-Recommendations.pdf](papers/Meta-Lattice-Model-Space-Redesign-for-Cost-Effective-Industry-Scale-Ads-Recommendations.pdf)
- [ ] **LoRa-RL: Deep Reinforcement Learning for Resource Management in Hybrid Energy**
  📄 [papers/LoRa-RL: Deep Reinforcement Learning for Resource Management in Hybrid Energy.pdf](papers/LoRa-RL-Deep-Reinforcement-Learning-for-Resource-Management-in-Hybrid-Energy-LoRa-Wireless-Networks.pdf)

### 24.3 Document and Information Processing
- [ ] **Document Intelligence in the Era of Large Language Models: A Survey**
  📄 [papers/Document-Intelligence-in-the-Era-of-Large-Language-Models-A-Survey.pdf](papers/Document-Intelligence-in-the-Era-of-Large-Language-Models-A-Survey.pdf)
- [ ] **Information Extraction From Fiscal Documents Using LLMs**
  📄 [papers/Information-Extraction-From-Fiscal-Documents-Using-LLMs.pdf](papers/Information-Extraction-From-Fiscal-Documents-Using-LLMs.pdf)
- [ ] **A Survey on LLM-based Conversational User Simulation**
  📄 [papers/A-Survey-on-LLM-based-Conversational-User-Simulation.pdf](papers/A-Survey-on-LLM-based-Conversational-User-Simulation.pdf)

### 24.4 Education
- [ ] **Towards an AI-Augmented Textbook**
  📄 [papers/Towards-an-AI-Augmented-Textbook.pdf](papers/Towards-an-AI-Augmented-Textbook.pdf)

### 24.5 NLP
- [ ] **Causality for Natural Language Processing**
  📄 [papers/Causality-for-Natural-Language-Processing.pdf](papers/Causality-for-Natural-Language-Processing.pdf)
- [ ] **Advances in Pre-trained Language Models for Domain-Specific Text Classification**
  📄 [papers/Advances-in-Pre-trained-Language-Models-for-Domain-Specific-Text-Classification-A-Systematic-Review.pdf](papers/Advances-in-Pre-trained-Language-Models-for-Domain-Specific-Text-Classification-A-Systematic-Review.pdf)

---

## 25. Theory and Foundations

- [ ] **Foundations of Reinforcement Learning and Interactive Decision Making**
  📄 [papers/Foundations-of-Reinforcement-Learning-and-Interactive-Decision-Making.pdf](papers/Foundations-of-Reinforcement-Learning-and-Interactive-Decision-Making.pdf)
- [ ] **Position: We Need An Algorithmic Understanding of Generative AI**
  📄 [papers/Position-We-Need-An-Algorithmic-Understanding-of-Generative-AI.pdf](papers/Position-We-Need-An-Algorithmic-Understanding-of-Generative-AI.pdf)
- [ ] **Understanding Self-attention Mechanism via Dynamical System Perspective**
  📄 [papers/Understanding-Self-attention-Mechanism-via-Dynamical-System-Perspective.pdf](papers/Understanding-Self-attention-Mechanism-via-Dynamical-System-Perspective.pdf)
- [ ] **From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures**
  📄 [papers/From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures.pdf](papers/From-Memorization-to-Creativity-LLM-as-a-Designer-of-Novel-Neural-Architectures.pdf)
- [ ] **Systems and Methods for Improving Large Language**
  📄 [papers/Systems-and-Methods-for-Improving-Large-Language.pdf](papers/Systems-and-Methods-for-Improving-Large-Language.pdf)
- [ ] **A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive**
  📄 [papers/A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive.pdf](papers/A-Theory-of-Response-Sampling-in-LLMs-Part-Descriptive-and-Part-Prescriptive.pdf)
- [ ] **Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth**
  📄 [papers/Drivel-ology-Challenging-LLMs-with-Interpreting-Nonsense-with-Depth.pdf](papers/Drivel-ology-Challenging-LLMs-with-Interpreting-Nonsense-with-Depth.pdf)
- [ ] **Artificial Hivemind: The Open-Ended Homogeneity of Language Models and Beyond**
  📄 [papers/Artificial-Hivemind-The-Open-Ended-Homogeneity-of-Language-Models-and-Beyond.pdf](papers/Artificial-Hivemind-The-Open-Ended-Homogeneity-of-Language-Models-and-Beyond.pdf)
- [ ] **From AGI to ASI**
  📄 [papers/From-AGI-to-ASI.pdf](papers/From-AGI-to-ASI.pdf)

---

## 26. Domain-Specific LLMs

### 26.1 Surveys and Cross-Domain
- [ ] **Large Language Model Enhanced Knowledge Representation Learning: A Survey**
  📄 [papers/Large-Language-Model-Enhanced-Knowledge-Representation-Learning-A-Survey.pdf](papers/Large-Language-Model-Enhanced-Knowledge-Representation-Learning-A-Survey.pdf)
- [ ] **Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework**
  📄 [papers/Large-Language-Models-Meet-Extreme-Multi-label-Classification-Scaling-and-Multi-modal-Framework.pdf](papers/Large-Language-Models-Meet-Extreme-Multi-label-Classification-Scaling-and-Multi-modal-Framework.pdf)

### 26.2 Medical
- [ ] **Multimodal Large Language Models for Medicine**
  📄 [papers/Multimodal-Large-Language-Models-for-Medicine-A-Comprehensive-Survey.pdf](papers/Multimodal-Large-Language-Models-for-Medicine-A-Comprehensive-Survey.pdf)
- [ ] **Rationale-Guided Retrieval Augmented Generation for Medical Question Answering**
  📄 [papers/Rationale-Guided-Retrieval-Augmented-Generation-for-Medical-Question-Answering.pdf](papers/Rationale-Guided-Retrieval-Augmented-Generation-for-Medical-Question-Answering.pdf)
- [ ] **SymptomAI** Section 12.10
  📄 [papers/SymptomAI-Towards-a-Conversational-AI-Agent-for-Everyday-Symptom-Assessment.pdf](papers/SymptomAI-Towards-a-Conversational-AI-Agent-for-Everyday-Symptom-Assessment.pdf)

### 26.3 Scientific
- [ ] **Intern-S1: A Scientific Multimodal Foundation Model**
  📄 [papers/Intern-S1: A Scientific Multimodal Foundation Model.pdf](papers/Intern-S1-A-Scientific-Multimodal-Foundation-Model.pdf)
- [ ] **A Survey of Scientific Large Language Models**
  📄 [papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf](papers/A-Survey-of-Scientific-Large-Language-Models-From-Data-Foundations-to-Agent-Frontiers.pdf)

---


## 27. Miscellaneous and Other

- [ ] **Quantum-ML-II.pdf** course slides
  📄 [papers/Quantum-ML-II.pdf](papers/Quantum-ML-II.pdf)
- [ ] **Course-Summary && Quantum Machine Learning I** course notes
  📄 [papers/Course-Summary-&&-Quantum-Machine-Learning-I.pdf](papers/Course-Summary-&&-Quantum-Machine-Learning-I.pdf)
