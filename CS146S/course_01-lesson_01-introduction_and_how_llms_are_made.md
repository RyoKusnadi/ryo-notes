# CS146S: The Modern Software Developer
---
## Lecture 1: Introduction & How LLMs Are Made

## 🌍 State of the World

### The Paradigm Shift
Software development has transitioned from **0→1 code creation** to an **iterative AI-augmented workflow**:

```Plan → Generate with AI → Validate → Refine → Deploy → Monitor```


### The Reality Check

#### Challenges
- **Rapid tooling churn**: AI dev tools evolve weekly; no stable "best practices" yet
- **Hallucination risk**: AI-generated code requires rigorous validation before deployment
- **Cost complexity**: Reasoning tokens can cost 10–100× more than standard inference
- **Context management**: Poor context curation leads to degraded output quality ("context rot")

#### Opportunities
- **Unprecedented productivity**: Engineers using modern AI tooling ship features 10–100× faster
- **Accelerated learning**: AI acts as a force multiplier for adopting new languages, frameworks, and paradigms
- **Democratized expertise**: Junior engineers can leverage AI to produce senior-level output with proper guidance
- **Strategic leverage**: Focus shifts from writing code to orchestrating agents, designing systems, and validating outcomes

> **Core mantra**: *"You won't be replaced by AI. You'll be replaced by a competent engineer who knows how to use AI."*

---

## 🎯 Core Philosophy: What This Course Teaches

This is **not** the "vibe coding" class. We focus on deliberate, high-leverage engineering practices.

| Principle | Technical Meaning | Why It Matters in 2026 |
|-----------|------------------|------------------------|
| **Human-Agent Engineering** | Focus on skills AI hasn't mastered: business context, architectural judgment, stakeholder alignment, system design | Positions you as the orchestrator/tech lead, not just a prompter or code reviewer |
| **Context = Quality** | "Good context leads to good code. If you can't understand your codebase, neither will an LLM." | LLMs mirror your input. Context engineering—curating the right information—is now a core engineering skill |
| **Develop "Taste"** | Read and review massive amounts of code. Learn to instantly spot clean, maintainable, correct implementations vs. fragile or wrong ones | You are the quality gatekeeper when AI generates at scale. Taste is your defensible, non-automatable skill |
| **Experiment Aggressively** | No established AI dev patterns exist yet. Everyone is iterating in real-time. | Treat tools, prompts, and workflows as hypotheses. Test, measure, and adopt what works for your context |

---

## ⚙️ How LLMs Work: Engineer's Deep Dive

### 🔁 Core Architecture: Autoregressive Next-Token Prediction

LLMs are next-token prediction engines. The inference pipeline:
```
Input Text
↓
Tokenization (BPE/Unigram → fixed vocabulary chunks)
↓
Embedding Layer (tokens → 1K–3K dimensional vectors)
↓
Transformer Layers (12–96+ layers)
↓
Probability Distribution (softmax over vocab for next token)
↓
Sampling (temperature/top-p control randomness; greedy/beam for deterministic)
↓
Append token to context → Repeat until stop condition

```


### 🚀 Key Architecture Innovations

#### Multi-Head Latent Attention (MLA)
- Compresses key/value tensors into lower-dimensional latent space before storing in KV cache
- Projects back to full dimension during attention computation
- **Benefit**: ~40% KV cache memory reduction with maintained or improved modeling performance
- **Used in**: DeepSeek V3, DeepSeek R1

#### Mixture-of-Experts (MoE) — Industry Standard
Sparse MoE activates only a subset of experts per token, scaling capacity without linear compute cost.

| Model | Total Params | Active Params | Experts | Routing Strategy |
|-------|-------------|---------------|---------|-----------------|
| DeepSeek V3/R1 | 671B | 37B | 256 (9 active) | Top-9 + 1 shared expert |
| Llama 4 Maverick | 400B | 17B | 128 (2 active) | Alternating MoE/dense layers |
| Qwen3-235B-A22B | 235B | 22B | 160 (8 active) | Load-balanced top-k routing |
| Mixtral 8x22B | 141B | 39B | 8 (2 active) | Token-level expert selection |

> Only active parameters consume compute per token. MoE enables frontier quality at manageable inference cost.

#### Sliding Window Attention
- Gemma 3 uses a 5:1 ratio of local (sliding window) to global attention layers
- Window size reduced from 4K (Gemma 2) to 1K
- **Benefit**: ~60% KV cache memory reduction for long-context tasks with minimal perplexity loss
- **Use case**: Code completion, chat history, document synthesis

#### Normalization Innovations
| Technique | Placement | Purpose | Models Using |
|-----------|-----------|---------|-------------|
| QK-Norm | Inside attention, before RoPE | Stabilizes training, prevents gradient explosion | OLMo 2, Gemma 3, Qwen3 |
| Post-Norm RMSNorm | After attention/FFN (inside residual) | Improves training stability vs. Pre-Norm | OLMo 2 |
| Dual Pre/Post-Norm | Both before and after attention | Redundancy for maximum stability | Gemma 3 |

#### Inference Optimizations
| Optimization | What It Does | Latency Impact | Adoption |
|-------------|-------------|----------------|----------|
| FlashAttention-3 | Fused, tiled attention with block-sparse KV cache | 2–4× faster prefill; 30–70% lower decode | vLLM, SGLang, TensorRT-LLM |
| PagedAttention | Recycles KV cache memory blocks like virtual memory | Enables 100K+ context without OOM | vLLM (standard) |
| Speculative Decoding | Small draft model proposes tokens; large model verifies | 2–3× speedup for deterministic tasks | Claude Code, Cursor, Windsurf |
| QuantSpec | Hierarchical quantized KV cache for draft verification | 2.5× speedup vs. baseline speculative | Production Q2 2026 |
| Ring Attention | Distributes context across GPUs via coordinated communication | Enables 1M+ token contexts | Gemini 2.5 Pro, research prototypes |

---

## 📚 Training Process

### Stage 1: Self-Supervised Pretraining
```
Goal: Teach model the "notion" of language & code structure
Data: 100s of billions → trillion+ tokens
Sources: Common Crawl, Wikipedia, StackExchange, Public GitHub, curated code corpora
Loss: Next-token prediction (cross-entropy)

Example:
Prompt: "Write a for loop"
Output: "that could be used in a piece of code" (continuation, not instruction-following)
```

### Stage 2: Supervised Fine-Tuning (SFT)
```
Goal: Teach model to follow instructions & format outputs
Data: Tens → hundreds of thousands of high-quality, curated prompt-response pairs

Enhancements:
• Synthetic data generation with execution verification loops
• Code-specific SFT using test-validated examples
• Multi-turn dialogue SFT for agent workflows
Example:
Prompt: "Write a for loop in Python that prints 0–9"
Output: "for i in range(10): print(i)" (instruction-following)
```

### Stage 3: Preference Tuning
| Method | Core Idea | Best Use Case | Key Advantage |
|--------|-----------|---------------|---------------|
| **DPO** | Logistic loss comparing preferred vs. rejected; no separate reward model | Baseline alignment; stable, simple | Removes RL instability; scales linearly with data |
| **SimPO** | Removes explicit reference log-ratio; softer gradients | Noisy/crowd-sourced preference data | Tolerates label noise; prevents mode collapse |
| **ORPO** | Reframes preference in odds-space; normalizes class imbalance | Highly imbalanced data (compliance, moderation) | Prevents minority-class gradient vanishing |
| **KTO** | Asymmetric loss: penalizes high-impact failures more heavily | High-risk domains (legal, healthcare, finance) | Models human loss-aversion; risk-sensitive alignment |
| **GRPO** | Uses group-average as baseline instead of learned critic (DeepSeek-R1) | Pure RL training without SFT demonstrations | Cuts training memory by ~50%; no critic model needed |
| **IRPO** | Implicit policy regularization for smoother optimization | Production fine-tuning with limited preference data | Better generalization; less overfitting to preference labels |

> **Modern Pipeline**: SFT → SimPO (broad shaping) → ORPO (handle imbalance) → KTO (risk domains) → DPO/IRPO (final polish)

---

## 🧠 Reasoning Models: Test-Time Compute Scaling

### Model Comparison
| Model | Key Innovation | AIME | Cost Tier | Open Weights | Thinking Mode |
|-------|---------------|-----------|-----------|-------------|---------------|
| GPT-4o | Standard decoder-only | 13% | Low | No | None |
| o3 (OpenAI) | Hidden CoT + large-scale RL + process reward model | 96.7% | High | No | Fully hidden tokens |
| o4-mini | Optimized for STEM/code at o3-class quality, lower cost | ~90% | Medium | No | Hidden tokens |
| DeepSeek-R1 | Pure RL (R1-Zero) + GRPO + cold-start data shaping | 79.8% | Low | ✅ Yes | Streamed inline CoT |
| Claude Opus 4.7 | Extended thinking budget + interleaved tool-use reasoning | ~94% | High | No | Inspectable thinking block |
| Gemini 2.5 Pro Deep Think | 128K long-context RL + parallel hypothesis exploration | 92.0% | Medium | No | Parallel exploration streams |
| GPT-5.5 | Controllable thinking budget + hidden trace summary | *landing* | Highest | No | Configurable hidden/visible |

### R1-Zero Breakthrough: Pure RL Without SFT Demonstrations
DeepSeek proved reasoning can emerge from **pure RL without SFT demonstrations**:
```
Start from pretrained base model (DeepSeek-V3-Base)
Apply RL with simple outcome reward:
• Correct final answer → +1
• Incorrect → 0
• No process rewards, no human-written reasoning traces

Use GRPO for advantage estimation (no critic model):
• Generate G responses per prompt
• Compute advantage_i = reward_i - mean(rewards)
• Update policy based on relative performance

Emergent behaviors (not programmed!):
• Self-verification: "Wait, let me check step 3 again..."
• Backtracking: "This approach is getting complicated. Try different method..."
• Strategy selection: "Direct computation too complex. Find pattern first..."
```

### Practical Implications for Engineers
- **Thinking tokens cost money**: One o3 request for a hard problem may use 5K–20K internal tokens
- **Latency trade-off**: Reasoning models take 15–90s for hard problems vs. 1–3s for standard models
- **Distillation is your friend**:
```
R1-Distill-Qwen-7B: 55.5% AIME, runs on RTX 4090 (~14 GB VRAM)
R1-Distill-Qwen-1.5B: 28.9% AIME, runs on CPU (~4 GB RAM)
```

Distillation beats direct RL on the small model and is far cheaper.
- **Hybrid workflows dominate**: Use reasoning model for planning/reflection, fast model for execution.

---

## ⚖️ Strengths & Limitations in Practice — 2026 Mitigations

### ✅ What LLMs Excel At
- Expert-level code completion & autocomplete across languages
- Deep code understanding across multiple files/languages (with proper context)
- Automated code fixing, refactoring, & bug triage (when given error traces + docs)
- Rapid prototyping: PRD → architecture → scaffold in minutes
- Multi-document synthesis and legal/contract analysis with long context

### ⚠️ Critical Limitations & Engineering Mitigations

| Limitation | Technical Reality (2026) | Practical Mitigation |
|------------|--------------------------|---------------------|
| **Hallucinations** | Generates non-existent/deprecated APIs; logic errors in novel domains | • Context engineering: Provide exact API docs, type signatures<br>• RAG + verification: Retrieve authoritative docs; ask model to cite sources<br>• Tool-use integration: Let model call APIs/docs at runtime |
| **Context Window Limits** | ~128K–1M tokens capacity, but attention isn't uniform | • Strategic context slicing: Send only relevant files, use RAG<br>• PagedAttention / Streaming KV: Recycle memory blocks<br>• Ring Attention: Distribute context across GPUs |
| **Latency** | Seconds to minutes per request; reasoning models add 15–90s | • Speculative decoding: Small draft model proposes, large model verifies<br>• QuantSpec: Hierarchical quantized KV cache → 2.5× speedup<br>• Plan & delegate: Break tasks into parallel sub-prompts |
| **Cost** | $1–3/M input tokens, $10+/M output tokens; reasoning tokens cost 10–100× more | • Token optimization: Cache prompts, strip boilerplate<br>• Model routing: Simple queries → small models; hard tasks → large models<br>• Distillation: Deploy distilled small models for common patterns |
| **Knowledge Cutoff** | May not know latest libraries/frameworks post-training | • Enable web search tool: Let model retrieve up-to-date docs<br>• Provide docs in context: Attach relevant API references, changelogs<br>• Fine-tune on internal docs: Use LoRA/QLoRA for domain adaptation |
| **Context Rot** | Performance degrades as context windows grow with poorly curated information | • Active context curation: Continuously validate and prune context<br>• Hierarchical memory: Separate short-term, working, and long-term memory<br>• Adaptive retrieval: Decide *when* to retrieve, not just *what* |

---

## 🛠️ Practical Engineering Checklist: Before You Prompt
### Context Hygiene (Non-Negotiable)
```markdown
✅ File Selection
 • Include only relevant files + imports (avoid dumping entire repo)
 • Use semantic search / embeddings to retrieve semantically similar code

✅ Documentation Injection
 • Attach API specs, type definitions, error codes
 • Include design docs, PRDs, or architecture diagrams for complex tasks

✅ Execution Context
 • Provide error logs, stack traces, or test failures when debugging
 • Include environment details: framework version, deployment target, feature flags

✅ Security Boundaries
 • Never paste secrets/API keys into prompts
 • Use environment variables + .env files; sanitize user input before agent processing
```

### Prompt Structure for High-Quality Output
```
## Role
[Define agent's expertise: "You are a senior backend engineer specializing in Go and distributed systems"]

## Objective
[Clear, measurable goal: "Refactor the auth middleware to support JWT refresh tokens with zero downtime"]

## Constraints
- Language/Framework: Go 1.23+, Gin framework v1.10
- Performance: <5ms p99 latency impact; <1% CPU overhead
- Security: Follow OWASP API Security Top 10 2026; no hardcoded secrets
- Compatibility: Must not break existing /v1/auth endpoints

## Context Provided
- File: `middleware/auth.go` (attached)
- API Spec: `openapi.yaml` v3.1 (attached)
- Error Logs: `auth_errors.log` last 24h (attached)

## Output Format
- Return only Go code + brief explanation (<100 words)
- Include unit tests for new functionality (table-driven tests)
- Flag any breaking changes explicitly with migration steps
```

```
Iteration Loop: Validate → Refine → Ship
```

```
1. Generate: Agent proposes solution with reasoning trace
2. Validate: 
   • Run linter + formatter
   • Execute unit/integration tests
   • Run SAST/DAST tools (Semgrep, CodeQL)
3. Refine: 
   • If tests fail: provide error trace + ask for fix
   • If security flag: ask for alternative implementation
   • If performance concern: request optimization pass
4. Document: 
   • Update PR description with AI-assisted summary
   • Add comments for non-obvious logic
   • Record prompt + output pair for future reuse
5. Deploy & Monitor:
   • Canary release with feature flag
   • Monitor: error rates, latency, security alerts
   • Feed observations back into context engineering loop
```

> **Golden Rule**: The LLM is a junior engineer with encyclopedic knowledge but no business context. You are the tech lead. Your job is to provide context, set constraints, and validate output — and to know when to use reasoning models vs. fast models.

## 📚 Recommended Technical Resources (2026)

### Core References
- [Context Engineering Guide](https://www.promptingguide.ai/guides/context-engineering-guide)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [AI Reasoning Models: Complete 2026 Prompting Guide](https://sureprompts.com/blog/ai-reasoning-models-prompting-complete-guide-2026)
- [Post-Training Stack: SimPO/ORPO/KTO/IRPO](https://mbrenndoerfer.com/writing/dpo-variants-ipo-kto-orpo-cdpo-llm-alignment)

### Tools & Frameworks

| Tool | Best For | Link |
|------|----------|------|
| **vLLM** | High-throughput serving with PagedAttention | [vllm.ai](https://vllm.ai) |
| **SGLang** | Structured generation + constrained decoding | [sgl-project.github.io](https://sgl-project.github.io) |
| **Semgrep** | AI-powered static analysis + security scanning | [semgrep.dev](https://semgrep.dev) |
| **Cursor / Windsurf** | Polished AI IDEs with context-aware generation | [cursor.com](https://cursor.com) |
| **Claude Code** | Complex reasoning, tool use, long-context debugging | [anthropic.com](https://www.anthropic.com) |
| **DeepSeek-R1 Distilled** | Local reasoning models (7B–32B) for edge deployment | [huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai) |
| **MCP (Model Context Protocol)** | Standardized tool/data connection for agents | [modelcontextprotocol.io](https://modelcontextprotocol.io) |

---

## ✅ Action Items

- [ ] Experiment with 3 prompt structures on the same coding task; document which yields best output quality + lowest token cost
- [ ] Set up a local distilled reasoning model (e.g., R1-Distill-Qwen-7B) and benchmark vs. API calls
- [ ] Implement context slicing: retrieve only relevant files via embeddings before prompting
- [ ] Add a "validation step" to your workflow: run linter + tests + SAST before accepting AI output
- [ ] Start a "prompt journal": log what prompt structures + model selections work for different task types
- [ ] Explore MCP connectors for your stack: connect your AI agent to databases, APIs, and internal tools

---

> **Final Thought**: This isn't about replacing your engineering judgment. It's about **amplifying it with deliberate context engineering and strategic model selection**.

The engineers who thrive in 2026+ won't be those who prompt best—they'll be those who understand:
>
> - **When** to use reasoning models vs. fast models
> - **How** to provide the right context (γ-covering, hierarchical memory, adaptive retrieval)
> - **What** to validate before shipping (security, performance, context citation accuracy)
> - **Which** architectural patterns to apply (MoE routing, MLA vs. GQA, sliding window)


Focus on the skills that aren't yet automated: problem framing, architectural tradeoffs, stakeholder communication, and quality judgment. The tools will keep evolving—your ability to learn, adapt, and lead will not.
