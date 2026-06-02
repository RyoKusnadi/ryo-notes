# LLM Power Prompting

## 1. The Core Paradigm
*   **Prompts as the Primary Instruction Interface:** Prompts are the *lingua franca* for instructing LLMs (Courtesy of Andrej Karpathy). Prompting is effectively programming using natural language intent. It requires the same rigor, version control, testing, and iterative debugging as traditional development.
*   **The Evolution of Human-Machine Interaction:** To understand the value of prompting, we must look at the factual evolution of how computational outcomes (like sentiment analysis) are achieved:
    *   **Software 1.0 (Explicit Rules):** Developers manually defined arrays of positive/negative keywords and wrote explicit logic to score text. Highly inefficient and impossible to scale for linguistic nuances.
    *   **Machine Learning (Pattern Training):** Developers fed thousands of labeled examples into a classifier. The algorithm mathematically found patterns to distinguish text, requiring massive datasets and training time.
    *   **Software 3.0 (Natural Language Instruction):** With LLMs, we simply provide a natural language instruction ("You are a sentiment classifier") and a few examples directly in the text input. The LLM processes this context and generates the classification instantly, without explicit rule-writing or model training.
*   **Search Engine Evolution:** This mirrors how search interactions evolved. Users previously had to use strict boolean operators (e.g., `corgi AND (cute OR funny)`). Today, users simply type natural language queries, and the system understands the semantic intent.
*   **The Black Box Reality & Empirical Control:** LLMs are opaque; the internal mathematical transformations during inference cannot be scientifically verified by the user. However, empirical evidence consistently shows that specific phrasing and techniques improve output quality, accuracy, and human satisfaction. Techniques that started as external prompting tricks (like Chain-of-Thought) were so effective that AI companies have now baked them directly into model architectures (e.g., modern "thinking" models).

---

## 2. Foundational Prompting Techniques
These techniques dictate how we structure the core request to maximize accuracy and adherence to patterns.

### Zero-Shot Prompting
*   **Definition:** Asking the LLM to perform a task without providing any supporting context, background information, or examples. The model relies entirely on its pre-trained weights.
*   **When to Use:** Simple, ubiquitous tasks where the model has seen massive amounts of training data.
    *   *Example:* "Write me a Rust for-loop that iterates over a list of strings, printing every value in an even index."
*   **Limitation:** Fails on proprietary logic, internal company conventions, or tasks requiring up-to-date knowledge not present in the training data.

### K-Shot (Few-Shot) Prompting
*   **Definition:** Also known as *In-Context Learning*. Providing the model with $k$ examples (commonly 1, 3, or 5) of the desired input-output behavior before asking it to perform the actual task.
*   **When to Use:** Pattern matching, enforcing specific naming conventions, formatting outputs, or mimicking a specific tone. Highly effective for subjective tasks (e.g., "make it funny") where textual descriptions fail, but direct examples succeed.
    *   *Example (Enforcing Naming Conventions):* Providing examples like `<example> var StRaRrAy=['cat', 'dog', 'wombat'] </example>` to teach the alternating capitalization pattern.
*   **Crucial Engineering Detail:** **Example Diversity.** If you provide 5 examples that are too similar, smaller models will over-index on the specific text rather than the underlying rule, leading to repetitive or incorrect outputs. Examples must cover different variations to teach the actual logic. Avoid over-constraining the model with too many edge-case rules in the examples, as this degrades general performance.

### Chain-of-Thought (CoT) Prompting
*   **Definition:** Forcing the model to generate intermediate logical steps before arriving at the final answer.
*   **Multi-shot CoT:** Providing examples that include the full logical trace. The examples do not need to be about the exact same task; they just need to demonstrate the *structure of logical reasoning*.
    *   *Example:* Asking to check if a number is a perfect cube and square. The prompt provides reasoning examples for *different* tasks (e.g., finding the maximum element, checking for a palindrome) to teach the model *how* to break down a coding problem step-by-step.
*   **Zero-shot CoT:** Simply adding the phrase *"Let's think step-by-step"* before the task. This triggers the model to access reasoning-heavy patterns from its training data (such as textbooks or analytical blogs), significantly improving accuracy on complex tasks.
    *   *Example:* "Think step by step about the subproblems before coding. Include worked out examples of subarrays you are considering..."
*   **Implementation:** Use explicit XML tags (e.g., `<reasoning>` for the thought process, `<answer>` for the final output) to separate the logic from the deliverable.

### Self-Consistency Prompting
*   **Definition:** Querying the model multiple times (e.g., 5 times) for the same task, typically combined with CoT, and selecting the most frequent final answer (majority vote).
*   **Mechanism:** By sampling diverse reasoning paths, the model is less likely to confidently commit to a single hallucinated or incorrect logical leap. The correct answer usually emerges as the statistical consensus.
    *   *Example:* Asking "What’s the root cause for this `IndexError`?" 5 times and taking the majority result.
*   **Trade-offs:** Increases API costs and latency. In production environments, these parallel requests must be executed simultaneously to maintain acceptable response speeds.

---

## 3. Autonomy & Environmental Interaction
These techniques allow the LLM to break out of its isolated text-generation bubble and interact with the real world or deterministic systems.

### Tool Use
*   **Definition:** Allowing the LLM to defer tasks to external, deterministic systems (e.g., web search, calculators, CI/CD test runners).
*   **Mechanism:** The prompt explicitly defines available tools and their required parameters. The LLM generates the exact syntax to invoke the tool, the external system executes it, and the result is fed back to the LLM.
*   **Value:** Drastically reduces hallucinations. The model does not guess the result of a test run; it lets the deterministic tool execute it. This is the foundation for enabling autonomous agent behavior.
    *   *Example:* `<tools> pytest -s /path/to/unit_tests </tools>` to ensure CI tests pass after fixing an `IndexError`.

### Retrieval-Augmented Generation (RAG)
*   **Definition:** Infusing the LLM with specific, up-to-date contextual data (code snippets, documentation, user profiles) *before* it generates a response.
*   **Mechanism:** Instead of relying solely on pre-trained weights, the model reads the provided context and bases its answer on that immediate information.
*   **Benefits:** 
    1. Keeps the model current without the massive cost of retraining (faster iteration).
    2. Provides free interpretability and citations (you can see exactly which document the model referenced).
    3. Grounds the answer in facts, significantly reducing hallucinations.
    *   *Example:* Extending `UserAuthService` by providing the current `<code_snippet>` and the `<url>` to the `requests-oauthlib` documentation.

### Reflexion (Self-Correction)
*   **Definition:** Forcing the model to critique its own output based on environmental feedback before finalizing it.
*   **The Loop:** 
    1. **Act:** Model generates an initial response or code.
    2. **Observe:** The environment provides a signal (e.g., a unit test fails).
    3. **Reflect:** Append a prompt suffix: *"Now critique your answer. Was it correct? If not, explain why and try again."*
    4. **Extend & Retry:** The model incorporates the failure reason into its context and generates a revised output.
*   **Example:** Extending `company_location` to handle string/JSON. The environment throws a `JSONDecodeError`. The model observes this, reflects ("I must ensure that when a string is provided... it doesn't throw this error"), extends the prompt, and rewrites the code.
*   **Application:** This multi-turn reflection loop is the core mechanism behind modern autonomous coding agents, allowing them to self-correct without human intervention.

---

## 4. Prompt Architecture & Terminology
Understanding how the LLM parses the conversation history is critical for structuring complex prompts.

*   **System Prompt:** The foundational, hidden instructions provided at the very beginning of the context. It defines the persona, rules, baseline constraints, and safety guardrails. 
    *   *Critical Detail:* Always dynamically inject the **current date and time** into the system prompt. The model has no inherent awareness of the current date unless explicitly told.
    *   *Example:* Leaked system prompts from major providers show they define the model's identity and core directives (similar to Asimov's laws of robotics), instructing it on how to handle users attempting to anthropomorphize the AI.
*   **User Prompt:** The explicit instructions, tasks, and context provided by the human (or the application wrapper). This is where techniques like Few-Shot and CoT are usually injected.
*   **Assistant Prompt:** The actual text generated by the LLM.
*   **The Interaction Loop & Tension:** The flow is System -> User -> Assistant -> User -> Assistant. There is often a tension between the System Prompt (which might instruct the model not to reveal its rules) and the User Prompt (where users try to extract the system prompt or bypass safety guardrails via "jailbreaking"). Major providers handle this via extensive alignment training, but custom implementations require careful tuning.
*   **Structural Formatting:** Use XML tags (e.g., `<log>`, `<error>`, `<context>`, `<code_snippet>`) to clearly delineate different sections of your input. LLMs are highly optimized to recognize and parse these structural patterns from their training data.

---

## 5. Production Best Practices
*   **Clear Prompting (The "Stranger" Test):** Give your prompt to someone with minimal context about your project. If they are confused, the LLM will be too. Avoid implicit assumptions.
*   **Aggressive Role Prompting:** Defining a specific persona drastically alters the verbosity, tone, and formatting of the output. 
    *   *Example 1 (Pedantic):* "You are a helpful assistant that loves programming at the level of a senior software developer and is very detailed and pedantic..." $\rightarrow$ *Yields exhaustive, strict code.*
    *   *Example 2 (Casual):* "You are a Gen Z digital bestie. Always sound like you’re texting on Snapchat at 2am." $\rightarrow$ *Yields casual, emoji-heavy, conversational responses.*
*   **Explicit Constraints:** Never assume the model knows your tech stack, library versions, or formatting rules. State them explicitly in the System or User prompt.
*   **Task Decomposition:** Break complex, multi-step requests into smaller, sequential sub-tasks. 

---

## 6. 2025/2026 Industry Evolution (State-of-the-Art)
To ensure this playbook reflects the current engineering landscape, the following concepts have evolved from the foundational lecture material:

*   **From Prompt Engineering to Context Engineering:** 
    The discipline has expanded beyond writing a single text prompt. Context Engineering is now the systematic design, management, and optimization of *all* information fed into the LLM during inference. This includes dynamically managing the context window limits, integrating real-time memory, orchestrating RAG pipelines, and formatting tool outputs so the LLM can plausibly execute the task.
*   **Model Context Protocol (MCP) for Tool Use:**
    While the lecture explains the *concept* of Tool Use, the current industry standard for *implementing* it is the Model Context Protocol (MCP). MCP is an open specification that standardizes how AI applications discover and interact with external data sources and tools. Instead of writing custom integration code for every new tool or database, developers now use MCP servers to securely expose capabilities to the LLM, making Tool Use highly modular, secure, and enterprise-ready.

---