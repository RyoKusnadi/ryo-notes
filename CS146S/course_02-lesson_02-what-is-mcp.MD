
# What is MCP? How to build an MCP?

## Part 1: What is MCP?
**Model Context Protocol (MCP)** is an open protocol that allows systems to provide context to AI models in a manner that is generalizable across integrations. In plain English, it is a **standard format for exposing tools to LLMs**.

### The Problem It Solves
* **Static Knowledge:** LLMs possess vast world knowledge, but it is static and only updates when the model is retrained. 
* **Need for Dynamic Data:** To build fully autonomous systems, models need robust ways to access real-time, dynamic data (e.g., today's weather, the current president, the price of Bitcoin, or specific details about a new ad campaign). While RAG and tool-calling are the best answers today, implementing them was historically chaotic.
* **The "$M \times N$" Nightmare:** Before MCP (pre-November 2024), integrating multiple LLM applications ($M$) with multiple third-party APIs ($N$) required building $M \times N$ custom connectors. Developers had to repeatedly reimplement authentication, error handling, and rate-limiting for every single integration.

### The MCP Solution
MCP eliminates the need to build $M \times N$ connectors, reducing the integration complexity to **$M + N$**. It enforces a consistent output format using **JSON-RPC** and extends concepts from Language Server Protocols (LSP), but crucially, it allows for **proactive agentic workflows** rather than purely reactive ones.

---

## Part 2: Core Terminology
To understand how to build or integrate an MCP, you must understand its four core components:
1. **Host:** The end-user application interfacing with the AI (e.g., Cursor, Claude Desktop).
2. **MCP Client:** A library embedded within the Host that maintains a stateful session with each connected server.
3. **MCP Server:** A lightweight wrapper that sits in front of a specific tool or data source.
4. **Tool:** The actual callable function, which could be a local script, an external API, or a database query.

---

## Part 3: How to Build an MCP (Architecture & Flow)
Building a custom MCP server involves setting up the server to communicate with the client using a standardized flow. MCP supports two primary transport layers: **`stdio`** (for local, same-machine execution) and **`SSE`** (Server-Sent Events, for remote/network-based execution).

### The Standard MCP Workflow:
1. **Ask Query:** The user submits a prompt to the Host application.
2. **Get Tools (`tools/list`):** The MCP Client calls the `tools/list` endpoint on the MCP Server to ask, *"What can you do?"*
3. **Receive Tool Definitions:** The MCP Server returns a JSON payload describing each available tool, including its **name**, a **summary**, and its **JSON schema** (required parameters).
4. **Context Injection & Tool Selection:** The Host injects this JSON tool description into the model’s context window. The user's prompt triggers the model, which then emits a structured tool call with the necessary parameters.
5. **Execution & Response:** The MCP Server executes the underlying function for the tool, gets the response, sends it back to the Host, and the conversation resumes with the new data.

*(Note: In practice, building the server involves using an MCP SDK/library—like Python's `FastMCP`—to wrap your existing functions with a tool decorator, automatically registering them to be served via the protocol).*

---

## Part 4: Current Limitations & Best Practices
When building and designing MCP tools, keep the following constraints in mind:
* **Tool Overload:** Agents do not handle a large number of tools very well today. Performance degrades as the number of available tools increases.
* **Context Window Consumption:** API descriptions and tool schemas are verbose. Feeding too many tool definitions into the prompt will quickly eat up the LLM's context window.
* **AI-Native Design:** Design APIs to be **AI-native** rather than rigid. Instead of forcing the LLM to parse complex, deeply nested JSON meant for human developers, consider returning simplified formats (like natural language summaries or CSV) that the LLM can easily comprehend and act upon.