<div align="center">
    <a name="readme-top"></a>
    <img src="assets/storia-logo.png" alt="Logo" width="50" style="border-radius: 15px;">
    <h1 align="center">Sage: Chat with any codebase</h1>
    <div>
        <a href="https://discord.gg/zbtZe7GcVU" target=="_blank"><img alt="Discord" src="https://img.shields.io/discord/1286056351264407643?logo=discord&label=discord&link=https%3A%2F%2Fdiscord.gg%2FzbtZe7GcVU"></a>
        <a href="https://x.com/StoriaAI" target=="_blank"><img alt="X (formerly Twitter) Follow" src="https://img.shields.io/twitter/follow/Storia-AI?logo=x&link=https%3A%2F%2Fx.com%2FStoriaAI"></a>
        <a href="https://github.com/Storia-AI/sage/stargazers" target=="_blank"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Storia-AI/sage?logo=github&link=https%3A%2F%2Fgithub.com%2FStoria-AI%2Fsage%2Fstargazers"></a>
        <a href="https://github.com/Storia-AI/sage/blob/main/LICENSE" target=="_blank"><img alt="GitHub License" src="https://img.shields.io/github/license/Storia-AI/sage" /></a>
    </div>
    <div>
        <a href="https://sage-docs.storia.ai">Documentation</a>
        <span>&#183;</span>
        <a href="https://sage.storia.ai">Hosted app</a>
    </div>
    <br />
    <figure>
        <!-- The <kbd> and <sub> tags are work-arounds for styling, since GitHub doesn't take into account inline styles. Note it might display awkwardly on other Markdown editors. -->
        <kbd><img src="assets/chat_screenshot2.png" alt="screenshot" /></kbd>
        <sub><figcaption align="center">Our chat window, showing a conversation with the Transformers library. 🚀</sub></figcaption>
    </figure>
</div>

***

**Sage** is like an open-source GitHub Copilot that helps you learn how a codebase works and how to integrate it into your project without spending hours sifting through the code.

# Main features
- **Dead-simple setup**. Follow our [quickstart guide](https://sage-docs.storia.ai/quickstart) to get started.
- **Runs locally or on the cloud**. When privacy is your priority, you can run the entire pipeline locally using [Ollama](https://ollama.com) for LLMs and [Marqo](https://github.com/marqo-ai/marqo) as a vector store. When optimizing for quality, you can use third-party LLM providers like OpenAI and Anthropic.
- **Wide range of built-in retrieval mechanisms**. We support both lightweight retrieval strategies (with nothing more but an LLM API key required) and more traditional RAG (which requires indexing the codebase). There are many knobs you can tune for retrieval to work well on your codebase.
- **Well-documented experiments**. We profile various strategies (for embeddings, retrieval etc.) on our own benchmark and thoroughly [document the results](benchmarks/retrieval/README.md).

# Want your repository hosted?

We're working to make all code on the internet searchable and understandable for devs. You can check out [hosted app](https://sage.storia.ai). We pre-indexed a slew of OSS repos, and you can index your desired ones by simply pasting a GitHub URL.

If you're the maintainer of an OSS repo and would like a dedicated page on Code Sage (e.g. `sage.storia.ai/your-repo`), then send us a message at [founders@storia.ai](mailto:founders@storia.ai). We'll do it for free!

![](assets/sage.gif)

# Extensions & Contributions

We built the code purposefully modular so that you can plug in your desired embeddings, LLM and vector stores providers by simply implementing the relevant abstract classes.

Feel free to send feature requests to [founders@storia.ai](mailto:founders@storia.ai) or make a pull request!

# Contributors

<a href="https://github.com/Storia-AI/sage/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=Storia-AI/sage"/>
</a>

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
