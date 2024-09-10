<div align="center">
  <h1 align="center">repo2vec</h1>
  <p align="center">An open-source pair programmer for chatting with any codebase.</p>
  <figure>
    <img src="assets/chat_screenshot2.png" alt="screenshot" style="max-height: 500px; border: 1px solid black;">
    <figcaption align="center" style="font-size: smaller;">Our chat window, showing a conversation with the Transformers library. 🚀</figcaption>
  </figure>
</div>

# Getting started

## Installation

To install the library, simply run `pip install repo2vec`!

:exclamation: **Please make sure you have the latest version installed (as indicated [here](https://github.com/Storia-AI/repo2vec/blob/ca8ff43b2993ab0da3f0b807513cb4e3f7b0955f/setup.py#L11)). We're under rapid development and older versions quickly become obsolete.** To check your latest version, you can run `pip list | grep repo2vec`.

## Prerequisites

`repo2vec` performs two steps:

1. Indexes your codebase (requiring an embdder and a vector store)
2. Enables chatting via LLM + RAG (requiring access to an LLM)

<details open>
<summary><strong>:computer: Running locally</strong></summary>

1. To index the codebase locally, we use the open-source project <a href="https://github.com/marqo-ai/marqo">Marqo</a>, which is both an embedder and a vector store. To bring up a Marqo instance:

    ```
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

    This will open a persistent Marqo console window. This should take around 2-3 minutes on a fresh install.

2. To chat with an LLM locally, we use <a href="https://github.com/ollama/ollama">Ollama</a>:

    - Head over to [ollama.com](https://ollama.com) to download the appropriate binary for your machine.
    - Open a new terminal window
    - Pull the desired model, e.g. `ollama pull llama3.1`.

</details>

<details>
<summary><strong>:cloud: Using external providers</strong></summary>

1. We support <a href="https://openai.com/">OpenAI</a> for embeddings (they have a super fast batch embedding API) and <a href="https://www.pinecone.io/">Pinecone</a> for the vector store. So you will need two API keys:

    ```
    export OPENAI_API_KEY=...
    export PINECONE_API_KEY=...
    ```

2. Create a Pinecone index [on their website](https://pinecone.io) and export the name:
    ```
    export PINECONE_INDEX_NAME=...
    ```

2. For chatting with an LLM, we support OpenAI and Anthropic. For the latter, set an additional API key:

    ```
    export ANTHROPIC_API_KEY=...
    ```

</details>

<br>
<summary><strong>Optional</strong></summary>
If you are planning on indexing GitHub issues in addition to the codebase, you will need a GitHub token:

    export GITHUB_TOKEN=...

## Running it

<details open>
<summary><strong>:computer: Run locally</strong></summary>

1. Select your desired repository:
    ```
    export GITHUB_REPO=huggingface/transformers
    ```

2. Index the repository. This might take a few minutes, depending on its size.
    ```
    r2v-index $GITHUB_REPO
    ```

3. Chat with the repository, once it's indexed:
    ```
    r2v-chat $GITHUB_REPO
    ```
    To get a public URL for your chat app, set `--share=true`.

</details>

<details>
<summary><strong>:cloud: Use external providers</strong></summary>

1. Select your desired repository:
    ```
    export GITHUB_REPO=huggingface/transformers
    ```

2. Index the repository. This might take a few minutes, depending on its size.
    ```
    r2v-index $GITHUB_REPO \
        --embedder-type=openai \
        --vector-store=pinecone \
        --index-name=$PINECONE_INDEX_NAME
    ```

3. Chat with the repository, once it's indexed:
    ```
    r2v-chat $GITHUB_REPO \
        --vector-store-type=pinecone \
        --index-name=$PINECONE_INDEX_NAME \
        --llm-provider=openai \
        --llm-model=gpt-4
    ```
    To get a public URL for your chat app, set `--share=true`.
</details>

## Additional features

<details>
<summary><strong>:hammer_and_wrench: Control which files get indexed</strong></summary>

You can specify an inclusion or exclusion file in the following format:
```
# This is a comment
ext:.my-ext-1
ext:.my-ext-2
ext:.my-ext-3
dir:my-dir-1
dir:my-dir-2
dir:my-dir-3
file:my-file-1.md
file:my-file-2.py
file:my-file-3.cpp
```
where:
- `ext` specifies a file extension
- `dir` specifies a directory. This is not a full path. For instance, if you specify `dir:tests` in an exclusion directory, then a file like `/path/to/my/tests/file.py` will be ignored.
- `file` specifies a file name. This is also not a full path. For instance, if you specify `file:__init__.py`, then a file like `/path/to/my/__init__.py` will be ignored.

To specify an inclusion file (i.e. only index the specified files):
```
r2v-index $GITHUB_REPO --include=/path/to/inclusion/file
```

To specify an exclusion file (i.e. index all files, except for the ones specified):
```
r2v-index $GITHUB_REPO --exclude=/path/to/exclusion/file
```
By default, we use the exclusion file [sample-exclude.txt](repo2vec/sample-exclude.txt).
</details>

<details>
<summary><strong>:bug: Index open GitHub issues</strong></summary>
You will need a GitHub token first:
```
export GITHUB_TOKEN=...
```

To index GitHub issues without comments:
```
r2v-index $GITHUB_REPO --index-issues
```

To index GitHub issues with comments:
```
r2v-index $GITHUB_REPO --index-issues --index-issue-comments
```

To index GitHub issues, but not the codebase:
```
r2v-index $GITHUB_REPO --index-issues --no-index-repo
```
</details>

# Why chat with a codebase?

Sometimes you just want to learn how a codebase works and how to integrate it, without spending hours sifting through
the code itself.

`repo2vec` is like an open-source GitHub Copilot with the most up-to-date information about your repo.

Features:

- **Dead-simple set-up.** Run *two scripts* and you have a functional chat interface for your code. That's really it.
- **Heavily documented answers.** Every response shows where in the code the context for the answer was pulled from. Let's build trust in the AI.
- **Runs locally or on the cloud.**
- **Plug-and-play.** Want to improve the algorithms powering the code understanding/generation? We've made every component of the pipeline easily swappable. Google-grade engineering standards allow you to customize to your heart's content.

# Changelog

- 2024-09-06: Updated command names to `r2v-index` and `r2v-chat` to avoid clash with local utilities.
- 2024-09-03: `repo2vec` is now available on pypi.
- 2024-09-03: Support for indexing GitHub issues.
- 2024-08-30: Support for running everything locally (Marqo for embeddings, Ollama for LLMs).

# Want your repository hosted?

We're working to make all code on the internet searchable and understandable for devs. You can check out our early product, [Code Sage](https://sage.storia.ai). We pre-indexed a slew of OSS repos, and you can index your desired ones by simply pasting a GitHub URL.

If you're the maintainer of an OSS repo and would like a dedicated page on Code Sage (e.g. `sage.storia.ai/your-repo`), then send us a message at [founders@storia.ai](mailto:founders@storia.ai). We'll do it for free!

![](assets/sage.gif)

# Extensions & Contributions

We built the code purposefully modular so that you can plug in your desired embeddings, LLM and vector stores providers by simply implementing the relevant abstract classes.

Feel free to send feature requests to [founders@storia.ai](mailto:founders@storia.ai) or make a pull request!
