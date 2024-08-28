# What is this?

![screenshot](assets/chat_screenshot.png)

*TL;DR*: `repo2vec` is a simple-to-use, modular library enabling you to chat with any public or private codebase.

**Ok, but why chat with a codebase?**

Sometimes you just want to learn how a codebase works and how to integrate it, without spending hours sifting through
the code itself. 

`repo2vec` is like GitHub Copilot but with the most up-to-date information about your repo. 

Features: 
- **Dead-simple set-up.** Run *two scripts* and you have a functional chat interface for your code. That's really it.
- **Heavily documented answers.** Every response shows where in the code the context for the answer was pulled from. Let's build trust in the AI.
- **Plug-and-play.** Want to improve the algorithms powering the code understanding/generation? We've made every component of the pipeline easily swappable. Customize to your heart's content.

Here are the two scripts you need to run:
```
pip install -r requirements.txt

export GITHUB_REPO_NAME=...
export OPENAI_API_KEY=...
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=...

python src/index.py $GITHUB_REPO_NAME --pinecone_index_name=$PINECONE_INDEX_NAME
python src/chat.py $GITHUB_REPO_NAME --pinecone_index_name=$PINECONE_INDEX_NAME
```
This will index your entire codebase in a vector DB, then bring up a `gradio` app where you can ask questions about it. 

The assistant responses always include GitHub links to the documents retrieved for each query.

If you want to publicly host your chat experience, set `--share=true`:
```
python src/chat.py $GITHUB_REPO_NAME --share=true ...
```

That's it.

Here is, for example, a conversation about the repo [Storia-AI/image-eval](https://github.com/Storia-AI/image-eval):
![screenshot](assets/chat_screenshot.png)

# Peeking under the hood

## Indexing the repo
The `src/index.py` script performs the following steps:
1. **Clones a GitHub repository**. See [RepoManager](src/repo_manager.py).
    - Make sure to set the `GITHUB_TOKEN` environment variable for private repositories.
2. **Chunks files**. See [Chunker](src/chunker.py).
    - For code files, we implement a special `CodeChunker` that takes the parse tree into account.
3. **Batch-embeds chunks**. See [Embedder](src/embedder.py).
    - By default, we use OpenAI's [batch embedding API](https://platform.openai.com/docs/guides/batch/overview), which is much faster and cheaper than the regular synchronous embedding API.
4. **Stores embeddings in a vector store**. See [VectorStore](src/vector_store.py).
    - By default, we use [Pinecone](https://pinecone.io) as a vector store, but you can easily plug in your own.

Note you can specify an inclusion or exclusion set for the file extensions you want indexed. To specify an extension inclusion set, you can add the `--include` flag:
```
python src/index.py repo-org/repo-name --include=/path/to/file/with/extensions
```
Conversely, to specify an extension exclusion set, you can add the `--exclude` flag:
```
python src/index.py repo-org/repo-name --exclude=src/sample-exclude.txt
```
Extensions must be specified one per line, in the form `.ext`.

## Chatting via RAG
The `src/chat.py` brings up a [Gradio app](https://www.gradio.app/) with a chat interface as shown above. We use [LangChain](https://langchain.com) to define a RAG chain which, given a user query about the repository:

1. Rewrites the query to be self-contained based on previous queries
2. Embeds the rewritten query using OpenAI embeddings
3. Retrieves relevant documents from the vector store
4. Calls an OpenAI LLM to respond to the user query based on the retrieved documents.

The sources are conveniently surfaced in the chat and linked directly to GitHub.

# Want your repository hosted?

We're working to make all code on the internet searchable and understandable for devs. If you would like help hosting
your repository, we're onboarding a handful of repos onto our infrastructure **for free**. 

You'll get a dedicated url for your repo like `https://sage.storia.ai/[REPO_NAME]`. Just send us a message at [founders@storia.ai](mailto:founders@storia.ai)!

![](assets/sage.gif)

# Extensions & Contributions
We built the code purposefully modular so that you can plug in your desired embeddings, LLM and vector stores providers by simply implementing the relevant abstract classes.

Feel free to send feature requests to [founders@storia.ai](mailto:founders@storia.ai) or make a pull request!
