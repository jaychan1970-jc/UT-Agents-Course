---
name: refactor-for-windows-and-openrouter
description: Refactor Jupyter notebooks for Windows PC compatibility and OpenRouter API integration with free Gemma 4 31B model
---

# Refactor for Windows PC and OpenRouter

This skill refactors an existing Jupyter notebook to:
- Run on Windows PC (replace `!pip` with `%pip`)
- Use OpenRouter API with free Gemma 4 31B model instead of local LLM downloads
- Maintain the original logic and structure as closely as possible

## Instructions

### 1. Analyze the Source Notebook
- Read the source Jupyter notebook file
- Identify all cells that use `!pip` for package installation
- Identify LLM loading code (typically using LlamaCpp, HuggingFace models, or other local inference)
- Identify environment configuration and API key management
- Note the overall structure and flow of the notebook

### 2. Create New Requirements File
- Create a new requirements file (e.g., `requirements_windows.txt`)
- Remove GPU-specific dependencies (CUDA, cuDNN, torch+cuda, etc.)
- Remove local LLM dependencies (llama-cpp-python, hf_hub_download for models)
- Add OpenRouter integration dependencies:
  - `openai>=1.0.0` (for OpenRouter API compatibility)
  - `python-dotenv>=1.0.0` (for environment variable management)
- Keep core dependencies needed for the notebook's logic (langchain, vector stores, embeddings, etc.)
- Ensure all versions are CPU-compatible for Windows

### 3. Refactor Notebook Cells

#### Installation Cells
- Replace all `!pip install ...` with `%pip install ...`
- Update installation commands to use the new requirements file
- Remove GPU-specific installation commands (e.g., CMAKE_ARGS for CUDA)

#### Import Cells
- Replace local LLM imports with OpenRouter-compatible imports:
  - Replace `from langchain.llms import LlamaCpp` with `from langchain_community.chat_models import ChatOpenAI`
  - Replace `from langchain.llms import OpenAI` with `from langchain_community.chat_models import ChatOpenAI`
  - Keep all other imports unchanged

#### LLM Configuration Cells
- Replace local LLM loading code with OpenRouter ChatOpenAI configuration:
  ```python
  from langchain_community.chat_models import ChatOpenAI
  from dotenv import load_dotenv
  import os

  load_dotenv()

  llm = ChatOpenAI(
      openai_api_key=os.getenv("OPENROUTER_API_KEY"),
      openai_api_base="https://openrouter.ai/api/v1",
      model_name="google/gemma-4-31b-it",
      temperature=0.01,
      max_tokens=2048
  )
  ```
- Remove model download code (hf_hub_download, local model paths)
- Remove GPU-specific LLM parameters (n_gpu_layers, etc.)
- Keep temperature, top_p, and other inference parameters if present

#### Environment Configuration
- Create or update `.env.example` file with:
  ```
  OPENROUTER_API_KEY=your_openrouter_api_key_here
  ```
- Ensure the notebook uses `load_dotenv()` to load environment variables

### 4. Maintain Original Structure
- Keep all markdown cells unchanged
- Keep cell order identical to original
- Keep variable names and logic flow the same
- Only modify what's necessary for Windows compatibility and OpenRouter integration
- Preserve all comments and explanations

### 5. Create New Notebook
- Create a new Jupyter notebook file with a descriptive name (e.g., add `_Windows_OpenRouter` suffix)
- Include all refactored cells in the correct order
- Add a markdown cell at the beginning explaining the changes:
  - Windows compatibility
  - OpenRouter integration
  - Benefits (no GPU required, no large downloads, free model)

### 6. Verify Dependencies
- Ensure all LangChain imports use the correct module paths for newer versions:
  - `langchain_community.chat_models` for ChatOpenAI
  - `langchain_community.llms` for OpenAI (if needed)
  - `langchain.embeddings` for SentenceTransformerEmbeddings
  - `langchain.vectorstores` for FAISS

### 7. Test Checklist
Before completing, verify:
- [ ] All `!pip` replaced with `%pip`
- [ ] All local LLM loading removed
- [ ] ChatOpenAI configured for OpenRouter
- [ ] Gemma 4 31B model specified
- [ ] No GPU-specific dependencies in requirements
- [ ] Original notebook structure preserved
- [ ] Environment variables properly configured
- [ ] All imports are correct for the LangChain version

## Important Notes

- **Never modify the original notebook** - always create a new file
- **Do not provide guidance on obtaining OpenRouter API keys** - assume the user already has one
- **Keep the logic exactly the same** - only change the implementation details
- **Use ChatOpenAI for chat models** like Gemma 4 31B (not the legacy OpenAI class)
- **Preserve all chain configurations** (RetrievalQA, etc.) - only change the LLM object
