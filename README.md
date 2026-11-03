# Unified LLM Service (Pydantic-AI with Groq & Gemini Fallbacks)

A lightweight FastAPI service providing unified, curl-friendly endpoints for interacting with Large Language Models via **Pydantic-AI**.

It features an automatic **3-tier fallback chain** on `/prompt`, plus dedicated single-model endpoints. All routes require no headers, no auth tokens, and no JSON wrapping: just pass your prompt text in the body and get the model's text response.

---

## Route Overview

| Endpoint | Provider | Model | Description |
|---|---|---|---|
| `/ping` | Internal | N/A | Health check; returns `pong\n` |
| `/files` | Internal | N/A | File explorer UI; renders HTML listing all files in `files/` folder and opens them on click |
| `/prompt` | Groq $\rightarrow$ Gemini $\rightarrow$ Gemini Lite | `llama-3.3-70b-versatile` $\rightarrow$ `gemini-3.5-flash` $\rightarrow$ `gemini-3.5-flash-lite` | **3-tier fallback**: Groq primary, Gemini Flash 1st fallback, Gemini Flash Lite 2nd fallback |
| `/qwen` | Groq API | `qwen/qwen3.8-27b` | Dedicated Qwen 3.8 27B model |
| `/gemini-lite` | Gemini API | `gemini-3.5-flash-lite` | Dedicated Gemini 3.5 Flash Lite model |
| `/gemini` | Gemini API | `gemini-3.5-flash` | Dedicated Gemini 3.5 Flash model |
| `/gpt` | Groq API | `openai/gpt-oss-120b` | Dedicated GPT-OSS 120B model |
| `/gemma` | Gemini API | `gemma-4-31b-it` | Dedicated Gemma 4 31B IT model |

---

## Architecture Flow (`/prompt`)

```
                      +-------------------+
                      |   curl Request    |
                      |  (Plain Text Body)|
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |  FastAPI /prompt  |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   UnifiedLLM      |
                      |  (Pydantic-AI)    |
                      +---------+---------+
                                |
             +------------------+------------------+
             | (1. Try First)                      |
             v                                     |
    +-----------------+                            |
    |    Groq API     |                            |
    | (Llama-3.3-70b) | ----(On Failure)----+      |
    +-----------------+                     |      |
             |                              v      |
             |                     +-----------------+
             |                     |   Gemini API    |
             |                     | (Gemini-3.5-Fl) | ----(On Failure)----+
             |                     +-----------------+                     |
             |                              |                              v
             |                              |                     +-----------------+
             |                              |                     |   Gemini API    |
             |                              |                     | (Gemini-3.5-Lite|
             |                              |                     +-----------------+
             |                              |                              |
             +------------------------------+------------------------------+
                                            | (Success)
                                            v
                                  +-------------------+
                                  | Plain Text Return |
                                  +-------------------+
```

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the sample environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```dotenv
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
```

*(Note: `GOOGLE_API_KEY` is also supported as an alias for `GEMINI_API_KEY`)*

### 3. Run the Server

```bash
python main.py
```
Or with uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Curl Usage Examples

No headers or auth tokens required.

### 1. Health Check (`/ping`)

```bash
curl http://localhost:8000/ping
```
**Output:**
```
pong
```

---

### 2. Multi-Tier Fallback (`/prompt`)

Automatically attempts Groq first, then Gemini 3.5 Flash, and finally Gemini 3.5 Flash Lite if earlier models fail:

```bash
curl -d "Why is the sky blue?" http://localhost:8000/prompt
```

---

### 3. Dedicated Model Routes

#### Qwen (`qwen/qwen3.8-27b` via Groq)
```bash
curl -d "Write a quicksort in Python" http://localhost:8000/qwen
```

#### Gemini 3.5 Flash Lite (`gemini-3.5-flash-lite` via Gemini)
```bash
curl -d "Summarize the history of Unix in 2 sentences" http://localhost:8000/gemini-lite
```

#### Gemini 3.5 Flash (`gemini-3.5-flash` via Gemini)
```bash
curl -d "Explain relativity to a 10 year old" http://localhost:8000/gemini
```

#### GPT-OSS 120B (`openai/gpt-oss-120b` via Groq)
```bash
curl -d "Compare C++ and Rust memory models" http://localhost:8000/gpt
```

#### Gemma 4 31B IT (`gemma-4-31b-it` via Gemini)
```bash
curl -d "Draft a haiku about programming" http://localhost:8000/gemma
```

---

### 4. Multiline & File Payloads

```bash
# Multiline text
curl --data-binary "Compare Python and Rust in 3 bullet points." http://localhost:8000/prompt

# From file
curl --data-binary @prompt.txt http://localhost:8000/prompt
```

---

### 5. File Explorer (`/files`)

Navigate in your browser to view all files located in the `files/` directory:

```
http://localhost:8000/files
```

- Renders an HTML page displaying all files (including size, modified date, and icon).
- Clicking any file or its "Open" button opens the file directly in the browser or a new tab.
- Supports nested subdirectories and safe path resolution preventing traversal.

---

## Running Tests

Run the automated test suite:

```bash
python -m pytest -v
```

All 19 tests verify:
- Groq primary success
- Gemini 3.5 Flash 1st fallback
- Gemini 3.5 Flash Lite 2nd fallback rescue
- All 3 tiers failing simultaneously (502 error aggregation)
- Dedicated `/qwen`, `/gemini-lite`, `/gemini`, `/gpt`, `/gemma` routing
- Curl formats (x-www-form-urlencoded, raw binary, JSON, query params)
- Empty body validation (400) and missing API key error responses
