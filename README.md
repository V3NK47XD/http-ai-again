# Unified LLM Service (Pydantic-AI with Groq & Gemini Fallback)

A lightweight FastAPI service providing a unified interface for interacting with Large Language Models via **Pydantic-AI**.

It routes prompts to **Groq API** first (primary) and automatically falls back to **Gemini API** if Groq encounters any rate limit, downtime, or runtime errors. Exposes curl-friendly `/ping` and `/prompt` endpoints requiring no headers, auth tokens, or JSON ceremony.

---

## Features

- **Unified Model Interaction**: Built on top of `pydantic-ai` using `FallbackModel`.
- **Primary + Fallback Strategy**:
  - Primary: **Groq** (`llama-3.3-70b-versatile` by default)
  - Fallback: **Gemini** (`gemini-2.5-flash` by default)
  - Automatic failover on API rate limits (429), timeouts, service errors (5xx), or connection failures.
- **Curl Friendly**: Send plain text directly with `curl -d "prompt"`. No `Content-Type` headers or auth headers required.
- **FastAPI Endpoints**:
  - `GET /ping`: Instant health check (`pong`).
  - `POST /prompt`: Takes plain text body and returns clean text response.

---

## Architecture Flow

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
                   +------------+------------+
                   |                         |
            (1. Try First)             (2. On Error)
                   v                         v
          +-----------------+       +-----------------+
          |    Groq API     |       |   Gemini API    |
          | (Llama-3.3-70b) | ----> | (Gemini-2.5-Fl) |
          +-----------------+ (Fail)+-----------------+
                   |                         |
                   +------------+------------+
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

### 1. Health Check (`/ping`)

```bash
curl http://localhost:8000/ping
```
**Output:**
```
pong
```

---

### 2. Prompt LLM (`/prompt`)

No headers or auth needed. Just pass your prompt in the body:

```bash
curl -d "Why is the ocean salty?" http://localhost:8000/prompt
```

#### Multiline prompts:
```bash
curl --data-binary "Compare Python and Rust in three bullet points." http://localhost:8000/prompt
```

#### From a text file:
```bash
curl --data-binary @prompt.txt http://localhost:8000/prompt
```

#### Also supports JSON or Query params if desired:
```bash
# JSON
curl -d '{"prompt": "Hello world"}' http://localhost:8000/prompt

# Query parameter
curl "http://localhost:8000/prompt?text=What+is+2%2B2"
```

---

## Running Tests

Run the complete automated test suite:

```bash
python -m pytest -v
```

All 14 unit and integration tests verify:
- Primary Groq invocation
- Automatic Gemini fallback when Groq fails
- Error aggregation when all providers fail
- Curl request formats (`application/x-www-form-urlencoded`, raw multiline binary, JSON, query params)
- Empty body validation (400) and configuration errors (502)

---

## Project Structure

```
.
├── .env.example         # Sample environment configuration
├── pytest.ini           # Pytest async configuration
├── requirements.txt     # Python dependencies
├── main.py              # Server bootstrap entry point
├── README.md            # Documentation and curl guide
├── app/
│   ├── __init__.py
│   ├── config.py        # Settings management via Pydantic
│   ├── llm.py           # Unified Pydantic-AI agent with Groq + Gemini fallback
│   └── main.py          # FastAPI app exposing /ping and /prompt
└── tests/
    ├── __init__.py
    ├── test_llm.py      # LLM fallback logic and exception handling tests
    └── test_api.py      # API endpoints and curl format compatibility tests
```
