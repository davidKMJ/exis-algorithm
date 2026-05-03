# exis-algorithm

![Static Badge](https://img.shields.io/badge/status-done-brightgreen?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/type-side_project-blue?style=for-the-badge)

An AI agent system for EXIS that generates personalized workout plans based on user input. The system consists of three specialized AI agents: a consultant that collects workout profiles, a planner that generates daily workout plans, and a professional trainer that answers fitness-related questions.

## How to Start

### Environment

- Python 3.13 or higher
- OpenAI API key (set in `.env` file)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/davidKMJ/exis-algorithm.git
cd exis-algorithm

# Install dependencies (using uv)
uv sync

# Set up environment variables
# Create a .env file with your OpenAI API key:
# OPENAI_API_KEY=your_api_key_here

# Run the FastAPI server
uv run python main.py
# Or using uvicorn directly:
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

## Project Structure

```
exis-algorithm/
├── main.py                 # FastAPI application with three agent endpoints
├── consultant.py           # AI agent for collecting workout profiles
├── planner.py              # AI agent for generating workout plans
├── professional.py         # AI agent for answering fitness questions
├── pyproject.toml          # Project dependencies and configuration
├── langgraph.json          # LangGraph configuration
└── README.md               # Project documentation
```
