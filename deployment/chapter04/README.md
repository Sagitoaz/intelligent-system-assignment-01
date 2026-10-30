# Local Knowledge & RAG Agent

Run from the repository root after executing the notebooks:

```powershell
.venv/Scripts/python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

Open http://127.0.0.1:8044 for the light-themed question page and http://127.0.0.1:8044/docs for Swagger UI. Models and corpus are local; no API key is needed. The fourth notebook starts a real uvicorn subprocess, makes HTTP requests and stops the process after verification. Startup with generation loads the two CPU models on the first RAG request.

The agent is a deterministic tool router with RAG generation, CSV statistics, graph lookup, arithmetic and citation ID checks. It is not a general autonomous LLM planner. Generated answers remain unverified and are not clinical recommendations or trading advice. The calculator accepts structured operands from the caller and does not automatically extract them from reports.

The agent runs locally only; see RUN.md at the repository root for all commands.
