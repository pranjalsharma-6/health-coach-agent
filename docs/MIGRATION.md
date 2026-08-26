# How to Begin — Migrating From the Streamlit Version

You have working code in VS Code and on GitHub. Nothing here throws it away.
This document explains exactly what happens to it and what you do on your machine.

---

## 1. The mental model

Your confusion is normal, and it comes from one worry: *"if I restructure, do I lose the
thing that currently works?"*

You don't, for three reasons:

1. **Git history is permanent.** Every commit you've ever made — the Streamlit app, the
   old `run_agent.py`, all of it — stays in the repository forever. Even if a file is
   deleted from the latest commit, `git log` and `git checkout <old-commit>` bring it back.
2. **We work on a branch.** All the new work happens on `claude/agentic-project-review-m7fjc1`.
   Your `main` branch is untouched until you decide to merge. Your deployed Streamlit app
   keeps running off `main` the entire time.
3. **The old app moves, it doesn't die.** The Streamlit files move to `legacy/streamlit/`
   and still run. You keep a working demo while the new one is half-built.

So: **your live demo stays up throughout.** You only switch the link when the new one is better.

---

## 2. What the repository looks like now

**Before:**

```
health-coach-agent/
├── app.py              ← Streamlit UI
├── agent.py            ← LLM setup + prompts
├── run_agent.py        ← LangGraph workflow
├── tools.py            ← Mongo + calculations
├── models.py           ← Pydantic schemas
├── requirements.txt
└── README.md
```

**After:**

```
health-coach-agent/
├── backend/                    ← FastAPI service (Python)
│   ├── app/
│   │   ├── main.py             ← FastAPI entrypoint
│   │   ├── core/               ← config, security, logging
│   │   ├── db/                 ← Mongo connection + repositories
│   │   ├── models/             ← Pydantic schemas (from models.py)
│   │   ├── api/                ← route handlers
│   │   ├── agent/              ← LangGraph workflow (from run_agent.py)
│   │   └── services/           ← nutrition math (from tools.py)
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   ← Next.js app (TypeScript)
│   ├── app/                    ← routes
│   ├── components/             ← UI components
│   ├── lib/                    ← API client, types
│   └── package.json
│
├── legacy/streamlit/           ← your old app, still runnable
├── docs/                       ← problem statement, architecture
└── README.md
```

Nothing is rewritten from scratch. `tools.py` becomes `services/nutrition.py`, `models.py`
becomes `models/plan.py`, `run_agent.py` becomes `agent/graph.py`. The logic you already
wrote and debugged carries over — it just gets a better home and an API in front of it.

---

## 3. What you do on your machine

### 3.1 Prerequisites to install

You already have Python. You'll additionally need **Node.js 18+** for the frontend:

- Download from [nodejs.org](https://nodejs.org) (pick the LTS version)
- Verify in a terminal: `node --version` and `npm --version`

Optional but recommended:
- **VS Code extensions**: Python, Pylance, ESLint, Prettier, Tailwind CSS IntelliSense
- **MongoDB Compass** — a GUI for browsing your Atlas data, far easier than the web console

### 3.2 Pull the new branch

Open your project in VS Code, then in the integrated terminal (`` Ctrl+` ``):

```bash
git fetch origin
git checkout claude/agentic-project-review-m7fjc1
```

If you have uncommitted local changes, save them first:

```bash
git stash -u          # tucks your changes away safely
git checkout claude/agentic-project-review-m7fjc1
git stash pop         # bring them back if you still want them
```

To get back to your old working state at any time:

```bash
git checkout main
```

That's the escape hatch. It always works.

### 3.3 Run the backend

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
cp ../.env.example .env      # then fill in your real keys
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> — FastAPI generates interactive API documentation
automatically. You can click "Try it out" on any endpoint and hit it from the browser.
This is one of the nicest parts of FastAPI and a good thing to screenshot for your README.

### 3.4 Run the frontend

In a **second terminal** (leave the backend running):

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

You now have two servers running side by side. The frontend calls the backend at
`localhost:8000`. This two-terminal workflow is what normal full-stack development looks
like — get comfortable with it.

---

## 4. The workflow from here

We build in phases, and each phase ends with a commit that leaves the app working:

1. **Foundation** — backend skeleton, auth, database. *Testable via `/docs`.*
2. **Agent v2** — preference-aware planning with safety guardrails. *Testable via `/docs`.*
3. **Frontend** — onboarding, dashboard, logging. *Now it looks like a product.*
4. **Adaptive loop** — skip detection and replanning. *Now it's the thing we pitched.*
5. **Polish** — grounding, tests, deployment.

You can stop and demo at the end of any phase. Nothing is left half-broken between them.

---

## 5. Things that will confuse you (and shouldn't)

**"Why are there two `requirements.txt`-like files?"**
`backend/requirements.txt` is Python dependencies. `frontend/package.json` is JavaScript
dependencies. Different languages, different package managers. Normal.

**"Why do I need two terminals?"**
The backend and frontend are separate programs. In production they run on separate
machines. Running them separately locally mirrors that.

**"CORS error in the browser console"**
The browser blocks requests between different ports by default. The backend is configured
to allow `localhost:3000`. If you see this, check that the backend is actually running and
that the frontend's `NEXT_PUBLIC_API_URL` points at the right port.

**"My old Streamlit app broke"**
It didn't — it moved. Run it with `streamlit run legacy/streamlit/app.py`, or check out
`main` where it's still at the root.

**"I don't understand the frontend code"**
That's expected and fine. You'll learn it by changing it. Start by editing text and
colors, watch the browser hot-reload, and build intuition from there. Nobody learns React
by reading about React.

---

## 6. Safety rules

- **Never commit `.env`.** It's gitignored. If a key ever lands in a commit, rotate the key
  immediately — deleting the file later does not remove it from git history.
- **Commit often**, with real messages. `git commit -m "add onboarding form"` beats
  `git commit -m "update"` when you're trying to find where something broke.
- **Push to the feature branch**, not `main`, until a phase is genuinely done.
