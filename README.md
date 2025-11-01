# 🍏 Autonomous Health & Nutrition Coach Agent

> An intelligent AI agent that autonomously generates and adapts personalized health plans using agentic reasoning with LangGraph

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**🔗 [Live Demo - Coming Soon](#)** | **📹 [Video Demo - Coming Soon](#)**

---

## 🎯 Overview

An agentic AI system that demonstrates autonomous decision-making for personalized health coaching. The agent evaluates user progress, decides when replanning is needed, and generates adaptive meal and activity plans **without human intervention**.

### Why This Matters

Traditional health apps require constant manual input. This agent:
- ✅ **Autonomously evaluates** progress using health metrics
- ✅ **Decides independently** when plan adjustments are needed
- ✅ **Generates personalized plans** using LLM reasoning
- ✅ **Adapts continuously** based on compliance data

---

## ✨ Key Features

### 🤖 Autonomous Reasoning
Uses **LangGraph** state machines for complex decision-making workflows. The agent autonomously:
- Fetches current health data and plans
- Evaluates compliance against goals
- Decides if replanning is necessary
- Generates new personalized plans when needed

### 🧠 LLM-Powered Planning
- **OpenAI GPT-4** integration with structured outputs
- Natural language meal suggestions
- Context-aware activity recommendations
- Reasoning explanations for every decision

### 💾 Persistent Memory
- **MongoDB Atlas** for long-term plan storage
- Historical tracking of user progress
- Plan versioning and retrieval

### 🔄 Adaptive Learning
Agent automatically adjusts plans when:
- Calorie consumption exceeds target by 20%+
- Daily step count falls below 5,000
- Other compliance thresholds are breached

### 📊 Interactive UI
- Real-time plan visualization with **Streamlit**
- Daily meal and activity breakdown
- Progress tracking (coming soon)

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface
└────────┬────────┘
         │
┌────────▼─────────────────────────┐
│   LangGraph Agent Workflow       │
│  ┌──────────────────────────┐   │
│  │ 1. Fetch Data Node       │   │
│  │ 2. Evaluate Progress     │   │
│  │ 3. Planning Agent Node   │   │
│  └──────────────────────────┘   │
└────────┬─────────────────────────┘
         │
    ┌────▼─────┐      ┌──────────┐
    │ OpenAI   │      │ MongoDB  │
    │ GPT-4    │      │  Atlas   │
    └──────────┘      └──────────┘
```

### Agent Workflow

1. **Data Collection**: Fetches current plan and daily health logs
2. **Progress Evaluation**: Analyzes compliance using custom metrics
3. **Decision Making**: Autonomously decides if replanning is needed
4. **Plan Generation**: Uses LLM to create personalized health plans
5. **Persistence**: Saves plan to MongoDB for future reference

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **AI Framework** | LangChain, LangGraph |
| **LLM** | OpenAI GPT-4 |
| **Database** | MongoDB Atlas |
| **Frontend** | Streamlit |
| **Language** | Python 3.10+ |
| **Data Validation** | Pydantic |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- MongoDB Atlas account ([Free tier available](https://cloud.mongodb.com))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/pranjalsharma-6/health-coach-agent.git
cd health-coach-agent
```

2. **Create virtual environment**
```bash
python -m venv coach-env

# Windows
coach-env\Scripts\activate

# Mac/Linux
source coach-env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
OPENAI_API_KEY=your_actual_openai_key_here
MONGODB_URI=your_mongodb_connection_string_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 How It Works

### Agentic Decision Flow

```python
# Simplified pseudocode of agent reasoning

if no_active_plan:
    decision = "CREATE_NEW_PLAN"
elif calories_consumed > target * 1.2:
    decision = "REPLAN_HIGH_CALORIES"
elif steps < 5000:
    decision = "REPLAN_LOW_ACTIVITY"
else:
    decision = "MAINTAIN_CURRENT_PLAN"
```

The agent uses **LangGraph** to manage this decision flow as a state machine, allowing for complex, multi-step reasoning.

### Example Generated Plan

```json
{
  "plan_title": "Week 1: Aggressive Fat Loss & Muscle Gain",
  "duration_days": 7,
  "agent_reasoning": "Based on current weight and activity levels...",
  "daily_plans": [
    {
      "day": 1,
      "meals": [
        {
          "meal_type": "Breakfast",
          "recipe_suggestion": "Scrambled eggs with spinach",
          "estimated_kcal": 350
        }
      ],
      "activity": {
        "activity_type": "Strength Training",
        "duration_minutes": 45,
        "description": "Upper body focus"
      }
    }
  ]
}
```

---

## 📁 Project Structure

```
health-coach-agent/
├── app.py                 # Streamlit UI
├── run_agent.py          # LangGraph workflow implementation
├── agent.py              # Agent configuration & prompts
├── tools.py              # Database & utility functions
├── models.py             # Pydantic data models
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── .gitignore           # Git ignore rules
```

---

## 🎨 Features Showcase

### Current Features ✅
- ✅ Autonomous plan generation
- ✅ Progress evaluation logic
- ✅ MongoDB persistence
- ✅ Interactive Streamlit UI
- ✅ Structured LLM outputs
- ✅ Error handling & logging

### Coming Soon 🚧
- 📊 Progress tracking dashboard with charts
- 🔗 Real wearable API integration (Fitbit, Google Fit)
- 🗣️ Voice interface with speech recognition
- 👥 Multi-user support with authentication
- 📧 Email/SMS notifications
- 📱 Mobile app (React Native)

---

## 🧪 Technical Challenges & Solutions

### Challenge 1: Inconsistent LLM Outputs
**Problem**: Free-form LLM responses were unreliable for structured data

**Solution**: Implemented structured output with Pydantic validation
```python
agent_chain = LLM.with_structured_output(schema=HealthPlan)
```

### Challenge 2: State Management in Agent Loop
**Problem**: Tracking agent decisions across multiple steps

**Solution**: Used LangGraph's StateGraph for explicit state management
```python
workflow = StateGraph(AgentState)
workflow.add_conditional_edges("evaluate", decision_function)
```

### Challenge 3: MongoDB BSON to JSON Conversion
**Problem**: MongoDB's BSON types broke Streamlit rendering

**Solution**: Clean conversion using bson.json_util
```python
clean_plan = json.loads(json_util.dumps(plan_document))
```

---

## 🔮 Future Enhancements

- **Real-time Integration**: Connect with Fitbit/MyFitnessPal APIs
- **RAG Implementation**: Add nutrition database for evidence-based suggestions
- **Multi-tenancy**: Support multiple users with role-based access
- **A/B Testing**: Test different agent reasoning strategies
- **GDPR Compliance**: Data anonymization and export features
- **Microservices**: Break into separate services for scalability

---

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

MIT License - feel free to use this project as inspiration for your own work.

---

## 👨‍💻 About

Built to demonstrate practical applications of agentic AI systems. This project showcases:

- **Autonomous reasoning** with state machines
- **LLM integration** for natural language generation
- **Production-ready architecture** with proper error handling
- **Full-stack development** from database to UI
- **System design** thinking for scalability

### Key Learnings

- Designing autonomous agent workflows
- Managing LLM prompt engineering for consistency
- Integrating multiple APIs (OpenAI, MongoDB)
- Building responsive UIs with Streamlit
- Handling state in complex AI systems

---

## 📧 Contact

**Pranjal Sharma**
- GitHub: [@pranjalsharma-6](https://github.com/pranjalsharma-6)
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]

**Project Link**: https://github.com/pranjalsharma-6/health-coach-agent

---

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- OpenAI for GPT-4 API access
- MongoDB Atlas for free tier database hosting
- Streamlit for rapid UI development

---

⭐ **If you found this project interesting, please consider giving it a star!**
