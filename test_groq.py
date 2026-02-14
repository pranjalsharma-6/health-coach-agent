import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env file!")
else:
    print(f"✅ API key found: {api_key[:20]}...")
    
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello!"}],
            model="llama-3.3-70b-versatile",
        )
        print("✅ GROQ API key is working!")
        print(f"Response: {chat_completion.choices[0].message.content}")
    except Exception as e:
        print(f"❌ GROQ API key failed: {e}")