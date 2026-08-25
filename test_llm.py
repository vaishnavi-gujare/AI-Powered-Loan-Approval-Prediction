import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Get Groq API key
groq_key = os.getenv("GROQ_API_KEY")

print("=========================================")
print("🌐 GROQ API KEY CONNECTION TESTER")
print("=========================================\n")

# Check API key
if not groq_key or groq_key.strip() == "" or groq_key == "your_groq_api_key_here":
    print("❌ GROQ: No valid API key found in .env")
    print("Please check your GROQ_API_KEY in the .env file.")
else:
    print("🔍 GROQ: API key detected. Testing connection...\n")

    try:
        # Create Groq client
        client = Groq(api_key=groq_key)

        # Send test request
        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one short sentence."
                }
            ],
        )

        # Get response
        response = chat_completion.choices[0].message.content

        print("✅ GROQ CONNECTION SUCCESSFUL!")
        print(f"🤖 Model: openai/gpt-oss-120b")
        print(f"💬 Response: {response}")

    except Exception as e:
        print("❌ GROQ CONNECTION FAILED!")
        print(f"Error: {e}")

print("\n=========================================")