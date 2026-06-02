import os
import sys
import dotenv
from pathlib import Path

# Setup paths and load environment variables
# Since this script is inside the scoping folder, the project root is one level up
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
dotenv.load_dotenv(dotenv_path=ROOT_DIR / ".env")

from langchain_core.messages import HumanMessage
from scoping.scoping_graph import scope_research

def chat_loop():
    print("==================================================")
    print("  Welcome to the Research Agent!")
    print("  Type your research request, or 'exit' to quit.")
    print("==================================================")
    
    # Use a configuration with a thread_id for the memory checkpointer
    config = {"configurable": {"thread_id": "chat_session_1"}, "run_name": "Scoping_Agent_Standalone_Test"}
    
    while True:
        try:
            user_input = input("\n👤 You: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Invoke the graph with only the new message; the checkpointer handles history
        print("🤖 Agent is thinking...")
        result = scope_research.invoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config
        )
        
        # If the graph reached the end and generated a brief, we are done scoping
        if result.get("research_brief"):
            research_brief = result["research_brief"]
            print(f"\n✅ Scoping complete! Generated Research Brief:\n{research_brief}")
            break
            
        # Otherwise, the agent has a clarifying question
        messages = result["messages"]
        
        # Print the last message (the AI's response)
        ai_response = messages[-1].content
        print(f"\n🤖 Agent: {ai_response}")

if __name__ == "__main__":
    chat_loop()
