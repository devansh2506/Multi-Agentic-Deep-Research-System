import os
import sys
import dotenv
from pathlib import Path

# Setup paths and load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
dotenv.load_dotenv(dotenv_path=ROOT_DIR / ".env")

from langchain_core.messages import HumanMessage
from research.agent import researcher_agent

def run_research():
    print("==================================================")
    print("  Welcome to the Research Agent Tester!")
    print("  Provide a research brief to start searching.")
    print("==================================================")
    
    try:
        brief = input("\n🔍 Enter a research brief: ")
    except (KeyboardInterrupt, EOFError):
        return
        
    if not brief.strip():
        print("No brief provided. Exiting.")
        return
        
    # Use a configuration with a thread_id for the memory checkpointer
    config = {"configurable": {"thread_id": "research_test_1"}, "run_name": "Researcher_Agent_Standalone_Test"}
    
    # Initialize ResearcherState
    state = {
        "researcher_messages": [HumanMessage(content=brief.strip())]
    }
    
    print(f"\n🤖 Agent is running research loop on your brief...")
    print("... (This will involve multiple web searches, so please be patient)")
    
    try:
        result = researcher_agent.invoke(state, config=config)
        
        print("\n" + "="*50)
        print("COMPRESSED RESEARCH FINDINGS:")
        print("="*50)
        print(result.get("compressed_research", "No findings returned."))
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    run_research()
