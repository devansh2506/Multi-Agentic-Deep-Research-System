import os
import sys
import dotenv
import asyncio
from pathlib import Path

# Setup paths and load environment variables
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
dotenv.load_dotenv(dotenv_path=ROOT_DIR / ".env")

from langchain_core.messages import HumanMessage
from scoping.scoping_graph import scope_research
from research_supervisor.agent import supervisor_agent

def main():
    print("==================================================")
    print("  Welcome to the Deep Researcher AI!")
    print("==================================================")
    
    # ---------------------------------------------------------
    # PHASE 1: SCOPING
    # ---------------------------------------------------------
    print("\n[PHASE 1: Scoping]")
    print("What would you like to research?")
    
    scoping_config = {"configurable": {"thread_id": "scoping_thread_1"}}
    research_brief = None
    
    while True:
        try:
            user_input = input("\n👤 You: ")
        except (KeyboardInterrupt, EOFError):
            return
            
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            return
            
        if not user_input.strip():
            continue
            
        print("🤖 Agent is thinking...")
        result = scope_research.invoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=scoping_config
        )
        
        # If the graph reached the end and generated a brief, we are done scoping
        if result.get("research_brief"):
            research_brief = result["research_brief"]
            print(f"\n✅ Scoping complete! Generated Research Brief:\n{research_brief}")
            break
            
        # Otherwise, the agent has a clarifying question
        messages = result["messages"]
        ai_response = messages[-1].content
        print(f"\n🤖 Agent: {ai_response}")
        
    if not research_brief:
        return
        
    # ---------------------------------------------------------
    # PHASE 2: DEEP RESEARCH
    # ---------------------------------------------------------
    print("\n==================================================")
    print("[PHASE 2: Deep Research via Multi-Agent Supervisor]")
    print(f"🤖 Supervisor is now coordinating research on: '{research_brief}'")
    print("... (It may spawn multiple parallel researchers, so please be patient)")
    print("==================================================\n")
    
    supervisor_config = {"configurable": {"thread_id": "supervisor_thread_1"}}
    
    # Initialize SupervisorState
    state = {
        "research_brief": research_brief,
        "supervisor_messages": [HumanMessage(content=research_brief)]
    }
    
    try:
        # The supervisor uses async nodes, so we must run it via asyncio.run and use .ainvoke()
        final_result = asyncio.run(supervisor_agent.ainvoke(state, config=supervisor_config))
        
        print("\n" + "="*50)
        print("AGGREGATED RESEARCH NOTES:")
        print("="*50)
        
        notes = final_result.get("notes", [])
        if not notes:
            print("No findings returned from the research phase.")
            return
            
        combined_notes = "\n\n---\n\n".join(notes)
            
    except Exception as e:
        print(f"\n❌ Error during supervisor execution: {e}")
        return
        
    # ---------------------------------------------------------
    # PHASE 3: FINAL SYNTHESIS (REPORTER AGENT)
    # ---------------------------------------------------------
    print("\n==================================================")
    print("[PHASE 3: Final Report Generation]")
    print("🤖 Reporter Agent is synthesizing the final report...")
    print("==================================================\n")
    
    from reporter.agent import report_writer_agent
    
    try:
        # The LCEL chain takes a simple dictionary as input
        final_report = report_writer_agent.invoke({
            "research_brief": research_brief,
            "research_notes": combined_notes
        })
        
        print("\n" + "="*80)
        print("FINAL RESEARCH REPORT")
        print("="*80 + "\n")
        print(final_report)
        print("\n" + "="*80)
        
        # Save to file
        report_path = ROOT_DIR / "final_research_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print(f"\n✅ Report successfully saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Error during report generation: {e}")

if __name__ == "__main__":
    main()
