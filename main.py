import argparse
import os
import sys
from dotenv import load_dotenv

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_graph

def main():
    parser = argparse.ArgumentParser(description="Academic Paper Analysis Agent")
    parser.add_argument("source", help="Arxiv URL or local PDF file path")
    args = parser.parse_args()
    
    # Check API Key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in .env file.")
        return

    print(f"Starting analysis for: {args.source}")
    print("This may take a minute depending on the paper length...")
    
    app = create_graph()
    
    try:
        # Run the graph
        final_state = app.invoke({"source": args.source})
        
        report = final_state.get("final_report")
        if report:
            output_file = "paper_analysis_report.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nAnalysis complete! Report saved to {output_file}")
            print("-" * 30)
            print(report[:500] + "...\n(See full report in file)")
        else:
            print("Error: Failed to generate report.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
