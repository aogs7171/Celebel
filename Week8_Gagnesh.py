import json
import re

# 🛠️ TOOL 1: Calculator
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error in calculation"

# 🛠️ TOOL 2: Keyword Extractor
def extract_keywords(text: str) -> list:
    """Extract keywords from text."""
    try:
        words = text.split()
        keywords = list(set([w.lower() for w in words if len(w) > 4]))
        return keywords[:5]
    except Exception:
        return []

# 🤖 AGENT FUNCTION
def agent(query: str) -> str:
    query_lower = query.lower()

    if "calculate" in query_lower:
        match = re.search(r"calculate\s+(.*)", query_lower)
        expression = match.group(1).strip() if match else ""
        result = calculator(expression)
        
        response = {
            "type": "calculation",
            "result": result
        }

    elif "keywords" in query_lower:
        # Clean up extraction text if the user specified "keywords from"
        match = re.search(r"keywords(?:\s+from)?\s+(.*)", query_lower)
        text = match.group(1).strip() if match else query_lower
        result = extract_keywords(text)
        
        response = {
            "type": "keywords",
            "result": result
        }

    else:
        response = {
            "type": "general",
            "result": "I am a simple smart assistant. Ask me to calculate a math expression or extract keywords from some text."
        }

    # Format all response packages as cleanly structured JSON outputs
    return json.dumps(response, indent=2)

if __name__ == "__main__":
    # 🧪 Test Cases
    queries = [
        "Calculate 20 + 5",
        "Extract keywords from Artificial Intelligence is transforming industries",
        "What is machine learning?"
    ]

    print("--- Running Automated Array Check ---")
    for q in queries:
        print("Query:", q)
        print("Response:", agent(q))
        print("-" * 50)

    # 🎯 Interactive Mode
    print("\n--- Interactive Mode ---")
    while True:
        try:
            user_input = input("Enter query (type 'exit' to stop): ")
            if user_input.strip().lower() == "exit":
                break
            print("Response:", agent(user_input))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
