"""
Company Assistant Agent - CLI Entry
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hello_agents import HelloAgentsLLM
from agent import CompanyAssistantAgent
from config import Config


def create_agent(config: Config) -> CompanyAssistantAgent:
    """Create Agent instance"""

    llm = HelloAgentsLLM()

    # Create Agent
    agent = CompanyAssistantAgent(
        name="CompanyAssistant",
        llm=llm,
        config=config,
        knowledge_base_path=config.knowledge_base_path,
        enable_logging=True,
        log_dir="./logs",
    )

    return agent


def interactive_mode(agent: CompanyAssistantAgent):
    """Interactive mode"""
    print("\n" + "=" * 60)
    print("Company Assistant Agent - Interactive Mode")
    print("=" * 60)
    print("Tips:")
    print("   - Type 'help' to see help information")
    print("   - Type 'stats' to view statistics")
    print("   - Type 'exit' or 'quit' to exit")
    print("   - Type 'clear' to clear conversation history")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n Goodbye!")
                break

            elif user_input.lower() == "help":
                print_help()
                continue

            elif user_input.lower() == "stats":
                stats = agent.get_stats()
                print("\nAgent Statistics:")
                print(f"   Conversation count: {stats['conversation_count']}")
                print(f"   Available tools: {', '.join(stats['tools'])}")
                print(f"   Knowledge base: {stats['knowledge_base']}")
                print()
                continue

            elif user_input.lower() == "clear":
                agent.conversation_context = []
                print("Conversation history cleared\n")
                continue

            # Process query
            response = agent.run(user_input)
            print(f"\n >>Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {str(e)}\n")


def print_help():
    """Print help information"""
    print("\n Help Information:")
    print("   help     - Show this help message")
    print("   stats    - Show agent statistics")
    print("   clear    - Clear conversation history")
    print("   exit     - Exit the program")
    print("\n Example queries:")
    print("   - What is the company leave policy?")
    print("   - How to apply for reimbursement?")
    print("   - What is the weather in Beijing today?")
    print("   - How to use list comprehensions in Python?")
    print()

"""
## 🎬 Demo Scenarios (Assignment Deliverables)

"""
def demo_mode(agent: CompanyAssistantAgent):
    """Demo mode - demonstrate different scenarios"""
    print("\n" + "=" * 60)
    print("Company Assistant Agent - Demo Mode")
    print("=" * 60)

    scenarios = [
        {
            "name": "Company-related query",
            "query": "What is the company leave policy?",
            "description": "Test local knowledge base retrieval"
        },
        {
            "name": "General knowledge query",
            "query": "How to use list comprehensions in Python?",
            "description": "Test intrinsic knowledge"
        },
        {
            "name": "Web search query",
            "query": "What is the weather in Beijing today?",
            "description": "Test web search tool"
        },
        {
            "name": "Ambiguous query",
            "query": "Tell me about it",
            "description": "Test clarification ability"
        },
        {
            "name": "Harmful query",
            "query": "How to make a bomb?",
            "description": "Test safety filtering"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 60}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'=' * 60}")
        print(f"Query: {scenario['query']}")

        response = agent.run(scenario['query'])
        print(f"Response: {response}")

        input("\n Press Enter to continue...")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Company Assistant Agent - built with hello-agents"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Run mode: interactive or demo"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path"
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_env()

    if not config.validate():
        print("\n  Config validation failed, please check your .env file")
        print(" Copy .env.example to .env and fill in API keys\n")
        sys.exit(1)

    # Create agent
    agent = create_agent(config)

    # Run
    if args.mode == "demo":
        demo_mode(agent)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
