from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

"""
This is a simple, stateless chatbot implementation using LangGraph.
Key characteristics of this implementation:

1. Stateless Nature:
   - Each interaction is independent
   - The bot doesn't remember previous conversations
   - Every new message starts with a fresh state

2. Single-Turn Architecture:
   - The graph processes one message at a time
   - Previous context is not maintained between turns
   - Suitable for simple question-answer scenarios

To add memory/context, we would need to:
1. Maintain a message history outside the graph
2. Pass the entire history with each new message
3. Modify the state management to persist between turns
"""

# Define the state structure for our chatbot
# Note: While we use add_messages, this only affects the current turn
# It doesn't persist between different user inputs
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph with our State type
# Each graph execution starts fresh with no memory of previous turns
graph_builder = StateGraph(State)

# Initialize our language model with environment variables directly
llm = ChatOpenAI(
    base_url=os.getenv("openai_api_base"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"  # Model specified directly in code
)

# Define the chatbot node function
# This function processes a single message without any context from previous interactions
# Each call to this function starts with a new, empty state
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add the chatbot node to our graph
graph_builder.add_node("chatbot", chatbot)

# Define the simple linear flow: START -> chatbot -> END
# This is a single-turn flow: get input -> process -> respond -> end
# No state is maintained between these cycles
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Stream updates function
# Note: Each call creates a new state with just the current message
# Previous messages are not included in the state
def stream_graph_updates(user_input: str):
    # Create a fresh state with only the current message
    # This means each response is based solely on the current input
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():
    """
    Main chat loop.
    
    Important: This implementation is 'memory-less' by design.
    Each iteration of the loop:
    1. Gets a new user input
    2. Creates a fresh state with just that input
    3. Gets a response based only on that input
    4. Discards all context before the next iteration
    
    This makes the bot unable to reference previous parts of the conversation.
    """
    print("Welcome to the stateless chatbot! Each response is independent of previous interactions.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Each call to stream_graph_updates starts fresh
            # No conversation history is maintained
            stream_graph_updates(user_input)
            
        except Exception as e:
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    main()
