from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

"""
Enhanced Chatbot with Web Search Capabilities

This implementation extends the basic chatbot by adding:
1. Web search functionality using Tavily Search
2. Tool integration with the LLM
3. Conditional routing based on whether the LLM needs to search for information

Key Components:
- TavilySearchResults: Web search tool
- ToolNode: Handles the execution of tools
- tools_condition: Routes the flow based on whether a tool needs to be used
"""

class State(TypedDict):
    # Same state structure as before, but now messages might include tool calls
    messages: Annotated[list, add_messages]

# Initialize the graph
graph_builder = StateGraph(State)

# Set up the search tool and LLM
tool = TavilySearchResults(max_results=2)  # Uses TAVILY_API_KEY from environment automatically
tools = [tool]
llm = ChatOpenAI(
    base_url=os.getenv("openai_api_base"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"  # Model specified directly in code
)
# Bind tools to the LLM so it knows how to use them
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node that can now use tools
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Set up the conditional routing
# This determines whether to use tools or end the interaction
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# After using tools, always return to chatbot for next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Compile the graph
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():
    print("Welcome to the enhanced chatbot with web search capabilities!")
    print("Ask me anything, and I'll search the web if needed.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

if __name__ == "__main__":
    main()