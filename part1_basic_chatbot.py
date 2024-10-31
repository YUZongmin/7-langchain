# part1_basic_chatbot.py

import os
import getpass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv



class State(TypedDict):
    messages: Annotated[list, add_messages]

def main():
    # Load environment variables
    load_dotenv()

    # Initialize the state graph
    graph_builder = StateGraph(State)

    # Initialize our language model with environment variables directly
    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"  # Model specified directly in code
    )

    # Define the chatbot node
    def chatbot(state: State):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Add the chatbot node to the graph
    graph_builder.add_node("chatbot", chatbot)

    # Set entry and finish points
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Compile the graph
    graph = graph_builder.compile()

    # Function to stream graph updates
    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    # Chat loop
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
