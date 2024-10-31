import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

class CustomToolNode(ToolNode):
    def __call__(self, inputs: dict):
        tool_results = super().__call__(inputs)
        messages = tool_results.get('messages', [])
        
        try:
            formatted_results = []
            for msg in messages:
                if isinstance(msg, dict) and 'url' in msg and 'content' in msg:
                    formatted_results.append(f"Source: {msg['url']}\n{msg['content']}")
            
            if formatted_results:
                summary = "Based on the search results:\n\n" + "\n\n".join(formatted_results)
                return {"messages": [AIMessage(content=summary)]}
            return {"messages": [AIMessage(content="I couldn't find any relevant information.")]}
            
        except Exception:
            return {"messages": [AIMessage(content="Sorry, I encountered an error processing the search results.")]}

def main():
    load_dotenv()
    
    graph_builder = StateGraph(State)
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    llm = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        try:
            filtered_messages = [
                msg for msg in state["messages"]
                if msg.content.strip() and isinstance(msg, (HumanMessage, AIMessage, SystemMessage))
            ]

            while True:
                response = llm_with_tools.invoke(filtered_messages)

                if hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                    tool_calls = response.additional_kwargs['tool_calls']
                    for tool_call in tool_calls:
                        try:
                            import json
                            arguments = json.loads(tool_call['function']['arguments'])
                            tool_result = tool.invoke(arguments['query'])
                            
                            filtered_messages.extend([
                                SystemMessage(content=str(tool_result)),
                                SystemMessage(content="Please summarize the above search results in a clear and concise way.")
                            ])
                        except Exception:
                            continue
                    continue
                return {"messages": [response]}

        except Exception:
            return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", CustomToolNode(tools=[tool]))
    graph_builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        messages = [HumanMessage(content=user_input)]
        for event in graph.stream({"messages": messages}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

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

