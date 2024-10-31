from dotenv import load_dotenv
load_dotenv()


from langchain_community.tools.tavily_search import TavilySearchResults

def test_tool():
    tool = TavilySearchResults(max_results=2)
    results = tool.invoke("What is LangGraph?")
    for result in results:
        print(f"URL: {result['url']}\nContent: {result['content']}\n")

if __name__ == "__main__":
    test_tool()

