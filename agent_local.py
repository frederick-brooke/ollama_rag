from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import datetime
from ddgs import DDGS

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time, formatted according to the provided Python strftime format string.
    Use this tool whenever the user asks for the current date, time, or both.
    Example format strings: '%Y-%m-%d' for date, '%H:%M:%S' for time.
    If no format is specified, defaults to '%Y-%m-%d %H:%M:%S'.
    """
    try:
        print(f"[tool] get_current_datetime invoked with format={format}")
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"

@tool
def web_search(query: str) -> str:
    """
    Performs a web search for the given query and returns the results.
    """
    try:
        print(f"[tool] web_search invoked with query='{query}'")
        with DDGS() as ddg:
            results = ddg.text(query, max_results=3)
        print(f"Web search results for '{query}': {results}")

        search_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            search_results.append(f"{i}. {title}: {body}\n")
        
        return "\n".join(search_results) if search_results else "No results found."
    except Exception as e:
        return f"Error performing web search: {e}"
    

# List of tools the agent can use
tools = [get_current_datetime, web_search]
print("Custom tool defined.")


def get_agent_llm(model_name="qwen3:4b", temperature=0):
    """Initializes the ChatOllama model for the agent."""
    # Ensure Ollama server is running (ollama serve)
    llm = ChatOllama(
        model=model_name,
        temperature=temperature # Lower temperature for more predictable tool use
        # Consider increasing num_ctx if expecting long conversations or complex reasoning
        # num_ctx=8192
    )
    
    print(f"Initialized ChatOllama agent LLM with model: {model_name}")
    return llm

def get_agent_llm_openrouter(model_name="nvidia/nemotron-3-ultra-550b-a55b:free", temperature=0):
    """Initializes the OpenRouter model for the agent."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "ollama_rag"),
        },
    )

    print(f"Initialized OpenRouter agent LLM with model: {model_name}")
    return llm

# agent_llm = get_agent_llm() 

def get_agent_prompt():
    """Returns a simple system prompt for the agent."""
    prompt = (
        """
        You are a helpful assistant that can use tools to answer user questions.
        Use the provided tools whenever the user asks for information that can be obtained through them (e.g., current date/time, web search).
        If the user asks for something that cannot be answered with the tools, respond with your best knowledge or say you don't know.
        Always try to use the tools when appropriate to provide accurate and up-to-date information. If web_search is used to answer a question, include the search results in your response to the user.
        """
    )
    print("Using local system prompt.")
    return prompt

# agent_prompt = get_agent_prompt() 

def build_agent(llm, tools, prompt):
    """Builds the tool-calling agent runnable."""
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt, debug=False)
    print("Agent runnable created.")
    return agent

# agent_runnable = build_agent(agent_llm, tools, agent_prompt) 

def create_agent_executor(agent, tools):
    """Returns the agent runnable for compatibility with existing flow."""
    _ = tools
    print("Using agent runnable directly (no AgentExecutor needed).")
    return agent

# agent_executor = create_agent_executor(agent_runnable, tools) 

def run_agent(executor, user_input):
    """Runs the agent executor with the given input."""
    print("\nInvoking agent...")
    print(f"Input: {user_input}")
    response = executor.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("\nAgent Response:")
    if isinstance(response, dict) and response.get("messages"):
        print(response["messages"][-1].content)
    else:
        print(response)


if __name__ == "__main__":
    # 1. Define Tools (already done above)

    # 2. Get Agent LLM
    agent_llm = get_agent_llm()

    # 3. Get Agent Prompt
    agent_prompt = get_agent_prompt()

    # 4. Build Agent Runnable
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)

    # 5. Create Agent Executor
    # agent_executor = create_agent_executor(agent_runnable, tools)

    # 6. Run Agent
    run_agent(agent_runnable, "Search for local cafes in London and rank the top 3 based on user reviews.") # Should use web_search tool
    run_agent(agent_runnable, "What is the current date?")
    run_agent(agent_runnable, "What time is it right now? Use HH:MM format.")
    run_agent(agent_runnable, "Tell me a joke.") # Should not use the tool