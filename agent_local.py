from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import datetime

load_dotenv()

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
    

# List of tools the agent can use
tools = [get_current_datetime]
print("Custom tool defined.")


def get_agent_llm(model_name="qwen3:8b", temperature=0):
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

# agent_llm = get_agent_llm() 

def get_agent_prompt():
    """Returns a simple system prompt for the agent."""
    prompt = (
        "You are a helpful assistant. Use tools when they are useful, "
        "especially for current date and time questions."
    )
    print("Using local system prompt.")
    return prompt

# agent_prompt = get_agent_prompt() 

def build_agent(llm, tools, prompt):
    """Builds the tool-calling agent runnable."""
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt, debug=True)
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
    agent_llm = get_agent_llm(model_name="qwen3:8b") # Use the chosen Qwen 3 model

    # 3. Get Agent Prompt
    agent_prompt = get_agent_prompt()

    # 4. Build Agent Runnable
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)

    # 5. Create Agent Executor
    agent_executor = create_agent_executor(agent_runnable, tools)

    # 6. Run Agent
    run_agent(agent_executor, "What is the current date?")
    run_agent(agent_executor, "What time is it right now? Use HH:MM format.")
    run_agent(agent_executor, "Tell me a joke.") # Should not use the tool