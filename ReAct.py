from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is a addition function that adds two numbers together"""
    return a + b

@tool
def subtract(a:int, b:int):
    """This is a subtraction function that subtracts two numbers together"""
    return a - b

@tool
def multiply(a:int, b:int):
    """This is a multiplication function that multiplies two numbers together"""
    return a * b


tools = [add, subtract, multiply]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def modal_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
            "You are My AI assistant, please answer my query to best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", modal_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "multiply 522 and 6983")]}
print_stream(app.stream(inputs, stream_mode = "values"))