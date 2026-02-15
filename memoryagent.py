from typing import TypedDict, List, Union
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model = "gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    """This node will solve your Request"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")

    print("CURRENT STATE:", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("conversation_history.txt", "w") as file:
    file.write("your conversation history: \n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("end of conversation")

print("Conversation history saved to conversation_history.txt")