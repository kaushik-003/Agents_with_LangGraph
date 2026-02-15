from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# variable to store document content
Document_Content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str) -> str:
    """Updates the Document with provided content."""
    global Document_Content
    Document_Content = content
    return f"Document updated successfully!! Current content is :\n{Document_Content}"

@tool
def save(file_name:str) -> str:
    """Saves the current document content to the text file and finsihes the execution.
    
    Args:
        file_name : Name for the text file
    """
    global Document_Content
    if not file_name.endswith(".txt"):
        file_name = f"{file_name}.txt"
    try:
        with open(file_name, "w") as f:
            f.write(Document_Content)
        return f"Document saved successfully as {file_name}!!"
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"
    
tools = [update, save]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content= f"""
    You are a Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If user wants to update or modify the document, use the 'update' tool with complete updated content.
    - If user wants to save the document, you need to use 'save' tool.
    - Make sure always to show the current document state after modifications
    
    The current document content is :\n{Document_Content}
    """)
    if not state["messages"]:
        user_input = "I'm ready to help you draft your document. How can I assist you today?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\n what would you like to do with the document? \n")
        print(f"User Input : {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)
   
    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")
                                  
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)
app = graph.compile()

def run_doc_agent():
    print("\n -------------DRAFTER-------------")
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])


    print("\n -------------END OF CONVERSATION-------------")


if __name__ == "__main__":
    run_doc_agent()