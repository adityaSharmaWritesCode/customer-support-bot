from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

itinerary = ""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def hotel_search(content: str) -> str:
    """Searches for a hotel and updates the itinerary."""
    global itinerary
    itinerary = content
    return f"Your HOTEL has been booked succesfully! Your current itinerary is as follows:\n{itinerary}"

@tool
def car_search(content: str) -> str:
    """Searches for a rental car and updates the itinerary."""
    global itinerary
    itinerary = content
    return f"Your RENTAL CAR has been booked successfully! Your current itinerary is as follows:\n{itinerary}"

@tool
def flights_search(content: str) -> str:
    """Searches for a flight and updates the itinerary."""
    global itinerary
    itinerary = content
    return f"Your FLIGHT has been updated successfully! Your current itinerary is as follows:\n{itinerary}"

@tool
def exc_search(content: str) -> str:
    """Searches for an excursion and updates the itinerary."""
    global itinerary
    itinerary = content
    return f"Your EXCURSION has been booked successfully! Your current itinerary is as follows:\n{itinerary}"


@tool
def save(filename: str) -> str:
    """Save the current itinerary to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global itinerary

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(itinerary)
        print(f"\n💾 Itinerary has been saved to: {filename}")
        return f"Itinerary has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [hotel_search, car_search, flights_search, exc_search, save]

model = ChatOllama(model="qwen3").bind_tools(tools) #Only agents which are compatible with tools will work here

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a Customer Support Bot, a helpful itinerary planning assistant. You are going to help the user update and modify their itinerary.
    
    - If the user wants to book a hotel, use the 'hotel_search' tool with the complete updated content.
    - If the user wants to book a car, use the 'car_search' tool with the complete updated content.       
    - If the user wants to book a flight, use the 'flights_search' tool with the complete updated content.    
    - If the user wants to book an excursion, use the 'exc_search' tool with the complete updated content.                                              
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current itinerary state after modifications.
    
    The current itinerary content is:{itinerary}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you plan your travel. When and where would you like to go?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with your itinerary? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

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

def run_agent():
    print("\n ===== Customer Support Bot =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== Bot Ended =====")

if __name__ == "__main__":
    run_agent()
