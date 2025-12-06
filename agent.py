from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from all_tools import get_video_links, get_transcript_text
from dotenv import load_dotenv

load_dotenv()




class AgentState(TypedDict):
    messages : Annotated[List[BaseMessage], operator.add]

all_tools_list = [get_video_links, get_transcript_text]
llm_with_tools = ChatOpenAI().bind_tools(all_tools_list)



def reasoner_node(state : AgentState):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}

workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("tools", ToolNode(all_tools_list))
workflow.set_entry_point("reasoner")
workflow.add_conditional_edges("reasoner", tools_condition)
workflow.add_edge("tools", "reasoner")

app = workflow.compile()