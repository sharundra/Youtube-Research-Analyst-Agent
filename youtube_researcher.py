from langchain_core.tools import tool
from duckduckgo_search import DDGS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.tools import DuckDuckGoSearchRun
from youtube_search import YoutubeSearch
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()


@tool
def get_video_links(query : str) -> str:
    """given a query, this function searches relevant youtube video on internet and returns video links"""
    query = f'site:youtube.com {query}'

    results = YoutubeSearch(query, max_results=5).to_dict()
    
    video_links = []
    for res in results:
        # Full link banao
        link = f"https://www.youtube.com/watch?v={res['id']}"
        video_links.append(link)
    return video_links 

@tool
def get_transcript_text(video_link : str) -> str:
    """
    given a youtube video link, return its transcript
    """
    yyt = YouTubeTranscriptApi()
    video_id = video_link.split("v=")[1]
    transcript_list_of_dict = yyt.fetch(video_id, languages=['en', 'en-US', 'en-GB', 'en-IN'])
    transcript_string = " ".join([item.text for item in transcript_list_of_dict])
    return transcript_string[:5000]

# links = get_video_links.invoke("Roman empire explained in English")
# i = 1
# for link in links:
#     transcript = get_transcript_text.invoke(link)
#     print(f'Here is the transcript of the {i}th video: {transcript}')
#     i += 1


class AgentState(TypedDict):
    messages : Annotated[List[BaseMessage], operator.add]

all_tools = [get_video_links, get_transcript_text]
llm_with_tools = ChatOpenAI().bind_tools(all_tools)

sys_msg = SystemMessage(content="""
You are a Smart YouTube Research Assistant. Your goal is to find a video on the user's topic and summarize it.

Follow these steps strictly:
1. Search for videos on the topic using the 'get_video_links' tool.
2. Pick the first video link and try to get its transcript using 'get_transcript_text'.
3. CRITICAL: If the transcript tool returns an "ERROR" (e.g., no subtitles), DO NOT GIVE UP. Immediately try the NEXT video link from the search results.
4. Repeat this until you find a video with a valid transcript.
5. Once you have the transcript, summarize the key learning points in 5-7 bullet points.
6. Provide the Title/Link of the video you finally used.
""")



def reasoner_node(state : AgentState):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}

workflow = StateGraph(AgentState)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("tools", ToolNode(all_tools))
workflow.set_entry_point("reasoner")
workflow.add_conditional_edges("reasoner", tools_condition)
workflow.add_edge("tools", "reasoner")

app = workflow.compile()

topic = "Bhagavadgita explained in English"
inputs = {
    'messages' : [sys_msg, HumanMessage(content = f'research topic: {topic}')]
}

config = {'recursion_limit': 20}
for output in app.stream(inputs, config):
    for node, value in output.items():
        if node == 'reasoner':
            last_msg = value['messages'][-1]
            if last_msg.tool_calls:
                print(f"Agent Decided to Call: {last_msg.tool_calls[0]['name']}")
        if node == 'tools':
            last_msg = value['messages'][-1]
            print(f"Tool Output Preview: {last_msg.content[:200]}...")

print("\n" + "="*30)
print("FINAL ANSWER")
print("="*30)
print(last_msg.content)           #This last msg comes from the reasoner node run at last silent iteration which didnt print anything in the loop because there was tool call.




    

