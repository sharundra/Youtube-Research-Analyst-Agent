import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from agent import app

st.title("🤖 Tube-RAG: YouTube Research Agent")
st.caption("Powered by LangGraph & OpenAI")

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
config = {'recursion_limit': 20}

query = st.text_input("Enter a topic (e.g., 'Agentic AI tutorial'):")


if st.button("start research"):
    with st.spinner("Agent is researching... (Checking YouTube, Transcripts, Summarizing)"):
        inputs = {"messages": [sys_msg, HumanMessage(content=f"Research about: {query}")]}
        
        # Stream output to UI
        placeholder = st.empty()
        full_response = ""
        
        for event in app.stream(inputs, config=config):
            for key, value in event.items():
                if key == "reasoner":
                    # Show thinking process
                    if value["messages"][-1].tool_calls:
                        st.info(f"Thinking: Calling Tool - {value['messages'][-1].tool_calls[0]['name']}")
                    else:
                        full_response = value["messages"][-1].content
                elif key == "tools":
                    st.success("Tool Executed Successfully!")

        st.markdown("### 📝 Final Research Report")
        st.write(full_response)


