from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
from langchain_core.tools import tool


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