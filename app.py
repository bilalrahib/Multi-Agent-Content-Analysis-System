import os
import openai
from openai import OpenAI
from typing import Dict, List, Optional
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import re

class NvidiaLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NVIDIA_NIM_API_KEY')
        if not self.api_key:
            raise ValueError("NVIDIA NIM API key is required")

        # Initialize OpenAI client with NVIDIA configuration
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

        # Configuration for chat completions
        self.chat_config = {
            "model": "tiiuae/falcon3-7b-instruct",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024
        }

        # Configure for Autogen
        self.config = {
            "config_list": [{
                "model": "tiiuae/falcon3-7b-instruct",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": self.api_key
            }]
        }

    def validate_message(self, message: str) -> str:
        """Validate and clean message content"""
        if not message or not isinstance(message, str):
            return "No input provided."
        cleaned = message.strip()
        if not cleaned:
            return "No valid input provided."
        return cleaned

    def create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create properly formatted message list for the API"""
        return [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

    def generate(self, prompt: str) -> str:
        """Generate response with proper error handling"""
        try:
            validated_prompt = self.validate_message(prompt)
            messages = self.create_messages(validated_prompt)
            
            response = self.client.chat.completions.create(
                messages=messages,
                **self.chat_config
            )
            
            if not response.choices:
                return "No response generated. Please try again."
                
            return response.choices[0].message.content or "Empty response received."
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)  # For debugging
            return error_msg

class YouTubeHelper:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        
    def get_video_transcript(self, video_id: str) -> str:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(t["text"] for t in transcript)
        except Exception as e:
            print(f"Error getting transcript: {e}")
            return "No transcript available."

    def search_videos(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            request = youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video"
            )
            response = request.execute()
            
            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                transcript = self.get_video_transcript(video_id)
                if transcript and transcript != "No transcript available.":
                    videos.append({
                        'title': item['snippet']['title'],
                        'url': f"https://youtube.com/watch?v={video_id}",
                        'transcript': transcript
                    })
            return videos
        except Exception as e:
            print(f"Error searching videos: {e}")
            return []

class MultiAgentSystem:
    def __init__(self, llm: NvidiaLLM, youtube: YouTubeHelper):
        self.llm = llm
        self.youtube = youtube
        self.agents = self._create_agents()

    def _create_agents(self):
        import autogen
        
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            system_message="A human user needing assistance with content analysis.",
            human_input_mode="NEVER"
        )

        youtube_agent = autogen.AssistantAgent(
            name="youtube_expert",
            llm_config=self.llm.config,
            system_message="""Expert at analyzing YouTube video content and creating summaries. 
            If no videos are found or transcripts are empty, provide helpful suggestions."""
        )

        return {
            "user": user_proxy,
            "youtube": youtube_agent
        }

    def process_request(self, query: str) -> str:
        """Process user requests with proper error handling"""
        if not query or not isinstance(query, str):
            return "Please provide a valid query."
            
        try:
            # Validate and clean input
            cleaned_query = query.strip()
            if not cleaned_query:
                return "Please provide a non-empty query."

            # Direct LLM generation for simple requests
            if len(cleaned_query) < 100:  # For short queries, use direct generation
                return self.llm.generate(cleaned_query)

            # For more complex queries, use the appropriate agent
            if "youtube" in cleaned_query.lower():
                return self.analyze_videos(cleaned_query)
            else:
                return self.llm.generate(cleaned_query)

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)  # For debugging
            return error_msg

    def analyze_videos(self, query: str) -> str:
        """Analyze YouTube videos with proper error handling"""
        try:
            videos = self.youtube.search_videos(query)
            if not videos:
                return "No videos found. Please try a different search query."

            prompt = self._create_analysis_prompt(videos)
            return self.llm.generate(prompt)

        except Exception as e:
            return f"Error analyzing videos: {str(e)}"

    def _create_analysis_prompt(self, videos: List[Dict]) -> str:
        """Create a well-structured prompt for video analysis"""
        prompt_parts = ["Please analyze these videos:\n"]
        
        for video in videos:
            prompt_parts.extend([
                f"\nTitle: {video['title']}",
                f"URL: {video['url']}",
                f"Content: {video['transcript'][:500]}...\n"  # Limit transcript length
            ])
            
        prompt_parts.append("\nProvide a clear summary of the key points and main takeaways.")
        return "\n".join(prompt_parts)

def create_interface():
    """Create Gradio interface with proper error handling"""
    try:
        api_key = os.getenv('NVIDIA_NIM_API_KEY')
        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY environment variable not set")

        llm = NvidiaLLM(api_key=api_key)
        youtube = YouTubeHelper()
        agent_system = MultiAgentSystem(llm, youtube)

        iface = gr.Interface(
            fn=agent_system.process_request,
            inputs=gr.Textbox(
                lines=3,
                placeholder="Enter your request (e.g., 'Tell me about quantum computing')"
            ),
            outputs=gr.Textbox(
                lines=10,
                label="Response"
            ),
            title="AI Content Analysis System",
            description="Ask questions or request analysis of topics. For YouTube analysis, include 'youtube' in your query."
        )
        return iface

    except Exception as e:
        print(f"Error creating interface: {e}")
        raise

if __name__ == "__main__":
    try:
        iface = create_interface()
        iface.launch(debug=True)
    except Exception as e:
        print(f"Error launching interface: {e}")
