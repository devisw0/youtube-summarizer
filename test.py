from youtube_transcript_api import YouTubeTranscriptApi

video_id = "y76wfHmySak"
transcript = YouTubeTranscriptApi.get_transcript(video_id)

for segment in transcript[:5]:
    print(segment["text"])
