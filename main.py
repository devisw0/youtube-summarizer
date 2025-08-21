from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

video_id = "zYrU0ZsWIhU"

api = YouTubeTranscriptApi()
fetched = api.fetch(video_id) #returns fetchedtranscript objecg
raw = fetched.to_raw_data() #converting the fetchedtranscript object to list of dictionaries

print("segments:", len(raw))
for seg in raw[:5]:
    print("-", seg["text"])

# Join all segment texts into one big string
full_text = " ".join(s["text"].strip() for s in raw if s["text"].strip())
print(f"\nTranscript length (chars): {len(full_text):,}")
print(full_text[:500])  # quick peek at the first 500 chars


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(full_text[:1200], max_length=200, min_length=60, num_beams=4, do_sample=False, truncation=True)
final_output = result[0]['summary_text']
print("\n=== SUMMARY ===\n", final_output)

