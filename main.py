from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

VIDEO_ID = "zYrU0ZsWIhU"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 1) transcript
raw = YouTubeTranscriptApi().fetch(VIDEO_ID).to_raw_data()
full_text = " ".join(s["text"].strip() for s in raw if s["text"].strip())

# 2) MAP: chunk → summarize
step = 3200         # ~safe for BART (≈1024 tokens ≈ ~3500 chars, but varies!)
overlap = 400       # small overlap to avoid cutting thoughts
partials = []

i = 0
while i < len(full_text):
    chunk = full_text[i:i+step]
    out = summarizer(
        chunk,
        max_length=210, min_length=70,
        num_beams=4, do_sample=False, truncation=True
    )[0]["summary_text"].strip()
    partials.append(out)
    i += (step - overlap)

# 3) REDUCE: summarize the summaries
stitched = " ".join(partials)
final = summarizer(
    stitched,
    max_length=260, min_length=90,
    num_beams=4, do_sample=False, truncation=True
)[0]["summary_text"].strip()

print("\n=== SUMMARY ===\n", final)
