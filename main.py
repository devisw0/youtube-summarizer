from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

VIDEO_ID = "zYrU0ZsWIhU"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 1) transcript
raw = YouTubeTranscriptApi().fetch(VIDEO_ID).to_raw_data()
full_text = " ".join(s["text"].strip() for s in raw if s["text"].strip())

# 2) MAP: chunk → summarize
step = 3200         # ~chars per chunk
overlap = 400       # ~chars repeated between chunks
partials = []

i = 0
while i < len(full_text):
    chunk = full_text[i:i+step]
    # --- scale output length to input size ---
    # rough char→token proxy: tokens ≈ chars/4 (very approximate)
    est_tokens = max(1, len(chunk) // 4)
    per_max = min(210, max(60, int(est_tokens * 0.7)))
    per_min = min(per_max - 10, max(30, int(est_tokens * 0.3)))

    out = summarizer(
        chunk,
        max_length=per_max, min_length=per_min,
        num_beams=4, do_sample=False, truncation=True
    )[0]["summary_text"].strip()
    partials.append(out)

    i += (step - overlap)  # slide window, leaving overlap

# 3) REDUCE: summarize the summaries (skip if only 1 chunk)
if len(partials) == 1:
    final = partials[0]
else:
    stitched = " ".join(partials)
    est_tokens = max(1, len(stitched) // 4)
    final_max = min(260, max(80, int(est_tokens * 0.6)))
    final_min = min(final_max - 20, max(50, int(est_tokens * 0.3)))

    final = summarizer(
        stitched,
        max_length=final_max, min_length=final_min,
        num_beams=4, do_sample=False, truncation=True
    )[0]["summary_text"].strip()

print("\n=== SUMMARY ===\n", final)
