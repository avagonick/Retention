import anthropic
import json
import os
import subprocess
import sys
import tempfile
from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# --- STYLE REFERENCE (locked for this test video) ---

STYLE_REFERENCE = """
- Flat 2D vector animation, no shading/gradients/3D
- Background: solid white
- Primary colors: light blue (~#5BC8E8) water/accents, yellow (~#F2D94E) fish, red (~#E84C3D) callout boxes, navy blue (~#1B2A57) bold text/numbers
- Line style: bold black/dark outlines, rounded shapes, no sharp corners
- Text: bold sans-serif, all-caps in callout boxes, large centered numbers
- Motion: minimal, gentle easing, no camera pans/zooms, no fast cuts
"""

# --- AUDIO EXTRACTION + TRANSCRIPTION ---

def download_audio(source: str) -> str:
    """Download audio from YouTube URL or return local path. Returns path to audio file."""
    if source.startswith(("http://", "https://")):
        print(f"Downloading audio from: {source}")
        out_path = os.path.join(tempfile.gettempdir(), "retention_audio.mp3")
        subprocess.run(
            ["yt-dlp", "--remote-components", "ejs:github", "-x", "--audio-format", "mp3", "-o", out_path, "--force-overwrites", source],
            check=True,
        )
        print(f"Audio saved to: {out_path}")
        return out_path
    if os.path.isfile(source):
        return source
    raise FileNotFoundError(f"Not a URL or valid file path: {source}")


def transcribe(audio_path: str) -> list[dict]:
    """Transcribe audio with Deepgram. Returns list of {start, end, text} segments."""
    dg = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    print("Transcribing with Deepgram...")
    response = dg.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        smart_format=True,
        utterances=True,
        punctuate=True,
    )

    segments = []
    for utterance in response.results.utterances:
        segments.append({
            "start": round(utterance.start, 2),
            "end": round(utterance.end, 2),
            "text": utterance.transcript,
        })

    print(f"Transcribed {len(segments)} segments, total duration {segments[-1]['end']:.1f}s")
    return segments


def format_transcript(segments: list[dict]) -> str:
    """Format segments into timestamped transcript string."""
    lines = []
    for seg in segments:
        start_m, start_s = int(seg["start"] // 60), seg["start"] % 60
        end_m, end_s = int(seg["end"] // 60), seg["end"] % 60
        lines.append(f'[{start_m}:{start_s:05.2f}-{end_m}:{end_s:05.2f}] "{seg["text"]}"')
    return "\n\n".join(lines)


def get_segment_context(segments: list[dict], start: float, end: float, context_window: float = 15.0):
    """Extract the flagged segment + surrounding context from transcript segments."""
    flagged_text = []
    context_segments = []

    for seg in segments:
        if seg["start"] >= (start - context_window) and seg["end"] <= (end + context_window):
            context_segments.append(seg)
        if seg["start"] >= start and seg["end"] <= end:
            flagged_text.append(seg["text"])
        elif seg["start"] < end and seg["end"] > start:
            flagged_text.append(seg["text"])

    return " ".join(flagged_text), format_transcript(context_segments)


# --- HARDCODED TEST CASE (fallback) ---

DEMO_TRANSCRIPT_CONTEXT = """
[0:60-0:75] "So if we have twelve fish and we want to split them equally into three bowls, how many fish go in each bowl? That's right — twelve divided by three equals four."

[0:75-0:90] "Now let's think about why that works. Division is really just repeated subtraction. We take away three, take away three, take away three, take away three — and we've done it four times. So twelve divided by three is four."

[0:90-1:05] "Let's try another one. What about fifteen divided by five?"
"""

DEMO_FLAGGED_SEGMENT = {
    "start": "0:75",
    "end": "0:90",
    "duration_seconds": 15,
    "transcript": "Now let's think about why that works. Division is really just repeated subtraction. We take away three, take away three, take away three, take away three — and we've done it four times. So twelve divided by three is four.",
    "visual_description": "Static image: 3 fishbowls with 4 fish each, equation '12 ÷ 3 = 4' overlaid in navy text. No motion or animation for the full 15 seconds while narration continues explaining repeated subtraction."
}


# --- STAGE 1: Pedagogical Diagnosis ---

def run_stage1(flagged_segment: dict, transcript_context: str):
    prompt = f"""You are an educational content analyst. A neuroscience-based attention model has flagged the following segment of a children's math lecture video as a "dip zone" — a moment where predicted student engagement drops.

FLAGGED SEGMENT ({flagged_segment['start']} - {flagged_segment['end']}, {flagged_segment['duration_seconds']}s):
Transcript: "{flagged_segment['transcript']}"
On-screen visual: {flagged_segment['visual_description']}

SURROUNDING CONTEXT:
{transcript_context}

Your task:
1. DIAGNOSE: In 1-2 sentences, explain WHY this segment likely loses student attention. Consider the mismatch between audio and visual channels, pacing, abstraction level, etc.

2. PROPOSE FIX: Describe ONE concrete, narrow visual fix — a short animation that could replace the static image during this segment. The fix should:
   - Directly illustrate the concept being narrated
   - Be approximately {flagged_segment['duration_seconds']} seconds long
   - Use only elements already present in the video style
   - Add meaningful motion that reinforces the narration

Respond in this exact JSON format:
{{
  "diagnosis": "...",
  "fix_description": "...",
  "fix_duration_seconds": {flagged_segment['duration_seconds']}
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


# --- STAGE 2: Style-Locked Pika Prompt Generation ---

def run_stage2(stage1_output: dict):
    prompt = f"""You are a prompt engineer specializing in AI video generation. Convert the following visual fix description into a single Pika prompt that will generate a clip matching the source video's style exactly.

FIX TO IMPLEMENT:
{stage1_output['fix_description']}
Duration: {stage1_output['fix_duration_seconds']} seconds

STYLE REFERENCE (the generated clip MUST match this exactly — no new colors, characters, or style elements):
{STYLE_REFERENCE}

RULES FOR THE PIKA PROMPT:
1. Open with the style description as a locked prefix (so the model understands the visual language)
2. Then describe ONLY the motion/action content of the fix
3. End with pacing/duration constraints
4. Do NOT introduce any elements outside the style reference
5. Keep it under 200 words total
6. Write it as a single continuous prompt (no line breaks or bullet points)

Respond with ONLY the Pika prompt text, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


# --- MAIN ---

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None

    if source:
        # --- LIVE MODE: YouTube URL or audio file ---
        audio_path = download_audio(source)
        segments = transcribe(audio_path)
        full_transcript = format_transcript(segments)

        print("\n" + "=" * 60)
        print("FULL TRANSCRIPT")
        print("=" * 60)
        print(full_transcript)

        # For now, prompt user for the dip zone timestamps
        print("\n" + "=" * 60)
        dip_start = float(input("Enter dip zone START time (seconds): "))
        dip_end = float(input("Enter dip zone END time (seconds): "))
        visual_desc = input("Describe what's on screen during this segment: ")

        flagged_text, context = get_segment_context(segments, dip_start, dip_end)

        flagged_segment = {
            "start": f"{int(dip_start//60)}:{dip_start%60:.0f}",
            "end": f"{int(dip_end//60)}:{dip_end%60:.0f}",
            "duration_seconds": round(dip_end - dip_start),
            "transcript": flagged_text,
            "visual_description": visual_desc,
        }
        transcript_context = context
    else:
        # --- DEMO MODE: hardcoded test case ---
        print("No source provided — running with demo test case.")
        print("Usage: python pika.py <youtube-url-or-audio-file>")
        flagged_segment = DEMO_FLAGGED_SEGMENT
        transcript_context = DEMO_TRANSCRIPT_CONTEXT

    print("\n" + "=" * 60)
    print("STAGE 1: Pedagogical Diagnosis")
    print("=" * 60)

    stage1 = run_stage1(flagged_segment, transcript_context)
    print(f"\nDiagnosis: {stage1['diagnosis']}")
    print(f"\nProposed Fix: {stage1['fix_description']}")
    print(f"\nDuration: {stage1['fix_duration_seconds']}s")

    print("\n" + "=" * 60)
    print("STAGE 2: Pika Prompt Generation")
    print("=" * 60)

    pika_prompt = run_stage2(stage1)
    print(f"\nGenerated Pika Prompt:\n")
    print(pika_prompt)

    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    output = {
        "stage1": stage1,
        "pika_prompt": pika_prompt,
        "metadata": {
            "source_timestamp": f"{flagged_segment['start']}-{flagged_segment['end']}",
            "duration": flagged_segment['duration_seconds'],
            "source": source or "demo",
        }
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
