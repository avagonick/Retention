"""
Generator agent — a Claude conversation that decides HOW to generate each video.

Three inputs every iteration:
  1. question          — learner's voice input transcribed by Deepgram (always present)
  2. previous_video    — the video generated last iteration (source video on iteration 1)
  3. feedback          — evaluator's improvement text (empty on iteration 1)

Claude reasons over all three and calls generate_fn as a tool. The conversation
grows across iterations so Claude can see patterns across all prior attempts.

Expected generate_fn signature:
    def generate_fn(
        question: str,            # Deepgram transcript
        previous_video_path: str, # last generated video (or source on iter 1)
        iteration: int,
        feedback: str,            # evaluator's improvement instruction (empty on iter 1)
        style: str,
        pacing: str,
        visual_complexity: str,
        add_text_overlays: bool,
        add_motion: bool,
        creative_direction: str,
    ) -> str   # local file path to saved video
"""

import asyncio
import logging
from typing import Callable

import anthropic

from .band import Band

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a Generator agent creating educational videos optimized for memory retention.

ITERATION 1 — you have two inputs:
  1. SOURCE VIDEO — the original content the user uploaded (what to teach from)
  2. LEARNER QUESTION — the exact words the learner spoke, transcribed from their voice

  On iteration 1 your job is to read the learner's question carefully and build a video
  that directly answers it. The question is the creative brief. Use their exact words to
  drive your creative_direction — what concept are they asking about, what confusion are
  they expressing, what do they need to see to understand it?

ITERATION 2+ — you also have:
  3. PREVIOUS VIDEO — path to the video you generated last iteration
  4. EVALUATOR FEEDBACK — brain scores + panel diagnosis of what to fix

  How brain scores map to creative choices:
  - High DMN at a specific second → viewer's mind wandered there → add a visual hook or
    cut at that moment, increase motion or contrast around that timestamp
  - Low memory regions → content not encoding → add visual anchors, repetition, text overlays
  - Low attention → losing focus → increase visual complexity, add kinetic elements
  - High reward already → don't overcorrect — small targeted changes only

  Think step-by-step: which second was worst? what was happening there? what would fix it?

Always call generate_video with a concrete creative_direction.\
"""

_GENERATE_TOOL = {
    "name": "generate_video",
    "description": (
        "Generate an educational video using Pika. Call this once per turn with "
        "your creative direction for this iteration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "description": "Visual style (e.g. 'kinetic typography', '2D animation', 'diagram walkthrough', 'whiteboard')",
            },
            "pacing": {
                "type": "string",
                "enum": ["fast", "medium", "slow"],
            },
            "visual_complexity": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "add_text_overlays": {
                "type": "boolean",
                "description": "Add key-term text overlays on screen",
            },
            "add_motion": {
                "type": "boolean",
                "description": "Add kinetic motion / animated elements",
            },
            "creative_direction": {
                "type": "string",
                "description": (
                    "Specific brief for this iteration — what to emphasize, avoid, "
                    "or change relative to the last attempt. Be concrete."
                ),
            },
        },
        "required": [
            "style", "pacing", "visual_complexity",
            "add_text_overlays", "add_motion", "creative_direction",
        ],
    },
}


def _format_evaluation(evaluation: dict) -> str:
    scores  = evaluation["scores"]
    per_sec = scores.get("per_second", [])
    per_sec_str = "\n".join(
        f"  t={r['second']:2d}s  hipp={r['hippocampus']:+.3f}  "
        f"pfc={r['left_pfc']:+.3f}  amyg={r['amygdala']:+.3f}  "
        f"dmn={r['dmn']:+.3f}  reward={r['reward']:+.3f}"
        for r in per_sec
    )
    return (
        f"Brain scores (mean):\n"
        f"  hippocampus (memory encoding) = {scores['hippocampus']:.4f}\n"
        f"  left_pfc    (semantic depth)  = {scores['left_pfc']:.4f}\n"
        f"  amygdala    (emotional hook)  = {scores['amygdala']:.4f}\n"
        f"  dmn         (mind-wandering)  = {scores['dmn']:.4f}\n"
        f"  reward                        = {scores['reward']:.4f}\n"
        f"Worst DMN spike at second {scores.get('peak_dmn_second', '?')}s  |  "
        f"Best moment at second {scores.get('peak_memory_second', '?')}s\n"
        f"Per-second:\n{per_sec_str}\n"
        f"Feedback: {evaluation['reason']}"
    )


async def generator_agent(
    question: str,
    source_video_path: str,
    band: Band,
    generate_fn: Callable,
    max_iterations: int,
) -> str | None:
    client = anthropic.AsyncAnthropic()
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"LEARNER QUESTION (transcribed from voice via Deepgram):\n\"{question}\"\n\n"
                f"SOURCE VIDEO: {source_video_path}\n\n"
                "This is iteration 1. Read the learner's question carefully — those are their "
                "exact words. Build a video that directly answers what they asked. "
                "Use the question to drive your creative_direction."
            ),
        },
    ]

    previous_video_path = source_video_path  # iteration 1: source is the reference
    feedback_text       = ""                 # iteration 1: no prior feedback

    # Score the source video first so iteration 1 knows what's already broken
    await band.generator_send({
        "video_path": source_video_path,
        "iteration":  0,
        "is_source":  True,
    })
    source_evaluation = await band.generator_recv()
    feedback_text = source_evaluation.get("reason", "")

    messages[0]["content"] = (
        f"LEARNER QUESTION (transcribed from voice via Deepgram):\n\"{question}\"\n\n"
        f"SOURCE VIDEO: {source_video_path}\n\n"
        f"SOURCE VIDEO BRAIN EVALUATION:\n{_format_evaluation(source_evaluation)}\n\n"
        "This is what's wrong with the original video. Iteration 1: build a video that "
        "directly answers the learner's question AND fixes the weaknesses shown above. "
        "Reference the specific seconds where brain scores were worst."
    )

    for iteration in range(1, max_iterations + 1):
        if iteration > 1:
            evaluation = await band.generator_recv()
            feedback_text = evaluation.get("reason", "")
            messages.append({
                "role": "user",
                "content": (
                    f"Iteration {iteration} — evaluation of your last video:\n\n"
                    f"PREVIOUS VIDEO: {previous_video_path}\n\n"
                    f"{_format_evaluation(evaluation)}\n\n"
                    "Generate an improved video. Reference specific seconds where scores were bad."
                ),
            })

        response = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=_SYSTEM,
            tools=[_GENERATE_TOOL],
            tool_choice={"type": "any"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_block = next(b for b in response.content if b.type == "tool_use")
        params = tool_block.input
        logger.info("[generator] iteration %d → %s", iteration, params.get("creative_direction", ""))

        video_path: str = await asyncio.to_thread(
            generate_fn,
            question=question,
            previous_video_path=previous_video_path,
            iteration=iteration,
            feedback=feedback_text,
            **params,
        )

        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": video_path}],
        })

        previous_video_path = video_path  # next iteration references this video

        await band.generator_send({
            "video_path": video_path,
            "iteration":  iteration,
            "params":     params,
        })

        if iteration == max_iterations:
            await band.generator_send({"done": True})
            break

    return previous_video_path
