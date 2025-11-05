#!/usr/bin/env python3
"""
GRPO training for BrowserGym tasks using TRL's `GRPOTrainer` and the OpenEnv HTTP
environment interface.

Usage:
    # First, start a BrowserGym server (Docker or local) and note its base URL.
    # For MiniWoB++ you can use the official BrowserGym container.

    # Then run this training script (customise env vars as needed):
    BROWSERGYM_BASE_URL=http://localhost:8001 \
        VLLM_ENDPOINT=http://localhost:8000/generate/ \
        python grpo.py
        
python -m http.server 8888
BROWSERGYM_BENCHMARK="miniwob" BROWSERGYM_TASK_NAME="buy-ticket" MINIWOB_URL="http://localhost:8888/miniwob/" BROWSERGYM_PORT=8001 python app.py
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="/fsx/benjamin_burtenshaw/OpenEnv/src:${PYTHONPATH}" VLLM_ENDPOINT="http://localhost:8010/generate/" python grpo.py
CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model "Qwen/Qwen3-VL-2B-Instruct" --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from itertools import zip_longest

import base64
from io import BytesIO
import os
import re
import textwrap

import numpy as np
from PIL import Image
import requests
from datasets import Dataset
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import prepare_multimodal_messages

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from envs.browsergym_env import BrowserGymAction, BrowserGymEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/generate/")
VLLM_SERVER_PORT = VLLM_ENDPOINT.split(":")[-1].split("/")[0]
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "16"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
TOP_K = int(os.getenv("TOP_K", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("PER_DEVICE_BATCH_SIZE", "1"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "1000"))
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "100"))   
OUTPUT_DIR = f"outputs/browsergym-grpo-{MODEL_ID.replace('/', '-')}-{NOW}"
RUN_ID = f"run-{NOW}"
PROJECT_ID = f"browsergym-{MODEL_ID.replace('/', '-')}"
SPACE_ID = "BrowserGym-GRPO"

STEP_DENSE_WEIGHT = float(os.getenv("STEP_DENSE_WEIGHT", "0.2"))
SHAPING_NOOP_PENALTY = float(os.getenv("SHAPING_NOOP_PENALTY", "0.05"))
SHAPING_ERROR_PENALTY = float(os.getenv("SHAPING_ERROR_PENALTY", "0.1"))
SHAPING_REPEAT_ACTION_PENALTY = float(
    os.getenv("SHAPING_REPEAT_ACTION_PENALTY", "0.05")
)
SHAPING_NOVEL_CLICK_BONUS = float(os.getenv("SHAPING_NOVEL_CLICK_BONUS", "0.05"))
SHAPING_REPEAT_CLICK_PENALTY = float(
    os.getenv("SHAPING_REPEAT_CLICK_PENALTY", "0.02")
)
SHAPING_NOVEL_TEXT_BONUS = float(os.getenv("SHAPING_NOVEL_TEXT_BONUS", "0.05"))
SHAPING_REPEAT_TEXT_PENALTY = float(
    os.getenv("SHAPING_REPEAT_TEXT_PENALTY", "0.02")
)

BROWSERGYM_BASE_URL = os.getenv("BROWSERGYM_BASE_URL", "http://localhost:8001")
MAX_DOM_CHARS = int(os.getenv("MAX_DOM_CHARS", "3500"))
FALLBACK_ACTION = os.getenv("FALLBACK_ACTION", "noop()")

ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:\-]\s*", re.IGNORECASE)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)
CLICK_ACTION_RE = re.compile(r"click\('([^']+)'\)", re.IGNORECASE)
TEXT_ENTRY_ACTION_RE = re.compile(r"(type|fill)\('([^']+)'", re.IGNORECASE)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser.
    Commands:
    - noop()
    - click('<BID>')
    - type('selector', 'text to enter')
    - fill('selector', 'text to enter')
    - send_keys('Enter')
    - scroll('down')
    """
).strip()

DEBUG = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes"}
if DEBUG:
    print("=" * 100)
    print("DEBUG mode enabled")
    print("=" * 100)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = getattr(processor, "tokenizer", None)

if tokenizer is None:
    raise RuntimeError("Processor does not expose a tokenizer instance.")
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

env = BrowserGymEnv(base_url=BROWSERGYM_BASE_URL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n... [truncated]"


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def extract_screenshot_image(observation) -> Optional[Image.Image]:
    screenshot = getattr(observation, "screenshot", None)
    array = np.array(screenshot, dtype=np.uint8)
    return Image.fromarray(array)


def extract_clickable_elements(observation) -> List[Dict[str, str]]:
    metadata = getattr(observation, "metadata", {}) or {}
    obs_dict = metadata.get("browsergym_obs", {}) or {}
    extra_props = obs_dict.get("extra_element_properties", {}) or {}

    clickables: List[Dict[str, str]] = []
    for bid, props in extra_props.items():
        if not props.get("clickable"):
            continue
        bbox = props.get("bbox") or []
        bbox_str = ", ".join(str(item) for item in bbox) if bbox else "?"
        clickables.append({"bid": str(bid), "bbox": bbox_str})

    clickables.sort(key=lambda item: item["bid"])
    return clickables


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    doc = (
        observation.axtree_txt
        or observation.pruned_html
        or observation.text
        or ""
    )
    doc = truncate_text(doc, MAX_DOM_CHARS)
    goal = observation.goal or "(not provided)"
    url = observation.url or "(unknown)"
    error_note = "Yes" if observation.last_action_error else "No"

    clickables = extract_clickable_elements(observation)
    if clickables:
        actions_hint = "\n".join(
            f"    - {item['bid']} (bbox: {item['bbox']})" for item in clickables
        )
    else:
        actions_hint = "    (none detected)"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Goal: {goal}
        Current URL: {url}
        Previous steps:
        {build_history_lines(history)}
        Last action error: {error_note}

        Available clickable element IDs:
{actions_hint}

        Page snapshot (truncated):
        {doc}

        Reply with exactly one BrowserGym action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            action = re.sub(r"\s+", " ", action)
            return action

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION


def compute_dense_step_reward(
    action_str: str,
    observation,
    extrinsic_reward: float,
    seen_click_ids: Set[str],
    seen_text_fields: Set[str],
    previous_action: Optional[str],
) -> float:
    """Compute dense shaping reward for a single step."""

    reward = 0.0
    normalized_action = (action_str or "").strip()
    if not normalized_action:
        return reward

    if previous_action and previous_action == normalized_action:
        reward -= SHAPING_REPEAT_ACTION_PENALTY

    lowered_action = normalized_action.lower()

    if lowered_action.startswith("noop"):
        reward -= SHAPING_NOOP_PENALTY

    if getattr(observation, "last_action_error", False):
        reward -= SHAPING_ERROR_PENALTY

    click_match = CLICK_ACTION_RE.search(normalized_action)
    if click_match:
        click_id = click_match.group(1).strip()
        if click_id:
            if click_id not in seen_click_ids:
                reward += SHAPING_NOVEL_CLICK_BONUS
                seen_click_ids.add(click_id)
            else:
                reward -= SHAPING_REPEAT_CLICK_PENALTY

    text_match = TEXT_ENTRY_ACTION_RE.search(normalized_action)
    if text_match:
        field_selector = text_match.group(2).strip()
        if field_selector:
            if field_selector not in seen_text_fields:
                reward += SHAPING_NOVEL_TEXT_BONUS
                seen_text_fields.add(field_selector)
            else:
                reward -= SHAPING_REPEAT_TEXT_PENALTY

    # Small positive reinforcement for receiving extrinsic reward.
    if extrinsic_reward > 0:
        reward += min(extrinsic_reward, 1.0) * 0.05

    return reward


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_vllm_completion(
    prompt: str, args: GRPOConfig, images: Optional[List[Image.Image]] = None
) -> Dict[str, List]:
    payload: Dict[str, object] = {
        "prompts": [prompt],
        "n": 1,
        "temperature": getattr(args, "temperature", TEMPERATURE),
        "max_tokens": getattr(args, "max_completion_length", MAX_NEW_TOKENS),
        "logprobs": True,
    }

    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        payload["top_k"] = top_k

    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        payload["top_p"] = top_p

    if images:
        payload["images"] = [pil_to_base64(img) for img in images]

    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    prompt_ids = data.get("prompt_ids") or data.get("prompt_token_ids") or [[]]
    completion_ids = (
        data.get("completion_ids") or data.get("completion_token_ids") or [[]]
    )
    logprobs = data.get("logprobs") or data.get("completion_logprobs") or [[]]
    texts = data.get("completions") or data.get("completion_texts") or data.get("texts")

    return {
        "prompt_ids": prompt_ids[0] if prompt_ids else [],
        "completion_ids": completion_ids[0] if completion_ids else [],
        "logprobs": [float(lp) for lp in (logprobs[0] if logprobs else [])],
        "text": (texts[0] if texts else None),
    }


def rollout_once(
    env: BrowserGymEnv,
    processor: AutoProcessor,
    args: GRPOConfig,
    dataset_prompt: str,
) -> Dict[str, List]:
    del dataset_prompt  # Not used for BrowserGym rollouts

    result = env.reset()
    observation = result.observation

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    history: List[str] = []
    step_rewards: List[float] = []
    dense_step_rewards: List[float] = []
    seen_click_ids: Set[str] = set()
    seen_text_fields: Set[str] = set()
    previous_action: Optional[str] = None

    tokenizer_inner = getattr(processor, "tokenizer", None)
    if tokenizer_inner is None:
        raise RuntimeError("Processor used in rollout does not expose a tokenizer.")

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(step, observation, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        screenshot_image = extract_screenshot_image(observation)
        if screenshot_image is not None:
            structured_messages = prepare_multimodal_messages(messages, [screenshot_image])
            images = [screenshot_image]
        else:
            structured_messages = messages
            images = None

        prompt_text = processor.apply_chat_template(
            structured_messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        vllm_result = request_vllm_completion(prompt_text, args, images=images)

        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])

        completion_text = vllm_result.get("text") or tokenizer_inner.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        action_str = parse_model_action(completion_text)

        result = env.step(BrowserGymAction(action_str=action_str))
        reward = float(result.reward or 0.0)
        step_rewards.append(reward)

        dense_reward = compute_dense_step_reward(
            action_str,
            result.observation,
            reward,
            seen_click_ids,
            seen_text_fields,
            previous_action,
        )
        dense_step_rewards.append(dense_reward)

        observation = result.observation
        history_line = f"Step {step}: {action_str} -> reward {reward:+.2f}"
        if observation.last_action_error:
            history_line += " ERROR"
        history.append(history_line)

        previous_action = action_str

        if DEBUG:
            print("=" * 100)
            print(f"Prompt:\n{user_prompt}")
            print(f"Action: {action_str}")
            print(f"Reward: {reward}")
            print(f"Done: {result.done}")
            print("=" * 100)

        if result.done:
            break

    episode_reward = float(sum(step_rewards))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "episode_reward": episode_reward,
        "step_rewards": step_rewards,
        "dense_step_rewards": dense_step_rewards,
    }


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def rollout_func(
    prompts: List[str], args: GRPOConfig, processing_class
) -> Dict[str, List]:
    all_prompt_ids: List[List[int]] = []
    all_completion_ids: List[List[int]] = []
    all_logprobs: List[List[float]] = []
    episode_rewards: List[float] = []
    per_step_rewards: List[List[float]] = []
    per_dense_rewards: List[List[float]] = []
    num_generations = args.num_generations or NUM_GENERATIONS

    for _ in range(num_generations):
        for prompt_text in prompts:
            rollout_stats = rollout_once(env, processing_class, args, prompt_text)
            all_prompt_ids.append(rollout_stats["prompt_ids"])
            all_completion_ids.append(rollout_stats["completion_ids"])
            all_logprobs.append(rollout_stats["logprobs"])
            episode_rewards.append(rollout_stats["episode_reward"])
            per_step_rewards.append(rollout_stats.get("step_rewards", []))
            per_dense_rewards.append(rollout_stats.get("dense_step_rewards", []))

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "episode_reward": episode_rewards,
        "step_rewards": per_step_rewards,
        "dense_step_rewards": per_dense_rewards,
    }


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reward_env(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    base_rewards = (kwargs or {}).get("episode_reward") or []
    dense_rewards = (kwargs or {}).get("dense_step_rewards") or []

    dense_sums = [sum(step_rewards or []) for step_rewards in dense_rewards]

    shaped: List[float] = []
    for base, dense_sum in zip_longest(base_rewards, dense_sums, fillvalue=0.0):
        total = float(base) + STEP_DENSE_WEIGHT * float(dense_sum)
        shaped.append(total)

    if len(shaped) < len(completions):
        shaped.extend([0.0] * (len(completions) - len(shaped)))

    if not shaped:
        return [0.0 for _ in completions]

    mean_reward = float(np.mean(shaped))
    normalized = [reward - mean_reward for reward in shaped]
    return normalized


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    train_dataset = Dataset.from_dict({"prompt": ["BrowserGym agent"] * DATASET_SIZE})

    grpo_config = GRPOConfig(
        vllm_mode="server",
        use_vllm=True,
        vllm_server_port=VLLM_SERVER_PORT,
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_INTERVAL,
        save_total_limit=None,
        report_to="trackio",
        trackio_space_id=SPACE_ID,
        run_name=RUN_ID,
        project=PROJECT_ID,
    )

    grpo_config.temperature = TEMPERATURE
    grpo_config.top_k = TOP_K
    grpo_config.run_name = RUN_ID
    grpo_config.project = PROJECT_ID
    grpo_config.trackio_space_id = SPACE_ID

    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=processor,
        reward_funcs=[reward_env],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training with BrowserGym environment...")
    print(f"Using {NUM_GENERATIONS} rollouts per dataset prompt")

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
