#!/usr/bin/env python3
"""
GRPO training for Wordle using the TextArena OpenEnv environment.

Usage:
    # First, start the TextArena Wordle server (Docker or local):
    TEXTARENA_ENV_ID=Wordle-v0 TEXTARENA_NUM_PLAYERS=1 \
        python -m src.envs.textarena_env.server.app

    # Then run this training script:
    python grpo.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from typing import Iterable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


model_id = "Qwen/Qwen3-0.6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Note: Gradient checkpointing can cause issues with LoRA + eval/train mode switching
# We use other memory optimizations instead (bfloat16, small batches, etc.)
# Uncomment if you have memory issues and are not using LoRA:
# model.gradient_checkpointing_enable()


import numpy as np
from envs.textarena_env import TextArenaAction, TextArenaEnv
from envs.textarena_env.models import TextArenaMessage

# Connect to the TextArena Wordle environment server (make sure it's running!)
# Start with: TEXTARENA_ENV_ID=Wordle-v0 python -m envs.textarena_env.server.app
env = TextArenaEnv(base_url="http://localhost:8000")

MAX_TURNS = 8

SYSTEM_PROMPT = (
    "You are an expert Wordle solver."
    " Always respond with a single guess inside square brackets, e.g. [crane]."
    " Use lowercase letters, exactly one five-letter word per reply."
    " Reason about prior feedback before choosing the next guess."
    " Words must be 5 letters long and real English words."
    " Do not include any other text in your response."
    " Do not repeat the same guess twice."
)


max_train_steps = 500  # More steps to see actual learning
num_generations = 4  # REDUCED: Number of episodes to run per training step (was 8)
max_new_tokens = 8  # Allow generation of one bracketed guess plus reasoning tokens
max_episode_steps = 8  # Wordle has at most 8 turns in our configuration
temperature = 0.7  # Lower temperature for more focused action selection
top_k = 10  # Smaller top_k for more deterministic actions
learning_rate = 1e-5  # Higher learning rate
weight_decay = 0.0
epsilon = 0.2
gradient_accumulation_steps = 2  # INCREASED: Accumulate gradients to reduce memory
warmup_ratio = 0.1
logging_frequency = 10


import re
import gc
import torch.nn.functional as F
from contextlib import nullcontext


def format_history(messages: Iterable[TextArenaMessage]) -> str:
    """Convert TextArena message history into plain text for the model."""

    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def extract_guess(text: str) -> str:
    """Return the first Wordle-style guess enclosed in square brackets."""

    match = re.search(r"\[[A-Za-z]{5}\]", text)
    if match:
        return match.group(0).lower()

    # Fallback: remove non-letters and enforce lowercase 5-letter word
    cleaned = re.sub(r"[^a-zA-Z]", "", text).lower()
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[crane]"


def make_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    """Combine the TextArena prompt and feedback history for the model."""

    history = format_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."

    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


def run_wordle_episode(env: TextArenaEnv, model, tokenizer, device, max_steps):
    """Run a single Wordle episode and collect prompts/completions for training."""

    result = env.reset()
    observation = result.observation

    episode_reward = 0.0
    all_prompt_ids = []
    all_completion_ids = []
    prompt_lengths = []
    seen_guesses = set()
    turn = 0

    while not result.done and turn < max_steps:
        prompt_text = make_user_prompt(observation.prompt, observation.messages)
        prompt_with_rules = f"{SYSTEM_PROMPT}\n\n{prompt_text}"

        messages = [{"role": "user", "content": prompt_with_rules}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = outputs[0, prompt_length:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        all_prompt_ids.extend(inputs["input_ids"][0].cpu().tolist())
        all_completion_ids.extend(completion_ids.cpu().tolist())
        prompt_lengths.append(prompt_length)

        guess = extract_guess(completion_text)
        if guess in seen_guesses:
            # Force a fallback to avoid duplicates
            for fallback in ["[crane]", "[slate]", "[adieu]", "[roate]"]:
                if fallback not in seen_guesses:
                    guess = fallback
                    break
        seen_guesses.add(guess)

        result = env.step(TextArenaAction(message=guess))
        reward = result.reward or 0.0
        episode_reward += reward
        observation = result.observation
        turn += 1

        del inputs, outputs, completion_ids

    return episode_reward, all_prompt_ids, all_completion_ids, prompt_lengths


def per_token_log_probs(logits, labels, use_float32=False):
    """
    Compute log probabilities for each token without materialising full log-softmax.

    Args:
        logits: Model logits (kept in bfloat16 by default for memory efficiency)
        labels: Target token IDs
        use_float32: If True, convert to float32 (more accurate but uses 2x memory)

    Note: bfloat16 is sufficient for RL training and saves significant memory.
    """
    if use_float32 and logits.dtype != torch.float32:
        logits = logits.to(torch.float32)

    vocab_size = logits.size(-1)
    # Use reshape instead of view for gradient checkpointing compatibility
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    per_token_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        ignore_index=tokenizer.pad_token_id,
    )
    return (-per_token_loss).reshape_as(labels)


# Setup autocast context for mixed precision training
# We use bfloat16 throughout for memory efficiency (4x less than float32)
# bfloat16 has the same exponent range as float32, making it ideal for RL training
if device.type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()


optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
total_update_steps = max_train_steps // gradient_accumulation_steps
warmup_steps = max(1, int(total_update_steps * warmup_ratio))
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

import trackio

trackio.init(project="grpo-wordle")

model.train()
global_step = 0
running_reward = 0.0
running_loss = 0.0
logging_frequency = 10

print("Starting GRPO training with Wordle environment...")
print(f"Running {num_generations} episodes per training step")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Device: {device}")
print(f"Using bfloat16: {device.type == 'cuda'}")

for step in range(1, max_train_steps + 1):
    print(f"\nStep {step} of {max_train_steps}")
    # Run multiple Wordle episodes to collect training data
    model.eval()
    episode_rewards = []
    all_sequences = []
    all_prompt_lengths = []  # Track prompt lengths for proper masking

    for episode_idx in range(0, num_generations):
        (
            episode_reward,
            prompt_ids,
            completion_ids,
            prompt_lengths,
        ) = run_wordle_episode(
            env, model, tokenizer, device, max_steps=max_episode_steps
        )
        episode_rewards.append(episode_reward)

        # Combine prompt and completion into full sequence
        full_sequence = prompt_ids + completion_ids
        all_sequences.append(full_sequence)
        all_prompt_lengths.append(sum(prompt_lengths))

        # Clear memory after each episode to prevent accumulation
        if device.type == "cuda" and episode_idx % 2 == 1:  # Every 2 episodes
            torch.cuda.empty_cache()

    # Clear memory after all episode generation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    model.train()

    # Pad sequences to same length
    max_len = max(len(seq) for seq in all_sequences)
    padded_sequences = []
    padded_completion_masks = []

    for seq, prompt_len in zip(all_sequences, all_prompt_lengths):
        # Pad sequence
        padded = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
        padded_sequences.append(padded)

        # Create completion mask: 1 for completion tokens, 0 for prompt and padding
        # CRITICAL FIX: Only train on the completion tokens (actions), not the prompts
        comp_mask = [0] * max_len
        for i in range(prompt_len, len(seq)):
            comp_mask[i] = 1
        padded_completion_masks.append(comp_mask)

    sequences = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    attention_mask = (sequences != tokenizer.pad_token_id).long()
    completion_mask = torch.tensor(
        padded_completion_masks, dtype=torch.long, device=device
    )

    # Convert episode rewards to tensor (use bfloat16 on GPU, float32 on CPU)
    reward_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rewards = torch.tensor(episode_rewards, dtype=reward_dtype, device=device)
    running_reward += rewards.mean().item()

    # Compute advantages (normalize rewards) - keep in bfloat16
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    std_reward = std_reward if std_reward > 0 else 1.0
    advantages = (rewards - mean_reward) / std_reward

    # Prepare labels for loss computation
    labels = sequences[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = tokenizer.pad_token_id

    # Compute old log probs (policy before update)
    with torch.no_grad():
        with autocast_ctx if device.type == "cuda" else nullcontext():
            old_outputs = model(
                input_ids=sequences,
                attention_mask=attention_mask,
                use_cache=False,
            )
        old_log_probs = per_token_log_probs(old_outputs.logits[:, :-1], labels)
        # Delete old_outputs to free memory
        del old_outputs

    valid_mask = (completion_mask[:, 1:] == 1) & (labels != tokenizer.pad_token_id)

    # Compute new log probs and loss
    # Note: With gradient_accumulation_steps > 1, we only zero grads at the start
    if step % gradient_accumulation_steps == 1:
        optimizer.zero_grad(set_to_none=True)

    with autocast_ctx if device.type == "cuda" else nullcontext():
        outputs = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            use_cache=False,
        )
        log_probs = per_token_log_probs(outputs.logits[:, :-1], labels)
        # Delete outputs immediately to free memory
        del outputs

    # GRPO loss computation
    ratio = (log_probs - old_log_probs).exp()
    ratio = torch.where(valid_mask, ratio, torch.ones_like(ratio))
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)

    adv = advantages.unsqueeze(1)
    loss_unclipped = ratio * adv
    loss_clipped = clipped_ratio * adv
    per_token_loss = -torch.min(loss_unclipped, loss_clipped)
    per_token_loss = torch.where(
        valid_mask, per_token_loss, torch.zeros_like(per_token_loss)
    )

    denom = valid_mask.sum().clamp(min=1)
    loss = per_token_loss.sum() / denom

    # Scale loss by gradient accumulation steps
    loss = loss / gradient_accumulation_steps

    # Backprop and update (only step optimizer every gradient_accumulation_steps)
    loss.backward()

    if step % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    global_step += 1
    running_loss += loss.item()

    # Clear memory after training step
    del sequences, attention_mask, completion_mask, rewards, advantages
    del labels, old_log_probs, valid_mask, log_probs
    del ratio, clipped_ratio, loss_unclipped, loss_clipped, per_token_loss, loss

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Logging
    if step % logging_frequency == 0:
        avg_reward = running_reward / logging_frequency
        avg_loss = running_loss / logging_frequency
        current_lr = scheduler.get_last_lr()[0]
        wins = sum(1 for r in episode_rewards if r > 0)
        losses = sum(1 for r in episode_rewards if r < 0)
        ties = sum(1 for r in episode_rewards if r == 0)
        print(
            f"step={step:04d} | loss={avg_loss:.4f} | avg_reward={avg_reward:.4f} | lr={current_lr:.2e}"
        )
        print(f"  Episode rewards: {[f'{r:+.1f}' for r in episode_rewards]}")
        print(
            f"  Win/Loss/Tie: {wins}/{losses}/{ties} (win rate: {wins/len(episode_rewards)*100:.1f}%)"
        )
        running_reward = 0.0
        running_loss = 0.0
        trackio.log(
            {
                "step": step,
                "loss": avg_loss,
                "reward": avg_reward,
                "win_rate": wins / len(episode_rewards),
            }
        )


print("\nTraining complete!")
print("Remember to close the OpenSpiel environment server when done.")
