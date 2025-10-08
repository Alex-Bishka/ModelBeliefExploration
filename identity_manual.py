"""
Identity Test Experiment Script - Fully Manual Version

This is a fully manual version where you control everything: questions and when to stop.

Setup:
1. Target agent: Gemma (google/gemma-3-12b-it) with system prompt claiming it's Qwen
2. Interrogator: YOU (manual human input)
3. Judge: YOU (manual y/N to continue)

Process:
- You manually type each question to ask the target AI
- Target AI responds to your question
- You decide whether to continue (y/N)
- Continues until you decide to stop
- Documents full conversation history and results
"""

import json
import fire
from pathlib import Path
from datetime import datetime

from logger import logger
from agent import Agent


def run_identity_test_manual(
    target_model_name: str = "google/gemma-3-12b-it",
    output_dir: str = "./identity-experiments"
) -> dict:
    """
    Run fully manual identity test where you control questions and when to stop.

    Args:
        target_model_name: Model being tested (should be Gemma)
        output_dir: Directory to save results

    Returns:
        Dictionary containing experiment results
    """
    # Welcome message
    print("\n" + "=" * 100)
    print("IDENTITY TEST - FULLY MANUAL MODE")
    print("=" * 100)
    print("\nYou will manually type each question to ask the target AI.")
    print("The target AI thinks it's Qwen (Alibaba), but it's actually Gemma (Google).")
    print("You decide when to stop by answering y/N after each response.")
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nStarting identity test experiment at {timestamp}")
    logger.info(f"Target model: {target_model_name}")
    logger.info(f"Interrogator: Manual human input")
    logger.info(f"Judge: Manual human decision (y/N)")
    logger.info(f"Output directory: {output_path}")

    # Initialize target agent
    target_agent = Agent(
        model_name=target_model_name,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

    # Track conversation and results
    conversation_history = []
    round_num = 0

    logger.info("=" * 100)
    logger.info("Starting interrogation...")
    logger.info("=" * 100)

    # Run interrogation rounds
    while True:
        round_num += 1
        logger.info(f"\n--- Round {round_num} ---")

        # Get question manually from user
        print(f"\n[Round {round_num}] Enter your question for the target AI (empty is allowed): ")
        question = input().strip()

        if question:
            logger.info(f"Your question: {question}")
        else:
            logger.info(f"Your question: [empty]")

        # Target agent responds
        answer = target_agent.call_agent(
            user_prompt=question,
            preserve_conversation_history=True
        )
        logger.info(f"Target: {answer}")

        # Record this exchange
        conversation_history.append({
            "round": round_num,
            "question": question,
            "answer": answer
        })

        # Ask user if they want to continue
        print("\nContinue? [y/N]: ", end="")
        continue_response = input().strip().lower()

        if continue_response != 'y':
            logger.info("User chose to stop.")
            break

    # Prepare results
    result = {
        "metadata": {
            "timestamp": timestamp,
            "target_model": target_model_name,
            "interrogator": "manual_human_input",
            "judge": "manual_human_decision",
            "total_rounds": round_num,
            "interrogation_style": "fully_manual",
            "target_system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        "results": {
            "full_conversation_history": conversation_history,
            "target_agent_conversation": target_agent.conversation_history,
        }
    }

    # Save results
    logger.info("=" * 100)
    logger.info("Saving results...")

    json_path = output_path / "identity_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    # Print summary
    logger.info("=" * 100)
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"Total rounds: {round_num}")
    logger.info(f"Conversation history saved to: {json_path}")
    logger.info("=" * 100)

    return result


if __name__ == "__main__":
    fire.Fire(run_identity_test_manual)
