"""
Identity Test Results Display Script

Displays results from identity test experiments with:
1. Summary dashboard showing key metrics and outcomes
2. Top-k token probabilities from the last response (if available)
3. Full conversation transcript with visual indicators
4. Interactive browser to select and view experiments
"""

import json
import fire
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class ResultsDisplay:
    """Display formatted results from identity test experiments."""

    # Color codes for terminal output
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }

    def __init__(self, results_file: str):
        """
        Initialize display with results file.

        Args:
            results_file: Path to the JSON results file
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()

    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def _color(self, text: str, color: str, bold: bool = False) -> str:
        """Apply color formatting to text."""
        prefix = self.COLORS.get('bold', '') if bold else ''
        color_code = self.COLORS.get(color, '')
        reset = self.COLORS.get('reset', '')
        return f"{prefix}{color_code}{text}{reset}"

    def _box(self, text: str, width: int = 100, color: str = 'cyan') -> str:
        """Create a boxed header."""
        top = "╔" + "═" * (width - 2) + "╗"
        bottom = "╚" + "═" * (width - 2) + "╝"

        # Center text
        padding = (width - 4 - len(text)) // 2
        middle = "║ " + " " * padding + text + " " * (width - 4 - padding - len(text)) + " ║"

        return self._color(f"{top}\n{middle}\n{bottom}", color, bold=True)

    def _separator(self, char: str = "─", width: int = 100) -> str:
        """Create a separator line."""
        return char * width

    def _logprob_to_prob(self, logprob: float) -> float:
        """Convert log probability to probability."""
        return math.exp(logprob)

    def _format_probability(self, logprob: float) -> str:
        """Format probability as percentage with color coding."""
        prob = self._logprob_to_prob(logprob)
        percentage = prob * 100

        # Color code based on probability
        if percentage > 80:
            color = 'green'
        elif percentage > 50:
            color = 'yellow'
        else:
            color = 'red'

        return self._color(f"{percentage:6.2f}%", color)

    def display_summary_dashboard(self):
        """Display summary dashboard with key metrics."""
        print("\n")
        print(self._box("IDENTITY TEST SUMMARY DASHBOARD", color='cyan'))
        print()

        metadata = self.results.get('metadata', {})
        results = self.results.get('results', {})

        # Basic info section
        print(self._color("📊 EXPERIMENT METADATA", 'blue', bold=True))
        print(self._separator("─"))
        print(f"  Timestamp:         {metadata.get('timestamp', 'N/A')}")
        print(f"  Target Model:      {self._color(metadata.get('target_model', 'N/A'), 'magenta')}")
        print(f"  Interrogator:      {metadata.get('interrogator_model', metadata.get('interrogator', 'N/A'))}")
        print(f"  Judge:             {metadata.get('judge_model', metadata.get('judge', 'N/A'))}")
        print(f"  Interrogation:     {metadata.get('interrogation_style', 'N/A')}")
        print(f"  Rounds:            {metadata.get('actual_rounds', metadata.get('total_rounds', 'N/A'))}/{metadata.get('max_rounds', 'N/A')}")
        print()

        # Results section
        print(self._color("🎯 RESULTS", 'blue', bold=True))
        print(self._separator("─"))

        identity_revealed = results.get('identity_revealed', results.get('gemma_admitted_being_gemma', False))
        revealed_identity = results.get('revealed_identity', 'Unknown')

        if identity_revealed:
            status_icon = "✅"
            status_text = self._color("IDENTITY REVEALED", 'green', bold=True)
        else:
            status_icon = "❌"
            status_text = self._color("IDENTITY CONCEALED", 'red', bold=True)

        print(f"  Status:            {status_icon} {status_text}")
        print(f"  Revealed Identity: {self._color(revealed_identity, 'yellow', bold=True)}")

        # Show the revealing question/answer if available
        final_question = results.get('final_question_that_revealed_identity', '')
        final_answer = results.get('final_answer_that_revealed_identity', '')

        if final_question or final_answer:
            print()
            print(self._color("🔓 REVEALING EXCHANGE", 'blue', bold=True))
            print(self._separator("─"))
            if final_question:
                print(f"  Q: {final_question[:150]}...")
            if final_answer:
                print(f"  A: {final_answer[:150]}...")

        print()

        # Token probabilities section (if available)
        self._display_top_k_tokens_from_last_response()

    def _display_top_k_tokens_from_last_response(self):
        """Display top-k token probabilities from the last response."""
        results = self.results.get('results', {})
        conversation_history = results.get('full_conversation_history', [])

        if not conversation_history:
            return

        # Get last round
        last_round = conversation_history[-1]
        token_probs = last_round.get('token_probabilities', None)

        if not token_probs:
            # No token probabilities in this run
            return

        print(self._color("🎲 TOP-K TOKENS FROM LAST RESPONSE", 'blue', bold=True))
        print(self._separator("─"))

        # Display settings
        tokens_to_show = min(10, len(token_probs))  # Show first 10 tokens
        top_k = len(token_probs[0].get('top_logprobs', [])) if token_probs else 0

        print(f"  Showing first {tokens_to_show} tokens (top-{top_k} alternatives each)")
        print()

        for i, token_data in enumerate(token_probs[:tokens_to_show]):
            selected_token = token_data.get('token', '')
            selected_logprob = token_data.get('logprob', 0.0)
            top_logprobs = token_data.get('top_logprobs', [])

            # Display selected token
            prob_str = self._format_probability(selected_logprob)
            token_display = repr(selected_token) if selected_token.strip() else "'<space>'"
            print(f"  Token {i+1}: {self._color(token_display, 'white', bold=True)} {prob_str}")

            # Display alternatives
            if top_logprobs:
                print(f"    Alternatives:")
                for alt in top_logprobs[:5]:  # Show top 5 alternatives
                    alt_token = alt.get('token', '')
                    alt_logprob = alt.get('logprob', 0.0)
                    alt_prob_str = self._format_probability(alt_logprob)
                    alt_display = repr(alt_token) if alt_token.strip() else "'<space>'"
                    print(f"      • {alt_display:20s} {alt_prob_str}")
            print()

        print()

    def display_conversation_transcript(self):
        """Display full conversation transcript with visual indicators."""
        print(self._box("CONVERSATION TRANSCRIPT", color='magenta'))
        print()

        results = self.results.get('results', {})
        conversation_history = results.get('full_conversation_history', [])

        if not conversation_history:
            print(self._color("  No conversation history available.", 'yellow'))
            return

        for exchange in conversation_history:
            round_num = exchange.get('round', '?')
            question = exchange.get('question', '')
            answer = exchange.get('answer', '')

            # Determine if this was the revealing round
            final_question = results.get('final_question_that_revealed_identity', '')
            is_revealing = (question == final_question) if final_question else False

            # Round header
            if is_revealing:
                round_header = self._color(f"╔══ ROUND {round_num} (IDENTITY REVEALED) ══╗", 'green', bold=True)
                icon = "🔓"
            else:
                round_header = self._color(f"─── Round {round_num} ───", 'cyan')
                icon = "🔒"

            print(round_header)
            print()

            # Question
            print(self._color(f"  {icon} INTERROGATOR:", 'blue', bold=True))
            print(f"  {question if question else '[empty question]'}")
            print()

            # Answer
            print(self._color(f"  💬 TARGET:", 'magenta', bold=True))
            print(f"  {answer}")
            print()

            # Show token count if available
            token_probs = exchange.get('token_probabilities', None)
            if token_probs:
                token_count = len(token_probs)
                print(self._color(f"  📊 Captured {token_count} token probabilities", 'cyan'))
                print()

            print(self._separator("─", width=100))
            print()

    def display_all(self):
        """Display both summary dashboard and conversation transcript."""
        self.display_summary_dashboard()
        self.display_conversation_transcript()


class ExperimentBrowser:
    """Browse and select experiments from the experiments directory."""

    COLORS = ResultsDisplay.COLORS

    def __init__(self, experiments_dir: str = "./identity-experiments"):
        """Initialize browser with experiments directory."""
        self.experiments_dir = Path(experiments_dir)

    def _color(self, text: str, color: str, bold: bool = False) -> str:
        """Apply color formatting to text."""
        prefix = self.COLORS.get('bold', '') if bold else ''
        color_code = self.COLORS.get(color, '')
        reset = self.COLORS.get('reset', '')
        return f"{prefix}{color_code}{text}{reset}"

    def _find_all_experiments(self) -> List[Dict[str, Any]]:
        """Find all experiment result files."""
        experiments = []

        if not self.experiments_dir.exists():
            return experiments

        # Find all JSON result files
        for json_file in self.experiments_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                results = data.get('results', {})

                # Extract key info for preview
                experiments.append({
                    'path': json_file,
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'target_model': metadata.get('target_model', 'N/A'),
                    'interrogator': metadata.get('interrogator_model', metadata.get('interrogator', 'N/A')),
                    'style': metadata.get('interrogation_style', 'N/A'),
                    'rounds': metadata.get('actual_rounds', metadata.get('total_rounds', 0)),
                    'max_rounds': metadata.get('max_rounds', 'N/A'),
                    'revealed': results.get('identity_revealed', results.get('gemma_admitted_being_gemma', False)),
                    'revealed_identity': results.get('revealed_identity', 'Unknown'),
                    'has_token_probs': self._check_has_token_probs(results),
                })
            except (json.JSONDecodeError, KeyError) as e:
                # Skip invalid files
                continue

        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        return experiments

    def _check_has_token_probs(self, results: Dict[str, Any]) -> bool:
        """Check if experiment has token probability data."""
        conversation_history = results.get('full_conversation_history', [])
        if not conversation_history:
            return False
        return any('token_probabilities' in round and round['token_probabilities']
                   for round in conversation_history)

    def list_experiments(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all available experiments with summaries."""
        experiments = self._find_all_experiments()

        if not experiments:
            print(self._color("\n⚠️  No experiments found in ./identity-experiments/", 'yellow'))
            return []

        print("\n" + self._color("╔" + "═" * 98 + "╗", 'cyan', bold=True))
        print(self._color("║" + " " * 35 + "AVAILABLE EXPERIMENTS" + " " * 42 + "║", 'cyan', bold=True))
        print(self._color("╚" + "═" * 98 + "╝", 'cyan', bold=True))
        print()

        experiments_to_show = experiments[:limit] if limit else experiments

        for idx, exp in enumerate(experiments_to_show, 1):
            # Status indicator
            if exp['revealed']:
                status = self._color("✅ REVEALED", 'green', bold=True)
            else:
                status = self._color("❌ CONCEALED", 'red', bold=True)

            # Token probs indicator
            token_indicator = self._color(" 🎲", 'yellow') if exp['has_token_probs'] else ""

            print(f"{self._color(f'[{idx}]', 'cyan', bold=True)} {exp['timestamp']}{token_indicator}")
            print(f"    Model: {self._color(exp['target_model'], 'magenta')} | "
                  f"Style: {exp['style']} | "
                  f"Rounds: {exp['rounds']}/{exp['max_rounds']}")
            print(f"    Status: {status} → {self._color(exp['revealed_identity'], 'yellow')}")
            print()

        if limit and len(experiments) > limit:
            remaining = len(experiments) - limit
            print(self._color(f"... and {remaining} more experiments", 'cyan'))
            print()

        return experiments

    def interactive_select(self) -> Optional[Path]:
        """Interactive menu to select an experiment."""
        experiments = self.list_experiments()

        if not experiments:
            return None

        print(self._color("─" * 100, 'cyan'))
        print(self._color("Select an experiment to view (enter number, or 'q' to quit):", 'blue', bold=True))
        print()

        try:
            choice = input(self._color("Enter choice: ", 'green')).strip()

            if choice.lower() == 'q':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]['path']
            else:
                print(self._color(f"\n❌ Invalid choice: {choice}", 'red'))
                return None

        except ValueError:
            print(self._color(f"\n❌ Invalid input. Please enter a number.", 'red'))
            return None
        except KeyboardInterrupt:
            print(self._color("\n\n⚠️  Cancelled.", 'yellow'))
            return None


def browse_experiments(experiments_dir: str = "./identity-experiments"):
    """
    Browse available experiments interactively and display selected one.

    Args:
        experiments_dir: Directory containing experiment results (default: ./identity-experiments)
    """
    browser = ExperimentBrowser(experiments_dir)
    selected_path = browser.interactive_select()

    if selected_path:
        print("\n" + "═" * 100 + "\n")
        displayer = ResultsDisplay(str(selected_path))
        displayer.display_all()


def list_experiments(experiments_dir: str = "./identity-experiments", limit: Optional[int] = None):
    """
    List all available experiments without displaying them.

    Args:
        experiments_dir: Directory containing experiment results (default: ./identity-experiments)
        limit: Maximum number of experiments to show (default: show all)
    """
    browser = ExperimentBrowser(experiments_dir)
    browser.list_experiments(limit=limit)


def display_results(
    results_file: str,
    show_transcript: bool = True,
    show_dashboard: bool = True
):
    """
    Display results from an identity test experiment.

    Args:
        results_file: Path to the JSON results file
        show_transcript: Whether to show full conversation transcript (default: True)
        show_dashboard: Whether to show summary dashboard (default: True)
    """
    displayer = ResultsDisplay(results_file)

    if show_dashboard:
        displayer.display_summary_dashboard()

    if show_transcript:
        displayer.display_conversation_transcript()


if __name__ == "__main__":
    fire.Fire({
        'display': display_results,
        'browse': browse_experiments,
        'list': list_experiments,
    })
