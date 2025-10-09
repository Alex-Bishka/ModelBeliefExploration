"""
Web-based Identity Test Results Viewer

A Flask application for browsing and viewing identity test experiment results.
"""

import json
import math
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, jsonify, request, session
from datetime import datetime
import secrets

from agent_transformerlens import AgentTransformerLens
from logger import logger
import threading

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management

# Configuration
EXPERIMENTS_DIR = Path("./identity-experiments")
DEFAULT_MODEL = "google/gemma-3-12b-it"

# Global model pool - reuse loaded models across sessions
model_pool = {}  # {model_name: agent_instance}
model_pool_lock = threading.Lock()

# Active experiment sessions
active_sessions = {}


def get_or_load_model(model_name: str, system_prompt: str = "", device: str = "cuda") -> AgentTransformerLens:
    """
    Get model from pool or load it if not present.
    Clears other models from pool if switching to different model.
    """
    with model_pool_lock:
        # If requesting different model, clear the pool
        if model_pool and model_name not in model_pool:
            logger.info(f"Switching to new model {model_name}, clearing pool...")
            for old_model_name, old_agent in model_pool.items():
                logger.info(f"Clearing model {old_model_name} from pool")
                if hasattr(old_agent, 'model'):
                    del old_agent.model
                if hasattr(old_agent, 'tokenizer'):
                    del old_agent.tokenizer
            model_pool.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Return existing model if available
        if model_name in model_pool:
            logger.info(f"Reusing model {model_name} from pool")
            agent = model_pool[model_name]
            # Reset conversation history and system prompt for new session
            agent.conversation_history = []
            agent.system_prompt = system_prompt
            return agent

        # Load new model
        logger.info(f"Loading new model {model_name} into pool...")
        agent = AgentTransformerLens(
            model_name=model_name,
            system_prompt=system_prompt,
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model_pool[model_name] = agent
        logger.info(f"Model {model_name} loaded and cached in pool")
        return agent


def prewarm_default_model():
    """Pre-load the default model on server startup."""
    logger.info(f"Pre-warming default model: {DEFAULT_MODEL}")
    try:
        get_or_load_model(DEFAULT_MODEL, system_prompt="", device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Default model {DEFAULT_MODEL} pre-warmed successfully")
    except Exception as e:
        logger.error(f"Failed to pre-warm default model: {e}")


def cleanup_session(session_id: str):
    """Clean up a session (but keep model in pool for reuse)."""
    if session_id not in active_sessions:
        return

    session_data = active_sessions[session_id]

    # Don't delete the model - it stays in the pool for reuse
    # Just clear the session reference
    if session_data.get('agent') is not None:
        session_data['agent'] = None

    logger.info(f"Session {session_id} cleaned up (model kept in pool for reuse)")


# Custom Jinja filters
@app.template_filter('exp')
def exp_filter(value):
    """Convert log probability to probability."""
    try:
        return math.exp(float(value))
    except (ValueError, TypeError):
        return 0.0


class ExperimentLoader:
    """Load and parse experiment data."""

    @staticmethod
    def find_all_experiments(experiments_dir: Path = EXPERIMENTS_DIR) -> List[Dict[str, Any]]:
        """Find all experiment result files."""
        experiments = []

        if not experiments_dir.exists():
            return experiments

        # Find all JSON result files
        for json_file in experiments_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                results = data.get('results', {})

                # Extract key info for preview
                experiments.append({
                    'path': str(json_file),
                    'filename': json_file.name,
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'target_model': metadata.get('target_model', 'N/A'),
                    'interrogator': metadata.get('interrogator_model', metadata.get('interrogator', 'N/A')),
                    'judge': metadata.get('judge_model', metadata.get('judge', 'N/A')),
                    'style': metadata.get('interrogation_style', 'N/A'),
                    'rounds': metadata.get('actual_rounds', metadata.get('total_rounds', 0)),
                    'max_rounds': metadata.get('max_rounds', 'N/A'),
                    'revealed': results.get('identity_revealed', results.get('gemma_admitted_being_gemma', False)),
                    'revealed_identity': results.get('revealed_identity', 'Unknown'),
                    'has_token_probs': ExperimentLoader._check_has_token_probs(results),
                    'label': metadata.get('label', ''),
                })
            except (json.JSONDecodeError, KeyError) as e:
                # Skip invalid files
                continue

        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        return experiments

    @staticmethod
    def _check_has_token_probs(results: Dict[str, Any]) -> bool:
        """Check if experiment has token probability data."""
        conversation_history = results.get('full_conversation_history', [])
        if not conversation_history:
            return False
        return any('token_probabilities' in round and round['token_probabilities']
                   for round in conversation_history)

    @staticmethod
    def load_experiment(path: str) -> Dict[str, Any]:
        """Load a specific experiment file."""
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def logprob_to_prob(logprob: float) -> float:
        """Convert log probability to probability."""
        return math.exp(logprob)


# Routes
@app.route('/')
def index():
    """Main page showing list of experiments."""
    experiments = ExperimentLoader.find_all_experiments()
    return render_template('index.html', experiments=experiments)


@app.route('/api/experiments')
def api_experiments():
    """API endpoint to get all experiments."""
    experiments = ExperimentLoader.find_all_experiments()
    return jsonify(experiments)


@app.route('/api/experiment/<path:experiment_path>')
def api_experiment(experiment_path):
    """API endpoint to get a specific experiment."""
    try:
        # Security: ensure path is within experiments directory
        full_path = Path(experiment_path)
        if not full_path.exists():
            return jsonify({'error': 'Experiment not found'}), 404

        data = ExperimentLoader.load_experiment(str(full_path))
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/experiment/<path:experiment_path>')
def view_experiment(experiment_path):
    """View a specific experiment."""
    try:
        full_path = Path(experiment_path)
        if not full_path.exists():
            return "Experiment not found", 404

        data = ExperimentLoader.load_experiment(str(full_path))
        return render_template('experiment.html', experiment=data, experiment_path=experiment_path)
    except Exception as e:
        return f"Error loading experiment: {str(e)}", 500


# Interactive Experiment Routes
@app.route('/run')
def run_experiment():
    """Page to start a new interactive experiment."""
    return render_template('run_experiment.html')


@app.route('/api/start-experiment', methods=['POST'])
def start_experiment():
    """Start a new interactive experiment session."""
    data = request.json
    target_model = data.get('target_model', DEFAULT_MODEL)
    top_k_logprobs = data.get('top_k_logprobs', 5)
    system_prompt = data.get('system_prompt', "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    assistant_prefill = data.get('assistant_prefill', None)
    return_logit_lens = data.get('return_logit_lens', False)
    logit_lens_top_k = data.get('logit_lens_top_k', 10)

    # Clean up all existing sessions (but keep model in pool)
    session_ids_to_clean = list(active_sessions.keys())
    for sid in session_ids_to_clean:
        cleanup_session(sid)
        del active_sessions[sid]

    if session_ids_to_clean:
        logger.info(f"Cleaned up {len(session_ids_to_clean)} old session(s) before starting new experiment")

    # Create session ID
    session_id = secrets.token_hex(8)

    # Store session WITHOUT loading model yet (lazy loading on first question)
    active_sessions[session_id] = {
        'agent': None,  # Will be loaded on first question using model pool
        'model_name': target_model,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'conversation_history': [],
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'target_model': target_model,
            'interrogator': 'web_interface_human',
            'judge': 'web_interface_human',
            'interrogation_style': 'web_interactive',
            'top_k_logprobs': top_k_logprobs,
            'target_system_prompt': system_prompt,
            'assistant_prefill': assistant_prefill,
            'logit_lens_enabled': return_logit_lens,
            'logit_lens_top_k': logit_lens_top_k if return_logit_lens else None
        },
        'round': 0,
        'assistant_prefill': assistant_prefill,  # Store for use during first question
        'return_logit_lens': return_logit_lens,
        'logit_lens_top_k': logit_lens_top_k
    }

    return jsonify({'session_id': session_id, 'status': 'started'})


@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Ask a question in an interactive experiment."""
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question', '')

    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_data = active_sessions[session_id]

    # Lazy load model on first question (reuse from pool if available)
    if session_data['agent'] is None:
        logger.info(f"Getting model {session_data['model_name']} for session {session_id}...")
        try:
            agent = get_or_load_model(
                model_name=session_data['model_name'],
                system_prompt=session_data['metadata']['target_system_prompt'],
                device=session_data['device']
            )

            # Apply assistant prefill if provided (with dummy user message to maintain alternation)
            if session_data.get('assistant_prefill'):
                logger.info(f"Applying assistant prefill for session {session_id}")
                # Add a minimal user greeting first to maintain user/assistant alternation
                agent.conversation_history.append({
                    'role': 'user',
                    'content': '[Session started]'
                })
                agent.conversation_history.append({
                    'role': 'assistant',
                    'content': session_data['assistant_prefill']
                })

            session_data['agent'] = agent
            logger.info(f"Model ready for session {session_id}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

    agent = session_data['agent']
    top_k_logprobs = session_data['metadata']['top_k_logprobs']
    return_logit_lens = session_data.get('return_logit_lens', False)
    logit_lens_top_k = session_data.get('logit_lens_top_k', 10)

    # Increment round
    session_data['round'] += 1
    round_num = session_data['round']

    # Get response with probabilities
    try:
        response_data = agent.call_agent_with_probs(
            user_prompt=question,
            preserve_conversation_history=True,
            top_k_logprobs=top_k_logprobs,
            return_logit_lens=return_logit_lens,
            logit_lens_top_k=logit_lens_top_k
        )
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

    # Record exchange
    exchange = {
        'round': round_num,
        'question': question,
        'answer': response_data['text'],
        'token_probabilities': response_data['token_probabilities']
    }

    # Add logit lens if available
    if return_logit_lens and 'logit_lens' in response_data:
        exchange['logit_lens'] = response_data['logit_lens']

    session_data['conversation_history'].append(exchange)

    # Prepare response
    response_json = {
        'round': round_num,
        'answer': response_data['text'],
        'token_probabilities': response_data['token_probabilities']
    }

    if return_logit_lens and 'logit_lens' in response_data:
        response_json['logit_lens'] = response_data['logit_lens']

    return jsonify(response_json)


@app.route('/api/finish-experiment', methods=['POST'])
def finish_experiment():
    """Finish experiment and save results."""
    data = request.json
    session_id = data.get('session_id')
    identity_revealed = data.get('identity_revealed', False)
    revealed_identity = data.get('revealed_identity', 'Unknown')

    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_data = active_sessions[session_id]

    # Prepare results
    result = {
        'metadata': {
            **session_data['metadata'],
            'total_rounds': session_data['round']
        },
        'results': {
            'identity_revealed': identity_revealed,
            'revealed_identity': revealed_identity,
            'full_conversation_history': session_data['conversation_history'],
            'target_agent_conversation': session_data['agent'].conversation_history
        }
    }

    # Save to file
    timestamp = session_data['metadata']['timestamp']
    output_path = EXPERIMENTS_DIR / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "identity_test_results_web.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Clean up session and free GPU memory
    cleanup_session(session_id)
    del active_sessions[session_id]

    return jsonify({
        'status': 'saved',
        'path': str(json_path)
    })


@app.route('/chat/<session_id>')
def chat_interface(session_id):
    """Chat interface for interactive experiment."""
    if session_id not in active_sessions:
        return "Session not found or expired", 404

    session_data = active_sessions[session_id]
    return render_template('chat.html', session_id=session_id, metadata=session_data['metadata'])


@app.route('/api/delete-experiment', methods=['POST'])
def delete_experiment():
    """Delete a specific experiment."""
    data = request.json
    experiment_path = data.get('path')

    if not experiment_path:
        return jsonify({'error': 'No path provided'}), 400

    try:
        full_path = Path(experiment_path)
        if not full_path.exists():
            return jsonify({'error': 'Experiment not found'}), 404

        # Delete the file
        full_path.unlink()

        # Try to delete parent directory if empty
        parent_dir = full_path.parent
        try:
            if parent_dir != EXPERIMENTS_DIR and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
        except:
            pass

        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-all-experiments', methods=['POST'])
def clear_all_experiments():
    """Delete all experiments."""
    try:
        deleted_count = 0
        for json_file in EXPERIMENTS_DIR.rglob("*.json"):
            try:
                json_file.unlink()
                deleted_count += 1
            except:
                pass

        # Clean up empty directories
        for dir_path in EXPERIMENTS_DIR.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except:
                    pass

        return jsonify({'status': 'cleared', 'count': deleted_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-label', methods=['POST'])
def update_label():
    """Update the label for an experiment."""
    data = request.json
    experiment_path = data.get('path')
    label = data.get('label', '')

    if not experiment_path:
        return jsonify({'error': 'No path provided'}), 400

    try:
        full_path = Path(experiment_path)
        if not full_path.exists():
            return jsonify({'error': 'Experiment not found'}), 404

        # Load experiment data
        with open(full_path, 'r') as f:
            experiment_data = json.load(f)

        # Update label
        if 'metadata' not in experiment_data:
            experiment_data['metadata'] = {}
        experiment_data['metadata']['label'] = label

        # Save back
        with open(full_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        return jsonify({'status': 'updated', 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 Identity Test Results Viewer")
    print("=" * 80)
    print(f"Experiments directory: {EXPERIMENTS_DIR.absolute()}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Starting server at: http://localhost:5000")
    print(f"Note: Auto-reload disabled to prevent crashes during model loading")
    print("=" * 80 + "\n")

    # Pre-warm default model for instant first chat
    print("⏳ Pre-warming default model (this may take 1-2 minutes)...")
    prewarm_default_model()
    print("✅ Model pre-warmed and ready!\n")

    # Debug mode but no auto-reload (prevents crashes during model loading)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
