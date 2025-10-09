import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
from logger import logger


class AgentTransformerLens:
    """
    Agent class using transformers library for model interaction and token probabilities.

    Provides:
    - Token probabilities (logprobs)
    - Full control over generation
    - Access to model outputs
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize agent with transformers library.

        Args:
            model_name: HuggingFace model name (e.g., "google/gemma-2-2b-it")
            system_prompt: System prompt for the agent
            device: Device to run on ("cuda" or "cpu")
            dtype: Model dtype (torch.float16 recommended for GPU)
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"Loading model {model_name}...")
        logger.info(f"Device: {device}, dtype: {dtype}")

        # Load model with transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # For Gemma-3-12B, we need to ensure proper loading
        if device == "cuda":
            # Load directly to GPU with proper config
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto"  # Let it handle device placement
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
            self.model = self.model.to(device)

        logger.info(f"Model loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")

        # Detect stop tokens for Gemma chat template
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        # Add <end_of_turn> as stop token for Gemma
        end_of_turn_ids = self.tokenizer.encode('<end_of_turn>', add_special_tokens=False)
        if end_of_turn_ids:
            self.stop_token_ids.extend(end_of_turn_ids)
        logger.info(f"Stop token IDs: {self.stop_token_ids}")

    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages into a prompt string.

        Handles models that don't support system messages (like Gemma) by
        prepending the system prompt to the first user message.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        # Gemma doesn't support system role, so we need to handle it specially
        processed_messages = []
        system_content = None

        for msg in messages:
            if msg['role'] == 'system':
                # Store system content to prepend to first user message
                system_content = msg['content']
            else:
                processed_messages.append(msg)

        # Prepend system content to first user message if present
        if system_content and processed_messages:
            for i, msg in enumerate(processed_messages):
                if msg['role'] == 'user':
                    processed_messages[i] = {
                        'role': 'user',
                        'content': f"{system_content}\n\n{msg['content']}"
                    }
                    break

        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt_parts = []
            for msg in processed_messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            prompt_parts.append("Assistant:")
            return "\n".join(prompt_parts)

    def _compute_logit_lens(
        self,
        hidden_states: tuple,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute logit lens: project hidden states at each layer through unembedding.

        This reveals what the model is "thinking" at each layer by seeing what tokens
        would be predicted if we stopped at that layer.

        Args:
            hidden_states: Tuple of hidden states from model (one per layer + embedding)
            top_k: Number of top tokens to return per layer

        Returns:
            Dictionary with logit lens data for each layer
        """
        # Get the LM head (unembedding matrix)
        lm_head = self.model.lm_head

        # hidden_states is a tuple: (embedding_output, layer_1, layer_2, ..., layer_N)
        # We want to look at each actual layer (skip the embedding output at index 0)
        num_layers = len(hidden_states) - 1  # -1 because first is embedding

        layer_predictions = []

        for layer_idx in range(1, len(hidden_states)):
            # Get hidden state at this layer for the last position
            # Shape: [batch, seq_len, hidden_size] -> we want [hidden_size]
            layer_hidden = hidden_states[layer_idx][0, -1, :]  # Last position

            # Project through LM head to get logits
            # Shape: [hidden_size] -> [vocab_size]
            layer_logits = lm_head(layer_hidden)

            # Convert to probabilities
            layer_probs = torch.softmax(layer_logits, dim=-1)
            layer_log_probs = torch.log_softmax(layer_logits, dim=-1)

            # Get top-k predictions at this layer
            top_k_probs, top_k_indices = torch.topk(layer_probs, k=min(top_k, len(layer_probs)))

            # Build top predictions list
            top_predictions = []
            for i in range(len(top_k_indices)):
                token_id = top_k_indices[i].item()
                token_str = self.tokenizer.decode([token_id])
                top_predictions.append({
                    "token": token_str,
                    "token_id": token_id,
                    "logprob": layer_log_probs[token_id].item(),
                    "prob": top_k_probs[i].item()
                })

            layer_predictions.append({
                "layer": layer_idx - 1,  # 0-indexed layer number (layer 0, 1, 2, ...)
                "top_predictions": top_predictions
            })

        return {
            "num_layers": num_layers,
            "layers": layer_predictions
        }

    def call_agent_with_probs(
        self,
        user_prompt: str,
        preserve_conversation_history: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k_logprobs: int = 5,
        return_logit_lens: bool = False,
        logit_lens_top_k: int = 10,
        assistant_prefill: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response with full probability and activation data.

        Args:
            user_prompt: The user's message
            preserve_conversation_history: Whether to keep conversation history
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k_logprobs: Number of top token probabilities to return
            return_logit_lens: Whether to compute logit lens at each layer
            logit_lens_top_k: Number of top tokens to return for logit lens at each layer
            assistant_prefill: Optional text to force the assistant to start with

        Returns:
            Dictionary containing:
                - text: The response text
                - token_probabilities: List of token probability data
                - tokens: List of generated token strings
                - logit_lens: List of logit lens data per token (if return_logit_lens=True)
        """
        # Build messages
        messages = []

        # Always include system prompt at the start if present
        # _format_messages_to_prompt() will handle prepending it to first user message for Gemma
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add new user message
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        # Format to prompt
        prompt = self._format_messages_to_prompt(messages)

        logger.info("Message:")
        logger.info(messages)

        # Debug: log the actual prompt being sent
        logger.info(f"Formatted prompt:\n{prompt}\n{'='*80}")

        # Add assistant prefill if provided (forces model to start with these tokens)
        prefill_tokens = []
        if assistant_prefill:
            # Ensure prefill ends with a space if it doesn't already
            prefill_text = assistant_prefill if assistant_prefill.endswith(' ') else assistant_prefill + ' '
            # Append prefill to the prompt so model continues from there
            prompt = prompt + prefill_text
            # Tokenize the prefill separately to track which tokens were prefilled
            prefill_tokens = self.tokenizer.encode(prefill_text, add_special_tokens=False)
            logger.info(f"Assistant prefill: '{prefill_text}' ({len(prefill_tokens)} tokens)")

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        logger.info(f"Generating with {input_ids.shape[1]} prompt tokens, max {max_new_tokens} new tokens")

        # Generate with full outputs
        generated_tokens = []
        token_probabilities = []
        logit_lens_data = [] if return_logit_lens else None

        with torch.no_grad():
            current_ids = input_ids

            for step in range(max_new_tokens):
                # Run model with output_hidden_states if we need logit lens
                outputs = self.model(current_ids, output_hidden_states=return_logit_lens)
                logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]

                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

                # Compute logit lens if requested
                logit_lens_entry = None
                if return_logit_lens:
                    logit_lens_entry = self._compute_logit_lens(
                        outputs.hidden_states,
                        top_k=logit_lens_top_k
                    )

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Clamp logits to prevent numerical instability
                next_token_logits = torch.clamp(next_token_logits, min=-1e9, max=1e9)

                # Check for NaN/Inf and replace with safe values
                if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                    logger.warning("NaN or Inf detected in logits, replacing with uniform distribution")
                    next_token_logits = torch.zeros_like(next_token_logits)

                # Convert to probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

                # Get top-k alternatives
                top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k_logprobs, len(probs)))

                # Decode tokens
                next_token_str = self.tokenizer.decode([next_token_id])

                # Store token probability info
                token_entry = {
                    "token": next_token_str,
                    "token_id": next_token_id,
                    "logprob": log_probs[next_token_id].item(),
                    "prob": probs[next_token_id].item(),
                    "top_logprobs": []
                }

                for i in range(len(top_k_indices)):
                    alt_token_id = top_k_indices[i].item()
                    alt_token_str = self.tokenizer.decode([alt_token_id])
                    token_entry["top_logprobs"].append({
                        "token": alt_token_str,
                        "token_id": alt_token_id,
                        "logprob": log_probs[alt_token_id].item(),
                        "prob": top_k_probs[i].item()
                    })

                token_probabilities.append(token_entry)
                generated_tokens.append(next_token_str)

                if return_logit_lens:
                    logit_lens_data.append(logit_lens_entry)

                # Check for stop tokens (EOS or end_of_turn)
                if next_token_id in self.stop_token_ids:
                    break

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        # Decode full response
        text_response = "".join(generated_tokens)

        # Strip stop tokens from response
        text_response = text_response.replace('<end_of_turn>', '').replace('<eos>', '').strip()

        # Prepend prefill to the response (since model continued from prefill)
        if assistant_prefill:
            # Use the same prefill_text with spacing that was used in the prompt
            prefill_text = assistant_prefill if assistant_prefill.endswith(' ') else assistant_prefill + ' '
            text_response = prefill_text + text_response

        # Update conversation history if requested
        if preserve_conversation_history:
            self.conversation_history.append({
                "role": "user",
                "content": user_prompt
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": text_response
            })

        result = {
            "text": text_response,
            "token_probabilities": token_probabilities,
            "tokens": generated_tokens
        }

        if return_logit_lens:
            result["logit_lens"] = logit_lens_data

        return result

    def call_agent(self, user_prompt: str, preserve_conversation_history: bool = False) -> str:
        """
        Standard agent call without detailed probability tracking.

        Args:
            user_prompt: The user's message
            preserve_conversation_history: Whether to keep conversation history

        Returns:
            String response from the model
        """
        response_data = self.call_agent_with_probs(user_prompt, preserve_conversation_history)
        return response_data["text"]
