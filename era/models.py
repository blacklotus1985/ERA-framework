"""
Model wrapper abstraction for ERA framework.
Supports different model architectures through a unified interface.
"""

from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    All model wrappers must implement:
    - get_token_probabilities()
    - get_full_distribution()
    - get_embedding()
    """
    
    @abstractmethod
    def get_token_probabilities(
        self,
        context: str,
        target_tokens: List[str],
    ) -> Dict[str, float]:
        """Get probability distribution over target tokens."""
        pass
    
    @abstractmethod
    def get_full_distribution(
        self,
        context: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Get full next-token probability distribution."""
        pass
    
    @abstractmethod
    def get_embedding(self, token: str) -> torch.Tensor:
        """Get embedding vector for a token."""
        pass


class HuggingFaceWrapper(ModelWrapper):
    """
    Wrapper for HuggingFace transformers models.
    
    Supports: GPT-2, GPT-Neo, Llama, Mistral, etc.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> wrapper = HuggingFaceWrapper(model, tokenizer)
        >>> probs = wrapper.get_token_probabilities("The CEO is", ["a", "the"])
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """
        Initialize wrapper.
        
        Args:
            model: HuggingFace model (e.g., GPTNeoForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> "HuggingFaceWrapper":
        """
        Load model and tokenizer from pretrained checkpoint.
        
        Args:
            model_name_or_path: Model name on HF Hub or local path
            device: Device to load model on
            **kwargs: Additional arguments for AutoModelForCausalLM
            
        Returns:
            Initialized HuggingFaceWrapper
            
        Example:
            >>> wrapper = HuggingFaceWrapper.from_pretrained("EleutherAI/gpt-neo-125M")
        """
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        return cls(model, tokenizer, device)
    
    def get_token_probabilities(
        self,
        context: str,
        target_tokens: List[str],
    ) -> Dict[str, float]:
        """
        Get probability distribution over target tokens.
        
        Args:
            context: Input context string
            target_tokens: List of tokens to get probabilities for
            
        Returns:
            Dictionary mapping token -> probability
        """
        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probabilities for target tokens
        result = {}
        for token in target_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_id) == 1:
                token_id = token_id[0]
                result[token] = float(probs[token_id].item())
            else:
                # Multi-token: use first token as proxy
                result[token] = 0.0
        
        # Normalize to sum to 1
        total = sum(result.values()) or 1.0
        result = {k: v / total for k, v in result.items()}
        
        return result
    
    def get_full_distribution(
        self,
        context: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get full next-token probability distribution.
        
        Args:
            context: Input context string
            top_k: If specified, return only top-k tokens
            
        Returns:
            Dictionary mapping token -> probability
        """
        # Tokenize context
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k if specified
        if top_k:
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        else:
            top_probs = probs
            top_indices = torch.arange(len(probs))
        
        # Convert to dictionary
        result = {}
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            token = self.tokenizer.decode([idx])
            result[token] = float(prob)
        
        return result
    
    def get_embedding(self, token: str) -> torch.Tensor:
        """
        Get embedding vector for a token.
        
        Args:
            token: Token string
            
        Returns:
            Embedding tensor (shape: [embedding_dim])
        """
        # Get token ID
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        
        if len(token_ids) == 0:
            raise ValueError(f"Token '{token}' not in vocabulary")
        
        # Use first token ID if multi-token
        token_id = token_ids[0]
        
        # Get embedding from model's embedding layer
        # Different models have different attribute names
        if hasattr(self.model, "transformer"):
            # GPT-2, GPT-Neo style
            embedding_layer = self.model.transformer.wte
        elif hasattr(self.model, "model"):
            # Llama, Mistral style
            embedding_layer = self.model.model.embed_tokens
        else:
            raise AttributeError("Cannot find embedding layer in model")
        
        with torch.no_grad():
            embedding = embedding_layer(torch.tensor([token_id]).to(self.device))
        
        return embedding[0]  # Return single embedding vector
