import torch
import torch.nn as nn
from typing import List, Callable, Optional
from functools import partial

class ExtractEmbedding:
    """
    Context manager to extract the layer input/output when forward is called.
    
    Args:
        layers (List[nn.Module]): The list of layers to extract input/output from.
        extract_input (bool): If True, extract input; otherwise, extract output.
        apply_func (Optional[Callable]): A function to apply to the extracted data for post-processing.
                                       If None, no post-processing will be applied.

    Attributes:
        extracted_data (List[torch.Tensor]): The extracted input/output data for each layer.
                                              The gradient tracking is controlled by the enable_grad argument.
    
    Example usage:
        def post_process_fn(data):
            return data.mean(dim=0)  # Example post-processing function

        with ExtractEmbedding(layers=[model.fc, model.conv], extract_input=True, enable_grad=False, apply_func=post_process_fn) as extractor:
            output = model(input)
        print(extractor.extracted_data[0].shape)  # First layer's input (after post-processing)
        print(extractor.extracted_data[1].shape)  # Second layer's input (after post-processing)

    Note:
        The gradient tracking for the extracted data is controlled by the `enable_grad` argument.
        If `enable_grad=False`, gradient tracking will be disabled for the extracted data.
    """
    def __init__(self, layers: List[nn.Module], extract_input: bool = True, 
                 apply_func: Optional[Callable] = None):
        # Ensure layers is a list of nn.Module instances
        assert all(isinstance(layer, nn.Module) for layer in layers), \
            "layers must be a list of nn.Module instances"

        self.layers = layers  # List of layers
        self.extract_input = extract_input  # Whether to extract inputs (True) or outputs (False)
        self.apply_func = apply_func  # Optional post-processing function
        self.extracted_data = [None] * len(layers)  # Placeholder for storing extracted data
        self.hooks = []  # Placeholder for storing hooks

    def __enter__(self):
        # Register a hook for each layer to capture input/output
        for idx, layer in enumerate(self.layers):
            hook_handle = layer.register_forward_hook(partial(self._save_embedding_hook, idx=idx))
            self.hooks.append(hook_handle)
        return self

    def _save_embedding_hook(self, module, input, output, idx):
        # Extract input or output depending on extract_input flag
        extracted = input if self.extract_input else output
        
        # Apply post-processing function if provided
        if self.apply_func is not None:
            extracted = self.apply_func(extracted)
        
        self.extracted_data[idx] = extracted  # Store the processed data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks when exiting the context manager
        for hook_handle in self.hooks:
            hook_handle.remove()
