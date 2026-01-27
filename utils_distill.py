"""
Integration utilities for using distilled frequency classifier in main training.

This module provides helper functions to load the distilled frequency classifier
and integrate it into the FDSM training pipeline.
"""

import os
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


def load_distilled_classifier(checkpoint_path, device='cuda'):
    """
    Load a distilled frequency classifier from checkpoint.
    
    Args:
        checkpoint_path: path to the saved classifier checkpoint
        device: device to load the model on
    
    Returns:
        classifier: FrequencyClassifier module (classifier head only)
        checkpoint_info: dict with training info
    """
    from model.dit import FrequencyClassifier
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract classifier state dict (only the classifier head, not text encoder)
    # The checkpoint from distill_frequency_classifier.py contains full model
    # We need to extract only the classifier part
    full_state_dict = checkpoint['model_state_dict']
    
    # Filter classifier weights
    classifier_state_dict = {
        k.replace('classifier.', ''): v 
        for k, v in full_state_dict.items() 
        if k.startswith('classifier.')
    }
    
    # Initialize classifier
    input_dim = checkpoint['args'].get('input_dim', 1024)  # Default SD 2.1 dimension
    classifier = FrequencyClassifier(input_dim=input_dim).to(device)
    classifier.load_state_dict(classifier_state_dict)
    
    # Freeze classifier
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    
    info = {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss'],
        'val_acc': checkpoint['val_acc'],
        'args': checkpoint['args']
    }
    
    return classifier, info


def get_frequency_gates(text_embeddings, classifier, device='cuda'):
    """
    Get frequency gates from text embeddings using distilled classifier.
    
    Args:
        text_embeddings: [B, seq_len, dim] or [B, dim] - text embeddings
        classifier: FrequencyClassifier module
        device: device to run on
    
    Returns:
        gates: [B, 1] - frequency gate values (probability of high-frequency)
    """
    # If text_embeddings has sequence dimension, take pooled output (last token)
    if text_embeddings.dim() == 3:
        # Assume pooled embedding is at position -1 (CLIP convention)
        pooled = text_embeddings[:, -1, :]  # [B, dim]
    else:
        pooled = text_embeddings
    
    with torch.no_grad():
        gates = classifier(pooled)
    
    return gates


class FrequencyGateProvider:
    """
    Wrapper class to provide frequency gates in training loop.
    
    This class can work in two modes:
    1. Distilled mode: use frozen distilled classifier
    2. Learnable mode: use learnable projection head (original approach)
    """
    def __init__(self, mode='distilled', classifier_path=None, device='cuda'):
        """
        Args:
            mode: 'distilled' or 'learnable'
            classifier_path: path to distilled classifier checkpoint (required if mode='distilled')
            device: device to run on
        """
        self.mode = mode
        self.device = device
        
        if mode == 'distilled':
            if classifier_path is None:
                raise ValueError("classifier_path must be provided in distilled mode")
            
            self.classifier, self.info = load_distilled_classifier(classifier_path, device)
            print(f"Loaded distilled classifier from {classifier_path}")
            print(f"  Epoch: {self.info['epoch']}, Val Loss: {self.info['val_loss']:.4f}, Val Acc: {self.info['val_acc']:.4f}")
        
        elif mode == 'learnable':
            # Will use external learnable head (KinematicProjectionHead)
            pass
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'distilled' or 'learnable'")
    
    def get_gates(self, text_embeddings):
        """
        Get frequency gates from text embeddings.
        
        Args:
            text_embeddings: [B, seq_len, dim] or [B, dim] - text embeddings
        
        Returns:
            gates: [B, 1] - frequency gate values
        """
        if self.mode == 'distilled':
            return get_frequency_gates(text_embeddings, self.classifier, self.device)
        else:
            # In learnable mode, gates are computed by external head
            # This is just a placeholder - actual computation happens in training loop
            raise NotImplementedError("In learnable mode, use KinematicProjectionHead directly")
    
    def is_frozen(self):
        """Check if gates are frozen (distilled mode) or learnable"""
        return self.mode == 'distilled'
