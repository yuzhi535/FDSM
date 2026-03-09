"""
LLM Knowledge Distillation for Frequency Classification

This script distills LLM's knowledge about whether an action is "high-frequency" 
(fast, rapid movements) into a lightweight classifier (KinematicProjectionHead).

The distilled classifier will be used to provide semantic gates for frequency enhancement 
in the main FDSM training.

Usage:
    python distill_frequency_classifier.py --dataset ntu60 --openai_api_key YOUR_KEY
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import openai
from openai import OpenAI

# Import KinematicProjectionHead from model module
from model.dit import KinematicProjectionHead


class DistillationModel(nn.Module):
    """
    Full model for distillation training: text_encoder + KinematicProjectionHead
    
    This wraps the text encoder and classifier together for end-to-end training.
    After training, only the KinematicProjectionHead weights are saved and used in main training.
    """
    def __init__(self, pretrained_model_path, freeze_text_encoder=False):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        
        # Freeze text encoder if specified
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # KinematicProjectionHead (same as in main training)
        self.head = KinematicProjectionHead(input_dim=1024, hidden_dim=256)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: tokenized text [B, seq_len]
        Returns:
            prob: probability of high-frequency [B, 1]
        """
        # Get text embeddings
        text_outputs = self.text_encoder(input_ids)
        pooled_output = text_outputs.pooler_output  # [B, 1024]
        
        # Classify
        prob = self.head(pooled_output)
        return prob


class ActionTextDataset(Dataset):
    """Dataset for action text descriptions with LLM-generated frequency labels"""
    
    def __init__(self, text_list, labels, tokenizer, max_length=77):
        self.texts = text_list
        self.labels = labels  # LLM-generated labels: 1 for high-freq, 0 for low-freq
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': text_inputs.input_ids.squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def get_llm_frequency_labels(texts, api_key, model="gpt-4o-mini", cache_path=None):
    """
    Query LLM to classify each action text as high-frequency or low-frequency
    
    Args:
        texts: list of action descriptions
        api_key: OpenAI API key
        model: LLM model name
        cache_path: path to cache LLM responses (to avoid repeated API calls)
    
    Returns:
        labels: list of soft labels [0, 1] indicating high-frequency probability
        reasoning: list of LLM's reasoning for each classification
    """
    # Load cache if exists
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached LLM labels from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        return cache['labels'], cache['reasoning']
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    labels = []
    reasoning = []
    
    print("Querying LLM for frequency classification...")
    for text in tqdm(texts):
        prompt = f"""You are an expert in human motion analysis. Your task is to classify whether the following action involves HIGH-FREQUENCY or LOW-FREQUENCY movements.

HIGH-FREQUENCY actions involve:
- Rapid, fast movements (e.g., running, jumping, clapping, shaking, punching)
- Quick changes in velocity or direction
- High temporal dynamics

LOW-FREQUENCY actions involve:
- Slow, gradual movements (e.g., walking slowly, sitting, standing, reading)
- Minimal velocity changes
- Smooth, continuous motions

Action description: "{text}"

Please respond in the following JSON format:
{{
    "classification": "high" or "low",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Only output valid JSON, no other text."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a motion analysis expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            # Remove potential markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # Convert to soft label
            if result['classification'].lower() == 'high':
                label = result['confidence']
            else:
                label = 1.0 - result['confidence']
            
            labels.append(label)
            reasoning.append(result['reasoning'])
            
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            print(f"Response content: {content if 'content' in locals() else 'N/A'}")
            # Default to medium frequency on error
            labels.append(0.5)
            reasoning.append("Error in LLM processing")
    
    # Cache results
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'labels': labels,
                'reasoning': reasoning,
                'texts': texts
            }, f, indent=2, ensure_ascii=False)
        print(f"Cached LLM labels to {cache_path}")
    
    return labels, reasoning


def distill_train(model, dataloader, optimizer, device, epoch):
    """Train the student model to mimic LLM's soft labels"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)  # [B, 1]
        
        # Forward pass
        pred = model(input_ids)
        
        # MSE loss for soft label distillation
        loss = F.mse_loss(pred, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy (threshold at 0.5)
        pred_binary = (pred > 0.5).float()
        label_binary = (labels > 0.5).float()
        correct += (pred_binary == label_binary).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate the student model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            pred = model(input_ids)
            loss = F.mse_loss(pred, labels)
            
            total_loss += loss.item()
            
            pred_binary = (pred > 0.5).float()
            label_binary = (labels > 0.5).float()
            correct += (pred_binary == label_binary).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ntu60', choices=['ntu60', 'ntu120'])
    parser.add_argument('--pretrained_model', default='stabilityai/stable-diffusion-2-1')
    parser.add_argument('--openai_api_key', default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--llm_model', default='gpt-4o-mini', help='LLM model for teacher')
    parser.add_argument('--cache_dir', default='./llm_cache', help='Directory to cache LLM responses')
    parser.add_argument('--work_dir', default='./work_dir/distill', help='Working directory')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze text encoder, only train classifier head')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load action texts
    print(f"Loading {args.dataset} action descriptions...")
    
    # Load sparse labels (CSV)
    csv_path = f'./data/class_lists/{args.dataset}.csv'
    df = pd.read_csv(csv_path)
    sparse_texts = df['label'].values.tolist()
    
    # Load rich descriptions (TXT)
    txt_path = f'./data/class_lists/{args.dataset}_llm.txt'
    with open(txt_path, 'r', encoding='utf-8') as f:
        rich_texts = [line.strip() for line in f.readlines()]
    
    # Combine both for distillation (we want classifier to work on both)
    all_texts = rich_texts  # Use rich descriptions for better LLM understanding
    
    print(f"Loaded {len(all_texts)} action descriptions")
    
    # Get OpenAI API key
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please provide OpenAI API key via --openai_api_key or OPENAI_API_KEY env var")
    
    # Get LLM frequency labels (with caching)
    cache_path = os.path.join(args.cache_dir, f'{args.dataset}_llm_labels.json')
    llm_labels, llm_reasoning = get_llm_frequency_labels(
        all_texts, 
        api_key=api_key,
        model=args.llm_model,
        cache_path=cache_path
    )
    
    # Print some examples
    print("\n=== LLM Classification Examples ===")
    for i in range(min(5, len(all_texts))):
        print(f"Text: {all_texts[i][:80]}...")
        print(f"Label: {llm_labels[i]:.3f} (High-freq prob)")
        print(f"Reasoning: {llm_reasoning[i]}")
        print()
    
    # Statistics
    high_freq_count = sum(1 for l in llm_labels if l > 0.5)
    print(f"High-frequency actions: {high_freq_count}/{len(llm_labels)} ({high_freq_count/len(llm_labels)*100:.1f}%)")
    
    # Initialize model
    print("\nInitializing frequency classifier...")
    model = DistillationModel(
        args.pretrained_model,
        freeze_text_encoder=args.freeze_encoder
    ).to(device)
    
    # Create dataset and dataloader
    tokenizer = model.tokenizer
    dataset = ActionTextDataset(all_texts, llm_labels, tokenizer)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("\nStarting distillation training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = distill_train(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.work_dir, 'frequency_classifier_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }, save_path)
            print(f"  Saved best model to {save_path}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.work_dir, f'frequency_classifier_epoch{epoch}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_path)
    
    # Final save
    final_path = os.path.join(args.work_dir, 'frequency_classifier_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'args': vars(args)
    }, final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    # Test on all data to see predictions
    print("\n=== Final Predictions on All Data ===")
    model.eval()
    with torch.no_grad():
        all_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        predictions = []
        for batch in all_loader:
            input_ids = batch['input_ids'].to(device)
            pred = model(input_ids)
            predictions.extend(pred.cpu().numpy().flatten().tolist())
    
    # Save predictions
    pred_path = os.path.join(args.work_dir, f'{args.dataset}_predictions.json')
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump({
            'texts': all_texts,
            'llm_labels': llm_labels,
            'model_predictions': predictions,
            'llm_reasoning': llm_reasoning
        }, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {pred_path}")
    
    # Print some comparisons
    print("\nLLM vs Model predictions (first 10):")
    for i in range(min(10, len(all_texts))):
        print(f"{i+1}. {all_texts[i][:60]}...")
        print(f"   LLM: {llm_labels[i]:.3f}, Model: {predictions[i]:.3f}, Diff: {abs(llm_labels[i]-predictions[i]):.3f}")


if __name__ == '__main__':
    main()
