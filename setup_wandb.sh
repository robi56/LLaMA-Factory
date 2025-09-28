#!/usr/bin/env bash

# Setup script for Weights & Biases (wandb) logging
set -euo pipefail

echo "🔧 Setting up Weights & Biases (wandb) for training logging"
echo "=========================================================="

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

echo ""
echo "You have several options to set up wandb:"
echo ""
echo "1️⃣  Export API key directly (recommended for scripts):"
echo "   export WANDB_API_KEY=your_api_key_here"
echo ""
echo "2️⃣  Login interactively (recommended for manual runs):"
echo "   wandb login"
echo ""
echo "3️⃣  Get your API key from: https://wandb.ai/settings"
echo ""

# Check if already logged in
if wandb whoami &> /dev/null; then
    echo "✅ You are already logged in to wandb!"
    echo "   User: $(wandb whoami)"
    echo ""
    echo "You can now run training scripts without additional setup."
else
    echo "❌ Not logged in to wandb yet."
    echo ""
    read -p "Do you want to login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Opening wandb login..."
        wandb login
        echo ""
        echo "✅ wandb setup complete!"
    else
        echo "You can login later with: wandb login"
        echo "Or export your API key with: export WANDB_API_KEY=your_key"
    fi
fi

echo ""
echo "📋 Next steps:"
echo "1. Run training: ./examples/train_lora/smollm2_bangla_pretrain.sh"
echo "2. Or run: ./examples/train_full/smollm2_bangla_pretrain_full.sh"
echo "3. Monitor training at: https://wandb.ai/your_username/smollm2-bangla-pretraining"
