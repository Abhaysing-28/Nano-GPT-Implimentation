
# nanoGPT

# Custom GPT Training Framework

A flexible GPT model training framework built on nanoGPT, optimized for training on domain-specific datasets with limited compute.

## Features
- 🚀 Easy-to-use dataset preparation pipeline
- 📊 Training progress visualization
- 🎯 Domain-specific fine-tuning (medical, legal, code, etc.)
- 💻 CPU and GPU support
- 📝 Comprehensive documentation

## Based On
This project extends [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy with additional features for practical deployment.

## Quick Start
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/my-gpt-project.git
cd my-gpt-project
pip install -r requirements.txt
python train.py config/my_config.py
\`\`\`

## My Contributions
- Added support for multiple dataset formats (CSV, JSON, TXT)
- Implemented real-time training metrics dashboard
- Created pre-configured setups for common domains
- Optimized for training on consumer hardware
- Added comprehensive testing suite

## Results
[Include charts, loss curves, example generations]

## License
MIT License (see LICENSE file)
Original work Copyright (c) 2022 Andrej Karpathy
Modified work Copyright (c) 2026 [Your Name]

#Windows CPU COMMAND
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0