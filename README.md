# FloraVision

An AI-powered flower identification app. Upload a photo and get the flower name, scientific name, care tips, bloom season, and fun facts in seconds.

## Live Demo

- **App:** [floravision-486x.onrender.com](https://floravision-486x.onrender.com)

## What It Does

Upload a flower photo and instantly receive a complete identification card including the common name, scientific name, care tips, bloom season, and fun facts  all powered by a custom-trained deep learning model.

## Tech Stack

**Frontend**
- HTML, CSS, JavaScript (vanilla, no frameworks)
- Responsive design, animated result cards, interactive bloom calendar

**Backend**
- Python, Flask, REST API
- Flask-CORS, Pillow, NumPy

**Machine Learning**
- PyTorch, ResNet50, fine-tuned on Oxford 102 Flower Dataset
- Transfer learning from ImageNet weights
- Custom classifier head: Linear 2048 → 512, ReLU, Dropout 0.4, Linear 512 → 102
- Cosine annealing scheduler, layerwise learning rates
- Test accuracy: 91.9% on Oxford 102 test set
- Smart preprocessing: saturation-based region detection, image enhancement, three-crop ensemble

**Training Environment**
- Google Colab, Tesla T4 GPU
- 25 epochs Phase 1, 15 epochs Phase 2
