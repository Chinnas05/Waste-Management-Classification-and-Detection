# Waste Management Classification and Detection

This project provides tools for training and running deep learning models to classify and detect different types of waste using image data.

## Project Structure

- `train_model.py`: Script for training a Keras/TensorFlow image classification model on the waste dataset.
- `live_detection.py`: Script for running live detection (details depend on implementation).
- `waste_classifier.h5`: Saved Keras/TensorFlow classification model.
- `best.pt`: PyTorch model checkpoint (YOLO format).
- `runs/detect/`: Contains YOLO training runs, including weights, logs, and results.
- `Waste_Dataset/`: Organized dataset for training and testing, with subfolders for each waste class.

## Dataset

The dataset is organized as follows:
```
Waste_Dataset/
    train/
        cardboard/
        glass/
        metal/
        organic/
        paper/
        plastic/
        trash/
    test/
        cardboard/
        glass/
        metal/
        organic/
        paper/
        plastic/
        trash/
```

## Training a Classification Model

To train a Keras/TensorFlow classification model:
```sh
python train_model.py
```
This will use the images in `Waste_Dataset/train/` and `Waste_Dataset/test/`.

## Training a YOLO Detection Model

YOLO training runs and configurations are stored in `runs/detect/`. Each run (e.g., `train4`) contains:
- `weights/`: Model checkpoints (`best.pt`, `last.pt`)
- `args.yaml`: Training configuration
- Results and logs

## Results

- Model weights and logs are saved in the corresponding `runs/detect/train*/` folders.
- Evaluation metrics and visualizations are also stored in these folders.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- PyTorch (for YOLO)
- OpenCV (for live detection)
- Other dependencies as required

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Usage

- Train a classifier: `python train_model.py`
- Run live detection: `python live_detection.py`
- Check YOLO results in `runs/detect/`