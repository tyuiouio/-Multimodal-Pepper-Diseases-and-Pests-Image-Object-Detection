# -Multimodal-Pepper-Diseases-and-Pests-Image-Object-Detection
 Multimodal Pepper Diseases and Pests Image Object Detection
 # A Multi-Modal Framework for Pepper Diseases and Pests Detection: Code and Dataset

This repository contains the dataset and code associated with the paper titled **"A Multi-Modal Framework for Pepper Diseases and Pests Detection"**. The code and dataset provided here are specifically designed for the experiments and results reported in the paper.

Due to confidentiality agreements made in collaboration with agricultural enterprises, only a subset of the dataset is publicly available. For access to the complete dataset and code, please contact the corresponding author of the paper.

## Repository Structure

The repository is organized as follows:

- **`code/`**: Contains all the code related to model training, evaluation, and data preprocessing.
- **`dataset/`**: The subset of the dataset used for training and testing the models, including images and associated metadata.
- **`figures/`**: The figures generated from experiments and used in the paper, including charts, plots, and diagrams.

## Requirements

To run the code in this repository, you will need the following dependencies:

- Python 3.6+
- PyTorch 1.8.0+
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- TensorFlow (if applicable)

You can install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
Dataset
The dataset contains images of pepper plants with various disease and pest symptoms. It is specifically tailored for training deep learning models to detect and classify different types of diseases and pests in pepper plants.

Please note that the dataset provided in this repository is a subset of the full dataset. Due to a confidentiality agreement with agricultural partners, the full dataset and code are not publicly available.
For access to the complete dataset and code, please contact the corresponding author of the paper.
Please refer to the dataset/ directory for further details on the dataset structure and contents.
Usage
1. Model Training
To train the model, use the following script:

python train.py --dataset ./dataset --epochs 50 --batch_size 32
This command will train the model using the dataset in the dataset/ directory for 50 epochs with a batch size of 32.

2. Model Evaluation
To evaluate the trained model on the test dataset:

python evaluate.py --model ./models/best_model.pth --testset ./dataset/test
3. Generating Figures
To regenerate the figures used in the paper (such as the performance comparison plots), run:

python generate_figures.py
This will generate the figures and save them in the figures/ folder.
