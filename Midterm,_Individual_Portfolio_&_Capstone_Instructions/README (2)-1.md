
---
# CIFAR-10 Image Classification with MobileNetV2

>This project demonstrates the use of MobileNetV2 for image classification on the CIFAR-10 dataset. The code includes preprocessing steps, model training, and fine-tuning.

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cifar10-mobilenetv2.git
   cd cifar10-mobilenetv2
   
2. Create a virtual environment and activate it:

   
  ```python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ````

4. Install the dependencies:
   
   
   pip install -r requirements.txt
   

## Running the Code

1. Preprocess the data and train the model:
   
  ```python train.py
  
3. Fine-tune the model:
   
  ````python fine_tune.py

## Results
The model achieved an accuracy of ~82% after fine-tuning.

## Project Structure

  train.py: Script to preprocess data and train the initial model.
  fine_tune.py: Script to fine-tune the pretrained model.
  README.md: Project documentation.
  requirements.txt: List of dependencies.


## Project Report

### Project Report

#### 1. Dataset Selection and Justification
**Dataset**: CIFAR-10 dataset, consisting of 60,000 32x32 color images in 10 different classes.

**Justification**:
- **Diversity**: Contains 10 different classes, suitable for testing model robustness.
- **Standard Benchmark**: Widely used, allowing for easy comparison with other models.
- **Size**: Sufficient number of images for training/testing while manageable computationally.

#### 2. Methodologies Used
**Initial Preprocessing**:
- Resized CIFAR-10 images to 96x96 pixels.
- Normalized pixel values to [-1, 1] using `preprocess_input` from TensorFlow.

**Model Training**:
- Used pre-trained MobileNetV2 (`include_top=False`).
- Added a `GlobalAveragePooling2D` layer and a `Dense` layer with 10 neurons (softmax activation).
- Optimizer: Adam with a 1e-4 learning rate.
- Initial Training: Trained for 5 epochs, then extended to 10 epochs.

**Model Enhancement**:
- Added dense layers with ReLU activation and dropout for regularization.
- Fine-Tuning: Unfroze MobileNetV2 layers and fine-tuned with a 1e-5 learning rate for 5 epochs.

**Training Strategy**:
- Initial Training: Established a baseline with 5 epochs.
- Extended Training: Increased to 10 epochs.
- Architectural Enhancements: Added additional dense and dropout layers.
- Fine-Tuning: Fine-tuned entire model with a low learning rate.

#### 3. Performance Metrics and Baseline Comparison
- **Baseline Accuracy**: ~70%
- **Extended Training Accuracy**: ~75%
- **Enhanced Architecture Accuracy**: ~78%
- **Fine-Tuned Model Accuracy**: ~82%

**Comparison to Baseline**:
- **Initial Improvement**: Extended training improved accuracy but led to overfitting.
- **Architectural Enhancements**: Reduced overfitting and improved accuracy.
- **Fine-Tuning**: Provided the best accuracy and overall performance improvement.

#### 4. Challenges Faced and Solutions
**Challenge**: Overfitting during initial extended training.
**Solution**: Introduced dropout layers and additional dense layers for regularization.

**Challenge**: Fine-tuning a pre-trained model without degrading performance.
**Solution**: Utilized a very low learning rate during fine-tuning to ensure minimal, beneficial weight adjustments.

**Challenge**: Efficiently managing computational resources.
**Solution**: Leveraged pre-trained MobileNetV2 to reduce training time and improve performance.

### Conclusion
>This project successfully demonstrates the use of a pre-trained MobileNetV2 model for image classification on the CIFAR-10 dataset. Through careful preprocessing, architectural enhancements, and fine-tuning, significant improvements in model accuracy were achieved. The approach balances computational efficiency with model performance, making it suitable for practical image classification tasks.

---

### Evaluation Criteria

**Emphasis on Dataset Selection and Adaptation**:
- **Dataset Selection**: CIFAR-10 chosen for relevance and challenge.
- **Adaptation**: Images resized and normalized for MobileNetV2 compatibility.

**Improvement Over Baseline Model**:
- **Initial Baseline**: Accuracy ~70%.
- **Enhancements**: Iterative improvements led to final accuracy ~82%.

**Efficient Use of Computational Resources**:
- **Pre-trained Model**: Used MobileNetV2 to reduce training time and resource usage.
- **Regularization Techniques**: Employed dropout and additional dense layers to prevent overfitting while maintaining efficiency.
