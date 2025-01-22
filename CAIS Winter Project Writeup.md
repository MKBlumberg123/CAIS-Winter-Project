## **Project Overview**

This project aims to build a Convolutional Neural Network (CNN) model to recognize emotions from facial expressions using the FER-2013 dataset. The model is designed to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## **Dataset**

The dataset used is **FER-2013**, a labeled dataset of grayscale images representing various facial expressions. Each image is 48x48 pixels and categorized into one of seven predefined emotion classes. Preprocessing steps included:

1) **Resizing**: Images were normalized to ensure consistency in size and input requirements for the CNN.  
2) **Normalization**: Pixel values were scaled to a range of \[0, 1\] for stable gradient updates during training.

These choices ensured the model received clean, uniformly processed data while minimizing the risk of overfitting.

## **Model Development and Training**

### **Implementation Choices**

The CNN consists of:

* **Four convolutional layers** with increasing filter counts (64, 128, 256, 512\) to progressively capture larger and more complex features.  
* **Batch normalization** added after each convolution to stabilize training and improve generalization.  
* **Max-pooling layers** to reduce dimensions while retaining key features.  
* **Three fully connected layers** for mapping the extracted features to the final emotion classes.  
* **Dropout** applied to the dense layers to prevent overfitting.

### **Training Procedure**

* **Optimizer**: Adam optimizer was chosen for its ability to adapt learning rates and converge efficiently.  
* **Loss Function**: Cross-entropy loss was used because it is well-suited for multi-class classification.  
* **Hyperparameters**:  
  * Learning rate: 0.001  
  * Batch size: 64  
  * Epochs: 15

**Reasoning for Choices:** These design choices ensure the model captures the hierarchical structure of facial features while minimizing overfitting. The Adam optimizer and cross-entropy loss are well-suited for multi-class tasks like FER-2013, while the selected hyperparameters balanceefficient training and model convergence.

## **Model Evaluation/Results**

**Evaluation Metrics:**

* **Overall Accuracy:** The percentage of correctly classified images.  
* **Per-Class Accuracy:** Accuracy for each emotion category to understand class-wise performance.

**Results:**

* Overall Test Accuracy: **51.80%**  
* Per-Class Accuracy:  
  * Accuracy for class: angry is **46.7 %**  
  * Accuracy for class: disgust is **14.4 %**  
  * Accuracy for class: fear is **34.8 %**  
  * Accuracy for class: happy is **71.0 %**  
  * Accuracy for class: neutral is **47.6 %**  
  * Accuracy for class: sad is **47.7 %**  
  * Accuracy for class: surprise is **57.4 %**

**Observations:** The model performs best on "Happy" and "Surprise" categories, likely due to distinct features and large datasets. Performance on "Angry" and "Disgust" is relatively low, potentially due to similarities between these emotions in facial expressions. There was also a significantly larger amount of “Happy” and “Neutral” images in the dataset compared to emotions like “Disgust” which led to the relatively good/poor performance of the model for those respective classes. If I continued to work on this project, I would likely trim the size of the data subsets for each respective class (to around 200-300) or would throw out the disgust emotion entirely.

## **Discussion**

**Dataset and Model Fit:** The FER-2013 dataset is suitable for this task, but the class imbalance and subtle differences between certain emotions (e.g. Angry and Disgust) make emotion classification very difficult. The model architecture is alright but could benefit from more advanced techniques such as residual connections or attention mechanisms. If I continued working on this project, I would also augment the images by rotation or using Affine transformations to prevent bias in unnecessary features. 

**Wider Implications and Social Good:** This project demonstrates the potential of AI in enhancing human well-being by enabling applications in fields such as mental health, education, and human-computer interaction. For example, emotion recognition systems could be used in mental health monitoring to detect signs of emotional distress or for inpatient hospital care to detect when patients are in pain. Additionally, emotion-aware human-computer interfaces can improve accessibility for individuals with speech or hearing impairments by enabling more natural interactions.

The FER-2013 dataset, while widely used, does not represent the full spectrum of human diversity in terms of ethnicity, age, and expression of emotions, which leads to incredibly biased model outputs. This exemplifies the need for more inclusive datasets to ensure fair and reliable performance across demographics. Furthermore, ethical concerns arise regarding privacy and consent, as deploying emotion recognition systems often involves analyzing personal and sensitive data.

**Limitations of this Project:**

* **Dataset Limitations:** FER-2013 lacks diversity in demographics and settings, limiting generalizability.  
* **Model Limitations:** The relatively low accuracy indicates the need for a more robust architecture.  
* **Ethical Concerns:** Misclassification in sensitive applications could lead to incorrect assessments. Privacy concerns also arise from emotion recognition systems. 

**Future Steps:**

1. **Architecture Improvements:** Incorporate deeper networks, residual blocks, and attention mechanisms.  
2. **Class Balancing:** Address class imbalance through techniques like oversampling or weighted loss functions.  
3. **Transfer Learning:** Use pre-trained models like ResNet.  
4. **Dataset Expansion:** Include more diverse datasets to improve generalizability.  
5. **Real-World Testing:** Evaluate the model in real-time scenarios or videos to assess practical feasibility.