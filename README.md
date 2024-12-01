<<<<<<< HEAD
# orthoai_poc
OrthoAI Project  - POC/POV System - AI for Dental Treatments
=======
# OrthoAI POC
OrthoAI Project  - POC/POV System - AI for Dental Treatments

## Project Description
OrthoAI is a POC/POV system that uses AI to assist in dental treatments. The system is designed to help dentists in the diagnosis and treatment of dental problems, as well as to provide insurance companies with a tool to assess the risk of dental claims. The system uses machine learning algorithms to analyze dental images and patient data, and provides an assessment of the patient's dental health.

## Data
The input data for the system consists of dental images and patient data.

* Dental images: The system uses dental images to analyze the patient's teeth and gums. The images has two types: intraoral images and x-ray radiographs. The intraoral images are taken from inside the mouth and show the teeth and gums in detail. The x-ray radiographs are taken from outside the mouth and show the teeth and jaw bones. The images are used to detect dental conditions such as cavities, gum disease, and malocclusion.
* Patient data: The system parses clinical notes and patient data to extract ground truth labels for the dental images. Examples of the labels include IOTN (Index of Orthodontic Treatment Need), Malocclusion, and other dental conditions.

Our data comprises 99 patients, each with five intraoral images and an x-ray radiograph. The data is split into training, validation, and test sets with a ratio of 70:15:15. The training set is used to train the model, the validation set is used to tune the hyperparameters, and the test set is used to evaluate the model.

## Model
We use a deep learning model to analyze the dental images and intraoral x-ray images. The model architecture has the following components:

* ResNet-50: A pre-trained ResNet-50 model is used to extract features from the radiograph images.
* EfficientNetV2B3: A pre-trained EfficientNetV2B3 model is used to extract features from the intraoral images.
* Multi-Modal Fusion: The features extracted from the ResNet-50 and EfficientNetV2B3 models are fused together using a multi-modal fusion layer.
* Attention Mechanism: An attention mechanism is used to focus on the relevant parts of the image and patient data.
* Classification Heads: The fused features are passed through the following classification heads to predict the dental conditions
    * IOTN Classification Head: Predicts whether or not the patient needs braces.
    * Malocclusion Classification Head: Predicts the severity of malocclusion (Class I, Class II, Class III).
    * Other Dental Conditions Classification Head: Predicts other dental conditions such as spacing, crowding, and missing teeth.

The model is trained using a multi-task learning approach, where the model is trained to predict multiple dental conditions simultaneously. The loss function is a summation of the losses from the individual classification heads. We used cross-entropy loss for the classification heads. We used the Adam optimizer with a learning rate of 0.0001 for training the model for 50 epochs with a batch size of 32. We saved the model weights with the best validation loss.

## Evaluation
The system is evaluated on the following metrics:

* Accuracy: The accuracy of the system in predicting the dental conditions.
* Precision: The precision of the system in predicting the positive cases (e.g., patients who need braces).
* Recall: The recall of the system in predicting the positive cases.
* F1 Score: The F1 score of the system, which is the harmonic mean of precision and recall.
* AUC-ROC: The Area Under the Receiver Operating Characteristic curve, which measures the trade-off between true positive rate and false positive rate.
* Confusion Matrix: A confusion matrix is used to visualize the performance of the system in predicting the different classes.

## Usage
For training the model, run the following command:
```
python train.py --batch_size BATCH_SIZE --epochs EPOCHS --lr LR --optimizer OPTIMIZER --exp_name EXP_NAME
```

and substitute the placeholders with the desired values. The script will create a new experiment directory under `experiments/` with the specified experiment name and save the model checkpoints and logs there.

For evaluating the model, run the following command:
```
python test.py --model_path MODEL_PATH --batch_size BATCH_SIZE
```

and substitute the placeholders with the path to the model checkpoint and the desired batch size.

## Pretrained Model
We provide a pretrained model for the OrthoAI system. You can download the model checkpoint and results from the following link: [Pretrained Model](https://drive.google.com/file/d/12zrJzAw_j-QX5dF3K_MvfQnuVpCc26UW/view?usp=sharing).
>>>>>>> feature_extraction_with_classification_fusion_layer
