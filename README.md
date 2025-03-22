# SMATO - Image-based Pedestrian Smartphone Usage Classification

In today's digital age, it is becoming increasingly common to witness pedestrians engrossed in their smartphones while navigating through bustling traffic or its vicinity [[1]](#1). Although regulations surrounding smartphone usage by drivers are prevalent in many countries, the impact of smartphone distraction on pedestrians should not be overlooked. Studies have revealed that smartphone use among pedestrians significantly hampers their situational awareness and attentiveness, consequently increasing the likelihood of accidents and injuries [[2]](#2). With the world gradually shifting towards autonomous driving, it is imperative to incorporate advanced safety measures that can identify smartphone usage among pedestrians. Leveraging widely available sensors like cameras, it becomes feasible to detect instances of smartphone engagement and take additional precautions. To facilitate this endeavor, this repository presents a carefully curated dataset and transfer-learning-based classifier.

![alt text](https://github.com/saadejazz/smato/blob/main/images/example_predictions.png)

The prediction scheme for the pictures above is **<predicted_label: (true|false)>, <true_label: (1|0)>**


This module is part of the following [paper](https://ieeexplore.ieee.org/abstract/document/10737454)
```
@article{ejaz2024trust,
  title={Trust-aware safe control for autonomous navigation: Estimation of system-to-human trust for trust-adaptive control barrier functions},
  author={Ejaz, Saad and Inoue, Masaki},
  journal={IEEE Transactions on Control Systems Technology},
  year={2024},
  publisher={IEEE}
}
```

## Sourcing the Dataset

The dataset utilized for this project has been compiled from various publicly available pedestrian datasets that encompass a wide range of images depicting pedestrians in diverse ambient environments, orientations, and engagements. These datasets include prominent sources such as PETA [[3]](#3).

To ensure the dataset's comprehensiveness, an additional collection of images was obtained through Open Source Intelligence (OSINT) techniques, specifically by leveraging image search functionalities provided by platforms like Google and Bing (example scraper: [icrawler](https://icrawler.readthedocs.io/en/latest/). This process facilitated the inclusion of images captured under varying lighting conditions and camera angles. In order to augment the dataset, most images were flipped horizontally, which effectively doubled the dataset's size while maintaining its diversity and correctness.

All images underwent a preliminary step of pedestrian isolation. This was achieved by employing a person detector, specifically [OpenPifPaf](https://openpifpaf.github.io/intro.html). By accurately detecting and segmenting pedestrians within the images, the subsequent annotation process was applied. The annotation process itself followed an iterative approach to gradually improve the dataset's quality and inclusiveness. It commenced with the curation of a small initial dataset, which was used to train an initial classifier. With this classifier in place, additional images were sourced from a variety of different sources. Subsequently, the previously trained classifier was utilized to automatically annotate these newly acquired images. However, to ensure accuracy, manual inspection of the annotations was conducted to identify any misclassifications (as the dataset grew in size, it was expected that the number of misclassifications would decrease). This manual inspection of misclasification also presented an opportunity to analyze the reasons behind misclassifications - knowledge that influenced future data sourcing strategies or inform adjustments to the model building process. The iterative nature of this approach, involving constant refinement through training, data sourcing, and manual inspection, ensured the dataset's continuous improvement, ultimately leading to a more reliable and comprehensive resource for smartphone usage detection in pedestrian images.

The dataset consist of a total of 13866 images of pedestrian (single pedestrian per image), from which 3770 are engaged with a smartphone while 10096 of them are not. This imbalance is kept to demonstrate real-world composition of smartphone users amongst pedestrians, while avoiding severe imbalances that might hinder training. The dataset can be downloaded from [here](https://drive.google.com/file/d/1cI6OcMlKPXcWCLZtmScGpkwnprvdVOLo/view?usp=sharing).

## Training the Classifier

The notebook ```train.ipynb``` contains comprehensive information regarding the steps followed to train the classifier. Transfer learning was employed to capitalize on the capabilities of pre-trained deep learning models. Specifically, MobileNet V2 was chosen as the feature extractor, enabling the extraction of a 1280-dimensional embedding vector from input images resized to 224x224 pixels. This embedding vector effectively captured the essential features and patterns related to smartphone usage in pedestrian images. The classifier's architecture consists of three fully connected layers following the feature extractor layer, with batch normalization and dropout applied after each layer. Dropout layers play a crucial role in mitigating overfitting issues by randomly disabling a fraction of neurons during training, thereby promoting model generalization.

Within the notebook, the train-validation-test split is detailed, highlighting how the dataset was divided to facilitate model training, validation, and final evaluation. Preprocessing steps, such as resizing the images and preparing the data for training, are also outlined. During training, the model was compiled, and the F1 score was chosen as the evaluation metric. The F1 score is particularly suited for imbalanced datasets, providing a comprehensive assessment of the model's performance by considering both precision and recall. After training, the model achieved an impressive F1 score of 87.68% and an accuracy of 92.71% on the test dataset. These results indicate that the classifier has learned to accurately detect instances of smartphone usage in pedestrian images, contributing to the advancement of safety considerations in autonomous driving and pedestrian environments.

The notebook ```train.ipynb``` provides further insights into the code implementation, training parameters, hyperparameter tuning, and any additional details necessary to reproduce the results or further improve the classifier's performance.

## Inference
The notebook ```infer.ipynb``` provides details on how to classify your images using this classifier. A saved model is already included in the repository in the folder ```saved_moels```.

## EfficientNet V2 as another model
A model using EfficientNet V2M as the feature extractor (has better benchmark performance) was also trained. This trained model performed slightly better than the one based on MobileNet V2, and can be downloaded from [here](https://drive.google.com/file/d/1IEBlPKuedAusiFGQOx-udnTLAt3-Aj2c/view?usp=sharing). However, it must be noted that the preprocessing steps for this feature extractor are included in the model and should be avoided in the inference code. More details on the training of this model are in the notebook: ```misc/train-efficientnet.ipynb```. Moreover, it should also be noted that this model is heavier and hence would require a greater computational burden and hence inference time.

## References
<a id="1">[1]</a> Nasar, J.L. and Troyer, D., 2013. Pedestrian injuries due to mobile phone use in public places. Accident Analysis & Prevention, 57, pp.91-95.

<a id="2">[2]</a>  Frej, D., Jaśkiewicz, M., Poliak, M. and Zwierzewicz, Z., 2022. Smartphone Use in Traffic: A Pilot Study on Pedestrian Behavior. Applied Sciences, 12(24), p.12676.

<a id="3">[3]</a> Y. Deng, P. Luo, C. C. Loy, X. Tang, "Pedestrian attribute recognition at far distance," in Proceedings of ACM Multimedia (ACM MM), 2014
