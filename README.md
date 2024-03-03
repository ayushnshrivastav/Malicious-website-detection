**Malicious URL Detection**

This repository contains an implementation of a system for detecting malicious URLs using machine learning techniques. The system is based on the research paper titled Malicious URL Detection, which proposes a comprehensive approach for categorizing URLs into harmful and non-malicious categories.


**Research Paper Link**

[Malicious URL Detection Research Paper](https://ieeexplore.ieee.org/document/10397872)


**Overview**

The suggested system utilizes Machine Learning techniques and analyses using different characteristics obtained from the URL for categorization purposes. It encompasses various stages including data collection, feature extraction, data preprocessing, model training, model testing, model selection, and final output phase.


**Detailed Description**

System Architecture:![training model chart](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/73db4dd6-5f59-4c1d-9a87-506163c16cab)

**Data Collection and Preparation**

1. Data Gathering:
  Trustworthy and illuminating datasets are crucial for solving learning-based classification problems. The dataset used includes both harmful and non-malicious URLs gathered from reputable sources to improve classification accuracy. Specifically, the dataset from [PhishTank](http://data.phishtank.com/data/online-valid.csv) containing hundreds of phishing URLs was considered. To reduce the possibility of data imbalance, a margin value of 10,000 phishing URLs and 5000 real URLs was taken into account.

2. Feature Selection and Extraction
  Feature extraction is a crucial and challenging phase when dealing with large datasets. A feature in machine learning is a distinct measurable quality, trait, or attribute of a phenomenon being observed. Various features were extracted from the URLs including domain extraction, presence of IP address, URL length categorization, number of slashes in the URL, presence of redirection, HTTPS token presence in the domain, presence of URL shortening services, domain age, and more.
![feature extraction chart](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/2ff88232-36ae-4dbe-982c-edb89ed0854f)


3. Creation of Datasets
  Legal and phishing datasets were created by extracting features from legitimate and phishing URLs respectively. These datasets were then concatenated to create the final dataset for model training.

**Machine Learning Modelling**

The system employs various machine learning models for detecting malicious URLs:

1. Decision Tree Classifier: A tree-based supervised learning algorithm, which generates a decision tree to make predictions based on the attributes of the URLs.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/d7fc05ed-6c85-4891-9657-371f919dfe1e)

2. Random Forest Classifier: An ensemble learning approach that builds numerous decision trees and averages their predictions to increase the model’s accuracy and stability.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/d058cf47-eefe-48df-bacc-327089c1d063)

3. Multilayer Perceptrons (MLPs): Artificial neural networks with multiple hidden layers, capable of handling intricate non-linear correlations between the input attributes and the output class.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/cb6a10a1-63e8-4e88-bce7-038cd3601e8c)

4. XGBoost Classifier: Utilizes a gradient boosting tree technique to construct a tree-based model, which can handle big datasets and increase accuracy by combining predictions of other trees.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/2b323501-1173-4d1b-96b7-f8b61c15405d)

5. Autoencoder Neural Network: Unsupervised deep learning methods known as autoencoder neural networks, which can learn to reassemble the input data and lower the dimensionality of the URL data to enhance the predictive model’s performance.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/5f9382f6-58eb-40df-850a-18c03726e2a5)

6. Support Vector Machines (SVM): Linear classifiers that locate the largest margin hyperplane to divide the data into multiple classes, capable of handling high-dimensional data and producing reliable findings even in the presence of outliers.
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/37002a67-44a5-4120-90d9-f2b99f01957e)


**Results and Decision**

The extracted features from the labelled data, which include both harmful and helpful URLs, are then used to separate the data into training and testing sets. Following their passage via several approaches for feature extraction and labeling, the training data are then fed to various models, such as Multilayer Perceptrons, XGBoost, Autoencoder, Random Forest, Decision Tree, and SVM.

The trained models are assessed using the training dataset to measure the model's accuracy. The dataset is split 80-20 into train and test sets for each classification model. The accuracy of the models is evaluated using the test data, and various metrics such as accuracy, precision, recall, and F1 score are calculated to assess the performance of each model.

The F1 score, which combines recall and accuracy into one number, offers a fair assessment of a classifier’s performance, especially when the dataset is unbalanced.

F1 Score:
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/ce60b99f-07f1-487b-be52-95f26ebb8397)

Accuracy:
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/4984cda1-5132-439a-9ede-6fdd2c35b7e9)

Precision:
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/c57b7388-0441-45e6-9e36-98f4e7d0e21d)

Recall:
![image](https://github.com/ayushnshrivastav/Malicious_website_detection/assets/71760784/60097329-fac7-46be-9b32-cec73f8e7fe9)


**Model Analysis**

1. Decision Tree: Accuracy was 81.5% and 82.4% for training and testing data respectively.
2. Random Forest: Accuracy was 81.5% and 82.4% for training and testing data respectively.
3. Multilayer Perceptrons (MLPs): Accuracy was approximately 84.55% for both training and testing data.
4. XGBoost: Accuracy was approximately 84.55% for both training and testing data.
5. SVM: Accuracy was 80% for both training and testing data.
   
**Conclusion**

The majority of online criminal operations are based on malicious websites, posing significant risks to end users. This repository provides an implementation of a system for detecting malicious URLs using machine learning techniques. It emphasizes the importance of trustworthy datasets, feature extraction, and selection, as well as the significance of model selection and evaluation in achieving accurate classification results.

A number of machine learning algorithms, including Multilayer Perceptrons and XGBoost, were deployed on the training dataset to solve the binary classification issue of detecting dangerous URLs. It has been shown that these classifiers outperform others for this specific challenge.

For more detailed information, please refer to the [research paper](https://ieeexplore.ieee.org/document/10397872)

Contributors:
Shakti Kinger; 
Pranav Nirmal; 
Ayush Shrivastav; 
Aniket Sharma; 
Sakshi Saindane
