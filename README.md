![image](https://www.investopedia.com/thmb/1hG4u1nwWnYnljZkoGq3dh7046o=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dark-web-4198212-bdd6dd31665e440789a47bd7f2b14460.jpg)

# Darknet Traffic Prediction

## Introduction
This repository contains a machine learning model and a Streamlit web application for predicting Darknet traffic. The project aims to classify internet traffic as either 'Malicious' or 'Safe', focusing on the analysis of Darknet traffic. Darknet traffic classification is crucial for early monitoring of malware and detection of malicious activities.

## Dataset - CIC-Darknet2020
The CIC-Darknet2020 dataset is used for training the model. It includes benign and darknet traffic data, with the latter consisting of various categories such as Audio-Stream, Browsing, Chat, Email, P2P, Transfer, Video-Stream, and VOIP. The dataset is an amalgamation of ISCXTor2016 and ISCXVPN2016 datasets, covering Tor and VPN traffic. 

### Dataset Features
- **Audio-Stream**: Traffic from Vimeo and Youtube.
- **Browsing**: Traffic using Firefox and Chrome.
- **Chat**: Traffic from applications like ICQ, AIM, Skype, Facebook, and Hangouts.
- **Email**: SMTPS, POP3S, and IMAPS protocols.
- **P2P**: Traffic from uTorrent and Transmission (BitTorrent).
- **Transfer**: Skype, SFTP, and FTPS using Filezilla and external services.
- **Video-Stream**: Traffic from Vimeo and Youtube.
- **VOIP**: Voice calls from Facebook, Skype, and Hangouts.

### License and Citation
The dataset can be redistributed, republished, and mirrored, but must include a citation to the CICDarknet2020 dataset and the associated research paper:

> Arash Habibi Lashkari, Gurdip Kaur, and Abir Rahali, “DIDarknet: A Contemporary Approach to Detect and Characterize the Darknet Traffic using Deep Image Learning”, 10th International Conference on Communication and Network Security, Tokyo, Japan, November 2020.

## The Model
The model is a deep neural network built using TensorFlow and Keras. It consists of dense layers and dropout for regularization, trained to classify traffic as 'Malicious' or 'Safe'.

### Model Structure
- **Input Layer**: Based on the number of features in the dataset.
- **Hidden Layers**: Two hidden layers with 64 and 32 neurons respectively, using ReLU activation.
- **Output Layer**: Single neuron with sigmoid activation for binary classification.
- **Regularization**: Dropout layers to prevent overfitting.
- **Optimizer**: Adam optimizer.

### Training and Evaluation
- The dataset is split into training and test sets, with standard scaling applied to the features.
- Early stopping is used during training to prevent overfitting.
- The model achieves high accuracy in classifying the test data.

## The Application
The Streamlit web application allows users to upload a CSV file containing network traffic data and get predictions on whether the traffic is 'Malicious' or 'Safe'. The application uses the trained model for predictions.

### Using the Application
1. Access the application at [Darknet Traffic Prediction App](https://darknetdetect.streamlit.app/).
2. Upload a CSV file with network traffic data.
3. View the predictions displayed in the application.

### Local Setup
To run the application locally:
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Run the Streamlit app using the command `streamlit run app.py`.

## Acknowledgements
Special thanks to the Mitacs Globalink Program and the Fredrik and Catherine Eaton Visitorship research fund from the University of New Brunswick for supporting this research.
