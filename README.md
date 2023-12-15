![image](https://www.investopedia.com/thmb/1hG4u1nwWnYnljZkoGq3dh7046o=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dark-web-4198212-bdd6dd31665e440789a47bd7f2b14460.jpg)

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3.4-blue?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21.2-blue?style=flat&logo=numpy&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.8.0-red?style=flat&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.7.0-orange?style=flat&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.0.2-green?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0.0-red?style=flat&logo=streamlit&logoColor=white)


# Darknet Traffic Prediction

## Introduction
This repository contains a machine learning model and a Streamlit web application for predicting Darknet traffic. The project aims to classify internet traffic as either 'Malicious' or 'Safe', focusing on the analysis of Darknet traffic. Darknet traffic classification is crucial for early monitoring of malware and detection of malicious activities.
CIC-Darknet2020
Darknet is the unused address space of the internet which is not speculated to interact with other computers in the world. Any communication from the dark space is considered sceptical owing to its passive listening nature which accepts incoming packets, but outgoing packets are not supported. Due to the absence of legitimate hosts in the darknet, any traffic is contemplated to be unsought and is characteristically treated as probe, backscatter or misconfiguration. Darknets are also known as network telescopes, sinkholes or blackholes.

Darknet traffic classification is significantly important to categorize real-time applications. Analyzing darknet traffic helps in early monitoring of malware before onslaught and detection of malicious activities after outbreak.

This research work proposes a novel technique to detect and characterize VPN and Tor applications together as the real representative of darknet traffic by amalgamating out two public datasets, namely, ISCXTor2016 and ISCXVPN2016, to create a complete darknet dataset covering Tor and VPN traffic respectively. In CICDarknet2020 dataset, a two-layered approach is used to generate benign and darknet traffic at the first layer. The darknet traffic constitutes Audio-Stream, Browsing, Chat, Email, P2P, Transfer, Video-Stream and VOIP which is generated at the second layer. To generate the representative dataset, we amalgamated our previously generated datasets, namely, ISCXTor2016 and ISCXVPN2016, and combined the respective VPN and Tor traffic in corresponding Darknet categories. Table 1 provides the details of darknet traffic categories, and the applications used to generate the network traffic.

## Dataset - CIC-Darknet2020
The CIC-Darknet2020 dataset is used for training the model. It includes benign and darknet traffic data, with the latter consisting of various categories such as Audio-Stream, Browsing, Chat, Email, P2P, Transfer, Video-Stream, and VOIP. The dataset is an amalgamation of ISCXTor2016 and ISCXVPN2016 datasets, covering Tor and VPN traffic. 

![img](https://github.com/abh2050/cybersecurity/blob/main/benign_vs_Malicious.png)
![img](https://github.com/abh2050/cybersecurity/blob/main/traffic.png)


### Dataset Features
- **Audio-Stream**: Traffic from Vimeo and Youtube.
- **Browsing**: Traffic using Firefox and Chrome.
- **Chat**: Traffic from applications like ICQ, AIM, Skype, Facebook, and Hangouts.
- **Email**: SMTPS, POP3S, and IMAPS protocols.
- **P2P**: Traffic from uTorrent and Transmission (BitTorrent).
- **Transfer**: Skype, SFTP, and FTPS using Filezilla and external services.
- **Video-Stream**: Traffic from Vimeo and Youtube.
- **VOIP**: Voice calls from Facebook, Skype, and Hangouts.

### Visualizations
![](https://github.com/abh2050/cybersecurity/blob/main/flow%20vs%20packet.png)
![](https://github.com/abh2050/cybersecurity/blob/main/corelation.png)

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

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 64)                4992      
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dropout_1 (Dropout)         (None, 32)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 33        
                                                                 
=================================================================
Total params: 7,105
Trainable params: 7,105
Non-trainable params: 0
_________________________________________________________________

### Training and Evaluation
- The dataset is split into training and test sets, with standard scaling applied to the features.
- Early stopping is used during training to prevent overfitting.
- The model achieves high accuracy in classifying the test data.
![](https://github.com/abh2050/cybersecurity/blob/main/training%20val.png)

## The Application
The Streamlit web application allows users to upload a CSV file containing network traffic data and get predictions on whether the traffic is 'Malicious' or 'Safe'. The application uses the trained model for predictions.

### Using the Application
1. Access the application at [Darknet Traffic Prediction App](https://darknetdetect.streamlit.app/).
2. Upload a CSV file with network traffic data called the holdout_set.csv file
3. View the predictions displayed in the application.

### Local Setup
To run the application locally:
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Run the Streamlit app using the command `streamlit run app.py`.

## Acknowledgements
Special thanks to the Mitacs Globalink Program and the Fredrik and Catherine Eaton Visitorship research fund from the University of New Brunswick for supporting this research.
