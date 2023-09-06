# Deep Learning Specialization Programming Exercises
These Programming Exercises are part of the deeplearning.ai Deep Learning Specialization on Coursera. It's a great course and I would highly recommend it for people who are interested in learning deep learning.<br>
**The Deep Learning Specialization Consists of 5 courses, divided into multiple weeks as shown below.**

# Course1 - Neural Networks and Deep Learning

## Week2
### [W2A2 - Logistic Regression with a Neural Network mindset](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%201%20Neural%20Networks%20and%20Deep%20Learning/Week2/W2A2/Logistic_Regression_with_a_Neural_Network_mindset.ipynb)
**Objective:**
- Build the general architecture of a learning algorithm, including:
    - Initializing parameters
    - Calculating the cost function and its gradient
    - Using an optimization algorithm (gradient descent) 
- Gather all three functions above into a main model function, in the right order.
## Week3
### [W3A1 - Planar data classification with one hidden layer](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%201%20Neural%20Networks%20and%20Deep%20Learning/Week3/W3A1/Planar_data_classification_with_one_hidden_layer.ipynb)
**Objective:**
- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh
- Compute the cross entropy loss
- Implement forward and backward propagation
## Week4
### [W4A1 - Building your Deep Neural Network: Step by Step](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%201%20Neural%20Networks%20and%20Deep%20Learning/Week4/W4A1/Building_your_Deep_Neural_Network_Step_by_Step.ipynb)
**Objective:**
- Use non-linear units like ReLU to improve your model
- Build a deeper neural network (with more than 1 hidden layer)
- Implement an easy-to-use neural network class

### [W4A2 - Deep Neural Network for Image Classification: Application](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%201%20Neural%20Networks%20and%20Deep%20Learning/Week4/W4A2/Deep%20Neural%20Network%20-%20Application.ipynb)
**Objective:**
- Build and train a deep L-layer neural network, and apply it to supervised learning

# Course2 - Improving Deep Neural Networks Hyperparameter Tuning, Regularization, and Optimization

## Week1
### [W1A1 - Initialization](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%2C%20and%20Optimization/week1/W1A1/Initialization.ipynb)
**Objective:**
Learning about how a well-chosen initialization can:
- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error

### [W1A2 - Regularization](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%2C%20and%20Optimization/week1/W1A2/Regularization.ipynb)
**Objective:**
- Using regularization in your deep learning models

### [W1A3 - Gradient Checking](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%2C%20and%20Optimization/week1/W1A3/Gradient_Checking.ipynb)
**Objective:**
- Implement gradient checking to verify the accuracy of your backprop implementation

## Week2
### [W2A1 - Optimization Methods](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%2C%20and%20Optimization/week2/W2A1/Optimization_methods.ipynb)
**Objective:**
- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization

## Week3
### [W3A1 - Introduction to TensorFlow](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuning%2C%20Regularization%2C%20and%20Optimization/week3/W3A1/Tensorflow_introduction.ipynb)
**Objective:**
- Use `tf.Variable` to modify the state of a variable
- Explain the difference between a variable and a constant
- Train a Neural Network on a TensorFlow dataset

# Course3 - Structuring Machine Learning Projects
This course consists of two weeks and it doesn't include any programming excercises.

## Week1
**Objective:**
- Explain why Machine Learning strategy is important
- Apply satisficing and optimizing metrics to set up your goal for ML projects
- Choose a correct train/dev/test split of your dataset
- Define human-level performance
- Use human-level performance to define key priorities in ML projects
- Take the correct ML Strategic decision based on observations of performances and dataset

## Week2
**Objective:**
- Describe multi-task learning and transfer learning
- Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

# Course4 - Convolutional Neural Networks

## Week1
### [W1A1 - Convolutional Neural Networks: Step by Step](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week1/W1A1/Convolution_model_Step_by_Step_v1.ipynb)
**Objective:**
- Explain the convolution operation
- Apply two different types of pooling operation
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build a convolutional neural network
  
### [W1A2 - Convolutional Neural Networks: Application](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week1/W1A2/Convolution_model_Application.ipynb)
**Objective:**
- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API
- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

## Week2
### [W2A1 - Residual Networks](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week2/W2A1/Residual_Networks.ipynb)
**Objective:**
- Implement the basic building blocks of ResNets in a deep neural network using Keras
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification
- Implement a skip connection in your network
  
### [W2A2 - Transfer Learning with MobileNetV2](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week2/W2A2/Transfer_learning_with_MobileNet_v1.ipynb)
**Objective:**
- Create a dataset from a directory
- Preprocess and augment data using the Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tune a classifier's final layers to improve accuracy

## Week3
### [W3A1 - Autonomous Driving - Car Detection](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week3/W3A1/Autonomous_driving_application_Car_detection.ipynb)
**Objective:**
- Detect objects in a car detection dataset
- Implement non-max suppression to increase accuracy
- Implement intersection over union
- Handle bounding boxes, a type of image annotation popular in deep learning
  
### [W3A2 - Image Segmentation with U-Net](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week3/W3A2/Image_segmentation_Unet_v2.ipynb)
**Objective:**
- Build your own U-Net
- Explain the difference between a regular CNN and a U-net
- Implement semantic image segmentation on the CARLA self-driving car dataset
- Apply sparse categorical crossentropy for pixelwise prediction

## Week4
### [W4A1 - Face Recognition](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week4/W4A1/Face_Recognition.ipynb)
**Objective:**
- Differentiate between face recognition and face verification
- Implement one-shot learning to solve a face recognition problem
- Apply the triplet loss function to learn a network's parameters in the context of face recognition
- Explain how to pose face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Perform face verification and face recognition with these encodings
  
### [W4A2 - Deep Learning & Art: Neural Style Transfer](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%204%20Convolutional%20Neural%20Networks/week4/W4A2/Art_Generation_with_Neural_Style_Transfer.ipynb)
**Objective:**
- Implement the neural style transfer algorithm 
- Generate novel artistic images using your algorithm 
- Define the style cost function for Neural Style Transfer
- Define the content cost function for Neural Style Transfer
  
# Course5 - Sequence Models

## Week1
### [W1A1 - Building your Recurrent Neural Network - Step by Step](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week1/W1A1/Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb)
**Objective:**
- Define notation for building sequence models
- Describe the architecture of a basic RNN
- Identify the main components of an LSTM
- Implement backpropagation through time for a basic RNN and an LSTM
- Give examples of several types of RNN 
  
### [W1A2 - Character level language model - Dinosaurus Island](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week1/W1A2/Dinosaurus_Island_Character_level_language_model.ipynb)
**Objective:**
- Store text data for processing using an RNN 
- Build a character-level text generation model using an RNN
- Sample novel sequences in an RNN
- Explain the vanishing/exploding gradient problem in RNNs
- Apply gradient clipping as a solution for exploding gradients

### [W1A3 - Improvise a Jazz Solo with an LSTM Network](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week1/W1A3/Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb)
**Objective:**
- Apply an LSTM to a music generation task
- Generate your own jazz music with deep learning
- Use the flexible Functional API to create complex models

## Week2
### [W2A1 - Operations on Word Vectors](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week2/W2A1/Operations_on_word_vectors_v2a.ipynb)
**Objective:**
- Explain how word embeddings capture relationships between words
- Load pre-trained word vectors
- Measure similarity between word vectors using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______. 
  
### [W2A2 - Emojify!](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week2/W2A2/Emoji_v3a.ipynb)
**Objective:**
- Create an embedding layer in Keras with pre-trained word vectors
- Explain the advantages and disadvantages of the GloVe algorithm
- Describe how negative sampling learns word vectors more efficiently than other methods
- Build a sentiment classifier using word embeddings
- Build and train a more sophisticated classifier using an LSTM

## Week3
### [W3A1 - Neural Machine Translation](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week3/W3A1/Neural_machine_translation_with_attention_v4a.ipynb)
**Objective:**
- Build a Neural Machine Translation (NMT) model to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25"). 
- Do this using an attention model, one of the most sophisticated sequence-to-sequence models. 
  
### [W3A2 - Trigger Word Detection](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week3/W3A2/Trigger_word_detection_v2a.ipynb)
**Objective:**
- Structure a speech recognition project
- Synthesize and process audio recordings to create train/dev datasets
- Train a trigger word detection model and make predictions

## Week4
### [W4A1 - Transformer Network](https://github.com/w-sliman/Deep_Learning_Specialization_Programming_Exercises/blob/main/Course%205%20Sequence%20Models/week4/W4A1/C5_W4_A1_Transformer_Subclass_v1.ipynb)
**Objective:**
- Create positional encodings to capture sequential relationships in data
- Calculate scaled dot-product self-attention with word embeddings
- Implement masked multi-head attention
- Build and train a Transformer model

