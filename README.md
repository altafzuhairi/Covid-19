# Malaysia Covid-19 New Cases Prediction Using LSTM Neural Network
# 1. Summary 
The aim of this project is to create a deep learning model using LSTM neural network to predict new cases in Malaysia using the past 30 days of number of cases. Deep learning model is trained and deployed for this task. The dataset is acquired from https://github.com/MoH-Malaysia/covid19-public
# 2. IDE and Framework
This project is completed mainly using VS Code IDE. The main frameworks used in this project are Numpy, Matplotlib,scikit-learn and TensorFlow Keras.
# 3. Methodology
The methodology for this project is inspired from official TensorFlow website which can be refer here:https://www.tensorflow.org/tfx/tutorials/transform/census

3.1 Model Pipeline

The model only use  LSTM, Dense, and Dropout layers. The Nodes in the LSTM layers is 64 and Window size set to 30 days. The simplified illustration of the model is shown in the figure below.

![model](https://user-images.githubusercontent.com/124944787/220149638-608908bd-6bcd-4b3e-8500-358e27299996.png)

# 4. Results
The model is evaluated with test data. The Mean Absolute Percentage Error(MAPE), Mean Absolute Error (MAE) and R-Squared score shown in figure below.

![performance of model](https://user-images.githubusercontent.com/124944787/221622146-e549eb49-724f-4e72-b426-0e0377e53199.png)
