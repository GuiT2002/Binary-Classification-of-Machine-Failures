import tensorflow as tf
import main
import pandas as pd
import numpy as np


loaded_model = tf.keras.models.load_model('Modelv7.h5')

file_path = "C:\\Users\\huilh\\OneDrive\\Área de Trabalho\\AI Training Models\\Kaggle Competitions\\Binary Classification of Machine Failures\\test.csv"

x_test = tf.transpose(main.get_training_data(file_path))

predictions = loaded_model.predict(x_test)
predictions = np.array(predictions).flatten()

print(predictions)

# Create a DataFrame with the ID and prediction columns
df = pd.DataFrame({'id': range(136429, 136429 + len(predictions)), 'Machine failure': predictions})

# Write the DataFrame to a CSV file
df.to_csv('predictions8.csv', index=False)

