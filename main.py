import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1_l2

file_path = "C:\\Users\\huilh\\OneDrive\\Área de Trabalho\\AI Training Models\\Kaggle Competitions\\Binary Classification of Machine Failures\\train.csv"


def get_training_data(file_path: str):
    
    df = pd.read_csv(file_path)
    
    # Define a mapping dictionary to convert letters to integers from the 'Types' column in the train.csv file
    mapping = {'L': 0, 'M': 1, 'H': 2}
    
    types = df['Type'].map(mapping).values
    air_temp = df['Air temperature [K]'].values
    process_temp = df['Process temperature [K]'].values
    rot_speed = df['Rotational speed [rpm]'].values
    torque = df['Torque [Nm]'].values
    tool_wear_time = df['Tool wear [min]'].values
    twf = df['TWF'].values
    hdf = df['HDF'].values
    pwf = df['PWF'].values
    osf = df['OSF'].values
    rnf = df['RNF'].values

    x_train = tf.stack([types, air_temp, process_temp, rot_speed, torque, tool_wear_time, twf, hdf, pwf, osf, rnf])
    
    return x_train


def get_target(file_path):

    df = pd.read_csv(file_path)

    y_train = df['Machine failure'].values

    return y_train


if __name__ == '__main__':
    x_train = tf.transpose(get_training_data(file_path))
    y_train = get_target(file_path)


    model = Sequential()

    model.add(Dense(128, activation='swish'))
    model.add(Dense(512, activation='swish', kernel_regularizer=l1_l2(0.01)))
    model.add(Dense(768, activation='swish', kernel_regularizer=l1_l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=256, epochs=8, validation_split=0.35, use_multiprocessing=True)

    model.save('Modelv7.h5')