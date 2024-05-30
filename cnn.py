import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def process_img(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = img.reshape(-1, 256, 256, 1)
    img = img.astype('float32') / 255

    return img

def data_generator(df, batch_size, path):

    num_samples = len(df)

    while True:

        for offset in range(0, num_samples, batch_size):

            batch_samples = df.iloc[offset:offset+batch_size]
            images = []
            labels = []

            for _, row in batch_samples.iterrows():
                img_path = f"{path}/{row['id']}.jpg"
                img = process_img(img_path)
                images.append(img)
                labels.append(row['breast density'])

            X = np.array(images)
            y = np.array(labels)

            yield X, y

def create_model():

    model = Sequential([

        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_size, name='input'),
        Dropout(rate=0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Dropout(rate=0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        Dropout(rate=0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dense(output_size, activation='softmax', name='density')

    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

                  
    return model

annotations_path = 'dataset/density_info.csv'
img_path = 'dataset/img'
pp_path = 'dataset/pp'
roi_path = 'dataset/roi'

df = pd.read_csv(annotations_path)

df['breast density'] = df['breast density'].replace(3, 2)
df['breast density'] = df['breast density'] - 1
df['breast density'] = df['breast density'].replace(3, 2)

y = df['breast density']

input_size = (256, 256, 1)
output_size = len(y.unique())

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)
batch_size = 32

train_generator = data_generator(x_train, batch_size, pp_path)
test_generator = data_generator(x_test, batch_size, pp_path)

model = create_model()
model.summary()

start = time.time()
history = model.fit(train_generator,
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=10,
                    validation_data=test_generator,
                    validation_steps=len(x_test) // batch_size
                    #callbacks=[EarlyStopping(patience=3)]
                    )
end = time.time() - start

print('Training time:', end)

x_testv, y_testv = next(test_generator)

loss, accuracy = model.evaluate(x_testv, y_testv)

# Save the model
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
model.save(f'cnn_pp_{timestamp}.h5')

# now with ROI

train_generator = data_generator(x_train, batch_size, roi_path)
test_generator = data_generator(x_test, batch_size, roi_path)

model = create_model()
model.summary()

start = time.time()
history = model.fit(train_generator,
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=10,
                    validation_data=test_generator,
                    validation_steps=len(x_test) // batch_size
                    #callbacks=[EarlyStopping(patience=3)]
                    )
end = time.time() - start

print('Training time:', end)

x_testv, y_testv = next(test_generator)

loss, accuracy = model.evaluate(x_testv, y_testv)

# Save the model
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
model.save(f'cnn_pp_{timestamp}.h5')