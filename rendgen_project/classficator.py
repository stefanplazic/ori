import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import cv2

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=60
EPOCHS=10

def load_data(file_uri='./chest_xray_metadata.csv'):
	csv_data = pd.read_csv(file_uri)
	csv_data = csv_data[csv_data.Label_1_Virus_category != 'Stress-Smoking']
	csv_data.drop(csv_data.columns[0], axis=1, inplace=True)
	csv_data['data_label'] = csv_data.apply (lambda row: _apply_label(row), axis=1)
	csv_data = csv_data.drop('Label', 1)
	csv_data = csv_data.drop('Label_2_Virus_category', 1)
	csv_data = csv_data.drop('Label_1_Virus_category', 1)

	return csv_data

def plot_data(csv_data):
	csv_data['data_label'].value_counts().plot.bar()
	plt.show()

def _apply_label(row):
	if row['Label'] == 'Normal' :
		return '0'

	if row['Label_1_Virus_category'] == 'Virus':
		return '1'

	if row['Label_1_Virus_category'] == 'bacteria':
		return '2'

def generate_train_test_data(csv_data):
	train_data, test_data = train_test_split(csv_data, test_size=0.20, random_state=42)
	train_data = train_data.reset_index(drop=True)
	test_data = test_data.reset_index(drop=True)
	return (train_data, test_data)

def generate_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
	model.save_weights("model.h5")
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def fit_model(model,train_generator,test_generator, total_train, total_test):
	history = model.fit_generator(
	    train_generator, 
	    epochs=EPOCHS,
	    validation_data=test_generator,
	    validation_steps=total_test//BATCH_SIZE,
	    steps_per_epoch=total_train//BATCH_SIZE
	)
	model.save_weights("model.h5")
	return history

def preprocess_images(dir_path='./chest_xray_data_set', output_dir_path='./resized_images'):
	files = os.listdir(dir_path)
	for file in files:
		image = cv2.imread(os.path.join(dir_path,file), cv2.IMREAD_UNCHANGED)
		resized_image = cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT))
		cv2.imwrite(os.path.join(output_dir_path,file),resized_image)
	print('-'*50)
	print('Finished resizing images')

def train_generator(data):
	data_gen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
	)
	data_generator = data_gen.flow_from_dataframe(
    data, 
    "./resized_images", 
    x_col='X_ray_image_name',
    y_col='data_label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
	)
	return data_generator

def validation_generator(data):
	validation_gen = ImageDataGenerator(rescale=1./255)
	validation_generator = validation_gen.flow_from_dataframe(
    data, 
    "./resized_images", 
    x_col='X_ray_image_name',
    y_col='data_label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE
	)
	return validation_generator

if __name__ == "__main__":
	csv_data = load_data()
	model = generate_model()
	train_data, test_data = generate_train_test_data(csv_data)
	train_generator = train_generator(train_data)
	test_generator = validation_generator(test_data)
	history = fit_model(model, train_generator, test_generator, train_data.shape[0], test_data.shape[0])
	print(history)