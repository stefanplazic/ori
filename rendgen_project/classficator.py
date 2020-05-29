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
		return 0

	if row['Label_1_Virus_category'] == 'Virus':
		return 1

	if row['Label_1_Virus_category'] == 'bacteria':
		return 2

def generate_train_test_data(csv_data):
	train_data, test_data = train_test_split(csv_data, test_size=0.20, random_state=42)
	train_data = train_data.reset_index(drop=True)
	test_data = test_data.reset_index(drop=True)
	return (train_data, test_data)

def generate_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return model

def show_images():
	pass

def preprocess_images(dir_path='./chest_xray_data_set', output_dir_path='./resized_images'):
	files = os.listdir(dir_path)
	for file in files:
		image = cv2.imread(os.path.join(dir_path,file), cv2.IMREAD_UNCHANGED)
		resized_image = cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT))
		cv2.imwrite(os.path.join(output_dir_path,file),resized_image)
	print('-'*50)
	print('Finished resizing images')

if __name__ == "__main__":
	csv_data = load_data()
	model = generate_model()
	train_data, test_data = generate_train_test_data(csv_data)
	print(train_data.tail())