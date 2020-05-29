import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
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
	csv_data['data_label'] = csv_data.apply (lambda row: _apply_label(row), axis=1)
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

def generate_train_test_data():
	pass

def generate_model():
	pass

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
	#preprocess_images()
	csv_data = load_data()
	plot_data(csv_data)