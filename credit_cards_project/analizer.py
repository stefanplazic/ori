import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(dataPath='./credit_card_data.csv'):
	credit_data = pd.read_csv(dataPath)
	return credit_data


def plot_data(credit_data):
	credit_data.plot(figsize=(20, 8))
	plt.show()

def plot_clusters(credit_data):
	df = DataFrame(credit_data,columns=['BALANCE','PURCHASES','BALANCEFREQUENCY','CASHADVANCE','CREDIT_LIMIT','CASH_ADVANCE_FREQUENCY','MINIMUM_PAYMENTS'])
	#print(df)
	df = df.replace(np.nan, 0, regex=True)
	kmeans = KMeans(n_clusters=4).fit(df)
	centroids = kmeans.cluster_centers_

	plt.scatter(df['BALANCE'], df['PURCHASES'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
	plt.show()
	
def find_correlation(credit_data):
	df = DataFrame(credit_data,columns=['BALANCE','PURCHASES','BALANCEFREQUENCY','CASHADVANCE','CREDIT_LIMIT','CASH_ADVANCE_FREQUENCY','MINIMUM_PAYMENTS'])
	print(df.corr(method ='pearson'))

if __name__ == "__main__":
	credit_data = load_data()
	#plot_data(credit_data)
	plot_clusters(credit_data)
	find_correlation(credit_data)