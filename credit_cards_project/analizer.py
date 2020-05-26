import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

cluster_number = 8

def load_data(dataPath='./credit_card_data.csv'):
	credit_data = pd.read_csv(dataPath)
	return credit_data

def clean_data(credit_data):
	credit_data = credit_data.apply(lambda x:x.fillna(x.value_counts().index[0]), axis=0)
	#drop the CUSTID column
	credit_data = credit_data.drop(['CUST_ID'], axis=1)
	return credit_data

def normilize_data(credit_data):
	min_max_scaler = preprocessing.MinMaxScaler()
	scaled_data = min_max_scaler.fit_transform(credit_data.values)
	return pd.DataFrame(scaled_data,columns=credit_data.columns)

'''
method for finding optimal cluser number
in this case it's 8
'''
def elbow(credit_data):
	sum_of_distances = []
	K = range(1,20)
	for k in K:
		km = KMeans(n_clusters=k)
		km = km.fit(credit_data)
		sum_of_distances.append(km.inertia_)
	
	plt.plot(K, sum_of_distances, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum_of_squared_distances')
	plt.title('Elbow Method For Optimal k')
	plt.show()

def split_data(credit_data,ratio=0.8):
	np.random.seed(0)
	mask = np.random.rand(len(credit_data)) < ratio
	train = credit_data[mask]
	test = credit_data[~mask]
	return (train,test)

def plot_clusters(credit_data, cluster_num=8):
	credit_data = credit_data.replace(np.nan, 0, regex=True)
	kmeans = KMeans(n_clusters=cluster_num,random_state=0).fit(credit_data)
	centroids = kmeans.cluster_centers_

	plt.scatter(credit_data['BALANCE'], credit_data['PURCHASES'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
	plt.show()

def predict_clusters(train,test, cluster_num=8):
	X = np.array(train)
	X_test = np.array(test)
	kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
	data_pred = kmeans.predict(X_test)
	return data_pred
	
def summery(data):
	train_summary = data.groupby(by='CLUSTER_PREDICTION').mean()
	train_summary = train_summary[['BALANCE', 'PURCHASES', 
                               'PURCHASES_FREQUENCY','CREDIT_LIMIT', 
                               'ONEOFF_PURCHASES_FREQUENCY', 
                              'MINIMUM_PAYMENTS','PRC_FULL_PAYMENT', 
                               'PAYMENTS']]
	print(train_summary)

def find_correlation(credit_data):
	df = DataFrame(credit_data,columns=['BALANCE','PURCHASES','BALANCEFREQUENCY','CASHADVANCE','CREDIT_LIMIT','CASH_ADVANCE_FREQUENCY','MINIMUM_PAYMENTS'])
	corr = df.corr(method ='pearson')
	print(corr)

if __name__ == "__main__":
	credit_data = load_data()
	credit_data = clean_data(credit_data)
	credit_data = normilize_data(credit_data)
	#elbow(credit_data)
	train,test = split_data(credit_data)
	plot_clusters(credit_data, cluster_number)
	prediction = predict_clusters(train,test,cluster_number)
	test['CLUSTER_PREDICTION'] = prediction
	summery(test)