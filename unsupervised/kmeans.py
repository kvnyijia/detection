from preprocessing import prepare_data
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.cluster import KMeans

class TextKmeans:
    def __init__(self, cluster_num) -> None:
        # text data vectorization tool from sklearn
        self.vectorizer = CountVectorizer()
        # normalization tool from sklearn
        self.transformer = TfidfTransformer(smooth_idf=False)
        self.cluster_num = cluster_num

    def __call__(self, text_data, text_labels, test_ratio =0.3):
        x_train, x_test, y_train, y_test = train_test_split(text_data, text_labels, test_size=test_ratio)
        self.train(x_train, y_train, self.cluster_num)
        predicted_labels = self.predict(x_test)
        self.evaluate(predicted_labels, y_test)

    def text_vectorization(self, text_data, train=True):
        if train:
            x_vect = self.vectorizer.fit_transform(text_data)
            x_vect = self.transformer.fit_transform(x_vect)
        else:
            x_vect = self.vectorizer.transform(text_data)
            x_vect = self.transformer.transform(x_vect)
        return x_vect

    def train(self, x_train, y_train, cluster_num):
        x_train_tf = self.text_vectorization(x_train)
        train_centers = []
        for i in range(cluster_num):
            y_i = np.where(y_train == i)
            x_train_mean = np.mean(x_train_tf[y_i], axis=0)
            train_centers.append(x_train_mean)
        train_centers = np.array(train_centers).reshape((cluster_num, -1))
        # print(train_centers.shape)
        self.kmeans = KMeans(n_clusters=self.cluster_num, init=train_centers)
        print(x_train_tf.shape)
        self.kmeans = self.kmeans.fit(x_train_tf)
    
    def predict(self, x_test):
        x_test_tf = self.text_vectorization(x_test, train=False)
        predict_labels = self.kmeans.predict(x_test_tf)
        return predict_labels

    def evaluate(self, predicted_labels, test_labels):
        print('Classification Report:',classification_report(test_labels,predicted_labels))
        print('Confusion Matrix:',confusion_matrix(test_labels,predicted_labels))
        print('Accuracy Score:',accuracy_score(test_labels,predicted_labels))
        print('Model Prediction Accuracy:',str(np.round(accuracy_score(test_labels,predicted_labels)*100,2)) + '%')
        

if __name__ == "__main__" :
    data = prepare_data()
    # x_train, x_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.2)
    review_kmeans = TextKmeans(cluster_num=2)
    review_kmeans(data['text'], data['category'])
    
