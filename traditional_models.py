from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from time import time

class RandomForest():
    def __init__(self,data_loader):
        self.data=data_loader
        self.model=RandomForestClassifier(n_estimators=100)
        print('_' * 80)
        print("Training: ")
        print(self.model)
    def fit(self):
        t0 = time()
        train_X,train_y=self.data.get_train_data()
        self.model.fit(train_X, train_y)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

    def accurancy(self):
        test_X,test_y=self.data.get_test_data()
        t0 = time()
        pred = self.model.predict(test_X)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        score = metrics.accuracy_score(test_y, pred)
        print("accuracy:   %0.3f" % score)

    def predict(self,text):
        pass








clf = RandomForestClassifier(n_estimators=100)
# clf = RidgeClassifier(tol=1e-2, solver="lsqr")
# clf = Perceptron(n_iter=50)
# clf = LinearSVC()
# clf = GradientBoostingClassifier()

# clf = SGDClassifier(alpha=.0001, n_iter=50,penalty="l1")
# clf = SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")

# clf = NearestCentroid()
# clf = MultinomialNB(alpha=.01)
# clf = BernoulliNB(alpha=.01)

# pipeline模型特征选择和分类模型结合在一起
# clf = Pipeline([ ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)), ('classification', LinearSVC())])

# benchmark(clf)