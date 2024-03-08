import gzip
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def evaluate(y_pred, y_true, title=''):
    print('==============', title, '==============')
    metric = metrics.classification_report(y_true, y_pred, target_names=idx_to_cls)
    confusion = metrics.confusion_matrix(y_true, y_pred)
    print(metric)

    # 绘制热度图
    plt.imshow(confusion, cmap=plt.cm.Greens)
    plt.title(title)
    indices = range(len(confusion))
    plt.xticks(indices, idx_to_cls)
    plt.yticks(indices, idx_to_cls)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.savefig(f"metric/{title}")
    # 显示图片
    plt.show()

dataset = []
f = gzip.open("renttherunway_final_data.json.gz")
n_data = 10000
count = 0
for l in f:
    try:
        d = eval(l)
    except:
        continue
    dataset.append(d)
    count += 1
    if count == n_data:
        break
f.close()

idx_to_cls = ['fit', 'large', 'small']
cls_to_idx = {label: i for i, label in enumerate(idx_to_cls)}

X = [d['review_text'] for d in dataset]
y = [cls_to_idx[d['fit']] for d in dataset]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_count, y_train)
y_pred = model.predict(X_test_count)
evaluate(y_pred, y_test, "LogisticRegression_CountVectorizer")

model = SVC()
model.fit(X_train_count, y_train)
y_pred = model.predict(X_test_count)
evaluate(y_pred, y_test, "SVC_CountVectorizer")


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
evaluate(y_pred, y_test, "LogisticRegression_TfidfVectorizer")

model = SVC()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
evaluate(y_pred, y_test, "SVC_TfidfVectorizer")

