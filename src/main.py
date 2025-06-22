from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt
import seaborn as sns

import time

# source: https://jasoneliu.github.io/pessoalab/mnist/mnist_sklearn.html

x, y = fetch_openml("mnist_784", return_X_y=True)

# feature scaling
# pixels can be of value 0 (=white), black (=255) or gray (=between 1 and 254)
# now pixels are only of value 0 or 1
x = x / 255.0

# split in test and train

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=361
)

# svm with linear kernel
svm_lin = SVC(kernel="linear")
start_time_lin = time.time()
svm_lin.fit(x_train, y_train)
end_time_lin = time.time()

# time how long training took
elapsed_time_lin = end_time_lin - start_time_lin
print(f"Training took {elapsed_time_lin:.2f} seconds")

# svm with rbf kernel
svm_rbf = SVC(kernel="rbf")
start_time_rbf = time.time()
svm_rbf.fit(x_train, y_train)
end_time_rbf = time.time()

# time how long training took
elapsed_time_rbf = end_time_rbf - start_time_rbf
print(f"Training took {elapsed_time_rbf:.2f} seconds")


def eval_model(model: SVC, x_test, y_test, title: str):
    y_pred = model.predict(x_test)
    print(f"Evaluation for model: {title}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    with open(f"stats/{secure_filename(title)}.md", "a", encoding="UTF-8") as f:
        f.write(f"Evaluation for model: {title}\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred))

    con_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(con_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix for {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"figures/{secure_filename(title)}.png", format="png")
    plt.close()


with open(
    f"stats/{secure_filename("SVM linear kernel")}.md", "a", encoding="UTF-8"
) as f:
    f.write(f"Training took {elapsed_time_lin:.2f} seconds. \n")

with open(
    f"stats/{secure_filename("SVM radial basis function kernel")}.md",
    "a",
    encoding="UTF-8",
) as f:
    f.write(f"Training took {elapsed_time_rbf:.2f} seconds. \n")

eval_model(svm_lin, x_test, y_test, "SVM linear kernel")
eval_model(svm_rbf, x_test, y_test, "SVM radial basis function kernel")
