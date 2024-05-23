import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from website_phising import WPSVCClassifier

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
REPEAT_NUM = 3
for kernel in kernels:
    train_acc, test_acc = 0, 0
    for idx in range(REPEAT_NUM):
        model = WPSVCClassifier(kernel=kernel)
        model.from_arff("data/PhishingData.arff")
        model.train()
        _train_acc, _test_acc = model.evaluate()
        train_acc += _train_acc
        test_acc += _test_acc
    train_acc /= REPEAT_NUM
    test_acc /= REPEAT_NUM
    print(f"{kernel}: train_acc({train_acc}), test_acc({test_acc})")