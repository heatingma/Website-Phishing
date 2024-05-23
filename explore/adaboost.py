import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from website_phising import WPAdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


base_models = [
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(max_depth=4)
]


REPEAT_NUM = 3
for base_model in base_models:
    train_acc, test_acc = 0, 0
    for idx in range(REPEAT_NUM):
        model = WPAdaBoostClassifier(base_model=base_model, n_estimators=4)
        model.from_arff("data/PhishingData.arff")
        model.train()
        _train_acc, _test_acc = model.evaluate()
        train_acc += _train_acc
        test_acc += _test_acc
    train_acc /= REPEAT_NUM
    test_acc /= REPEAT_NUM
    print(f"{base_model}: train_acc({train_acc}), test_acc({test_acc})")