import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from website_phising import WPMLPClassifier

hidden_dims_list = [
    [([16, 16]), 50],  
    [([16, 16]), 200],
    [([64, 64, 64]), 50],
    [([64, 64, 64]), 200],
    [([64, 128, 64]), 50],
    [([64, 128, 64]), 200]
]
REPEAT_NUM = 3


for (hidden_dims, max_iter) in hidden_dims_list:
    train_acc, test_acc = 0, 0
    for idx in range(REPEAT_NUM):
        model = WPMLPClassifier(hidden_layer_sizes=hidden_dims, max_iter=max_iter)
        model.from_arff("data/PhishingData.arff")
        model.train()
        _train_acc, _test_acc = model.evaluate()
        train_acc += _train_acc
        test_acc += _test_acc
    train_acc /= REPEAT_NUM
    test_acc /= REPEAT_NUM
    print(f"{hidden_dims} with max_iter({max_iter}): train_acc({train_acc}), test_acc({test_acc})")