from website_phising import(
    WebsitePhishingBaseModel, 
    WPRandomForestClassifier,
    WPDecisionTreeClassifier,
    WPMLPClassifier,
    WPSVCClassifier
)

model_list = [
    WPRandomForestClassifier,
    WPDecisionTreeClassifier,
    WPMLPClassifier,
    WPSVCClassifier
]


for model_class in model_list:
    model: WebsitePhishingBaseModel
    model = model_class()
    model.from_arff("data/PhishingData.arff")
    model.train()
    train_acc, test_acc = model.evaluate()
    print(f"{model.model_name}: train_acc({train_acc}), test_acc({test_acc})")
    