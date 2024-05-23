from sklearn.ensemble import AdaBoostClassifier
from website_phising.model.base import WebsitePhishingBaseModel



class WPAdaBoostClassifier(WebsitePhishingBaseModel):
    def __init__(
        self,
        base_model,
        train_ratio: float = 0.8,
        n_estimators: int = 16,
        random_state: int = 42
    ) -> None:
        model = AdaBoostClassifier(
            base_estimator=base_model, 
            n_estimators=n_estimators, 
            random_state=random_state
        )
        super(WPAdaBoostClassifier, self).__init__(
            model=model, 
            model_name="AdaBoostClassifier", 
            train_ratio=train_ratio
        )
