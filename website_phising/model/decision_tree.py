from sklearn.tree import DecisionTreeClassifier
from website_phising.model.base import WebsitePhishingBaseModel


class WPDecisionTreeClassifier(WebsitePhishingBaseModel):
    def __init__(
        self, 
        train_ratio: float = 0.8,
        max_depth: int = 8
    ) -> None:
        model = DecisionTreeClassifier(max_depth=max_depth)
        super(WPDecisionTreeClassifier, self).__init__(
            model=model, 
            model_name="DecisionTreeClassifier", 
            train_ratio=train_ratio
        )
