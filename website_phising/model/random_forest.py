from sklearn.ensemble import RandomForestClassifier
from website_phising.model.base import WebsitePhishingBaseModel


class WPRandomForestClassifier(WebsitePhishingBaseModel):
    def __init__(
        self, 
        train_ratio: float = 0.8,
        max_depth: int = 15
    ) -> None:
        model = RandomForestClassifier(max_depth=max_depth)
        super(WPRandomForestClassifier, self).__init__(
            model=model, 
            model_name="RandomForestClassifier", 
            train_ratio=train_ratio
        )