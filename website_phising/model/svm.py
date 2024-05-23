from sklearn.svm import SVC
from website_phising.model.base import WebsitePhishingBaseModel


class WPSVCClassifier(WebsitePhishingBaseModel):
    def __init__(
        self, 
        train_ratio: float = 0.8,
        kernel: str = "linear",
        C: float = 1.0,
        random_state: int = 12
    ) -> None:
        model = SVC(kernel=kernel, C=C, random_state=random_state)
        super(WPSVCClassifier, self).__init__(
            model=model, 
            model_name="SVCClassifier", 
            train_ratio=train_ratio
        )
