from sklearn.neural_network import MLPClassifier
from website_phising.model.base import WebsitePhishingBaseModel


class WPMLPClassifier(WebsitePhishingBaseModel):
    def __init__(
        self, 
        train_ratio: float = 0.8,
        hidden_layer_sizes: tuple = ([64, 64, 64])
    ) -> None:
        model = MLPClassifier(
            alpha=0.001, 
            hidden_layer_sizes=hidden_layer_sizes
        )
        super(WPMLPClassifier, self).__init__(
            model=model, 
            model_name="MLPClassifier", 
            train_ratio=train_ratio
        )
