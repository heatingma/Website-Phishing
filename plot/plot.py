import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from website_phising import WebsitePhishingBaseModel

model = WebsitePhishingBaseModel()
model.from_arff("data/PhishingData.arff")
model.show_data_distribution()
model.show_data_heatmap()