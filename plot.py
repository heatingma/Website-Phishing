from website_phising import WebsitePhishingBaseModel

model = WebsitePhishingBaseModel()
model.from_arff("data/PhishingData.arff")
model.show_data_distribution()
model.show_data_heatmap()