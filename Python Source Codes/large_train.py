import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import plotters
from data_clean import data_merged


raw = data_merged()

training_data = raw.sample(frac = 0.8)
test_data = raw.drop(training_data.index)

training_features = (training_data.copy()).drop(columns = ["Score"])
training_label = training_data["Score"]
test_features = (test_data.copy()).drop(columns = ["Score"])
test_label = test_data['Score']

normalizer = layers.Normalization()
normalizer.adapt(np.array(training_features))

multi_feature_model = keras.models.Sequential([
    normalizer,
    layers.Dense(units = 16),
    layers.Dense(units = 1)
])


loss_function = keras.losses.MeanSquaredError()
optim = keras.optimizers.SGD(learning_rate = 0.05)

multi_feature_model.compile(optimizer = optim, loss = loss_function)

print(training_features)
print(training_label)

loss_data = multi_feature_model.fit(
    training_features,
    training_label,
    epochs = 100,
    verbose = 1,
    validation_split = 0.2
)

plotters.plot_loss(loss_data, "full_trainingloss")
print("\nTest data:\n")
pc = pd.DataFrame(list(test_data['Score']), columns = ["Score"])
pc["Predicted"] = multi_feature_model.predict(test_features)
pc["% Difference"] = abs(pc["Score"] - pc["Predicted"])/pc["Score"] * 100

print(pc)
print("Average deviation: ", str(pc["% Difference"].mean())+"%")

raw1 = pd.read_csv(r"Data\2019.csv")
input = (raw1.copy()).drop(columns = ['Country or region', 'Overall rank', "Score", "Social support"])
features = (raw1.copy())["Score"]

print("\nDeployment data:\n")
pc1 = pd.DataFrame(list(raw1['Country or region']), columns = ["Country"])
pc1["Score"] = list(features)
pc1["Predicted"] = multi_feature_model.predict(input)
pc1["% Difference"] = abs(pc1["Score"] - pc1["Predicted"])/pc1["Score"] * 100

print(pc1)
print("Average deviation: ", str(pc1["% Difference"].mean())+"%")