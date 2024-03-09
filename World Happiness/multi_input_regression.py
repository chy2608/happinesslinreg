import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import plotters


raw = pd.read_csv(r"Data\2018.csv")
raw = raw.dropna()
print(raw.head(10))

training_data = raw.sample(frac = 0.8)
test_data = raw.drop(training_data.index)

training_features = training_data.copy()
training_label = training_features.pop("Score")
training_features.drop(columns = ['Country or region', 'Overall rank'], inplace=True)
test_features = test_data.copy()
test_features.drop(columns = ['Country or region', 'Overall rank'], inplace=True)
test_label = test_features.pop("Score")

print(training_features.columns)

normalizer = layers.Normalization()
normalizer.adapt(np.array(training_features))

multi_feature_model = keras.models.Sequential([
    normalizer,
    layers.Dense(units = 5),
    layers.Dense(units = 1)
])


loss_function = keras.losses.MeanSquaredError()
optim = keras.optimizers.SGD(lr = 0.01)

multi_feature_model.compile(optimizer = optim, loss = loss_function)

loss_data = multi_feature_model.fit(
    training_features,
    training_label,
    epochs = 100,
    verbose = 1,
    validation_split = 0.2
)

plotters.plot_loss(loss_data, "multi_trainingloss")
print("\nTest data:\n")
pc = pd.DataFrame(list(test_data['Country or region']), columns = ["Country"])
pc["Score"] = list(test_data['Score'])
pc["Predicted"] = multi_feature_model.predict(test_features)
pc["% Difference"] = abs(pc["Score"] - pc["Predicted"])/pc["Score"] * 100

print(pc)
print("Average deviation: ", str(pc["% Difference"].mean())+"%")

raw1 = pd.read_csv(r"Data\2019.csv")
input = (raw1.copy()).drop(columns = ['Country or region', 'Overall rank', "Score"])
features = (raw1.copy())["Score"]

print("\nDeployment data:\n")
pc1 = pd.DataFrame(list(raw1['Country or region']), columns = ["Country"])
pc1["Score"] = list(features)
pc1["Predicted"] = multi_feature_model.predict(input)
pc1["% Difference"] = abs(pc1["Score"] - pc1["Predicted"])/pc1["Score"] * 100

print(pc1)
print("Average deviation: ", str(pc1["% Difference"].mean())+"%")