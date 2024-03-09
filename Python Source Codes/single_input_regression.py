import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import plotters


raw = pd.read_csv(r"Data\2019.csv")
print(raw.head(10))

training_data = raw.sample(frac = 0.8)
test_data = raw.drop(training_data.index)

training_features = training_data.copy()
training_label = training_features.pop("Score")
test_features = test_data.copy()
test_label = test_features.pop("Score")

feature = "GDP per capita"
normalizer = layers.Normalization(input_shape = [1,])
normalizer.adapt(training_features[feature])

norm_comparison = pd.DataFrame(data = np.array(training_features[feature]), columns = [feature])
norm_comparison["Post normalisation"] = normalizer(training_features[feature]).numpy().T

print(norm_comparison.head(10))

single_feature_model = keras.models.Sequential([
    normalizer,
    layers.Dense(units = 1)
])


loss_function = keras.losses.MeanSquaredError()
optim = keras.optimizers.SGD(lr = 0.05)

single_feature_model.compile(optimizer = optim, loss = loss_function)

loss_data = single_feature_model.fit(
    training_features[feature],
    training_label,
    epochs = 100,
    verbose = 1,
    validation_split = 0.2
)

plotters.plot_loss(loss_data)

for k in [5, 6, 7, 8]:
    new_data = pd.read_csv(r"Data\201"+ f"{k}.csv")
    if k == 5 or k == 6:
        label = "Happiness Score"
        feature = "Economy (GDP per Capita)"
    elif k == 7:
        label = "Happiness.Score"
        feature = "Economy..GDP.per.Capita."
    else:
        label = "Score"
        feature = "GDP per capita"

    plotters.plot_apply([new_data[feature]], [new_data[label]], single_feature_model, feature, k)

    pred_comparison = pd.DataFrame(data = np.concatenate(([new_data[feature]], [new_data[label]]), axis = 0).T, columns = ["Inputs", "True Values"])
    pred_comparison["Predicted"] = single_feature_model.predict(tf.constant(new_data[feature])) 
    pred_comparison["Absolute Loss"] = abs(pred_comparison["Predicted"] - pred_comparison['True Values'])
    print(f"201{k} Predictions")
    print(pred_comparison.head(10), "\n")
