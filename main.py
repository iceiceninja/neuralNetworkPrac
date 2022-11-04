from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_csv("wine-reviews.csv", usecols= ['country', 'description', 'points', 'price', 'variety', 'winery'])
print(df.head())
df = df.dropna(subset = ["description", "points"])

# plt.hist(df.points, bins=20)
# plt.show()

df["label"] = (df.points >= 90).astype(int)
df = df[["description", "label"]]

print(df.tail())

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
  df = dataframe.copy()
  labels = df.pop('label')
  df = df["description"]
  ds = tf.data.Dataset.from_tensor_slices((df, labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

train_data = df_to_dataset(train)
valid_data = df_to_dataset(val)
test_data = df_to_dataset(test)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable =True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# print(model.predict(test_data))
# model.evaluate(train_data)

history = model.fit(train_data, epochs=5, validation_data= valid_data)

# model.evaluate(test_data)
# 345,Australia,"This wine contains some material over 100 years old, but shows no signs of fragility. Instead, it's concentrated through age and should hold in the bottle indefinitely. It's dark coffee-brown in color, with delectable aromas of rancio, dried fig, molasses and black tea, yet despite enormous concentration avoids excessive weight. And it's amazingly complex and fresh on the nearly endless finish.",Rare,100,350.0,Victoria,Rutherglen,,Joe Czerwinski,@JoeCz,Chambers Rosewood Vineyards NV Rare Muscat (Rutherglen),Muscat,Chambers Rosewood Vineyards
# testPredict = 0,Italy,"Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.",Vulkà Bianco,87,,Sicily & Sardinia,Etna,,Kerin O’Keefe,@kerinokeefe,Nicosia 2013 Vulkà Bianco  (Etna),White Blend,Nicosia)
# 120,Italy,"Slightly backward, particularly given the vintage, the wine has a complex nose with plenty of dark fruit, Turkish delight, smoked meat and earth. A full rich palate, slightly soft, it begins to run out of steam a little in mid palate, but returns with a flourish, and the finish is long and nuanced. Will repay 5+ years of cellaring.",Bricco Rocche Prapó,92,70.0,Piedmont,Barolo,,,,Ceretto 2003 Bricco Rocche Prapó  (Barolo),Nebbiolo,Ceretto
# 339,Spain,"Red in color, with berry and apple aromas, this is sweet, with a heavy body and light effervescence.",1887 Rosado,82,13.0,Catalonia,Cava,,Michael Schachner,@wineschach,Cavas Hill NV 1887 Rosado Sparkling (Cava),Sparkling Blend,Cavas Hill


# Below is an 82 Point wine. This shouldnt ever be seen as a good wine
Wine = {
    'country':["Spain"],
    'description' :["Red in color, with berry and apple aromas, this is sweet, with a heavy body and light effervescence."],
    'province':['Catalonia'],
    'designation':["1887 Rosado"],
    'label' : [1212]
}
df = pd.DataFrame(Wine)

print(model.predict(df_to_dataset(Wine)))

# Below is a 92 Point wine. 
# Good for seeing how accurate the computer can detect the value of the wine when 
# teetering on the edge of good/bad.
Wine = {
    'country':["Italy"],
    'description' :["Slightly backward, particularly given the vintage, the wine has a complex nose with plenty of dark fruit, Turkish delight, smoked meat and earth. A full rich palate, slightly soft, it begins to run out of steam a little in mid palate, but returns with a flourish, and the finish is long and nuanced. Will repay 5+ years of cellaring."],
    'province':['Piedmont'],
    'designation':["Bricco Rocche Prapó"],
    'label' : [1212]
}
df = pd.DataFrame(Wine)

print(model.predict(df_to_dataset(Wine)))

# Below is a 100 Points Dataframe that can be used
#  to see if machine can detect a high quality wine
Wine = {
    'country':["Austrailia"],
    'description' :["This wine contains some material over 100 years old, but shows no signs of fragility. Instead, it's concentrated through age and should hold in the bottle indefinitely. It's dark coffee-brown in color, with delectable aromas of rancio, dried fig, molasses and black tea, yet despite enormous concentration avoids excessive weight. And it's amazingly complex and fresh on the nearly endless finish."],
    'province':['Victoria'],
    'designation':["Rare"],
    'label' : [1212]
}
df = pd.DataFrame(Wine)
print(model.predict(df_to_dataset(Wine)))

# Below is a wine im making up for fun to see if I could make a good wine
Wine = {
    'country':["USA"],
    'description' :["This wine is divine. It has hints of apple and mint. It smells good."],
    'province':['Texas'],
    'designation':["Rose"],
    'label' : [1212]
}
df = pd.DataFrame(Wine)
print(model.predict(df_to_dataset(Wine)))
# model.predict([])

# print(test)
