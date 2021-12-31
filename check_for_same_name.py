import os

t = "audio_data_emotions_features"

datatypes = ["train", "test", "val"]

data = ["features", "labels"]

feature_names = []
label_names = []

for datatype in datatypes:
    filenames = os.listdir(os.path.join(t, datatype, "features"))
    for filename in filenames:
        feature_names.append(filename)

print(len(feature_names))
print(len(list(set(feature_names))))
    