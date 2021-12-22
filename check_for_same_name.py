import os

t = "data_emotions"

for video_type in os.listdir(t):
    train_names = []
    test_names = []
    val_names = []
    for data_type in os.listdir(os.path.join(t, video_type)):
        for file in os.listdir(os.path.join(t, video_type, data_type)):
            if data_type == "train":
                train_names.append(file)
            elif data_type == "test":
                test_names.append(file)
            else:
                val_names.append(file)
    
    print(len(train_names), len(list(set(train_names))))
    print(len(test_names), len(list(set(test_names))))
    print(len(val_names), len(list(set(val_names))))
    
    for name in test_names:
        print(name in train_names)
    
    print()
    
    for name in val_names:
        print(name in train_names)
    