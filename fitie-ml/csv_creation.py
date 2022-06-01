import pandas as pd
import os
import csv

train_datapath = r'D:\Dataset\Exercises dataset\train'
test_datapath = r'D:\Dataset\Exercises dataset\test'

dataset_path = os.listdir(train_datapath)

label_types = os.listdir(train_datapath)
print(label_types)

rooms = []

classes = ['Arm Raise','Bicycle Crunch','Bird Dog','Curl','Fly', 'Leg Raise', 'Overhead Press', 'Push Up', 'Squat',
           'Superman']

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir(train_datapath + '\\' + item)

    # Add them to the list
    for room in all_rooms:

        rooms.append((item, str(train_datapath + '\\' + item) + '\\' + room))

# Build a dataframe
train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(train_df.head())
print(train_df.tail())

df = train_df.loc[:, ['video_name', 'tag']]
df
df.to_csv('train.csv')

dataset_path = os.listdir(test_datapath)
print(dataset_path)

room_types = os.listdir(test_datapath)
print("Types of Fitness Movement: ", len(dataset_path))

rooms = []

for item in dataset_path:
    # Get all the file names
    all_rooms = os.listdir(test_datapath + '\\' + item)

    # Add them to the list
    for room in all_rooms:
        rooms.append((item, str(test_datapath + '\\' + item) + '\\' + room))

# Build a dataframe
test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
print(test_df.head())
print(test_df.tail())

df = test_df.loc[:, ['video_name', 'tag']]
df
df.to_csv('test.csv')