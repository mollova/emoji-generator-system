import emoji
import csv
import re

from typing import List

dataset_emoji_mapping = {
    "datasets/raw-data/face_holding_back_tears.csv": '🥹',
    "datasets/raw-data/face_with_steam_from_nose.csv": '😤',
    "datasets/raw-data/face_with_tears_of_joy.csv": '😂',
    "datasets/raw-data/loudly_crying_face.csv": '😭',
    "datasets/raw-data/smiling_face_with_heart-eyes.csv": '😍'
}

def sanitize_file_by_emoji(file_content: List[str], target_emoji: str) -> List[str]:
    filtered = []

    for row in file_content:
        row_str = str(row)
        dist = emoji.distinct_emoji_list(row_str)
        if len(dist) == 1 and target_emoji in dist:       
            result = re.sub(target_emoji,'',row_str) + target_emoji
            filtered.append(result)

    return filtered

for filename in dataset_emoji_mapping:
    #read file
    file = open(filename)
    csvreader2 = csv.reader(file)

    header = []
    header = next(csvreader2)

    rows = []
    for row in csvreader2:
        rows.append(row)


    # get the target emoji
    target_emoji = dataset_emoji_mapping.get(filename)

    # filter rows with only the target emoji
    filtered = sanitize_file_by_emoji(rows, target_emoji)

    print(filtered[0][-1])
    print(len(filtered))
