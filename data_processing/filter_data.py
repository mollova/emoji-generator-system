import emoji
import csv
import re
import random

from typing import List

dataset_emoji_mapping = {
    "datasets/raw-data/face_holding_back_tears.csv": '🥹',
    "datasets/raw-data/face_with_steam_from_nose.csv": '😤',
    "datasets/raw-data/face_with_tears_of_joy.csv": '😂',
    "datasets/raw-data/loudly_crying_face.csv": '😭',
    "datasets/raw-data/smiling_face_with_heart-eyes.csv": '😍',
    # "datasets/raw-data/clown_face.csv": '🤡',
    # "datasets/raw-data/hot_face.csv": '🥵',
    # "datasets/raw-data/skull.csv": '💀',
    # "datasets/raw-data/thinking_face.csv": '🤔',
    # "datasets/raw-data/winking_face.csv": '😉'
}

def clean_row(text: str) -> str:
    # remove remove links, anonymize user mentions and remove quotes
    text = re.sub(r'[\[\]\"\']', '', text)
    text = re.sub(r'&gt', ' ', text)
    text = re.sub(r'&lt', ' ', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'&gt', ' ', text)
    text = re.sub(r'&lt', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    clean = ""
    for word in text.split(" "):
        if (word.startswith('@') or word.startswith('\'@')) and len(word) >= 1:
            pass
        elif word.startswith('http') or 'http' in word:
            pass
        elif word.startswith('https') or 'https' in word:
            pass
        else:
            clean += word + " "

    return clean.strip()

def sanitize_file_by_emoji(file_content: List[str], target_emoji: str) -> List[str]:
    filtered = []
    for row in file_content:
        row_str = str(row)
        dist = emoji.distinct_emoji_list(row_str)
        if len(dist) == 1 and target_emoji in dist:
            result = re.sub(target_emoji,'',row_str)
            filtered.append(clean_row(result) + ' ' + target_emoji)

    return filtered

def sanitize_file_by_emoji_v2(file_content: List[str], target_emoji: str) -> List[str]:
    filtered = []
    for row in file_content:
        row_str = str(row)
        dist = emoji.distinct_emoji_list(row_str)
        for emoji2 in dist:
            row_str = re.sub(emoji2,'',row_str)
        filtered.append(clean_row(row_str) + ' ' + target_emoji)

    return filtered

def collect_all_data(dataset_emoji_mapping: dict) -> List[str]:
    all_data = []
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

        all_data.extend(filtered)

    return all_data


lines = collect_all_data(dataset_emoji_mapping=dataset_emoji_mapping)
random.shuffle(lines)

with open('datasets/data/tweets_with_five_emojis.csv', 'w') as f:
    f.writelines(line + '\n' for line in lines)
