import emoji
import csv
import re

file = open("datasets/raw-data/face_with_steam_from_nose.csv")
type(file)

csvreader2 = csv.reader(file)

header = []
header = next(csvreader2)

rows = []
for row in csvreader2:
    rows.append(row)


# Define the target emoji
target_emoji = '😤'

# Filter texts with only the target emoji

filtered = []
for row in rows:
   row_str = str(row)
   dist = emoji.distinct_emoji_list(row_str)
   if len(dist) == 1 and target_emoji in dist:       
       result = re.sub(target_emoji,'',row_str) + target_emoji
       filtered.append(result)

print(len(filtered))
