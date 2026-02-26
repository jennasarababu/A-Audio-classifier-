import os
from utils.transformations import create_nightcore

ORIGINAL = "dataset/gtzan_original"
NIGHTCORE = "dataset/gtzan_nightcore"

for genre in os.listdir(ORIGINAL):

    os.makedirs(os.path.join(NIGHTCORE, genre), exist_ok=True)

    for file in os.listdir(os.path.join(ORIGINAL, genre)):

        input_path = os.path.join(ORIGINAL, genre, file)
        output_path = os.path.join(NIGHTCORE, genre, file)

        try:
            create_nightcore(input_path, output_path)
            print(f"Converted: {file}")
        except Exception as e:
            print(f"Error: {file} -> {e}")##file is corrupted
