from pathlib import Path

directory = '/Volumes/Harshas ssd/development/ieee-ml-hack-data/archive/test'
mp4_count = len(list(Path(directory).glob('*.mp4')))
print(f"Number of MP4 files: {mp4_count}")