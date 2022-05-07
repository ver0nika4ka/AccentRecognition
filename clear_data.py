import constants
import shutil
from pathlib import Path

if __name__ == '__main__':
    metadata_file_path = Path(constants.METADATA_FILE_NAME)
    audio_directory_path = Path(constants.AUDIO_PATH[:-3])

    if Path.exists(metadata_file_path):
        Path.unlink(metadata_file_path)

    if Path.exists(audio_directory_path):
        shutil.rmtree(audio_directory_path)
