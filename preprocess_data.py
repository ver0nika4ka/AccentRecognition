import json
import subprocess
import threading
from pathlib import Path

import pandas as pd
from loguru import logger
from pydub import AudioSegment
from unsilence import Unsilence

import constants
from constants import languages, AUDIOS_INFO_FILE_NAME

locations = dict()
samples = []
audios_info = []


def get_sample_references(language):
    global samples
    with open(constants.METADATA_FILE_NAME) as metadata_file:
        samples = json.load(metadata_file)[language]


def convert_to_wav(path, destination):
    if Path.exists(Path(destination)):
        return
    logger.info(f'Saving wav file to {destination}...')
    sound = AudioSegment.from_mp3(path)
    sound.export(destination, format='wav')
    logger.info(f'File saved')


def unsilence(path):
    destination = path.replace('.wav', '_unsilenced.wav')
    if Path.exists(Path(destination)):
        return
    u = Unsilence(path)
    u.detect_silence()
    u.render_media(destination, audio_only=True)
    logger.info(f'Audio [{path}] unsilenced')


def process_files(language):
    get_sample_references(language)
    paths = [sample['file_location'] for sample in samples]
    for path in paths:
        destination = path.replace('mp3', 'wav')
        convert_to_wav(path, destination)
        unsilence(destination)
        audios_info.append(
            [language, path.replace('mp3', 'wav'), path.replace('.mp3', '_unsilenced.wav')])  # mfccs_processed


class PreprocessingThread(threading.Thread):
    def __init__(self, language):
        threading.Thread.__init__(self)
        self.language = language

    def run(self):
        process_files(self.language)


def main():
    # TODO: Make sure that ffmpeg is installed
    # logger.info("Checking if ffmpeg is installed")
    # try:
    #     subprocess.call(['ls'])  # 'ffmpeg -version'
    # except:
    #     logger.error("FFMPEG is not found. Install before proceeding.")
    #     return -1

    for language in languages:
        process_files(language)

    audios_info_df = pd.DataFrame(audios_info, columns=['language', 'path', 'path_unsilenced'])
    audios_info_df.to_csv(AUDIOS_INFO_FILE_NAME, index=False)


if __name__ == '__main__':
    main()
