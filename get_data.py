import json
import os
from pathlib import Path
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup
from loguru import logger

import constants
from sample import Sample

samples_references = []
file_location = ''


def get_samples_references():
    global samples_references

    language_url = constants.ROOT_URL + constants.BROWSE_URL.format(current_language)
    logger.debug(language_url)
    soup = BeautifulSoup(urlopen(language_url), 'lxml')
    references = soup.find_all('p')
    samples_references = []
    for reference in references:
        a_hyperlink = reference.find_all('a')
        for r in a_hyperlink:
            samples_references.append(r)  # .text.replace(',', '')
    logger.debug(f'[{samples_references}]')


def create_audio_directory():
    Path(constants.AUDIO_PATH.format(current_language)).mkdir(parents=True, exist_ok=True)


def retrieve_audio(sample_reference):
    global file_location

    create_audio_directory()

    sample_reference_text = sample_reference.text.replace(',', '')
    file_name = constants.SAMPLE_FILE_NAME.format(sample_reference_text)
    file_location = constants.AUDIO_PATH.format(current_language) + '/' + file_name
    file_path = Path(file_location)

    if Path.exists(file_path):
        return

    logger.debug(constants.ROOT_URL + constants.AUDIO_URL.format(sample_reference_text))
    request = requests.get(constants.ROOT_URL + constants.AUDIO_URL.format(sample_reference_text), stream=True)
    if request.ok:
        logger.info('Saving to {}'.format(Path.absolute(file_path)))
        with open(file_path, 'wb') as f:
            for chunk in request.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

        retrieve_sample_metadata(sample_reference)
    else:
        logger.error("Download failed: status code {}\n{}".format(request.status_code, request.text))


def retrieve_sample_metadata(sample_reference):
    sample_url = constants.ROOT_URL + sample_reference.attrs['href']
    soup = BeautifulSoup(urlopen(sample_url), 'lxml')
    sample_bio = soup.find('ul', {'class': 'bio'})
    sample = Sample(file_location=file_location)
    metadata_rows = sample_bio.find_all('li')

    for row in metadata_rows:
        row_key = row.find('em').text.rstrip().replace(' ', '_')
        row_value = row.text
        for c in '(),:':
            row_key = row_key.replace(c, '')
        sample.set_field(None, row_key, row_value)

    if not Path.exists(Path(constants.METADATA_FILE_NAME)):
        new_metadata_file = json.dumps({current_language: []})
        with open(constants.METADATA_FILE_NAME, 'w') as f:
            f.write(new_metadata_file)
            f.close()

    with open(constants.METADATA_FILE_NAME) as metadata_file:
        metadata = json.load(metadata_file)

    if current_language not in metadata:    # start array for lang if it has not been started before
        metadata[current_language] = []

    sample_json = dict(sample.to_struct())
    metadata[current_language].append(sample_json)

    with open(constants.METADATA_FILE_NAME, 'w') as metadata_file:
        json.dump(metadata, metadata_file)


if __name__ == '__main__':
    # TODO: Don't duplicate metadata information if the files were downloaded previously
    for current_language in constants.LANGUAGES.values():
        get_samples_references()

        logger.debug(f'num of samples {len(samples_references)} for lang {current_language}')
        for reference in samples_references:
            retrieve_audio(reference)
