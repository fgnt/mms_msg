import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import click

from mms_msg.databases.utils import check_audio_files_exist
from paderbox.io.audioread import audio_length
from paderbox.io import dump_json


def read_speakers(filepath):
    sub_dict = defaultdict(dict)
    for line in filepath.read_text().splitlines():
        if line.startswith(';'):
            continue
        speaker_id, gender, subset, _, _ = [
            segment.strip() for segment in line.split(' | ')]
        sub_dict[subset + '|' + speaker_id]['gender'] = \
            'male' if gender == 'M' else 'female'

    return sub_dict


def read_chapters(file_path, sub_dict):
    sub_dict_chapters = dict()
    for line in file_path.read_text().splitlines():
        if line.startswith(';'):
            continue
        chapter_id, speaker_id, _, subset, _, book_id, _, _ = [
            segment.strip() for segment in line.split(' | ')
        ]
        sub_id = subset + '|' + speaker_id + '|' + chapter_id
        sub_dict_chapters[sub_id] = {
            'speaker_id': speaker_id,
            'chapter_id': chapter_id,
            'gender': sub_dict[subset + '|' + speaker_id]['gender']
        }
    return sub_dict_chapters


def get_example_dict(in_tuple, transcription):
    audio_file, sub_dict, subset_id = in_tuple
    audio_id = audio_file.stem

    example_dict = {
        'audio_path': {"observation": str(audio_file)},
        'transcription': transcription[audio_id],
        'num_samples': audio_length(str(audio_file), unit='samples'),
    }

    example_dict.update(sub_dict)
    return subset_id, audio_id, example_dict


def get_transcription(segments, database_path, sub_dict):
    transcription = dict()
    del_list = list()
    for segment in segments:
        subset, speaker_id, chapter_id = segment.split('|')
        file_path = database_path / subset / speaker_id / chapter_id
        transcript_file = list(file_path.glob('*.txt'))
        if len(transcript_file) < 1:
            del_list.append(segment)
            continue
        else:
            assert len(transcript_file) == 1, transcript_file
        for line in transcript_file[0].read_text().splitlines():
            transcription[line.split()[0]] = ' '.join(line.split()[1:])
    for key in del_list:
        del sub_dict[key]

    return transcription


def get_audio_files(sub_dict, database_path, identifier):
    for segment, sub_dict in sub_dict.items():
        subset, speaker_id, chapter_id = segment.split('|')
        file_path = database_path / subset / speaker_id / chapter_id
        subset_id = subset.strip().replace('-', '_')
        for x in file_path.glob(f'*.{identifier}'):
            yield x, sub_dict, subset_id


def read_subset(database_path, sub_dict, wav):
    database = dict()
    examples = defaultdict(dict)
    transcription = get_transcription(sub_dict.keys(), database_path, sub_dict)
    identifier = 'wav' if wav else 'flac'
    audio_files = get_audio_files(sub_dict, database_path, identifier)
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for subset_id, example_id, example_dict in ex.map(
                partial(get_example_dict, transcription=transcription),
                audio_files
        ):
            examples[subset_id][example_id] = example_dict

    database['datasets'] = examples
    database['alias'] = {
        'train_960': ['train_clean_100', 'train_clean_360', 'train_other_500'],
        'train_460': ['train_clean_100', 'train_clean_360']}
    return database


def create_json(database_path, wav):
    sub_dict = read_speakers(database_path / 'SPEAKERS.TXT')
    sub_dict_chapters = read_chapters(database_path / 'CHAPTERS.TXT', sub_dict)
    database = read_subset(database_path, sub_dict_chapters, wav)
    return database


@click.command()
@click.option(
    '--json_path', '-j', default='librispeech.json',
    help=''
)
@click.option(
    '--data_dir', '-d',
    help='Path to the folder containing the LibriSpeech data',
)
@click.option(
    '--wav', default=False,
    help='Defines whether to look for wav or flac files. '
         'If True only wav files are written to the json '
         'otherwise only flac files are used',
    is_flag=True
)

def main(json_path, database_path, wav):

    database = create_json(Path(database_path).absolute(), wav)
    print('Check that all wav files in the json exists.')
    check_audio_files_exist(database, speedup='thread')
    print('Finished check. Write json to disk:')
    dump_json(Path(json_path), database,
              create_path=True,
              indent=4,
              ensure_ascii=False
              )


if __name__ == '__main__':
    main()
