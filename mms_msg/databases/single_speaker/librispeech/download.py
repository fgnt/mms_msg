import click

from pathlib import Path
from paderbox.io.download import download_file_list

url = 'http://www.openslr.org/resources/12/'

FILE_DICT = {
    'development':
        [url + package
         for package in [
             'dev-clean.tar.gz',
             'dev-other.tar.gz'
         ]
         ],
    'test':
        [url + package
         for package in [
             'test-clean.tar.gz',
             'test-other.tar.gz'
         ]
         ],
    'train':
        [url + package
         for package in [
             'train-clean-100.tar.gz',
             'train-clean-360.tar.gz',
             'train-other-500.tar.gz '
         ]
         ]
}


def check_files(datasets, database_path):
    """"
    checks if all download files are available

    Parameters:
        database_path (str): storage path
        datasets (list): datasets to be downloaded
    """
    file_list = list()

    for ds in datasets:
        assert ds in ['development', 'test', 'train'], 'Typo or dataset does not exit'
        for file in FILE_DICT[ds]:
            file_zip = file.split('/')[-1]
            file_name = file_zip.split('.')[0]
            if not Path(database_path + '/LibriSpeech/' + file_name + '/').exists():
                file_list.append(file)

    return file_list


@click.command()
@click.option('--datasets', default='development,test,train', help='String of datasets separated by ","')
@click.option('--database_path', default='LibriSpeech', help='Destination directory for the database')
def download(datasets, database_path):
    """
    Downloads the LibriSpeech Datasets from http://www.openslr.org/12 and save them in a local path

    Parameters:
        datasets (str): datasets to be downloaded, with seperator ','
        database_path (str): storage path

    Raises:
        IOError - Download failed
    """
    datasets = datasets.split(',')

    file_list = check_files(datasets, database_path)

    download_file_list(file_list, database_path, exist_ok=True)

    missing_files = check_files(datasets, database_path)

    assert not missing_files, 'The following files were not downloaded or extracted: \n' + str(missing_files)


if __name__ == '__main__':
    download()
