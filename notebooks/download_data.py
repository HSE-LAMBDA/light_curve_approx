#!/usr/bin/env python

import argparse
import multiprocessing
import os

import requests
import json
import hashlib
from itertools import repeat


DEST_PATH = '../data/plasticc'
URL_API = 'https://zenodo.org/api/records/2539456'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--nproc', type=int, help='number of parallel processes')
    args = parser.parse_args()
    return args


def download(description, destdir, session=requests):
    # zenodo_get maybe a better solution in general:
    #   https://gitlab.com/dvolgyes/zenodo_get/
    # but we'd like to practice a bit...

    url = description['links']['self']

    dest = os.path.join(destdir, description['key'])
    expected_size = description['size']

    # Download resuming is not possible at the moment.
    # Zenodo doesn't support it. So let's just test the size for equality.
    # See: https://github.com/zenodo/zenodo/issues/1599
    if not os.path.exists(dest) or expected_size != os.stat(dest).st_size:
        with session.get(url, stream=True) as response:
            print('Downloading {}'.format(dest))
            with open(dest, 'wb') as fh:
                for chunk in response:
                    fh.write(chunk)

    if not description['checksum'].startswith('md5:'):
        print('Skipping checksum of {} due to unknown hashing algorithm.'.format(dest))
        return

    expected_md5 = description['checksum'][4:]
    actual_md5 = md5sum(dest)
    if expected_md5 == actual_md5:
        print('Checksum test for {} PASSED.'.format(dest))
    else:
        raise RuntimeError('Checksum test for {} FAILED.'.format(dest))


def _download_stared(args):
    """Reason: python before 3.3 didn't have multiprocessing.Pool.starmap"""
    return download(*args)


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            md5.update(block)

    return md5.hexdigest()


def main():
    args = parse_args()

    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)

    r = requests.get(URL_API)
    if not r.ok:
        raise RuntimeError('Failed to download metadata from Zenodo.')

    jsons = json.loads(r.text)
    files = jsons['files']

    with multiprocessing.Pool(processes=args.nproc) as pool:
        pool.map(_download_stared, zip(files, repeat(DEST_PATH)))


if __name__ == '__main__':
    main()
