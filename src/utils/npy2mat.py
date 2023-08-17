#!/usr/bin/env python3

import argparse
import numpy as np
import os
import scipy.io

'''
Program to convert npy files in a given array to mat format.
'''
def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.path != False:
        if not os.path.isdir(args.path):
            print('Given path is not a dir: ' + args.path)
            print('Exiting...')
            return

    RESULT_DIR = 'mat'
    files = os.listdir(args.path)
    print('Found {} files at {}'.format(len(files), args.path))
    if args.path != False: # prepend file paths to load
        files = [ os.path.join(args.path, f) for f in files ]
        RESULT_DIR = os.path.join(args.path, RESULT_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for i, f in enumerate(files):
        print(f'{i+1}/{len(files)}')
        print('Loading file: {}'.format(f))
        a = np.load(f)
        fname = f.split('/')[-1]
        fname = fname[:-3] + 'mat' # replace npy extension with mat
        fname = os.path.join(RESULT_DIR, fname)
        print('Saving file: {}'.format(fname))
        scipy.io.savemat(fname, dict(x=a))

    print('Complete!')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=False)

    return parser

if __name__ == '__main__':
    main()
