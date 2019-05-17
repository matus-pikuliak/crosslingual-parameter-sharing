# Processing UD data into POS file format
import codecs
import os
import sys


def process(source_files):

    if not isinstance(source_files, list):
        source_files = [source_files]

    for file_path in source_files:
        with codecs.open(file_path, encoding='utf-8') as f:

            for line in f:
                if line.startswith("#"):
                    continue
                elif not line.strip():
                    print()
                else:
                    tokens = line.split('\t')
                    if tokens[0].isdigit():
                        print("%s\t%s\t%s" % (tokens[1], tokens[7].split(':')[0], tokens[6]))


if __name__ == '__main__':
    process(sys.argv[1])
