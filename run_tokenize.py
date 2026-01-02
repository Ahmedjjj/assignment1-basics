import argparse
import logging
import os
import time

import numpy as np

from cs336_basics import BPETokenizer, configure_logging
from cs336_basics._utils import read_in_chunks
from settings import settings

logger = logging.getLogger("cs336_basics")


def _parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text")
    parser.add_argument("--tokenizer-path", "-t", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--split-token", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, required=False, default=settings.bpe_count_max_workers)
    return parser.parse_args()


def _get_filename(path: str) -> str:
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


def _get_results_folder(input_path: str, tokenizer_path: str) -> str:
    path = os.path.join(settings.results_folder / f"{_get_filename(input_path)}_{_get_filename(tokenizer_path)}")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def _get_file_size(path: str) -> int:
    return os.path.getsize(path)


def main():
    args = _parse_args()
    results_folder = _get_results_folder(args.input, args.tokenizer_path)

    configure_logging(logger, file=os.path.join(results_folder, "log.txt"))

    logger.info("Input corpus %s", args.input)
    logger.info("Tokenizer path %s", args.tokenizer_path)
    logger.info("Results folder %s", results_folder)
    logger.info("Splitting on special token %s", args.split_token)
    logger.info("Data num chunks %i", args.num_chunks)

    tokenizer = BPETokenizer.from_folder(args.tokenizer_path)
    corpus = read_in_chunks(path=args.input, split_token=args.split_token, num_chunks=args.num_chunks)

    start_time = time.perf_counter()
    tokens = list(tokenizer.encode_iterable(corpus))
    end_time = time.perf_counter()

    input_size = _get_file_size(args.input)
    total_time = end_time - start_time
    logger.info("Tokenization took %.8f seconds", total_time)
    logger.info("Tokenization throughput %.8f bytes/s", input_size / total_time)
    logger.info("Tokenization compression ratio %.8f", input_size / len(tokens))

    with open(os.path.join(results_folder, "res.npy"), "wb") as f:
        np.save(f, np.array(tokens, dtype="uint16"))


if __name__ == "__main__":
    main()
