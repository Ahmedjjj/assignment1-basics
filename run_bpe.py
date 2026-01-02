import argparse
import logging
import os
import shutil
import tracemalloc
from collections.abc import Iterable

from cs336_basics import configure_logging, find_chunk_boundaries, train_bpe
from cs336_basics._utils import read_in_chunks
from settings import settings

logger = logging.getLogger("cs336_basics")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--vocab-size", "-v", type=int, required=True)
    parser.add_argument("--split-token", type=str, required=True)
    parser.add_argument("--special-tokens", type=str, nargs="*", default=[])
    parser.add_argument("--num-chunks", type=int, required=False, default=settings.bpe_count_max_workers)
    return parser.parse_args()


def _get_results_folder(input_path: str, vocab_size: int):
    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    path = os.path.join(settings.results_folder / f"{input_filename}_{vocab_size}")

    seq_num = 1

    while True:
        if os.path.exists(path):
            if settings.overwrite_results:
                shutil.rmtree(path)
                os.makedirs(path)
                return path

            path = os.path.join(settings.results_folder / f"{input_filename}_{vocab_size}_{seq_num}")
            seq_num += 1
        else:
            os.makedirs(path)
            return path


def main():
    if settings.track_memory:
        tracemalloc.start()

    args = _parse_args()

    results_folder = _get_results_folder(args.input, args.vocab_size)
    configure_logging(logger, file=os.path.join(results_folder, "log.txt"))

    logger.info("Input corpus %s", args.input)
    logger.info("Target vocab size %i", args.vocab_size)
    logger.info("Special tokens %s", args.special_tokens)
    logger.info("Results folder %s", results_folder)
    logger.info("Splitting on special token %s", args.split_token)

    logger.info("Data num chunks %i", args.num_chunks)

    try:
        corpus = read_in_chunks(path=args.input, split_token=args.split_token, num_chunks=args.num_chunks)

        tokenizer = train_bpe(
            texts=corpus,
            special_tokens=args.special_tokens,
            vocab_size=args.vocab_size,
            ptoken_count_n_jobs=settings.bpe_count_max_workers,
        )

        tokenizer.save_to_folder(results_folder)
    finally:
        if settings.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            logger.info("Peak memory usage %.3f MB", peak / (1024 * 1024))
            tracemalloc.stop()


if __name__ == "__main__":
    main()
