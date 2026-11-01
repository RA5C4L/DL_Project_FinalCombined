"""Prepare and persist the tiny IMDB subset so training can be run independently."""
import argparse
import numpy as np

from config import experiment_config
from data import (build_vocabulary, encode_texts, load_imdb, prepare_smoke_split, sequence_lengths,
                  save_vocabulary, set_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Prepare the required 20k/5k/25k experiment.")
    args = parser.parse_args()
    config = experiment_config(full=args.full)
    tag = "full" if args.full else "smoke"
    for directory in config.output_paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    print("Loading official IMDB data...", flush=True)
    train_texts, train_labels, test_texts, test_labels = load_imdb()
    indices = prepare_smoke_split(train_labels, test_labels, config.train_size, config.validation_size,
                                  config.test_size, config.seed, config.output_paths["splits"])
    train_subset = [train_texts[i] for i in indices["train"]]
    vocabulary = build_vocabulary(train_subset, config.vocab_size)
    save_vocabulary(vocabulary, config.output_paths["vocabulary"] / f"{tag}_vocab_30000.json")
    archive = config.output_paths["processed"] / f"{tag}_data_seed42.npz"
    train_x = encode_texts(train_subset, vocabulary, config.max_length)
    validation_x = encode_texts([train_texts[i] for i in indices["validation"]], vocabulary, config.max_length)
    test_x = encode_texts([test_texts[i] for i in indices["test"]], vocabulary, config.max_length)
    np.savez_compressed(
        archive,
        train_x=train_x,
        train_y=train_labels[indices["train"]],
        train_lengths=sequence_lengths(train_x),
        validation_x=validation_x,
        validation_y=train_labels[indices["validation"]],
        validation_lengths=sequence_lengths(validation_x),
        test_x=test_x,
        test_y=test_labels[indices["test"]],
        test_lengths=sequence_lengths(test_x),
        vocab_size=np.asarray([len(vocabulary)]),
    )
    print(f"Prepared {archive} with vocabulary size {len(vocabulary)}.", flush=True)


if __name__ == "__main__":
    main()
