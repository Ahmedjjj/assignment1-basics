train_bpe_tinystories_valid_258:
	uv run run_bpe.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 258 --split-token "<|endoftext|>" --special-tokens "<|endoftext|>" 

train_bpe_tinystories_valid_10000:
	uv run run_bpe.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --split-token "<|endoftext|>" --special-tokens "<|endoftext|>"

train_bpe_tinystories_train_258:
	uv run run_bpe.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 258 --split-token "<|endoftext|>" --special-tokens "<|endoftext|>"

train_bpe_tinystories_train_10000:
	uv run run_bpe.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --split-token "<|endoftext|>"  --special-tokens "<|endoftext|>"

train_owt_valid_258:
	uv run run_bpe.py --input data/owt_valid.txt --vocab-size 258 --split-token "<|endoftext|>"  --special-tokens "<|endoftext|>"

train_owt_train_32000:
	uv run run_bpe.py --input data/owt_train.txt --vocab-size 32000 --split-token "<|endoftext|>"  --num-chunks 100 --special-tokens "<|endoftext|>" 

clear_resuls_tiny_valid:
	rm -rf results/TinyStoriesV2-GPT4-valid_*

clear_resuls_tiny_train:
	rm -rf results/TinyStoriesV2-GPT4-train_*

clear_results_tiny:
	rm -rf results/TinyStoriesV2-GPT4-*