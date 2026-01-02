# `unicode1`
## a
`\x00`
## b
Printing this character does not display anything.
## c 
It is ignored (replaced by '').
# `unicode2`
## a
UTF-8 tends to produce shorter byte sequences for common characters as it uses variable width encoding.
## b
input: "الكرام" throws an error.
Function assumes each bytes decodes to 1 character. But utf-8 uses more than 1 byte for non-ascii characters
## c
\x80\x11
\x80 is a continuation byte, a valid utf-8 string cannot start with it
# `train_bpe_tinystories`
## a
- Training time: 65s
- Memory: 8 GB
- Longest token in vocab (in bytes):  ' accomplishment'. This makes sense.
## b
On an M3 Macbook Pro with 12 CPUs, most time is spent on the parallel counting of the tokens (30s) following by running all the merges (29s).
# `train_bpe_expts_owt`
## a
- Longest token in vocab (in bytes): '————————————————'. This token makes sense as it occurs often in the owt dataset.
## b
- There is a large overlap between learned vocabulary. Specifically, out of the 10000 vocabulary words of tiny stories, 7329 were learned on owt as well.
- The remaining non-overlapping words comprise a lot of proper nouns, which seem to occur often if the tiny stories, as well as other "story-themed" keywords.
- Owt tokenizer is richer and has longer tokens (expected).
# `tokenizer_experiments`
## a
- Tiny Stories tokenizer compression ratio: 4.08
- OWT tokenizer compression ratio: 4.32
## b
- When tokenizing the tiny stories sample with owt the compression ratio: 3.99 
- When tokenizing the owt sample with tiny stories the compression ratio: 3.08 
The owt tokenizer has a much larger vocab size, which explains that it handles the tiny stories well.
In contrast, the tiny stories tokenizer has much worse performance on owt as it doesn't generalize well to unseen text.
## c
Estimated throughput of the owt tokenizer: 4003146 B/s
It would take  approximatively 2.5 days to tokenize the Pile dataset with the owt tokenizer.
## d
`uint16` is an appropriate size as the vocab sizes are at least 10000, which don't fit in `uint8` but do in `uint16`.
The integers are all un-signed therefore an unsigned type is well suited.