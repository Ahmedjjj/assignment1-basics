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

