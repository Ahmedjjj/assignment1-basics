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
# `transformer_accounting`
- Embedding:
    - #Params = vocab_size * d_model
    - #MM-Flops = 0
- Transformer Block:
    - Input RMSNorm:
        - #Params: d_model
        - #MM-Flops: 0
    - Multi-head Self Attention
        - #Params:
            - Q, K, V projections: 3 d_model^2
            - Output projection: d_model^2
            - RoPE: 0
            - Total: 4 d_model^2
        - #MM-Flops:
            - Q, K, V Projection: 6 * seq_len * d_model^2
            - RoPE: 0
            - Attention: 4 * d_model * seq_len^2 
            - Output Projection: 2 * seq_len * d_model^2
            - Total: 8 seq_len d_model^2 + 4 d_model seq_len^2
    - FFN RMSNorm:
        - #Params: d_model
        - #MM-Flops: 0
    - FFN:
        - #Params: 3 * d_model * d_ff 
        - ##MM-Flops: 6 * seq_len * d_model * d_ff
    - #Params: num_layers * (2 d_model + 3 * d_model * d_ff + 4 d_model^2)
    - #MM-Flops: num_layers * (8 seq_len d_model^2 + 4 d_model seq_len^2 + 6 * seq_len * d_model * d_ff)
- Final FF Layer:
    - RMSNorm:
        - #Params: d_model
        - #MM-Flops: 0
    - FFN: 
        - #Params: d_model * vocab_size
        - #MM-Flops: 2 * seq_len * vocab_size * d_model
- #Params: num_layers * (2 d_model + 3 * d_model * d_ff + 4 d_model^2) + d_model + 2 d_model * vocab_size
- #MM-Flops: num_layers * (8 seq_len d_model^2 + 4 d_model seq_len^2 + 6 * seq_len * d_model * d_ff) + 2 * seq_len * vocab_size * d_model
# a
============================================================
PARAMETER COUNT BREAKDOWN
============================================================
GPT2-XL: d_model=1600, d_ff=6400, num_layers=48, vocab_size=50257
------------------------------------------------------------
Embedding:                    80.41M (80,411,200)

Per Transformer Block:
  Input RMSNorm:               1.60K (1,600)
  Attention:                  10.24M (10,240,000)
  FFN RMSNorm:                 1.60K (1,600)
  SwiGLU FFN:                 30.72M (30,720,000)
  Block Total:                40.96M (40,963,200)

All 48 Blocks:                  1.97B (1,966,233,600)
Final Layer:                  80.41M (80,412,800)
------------------------------------------------------------
TOTAL PARAMETERS:              2.13B (2,127,057,600)
============================================================
# b
======================================================================
MATRIX-MULTIPLY FLOP BREAKDOWN (Forward Pass)
======================================================================
GPT-2 XL: d_model=1600, d_ff=6400, num_layers=48, seq_len=1024
----------------------------------------------------------------------
Embedding:                         0    0.0%

Per Transformer Block:
  Input RMSNorm:                   0    0.0%
  Attention:                  27.68B    0.6%  (27,682,406,400)
  FFN RMSNorm:                     0    0.0%
  SwiGLU FFN:                 62.91B    1.4%  (62,914,560,000)
  Block Total:                90.60B    2.0%  (90,596,966,400)

All 48 Blocks:                  4.35T   96.4%
  - Total Attention:           1.33T   29.4%
  - Total FFN:                 3.02T   66.9%
Final Layer:                 164.68B    3.6%  (164,682,137,600)
----------------------------------------------------------------------
TOTAL FLOPS:                   4.51T  100.0%  (4,513,336,524,800)
======================================================================

FLOPs per parameter: 2121.87
Approximate 6*N*D rule: 6 * 2.13B * 1024 = 13.07T

# c
In this setting, the swiglu FFN takes the most flops, as the hidden dimension is large compared to the sequence length.
# d
======================================================================
MATRIX-MULTIPLY FLOP BREAKDOWN (Forward Pass)
======================================================================
GPT-2 Large: d_model=1280, d_ff=5120, num_layers=36, seq_len=1024
----------------------------------------------------------------------
Embedding:                         0    0.0%

Per Transformer Block:
  Input RMSNorm:                   0    0.0%
  Attention:                  18.79B    0.8%  (18,790,481,920)
  FFN RMSNorm:                     0    0.0%
  SwiGLU FFN:                 40.27B    1.8%  (40,265,318,400)
  Block Total:                59.06B    2.6%  (59,055,800,320)

All 36 Blocks:                  2.13T   94.2%
  - Total Attention:         676.46B   30.0%
  - Total FFN:                 1.45T   64.2%
Final Layer:                 131.75B    5.8%  (131,745,710,080)
----------------------------------------------------------------------
TOTAL FLOPS:                   2.26T  100.0%  (2,257,754,521,600)
======================================================================
======================================================================
MATRIX-MULTIPLY FLOP BREAKDOWN (Forward Pass)
======================================================================
GPT-2 Medium: d_model=1024, d_ff=4096, num_layers=24, seq_len=1024
----------------------------------------------------------------------
Embedding:                         0    0.0%

Per Transformer Block:
  Input RMSNorm:                   0    0.0%
  Attention:                  12.88B    1.2%  (12,884,901,888)
  FFN RMSNorm:                     0    0.0%
  SwiGLU FFN:                 25.77B    2.5%  (25,769,803,776)
  Block Total:                38.65B    3.7%  (38,654,705,664)

All 24 Blocks:                927.71B   89.8%
  - Total Attention:         309.24B   29.9%
  - Total FFN:               618.48B   59.9%
Final Layer:                 105.40B   10.2%  (105,396,568,064)
----------------------------------------------------------------------
TOTAL FLOPS:                   1.03T  100.0%  (1,033,109,504,000)
======================================================================
MATRIX-MULTIPLY FLOP BREAKDOWN (Forward Pass)
======================================================================
GPT-2 Small: d_model=768, d_ff=3072, num_layers=12, seq_len=1024
----------------------------------------------------------------------
Embedding:                         0    0.0%

Per Transformer Block:
  Input RMSNorm:                   0    0.0%
  Attention:                   8.05B    2.3%  (8,053,063,680)
  FFN RMSNorm:                     0    0.0%
  SwiGLU FFN:                 14.50B    4.1%  (14,495,514,624)
  Block Total:                22.55B    6.4%  (22,548,578,304)

All 12 Blocks:                270.58B   77.4%
  - Total Attention:          96.64B   27.6%
  - Total FFN:               173.95B   49.8%
Final Layer:                  79.05B   22.6%  (79,047,426,048)
----------------------------------------------------------------------
TOTAL FLOPS:                 349.63B  100.0%  (349,630,365,696)
======================================================================
- As the size increases, the final layer tends to take less % of the total flops. While attention stays almost constant, the FFN in each block grows in terms of the % of total flops.
# e
======================================================================
MATRIX-MULTIPLY FLOP BREAKDOWN (Forward Pass)
======================================================================
GPT-2 XL: d_model=1600, d_ff=6400, num_layers=48, seq_len=16384
----------------------------------------------------------------------
Embedding:                         0    0.0%

Per Transformer Block:
  Input RMSNorm:                   0    0.0%
  Attention:                   2.05T    1.4%  (2,053,531,238,400)
  FFN RMSNorm:                     0    0.0%
  SwiGLU FFN:                  1.01T    0.7%  (1,006,632,960,000)
  Block Total:                 3.06T    2.0%  (3,060,164,198,400)

All 48 Blocks:                146.89T   98.2%
  - Total Attention:          98.57T   65.9%
  - Total FFN:                48.32T   32.3%
Final Layer:                   2.63T    1.8%  (2,634,914,201,600)
----------------------------------------------------------------------
TOTAL FLOPS:                 149.52T  100.0%  (149,522,795,724,800)
======================================================================
Attention will take-over as the biggest % of FLOPs, this is normal as attention is quadratic in the sequence length.
# `learning_rate_tuning`
1, 10: Learning is very slow
100: learning is good and fast
1000: diverges