# How to run Kindi on each card

The installer ([INSTALL.md](INSTALL.md)) uses these settings. It selects them from the cards that it finds.

Blocking sync (`KINDI_BLOCKING_SYNC`, on by default) lets the host thread sleep while it waits for the card. It
costs 2-3% of the decode speed on the GTX 1080. With one server on its own card, set `KINDI_BLOCKING_SYNC=0`
(the installer does this). The RTX 3080 profile keeps it on: there, many servers share few CPU cores.

This page gives the settings for each tested card. It also gives the speeds that we measured with these settings.
All the cards use the same Docker image (CUDA 10.2, CuPy 9.6). The `kindi` folder of this repository is the code.

To make a pack, use `kindi convert` (see the README). Put the pack in `$HOME/.kindi/<pack>`. In the container, it
is at `/data/<pack>`. The examples serve on `127.0.0.1:8093`.

```
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
  -p 127.0.0.1:8093:8093 -v $HOME/.kindi:/data -e HOME=/tmp \
  -e KINDI_PACK=/data/pack -e KINDI_STATE_DIR=/data/state ... kindi:0.1.0
```

Replace `...` with the variables in the section for your card. To use one card of two, set `--gpus device=N`.

## Quadro K420, 1 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-2B |
| variables | `KINDI_CTX=24576` (the default) |
| generation | 22.9 tokens/s (fast head, 2,000 tokens of context); 20.8 tokens/s (exact head, 512 tokens) |
| prompt reading | 31.6 tokens/s at 3K tokens, 28.1 at 12K (one token at a time: 25.3 and 23.4; dev build of 10-21) |

- The prompt path uses chunks of 128 tokens on this card. Then the context stays at 24,576 tokens.
- To read prompts one token at a time, set `KINDI_CHUNK=0`.

## Quadro K420, 2 GB (Kepler, sm_30)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-4B (the installer's choice), or the 2B |
| variables | `KINDI_CTX=23808` for the 4B (the context that fits), `KINDI_CTX=65536` for the 2B |
| generation | 2B: 21.2 tokens/s (exact head), 24.8 (fast head); 4B: 9.8 tokens/s |

## Quadro K1200, 4 GB (Maxwell, sm_50)

| item | value |
|---|---|
| models | 2B (`/data/pack`), 4B (`/data/pack4b`) or 9B (`/data/pack9b`) |
| variables | `KINDI_CTX=24576` |
| generation (exact / fast head) | 2B 71.6 / 88.3 tokens/s, 4B 35.0 / 42.2, 9B 20.7 / 24.9 |
| prompt reading at 3K tokens | 2B 186.6 tokens/s, 4B 74.5, 9B 40.0 |

- The card in an x16 slot is faster than the card in an x4 slot.

## Two Quadro K1200 cards (tensor parallel)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (`/data/pack27b`) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CTX=24576`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=4096`, `KINDI_REPEAT_STOP=1` |
| docker | `--gpus all` (the engine uses both cards), `--memory 12g` |
| generation | 12.6 tokens/s |
| prompt reading | 20.9 tokens/s at 512 tokens, 18.4 at 22K (dev code of 10-21, the chunk-path copy) |

- The K/V cache is q8_0. The output matches llama.cpp with `-ctk q8_0 -ctv q8_0`.
- Set the host to text mode. A desktop session turns on the CUDA watchdog, and the watchdog stops the exchange
  kernels.

## Quadro K6000, 12 GB (Kepler, sm_35)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 20.4 tokens/s at 512 tokens, 16.4 at 8K, 13.1 at 24K |
| prompt reading | 36.2 tokens/s at 512 tokens, 30.2 at 8K, 26.2 at 24K |

- The card needs the NVIDIA 470 driver. The 470 driver does not report the compute capability, so the installer
  reads it from the card name.
- With all of the 12 GB, the engine fits a context of 156,672 tokens. The installer sets 65,536.
- Nemotron also runs on this card (experimental): 20.1 tokens/s of generation at 512 tokens. The installer does not
  offer it on Kepler cards.

## GeForce GTX TITAN X, 12 GB (Maxwell, sm_52)

Measured on Windows 10 with Docker Desktop (WSL2), driver 582.28, the card also driving the desktop, Kindi 0.3.0.

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B (`kindi verify` ALL PASS; the 2B also passes) |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536`, `KINDI_BLOCKING_SYNC=0` |
| generation | 22.7 tokens/s at 100 tokens, 21.1 at 8K, 18.8 at 24K, 17.3 at 32K, 14.0 at 60K (median of three `kindi bench` passes) |
| prompt reading | 68.5 tokens/s at 0 (a 256-token chunk), 52.8 at 8K, 49.2 at 16K, 43.3 at 32K |
| context | 119,040 tokens fit with `KINDI_MEM_RESERVE_MB=512` next to a desktop that takes ~1.8 GB; see below |

- The card runs the K1200 paths: 2-bit weights, fp32 chunk products (no dp4a below sm_61). `KINDI_G5=1` (5-trit
  weights) works and passes `verify` on this card, but generation is 18% slower (18.6 against 22.7 tokens/s), as on
  Kepler.
- **Leave about 2 GB of card memory free on Windows.** With the context fitted to 119,040 tokens (~220 MiB free), a
  server read new prompt text 2.8 times slower than with `KINDI_CTX=65536` (~2 GB free): 984 tokens in 52.4 s against
  18.9 s, and a 6,700-token opening at 28 against 56 tokens/s. The engine speeds above were measured with ~2 GB free.
  A monitor on another graphics adapter (for example the CPU's) frees the desktop's part of the card.
- Every start lays out the 2-bit weights again: about 4-6 minutes with the pack on a Windows drive.

## GeForce GTX 1080, 8 GB (Pascal, sm_61)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=65536`, `KINDI_HEAD=exact`, `KINDI_REPEAT_STOP=1`, `KINDI_MAX_TOKENS=65536` |
| generation | 28.4 tokens/s at a short context, 16.0 at 60K |
| prompt reading | 249 tokens/s at 512 tokens, 237 at 3K, 213 at 12K, 188 at 22K |

- The first start writes the 5-trit weights into the pack (`g5n_*.bin`, 5.6 GB). This takes about 12 minutes.
  Later starts take about 30 seconds.
- The prompt products use int8 activations on this card. For fp32, set `KINDI_CHUNK_Q8=0` (110 tokens/s at 512
  tokens).
- On a host with more than one GPU, set `CUDA_VISIBLE_DEVICES` to the GTX 1080.

## GeForce RTX 3080, 10 GB (Ampere, sm_86)

| item | value |
|---|---|
| model | Ternary-Bonsai-2-27B |
| variables | `KINDI_PACK=/data/pack27b`, `KINDI_CARDS=1`, `KINDI_CTX=131072`, `KINDI_MEM_RESERVE_MB=128`, `KINDI_HEAD=exact`, `KINDI_CHECKPOINT_BUDGET_MB=1024`, `KINDI_PARK_MAX=1`, `KINDI_REPEAT_STOP=1` |
| docker | `--memory 14g` (the first load needs it) |
| context | 110,592 tokens (the fit to the free memory) |
| generation | 47-49 tokens/s at a short context, 29 at 64K |
| prompt reading | 491 tokens/s at 512 tokens, 381 at 22K |

These figures are from kindi-rel, 10-21, not the 0.2.0 build.

Tensor cores and speculative decoding (sm_80 and later). This is the profile of the RTX 3080 cluster servers, and
the installer uses it:

| variable | value | |
|---|---|---|
| `KINDI_TC` | `1` | int8 tensor-core products and CUDA graphs for the decode |
| `KINDI_FD` | `1` | split flash-decode attention: the K/V is read once per query, in chunks of 512 positions |
| `KINDI_ATT_TC` | `1` | the attention of the prompt chunks on the tensor cores |
| `KINDI_SPEC` | `draft2b` | speculative decoding with the 2B as the draft model (also `ngram` or `both`). Off by default |
| `KINDI_SPEC_K` | `3` | the draft tokens per verify pass (default 4) |
| `KINDI_SPEC_PACK` | the 2B pack | for `draft2b` (for example `/data/pack2b`) |
| `KINDI_SPEC_WINDOW` | `8192` | the context of the draft model |
| `KINDI_TC_TMAX` | `4` | the most tokens in one verify pass |
| `KINDI_CKPT_VRAM` | `1` | a checkpoint is a copy in card memory; its copy to the host waits until the server is idle |
| `KINDI_MEM_RESERVE_MB` | `160` | the memory that the CUDA graphs need after the context fit |
| `KINDI_DENSE_SPACING` | `2048` | the spacing of the checkpoints in the opening of a conversation |
| `KINDI_CHECKPOINT_BUDGET_MB` | `1280` | the host memory for checkpoints and parked conversations (`1024` on the agent servers) |
| `KINDI_PARK_MAX` | `1` | parked conversations (`0` on the agent servers: one conversation each) |
| `KINDI_PREFIX_MAX` | `2` | saved openings on disk |

With this profile: context 84,992 tokens; at 190 W on agent prompts 99.9 tokens/s greedy and 91.3 sampled; prompt
reading 1,093 tokens/s at 512 to 855 at 22K.

- With `KINDI_TC=1`, the server reads prompts in passes of up to `KINDI_TC_TMAX` tokens.
- Speculative greedy output is the same as the output without speculation.
- The cluster runs seven servers on one 4-core host with `--memory 6g`. One server on a computer of its own can use
  the default budgets.

## CPU (x86-64 with AVX2)

| item | value |
|---|---|
| models | 2B, 4B or 9B |
| variables | `KINDI_DEVICE=cpu`, `OMP_NUM_THREADS=<cores of one socket>`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |

- The CPU engine needs gcc with OpenMP. It builds `libkcpu.so` at the first start (`KINDI_CPU_CACHE`).
- The output is token-exact with llama.cpp.
