# Validation of the Gaussian Jigsaw/CTC integration

Validated in this workspace with Python 3.12 and PyTorch 2.14.0+cpu.
The code targets Python 3.10+ / PyTorch 2.1+; the user's CUDA environment was not available.

- `python -m unittest discover -s tests -v`: 8 tests passed. The training test
  loops over all 16 arm/mode/noise combinations and includes both frozen stages.
- Checked actual encoder updates, frozen encoder/head invariance, decoder
  updates, lag-head gradients, and CTC gradients through noisy embeddings.
- Checked Gaussian standard deviation, zero-noise identity and no padding noise.
- Dense sequence encoding equals the supplied window encoder, with independent
  trial boundaries. Packed bidirectional decoding is invariant to batch padding.
- Repeated-character CTC alignment failures raise a clear error.
- Reconstruction targets stay clean and spans stay inside individual trials.
- Checkpoint reload reproduces the exact next CPU update, including RNG/data order.
- Source order-head resets clear the intended optimizer state; CER aggregates
  edit counts and target characters correctly.
- End-to-end CLI smoke run used synthetic `.mat` files in the original CORP
  directory layout and the original `get_input` loader. All four `order_full`
  cells completed training, validation, checkpoint writes, evaluation dispatch,
  JSON/CSV output and the four-column CER table.
- Completed frozen-checkpoint CLI resume was checked separately.
- Python syntax compilation and `bash -n submit_nlp21_jigsaw.sh` passed.

The synthetic smoke run used small dimensions and two updates per stage. It is
an integration check, not evidence of decoding quality. No CORP data, CUDA/H200
training, full 1024-hidden GRU run or RunAI submission was available/performed.
CPU backend warnings about unavailable processor metadata did not prevent the tests.

Legacy CEBRA paths were preserved, but were not tested; new entrypoints avoid importing CEBRA.
