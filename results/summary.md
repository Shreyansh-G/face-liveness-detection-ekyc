## Results

| Model | Fraud accepted (APCER) | Customers blocked (BPCER) | ACER | AUC | Blocked @ 1% fraud | Bank threshold: fraud / blocked | ms/face (PyTorch CPU) | Params |
|---|---|---|---|---|---|---|---|---|
| before | 51.37% | 6.76% | 29.07% | 0.8295 | 86.91% | 1.67% / 83.60% | 21.6 | 868,146 |
| baseline | 30.08% | 1.58% | 15.83% | 0.9556 | 52.66% | 30.42% / 1.51% | 22.3 | 868,146 |
| ours | 28.70% | 2.30% | 15.50% | 0.9496 | 62.01% | 27.82% / 2.23% | 21.7 | 868,146 |
| ours_no_fft | 28.58% | 2.30% | 15.44% | 0.9472 | 63.60% | 26.37% / 2.73% | 22.0 | 868,146 |
| ours_no_synthetic | 31.77% | 1.15% | 16.46% | 0.9533 | 53.09% | 28.85% / 2.09% | 21.8 | 868,146 |
| ours_no_degrade | 33.17% | 2.73% | 17.95% | 0.9250 | 76.55% | 28.90% / 3.96% | 21.7 | 868,146 |
| ours_ce_loss | 24.46% | 3.74% | 14.10% | 0.9509 | 57.63% | 21.30% / 5.04% | 20.9 | 868,146 |

## Impact per 1,000,000 checks (assumed: 1.0% fraud attempts, threshold 0.5, peak 50 checks/s, 9.4 ms/face)

| Model | Frauds let through | Genuine customers blocked | CPU hours | Workers at peak |
|---|---|---|---|---|
| before | 4,941 (+0) | 69,799 (+0) | 2.6 | 1 |
| baseline | 2,993 (-1,948) | 16,381 (-53,417) | 2.6 | 1 |
| ours | 2,552 (-2,390) | 32,050 (-37,748) | 2.6 | 1 |
| ours_no_fft | 2,620 (-2,321) | 30,626 (-39,173) | 2.6 | 1 |
| ours_no_synthetic | 3,005 (-1,936) | 17,806 (-51,993) | 2.6 | 1 |
| ours_no_degrade | 3,003 (-1,938) | 34,899 (-34,899) | 2.6 | 1 |
| ours_ce_loss | 2,436 (-2,505) | 37,036 (-32,763) | 2.6 | 1 |

## Deployment (CPU, both models per face)

| Model set | Format | ms/face | p95 ms | Size MB | ACER | AUC |
|---|---|---|---|---|---|---|
| original | pytorch | 22.91 | 25.08 | 3.71 | 28.23% | 0.8295 |
| original | onnx_fp32 | 9.60 | 10.19 | 3.49 | 28.23% | 0.8295 |
| original | onnx_int8 | 10.90 | 12.37 | 1.35 | 32.86% | 0.7571 |
| ours | pytorch | 23.25 | 24.98 | 3.81 | 14.38% | 0.9496 |
| ours | onnx_fp32 | 9.41 | 10.98 | 3.49 | 14.38% | 0.9496 |
| ours | onnx_int8 | 10.81 | 12.82 | 1.35 | 19.60% | 0.8837 |

## Early exit (2nd model only when the 1st is unsure)

| Model | Faces needing 2nd model | ACER with early exit | ACER always both | ms/face (est. vs always both) |
|---|---|---|---|---|
| before | 31.1% | 28.18% | 28.23% | 13.9 vs 21.6 |
| baseline | 15.2% | 16.04% | 15.79% | 12.4 vs 22.3 |
| ours | 29.7% | 14.51% | 14.38% | 13.6 vs 21.7 |
| ours_no_fft | 26.7% | 15.47% | 14.65% | 13.8 vs 22.0 |
| ours_no_synthetic | 14.0% | 16.01% | 15.93% | 12.1 vs 21.8 |
| ours_no_degrade | 36.1% | 16.80% | 16.78% | 14.7 vs 21.7 |
| ours_ce_loss | 31.2% | 14.13% | 14.05% | 13.4 vs 20.9 |
