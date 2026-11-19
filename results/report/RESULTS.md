# Results

## summary

| method                   |   n_seeds |   auroc |   auroc_sd | auroc_ci       |   auprc |   auprc_sd | auprc_ci       |     f1 |   f1_sd | f1_ci          |    mcc |   mcc_sd | mcc_ci         |
|:-------------------------|----------:|--------:|-----------:|:---------------|--------:|-----------:|:---------------|-------:|--------:|:---------------|-------:|---------:|:---------------|
| cadd                     |         1 |  0.9966 |          0 | [0.995, 0.998] |  0.9582 |          0 | [0.931, 0.979] | 0.922  |       0 | [0.902, 0.941] | 0.9126 |        0 | [0.891, 0.935] |
| nt_v2_250m_finetune      |         1 |  0.9659 |          0 | [0.957, 0.974] |  0.8601 |          0 | [0.835, 0.885] | 0.7729 |       0 | [0.742, 0.802] | 0.7453 |        0 | [0.713, 0.778] |
| nt_v2_250m_frozen        |         1 |  0.8599 |          0 | [0.839, 0.879] |  0.564  |          0 | [0.517, 0.612] | 0.5232 |       0 | [0.484, 0.560] | 0.4624 |        0 | [0.419, 0.502] |
| dnabert2_finetune        |         1 |  0.8452 |          0 | [0.823, 0.867] |  0.6482 |          0 | [0.607, 0.689] | 0.6131 |       0 | [0.573, 0.655] | 0.5821 |        0 | [0.540, 0.626] |
| dnabert2_frozen          |         1 |  0.7361 |          0 | [0.711, 0.764] |  0.3619 |          0 | [0.319, 0.411] | 0.3609 |       0 | [0.327, 0.397] | 0.2735 |        0 | [0.234, 0.313] |
| hyenadna_medium_finetune |         1 |  0.6141 |          0 | [0.585, 0.643] |  0.189  |          0 | [0.162, 0.220] | 0.2327 |       0 | [0.208, 0.256] | 0.1063 |        0 | [0.074, 0.137] |
| dnabert2_zeroshot        |         1 |  0.5637 |          0 | [0.530, 0.596] |  0.1381 |          0 | [0.118, 0.165] | 0.234  |       0 | [0.208, 0.262] | 0.0771 |        0 | [0.040, 0.114] |
| hyenadna_medium_frozen   |         1 |  0.5618 |          0 | [0.532, 0.590] |  0.1316 |          0 | [0.116, 0.153] | 0.2025 |       0 | [0.187, 0.219] | 0.0569 |        0 | [0.032, 0.080] |
| nt_v2_250m_zeroshot      |         1 |  0.5422 |          0 | [0.512, 0.573] |  0.1332 |          0 | [0.114, 0.158] | 0.2018 |       0 | [0.183, 0.222] | 0.046  |        0 | [0.015, 0.078] |

## delong_vs_cadd

| method                   | vs   |    n |   auc1 |   auc2 |    diff |        z |   p_value |
|:-------------------------|:-----|-----:|-------:|-------:|--------:|---------:|----------:|
| dnabert2_finetune        | cadd | 3995 | 0.846  | 0.9966 | -0.1506 | -13.6423 |         0 |
| dnabert2_frozen          | cadd | 3995 | 0.736  | 0.9966 | -0.2606 | -19.2311 |         0 |
| dnabert2_zeroshot        | cadd | 2255 | 0.5643 | 0.9967 | -0.4324 | -24.7831 |         0 |
| hyenadna_medium_finetune | cadd | 3995 | 0.6149 | 0.9966 | -0.3817 | -25.0836 |         0 |
| hyenadna_medium_frozen   | cadd | 3995 | 0.5614 | 0.9966 | -0.4352 | -30.2643 |         0 |
| nt_v2_250m_finetune      | cadd | 3995 | 0.9663 | 0.9966 | -0.0303 |  -6.5687 |         0 |
| nt_v2_250m_frozen        | cadd | 3995 | 0.8607 | 0.9966 | -0.1359 | -13.8379 |         0 |
| nt_v2_250m_zeroshot      | cadd | 3995 | 0.5433 | 0.9966 | -0.4533 | -31.1897 |         0 |

## by_consequence

| method                   | group      |    n |   pos_frac |   auroc |
|:-------------------------|:-----------|-----:|-----------:|--------:|
| cadd                     | intronic   | 1241 |     0.0056 |  0.9979 |
| cadd                     | missense   |  558 |     0.2796 |  0.9618 |
| cadd                     | noncoding  |  341 |     0.0029 |  1      |
| cadd                     | nonsense   |  165 |     0.9697 |  0.4994 |
| cadd                     | synonymous | 1459 |     0.0007 |  0.9986 |
| dnabert2_finetune        | intronic   | 1241 |     0.0056 |  0.7728 |
| dnabert2_finetune        | missense   |  558 |     0.2796 |  0.6898 |
| dnabert2_finetune        | noncoding  |  341 |     0.0029 |  0.5147 |
| dnabert2_finetune        | nonsense   |  165 |     0.9697 |  0.5425 |
| dnabert2_finetune        | synonymous | 1459 |     0.0007 |  1      |
| dnabert2_frozen          | intronic   | 1241 |     0.0056 |  0.6961 |
| dnabert2_frozen          | missense   |  558 |     0.2796 |  0.6826 |
| dnabert2_frozen          | noncoding  |  341 |     0.0029 |  0.7206 |
| dnabert2_frozen          | nonsense   |  165 |     0.9697 |  0.415  |
| dnabert2_frozen          | synonymous | 1459 |     0.0007 |  0.9822 |
| dnabert2_zeroshot        | intronic   |  695 |     0.0058 |  0.4931 |
| dnabert2_zeroshot        | missense   |  313 |     0.2843 |  0.5628 |
| dnabert2_zeroshot        | noncoding  |  190 |     0.0053 |  0.5873 |
| dnabert2_zeroshot        | nonsense   |  114 |     0.9825 |  0.4018 |
| dnabert2_zeroshot        | synonymous |  810 |     0.0012 |  0.7651 |
| hyenadna_medium_finetune | intronic   | 1241 |     0.0056 |  0.479  |
| hyenadna_medium_finetune | missense   |  558 |     0.2796 |  0.5948 |
| hyenadna_medium_finetune | noncoding  |  341 |     0.0029 |  0.9235 |
| hyenadna_medium_finetune | nonsense   |  165 |     0.9697 |  0.4838 |
| hyenadna_medium_finetune | synonymous | 1459 |     0.0007 |  0.2215 |
| hyenadna_medium_frozen   | intronic   | 1241 |     0.0056 |  0.4406 |
| hyenadna_medium_frozen   | missense   |  558 |     0.2796 |  0.594  |
| hyenadna_medium_frozen   | noncoding  |  341 |     0.0029 |  0.3147 |
| hyenadna_medium_frozen   | nonsense   |  165 |     0.9697 |  0.515  |
| hyenadna_medium_frozen   | synonymous | 1459 |     0.0007 |  0.4129 |
| nt_v2_250m_finetune      | intronic   | 1241 |     0.0056 |  0.9514 |
| nt_v2_250m_finetune      | missense   |  558 |     0.2796 |  0.7499 |
| nt_v2_250m_finetune      | noncoding  |  341 |     0.0029 |  0.9882 |
| nt_v2_250m_finetune      | nonsense   |  165 |     0.9697 |  0.785  |
| nt_v2_250m_finetune      | synonymous | 1459 |     0.0007 |  0.9979 |
| nt_v2_250m_frozen        | intronic   | 1241 |     0.0056 |  0.7142 |
| nt_v2_250m_frozen        | missense   |  558 |     0.2796 |  0.6426 |
| nt_v2_250m_frozen        | noncoding  |  341 |     0.0029 |  0.1853 |
| nt_v2_250m_frozen        | nonsense   |  165 |     0.9697 |  0.5775 |
| nt_v2_250m_frozen        | synonymous | 1459 |     0.0007 |  0.9472 |
| nt_v2_250m_zeroshot      | intronic   | 1241 |     0.0056 |  0.4106 |
| nt_v2_250m_zeroshot      | missense   |  558 |     0.2796 |  0.5724 |
| nt_v2_250m_zeroshot      | noncoding  |  341 |     0.0029 |  0.7618 |
| nt_v2_250m_zeroshot      | nonsense   |  165 |     0.9697 |  0.65   |
| nt_v2_250m_zeroshot      | synonymous | 1459 |     0.0007 |  0.9047 |
