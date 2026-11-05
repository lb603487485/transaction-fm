# Should we invest in a transaction foundation model to replace XGBoost? Evidence from a small end-to-end build

Bo Lin, November 2026. Code, notebooks and every number below: github.com/lb603487485/transaction-fm (results/metrics/*.json).

## 1. Summary and recommendation

An 11M-parameter GPT-style decoder was pretrained on 1.2 million card transactions with next-token prediction and fine-tuned to flag fraud on the transaction being judged, given the card's previous twelve. On an out-of-time test period it reaches PR-AUC 0.9857 (3 seeds, sd 0.0003), against 0.9751 for an XGBoost model with 29 hand-built features tuned on the same validation period. The difference is real but small: a paired bootstrap over cards gives +0.0106 with a 95% interval of +0.0059 to +0.0156, and at a fixed 0.6% alert rate it means 1.1 points more recall (0.989 vs 0.978) and 1.7% fewer false positives per fraud. Pretraining itself is worth +0.044 PR-AUC over the same model fine-tuned from random weights, fifty times the seed noise.

The ablations agree with the Wells Fargo talk on direction and disagree on emphasis. History is the largest driver, last-token pooling beats mean pooling by a wide margin, the decoder beats a same-size MLM encoder, and the time-gap token matters. But the history gain is almost entirely in the first five transactions (0.74 to 0.98 PR-AUC), twelve add 0.006, and thirty-two add nothing at four times the compute. Perplexity stops predicting downstream quality before pretraining stops improving it, and it cannot be compared across tokenizers at all, which supports the Wealthsimple position.

Recommendation on this evidence: do not replace XGBoost on the strength of a one-point gain on synthetic data. The case for investing is narrower and more specific. The sequence model's edge is concentrated in frauds that look normal by amount but sit inside a burst, which hand-built features do not capture; the representation is learned without labels and could serve other tasks; and the remaining gap in the other direction (isolated large amounts, where XGBoost wins) has a cheap fix under test. A sensible next step is a pilot on real transactions with the pretrained model as a challenger to the production XGBoost, an ensemble of fine-tuning seeds, the validator checks in section 6 as release gates, and XGBoost kept as champion until an out-of-time real-data test is passed.

## 2. Data and split

The data is the Sparkov synthetic credit card set from Kaggle: 1,852,394 transactions on 999 cards over 2019 to 2020, 14 merchant categories, 693 merchants, 0.52% fraud. Personally identifying columns (name, street, city, zip, job, date of birth, transaction number) are dropped at load; the card number becomes a dense id; a home-to-merchant distance is computed from the coordinates. Gender, age bucket, state and city-size bucket are kept in a separate card table that is never joined into a model input and is read only for the subgroup check in section 6.

The split is by time, not by file. Train is everything before 2020-05-22, validation is the following 30 days, test is from 2020-06-21 (the Kaggle test file): 1,212,704 / 83,971 / 555,719 rows, with 2,145 frauds in test (0.39%, lower than train's 0.58%, so test precision is naturally lower). All splits live in one table so that a test transaction can see its card's earlier history, as a production scorer would; the forbidden direction is blocked in code rather than by file layout. Every function that learns from data (bin edges, vocabulary, merchant list, category codes, scalers, models) calls a guard that raises if a validation or test row is present. History windows are built backward in time only. Model selection reads validation only. The XGBoost features are covered by tests that edit a later row and assert that no earlier row's features change, flip every label and assert the feature matrix is unchanged, and recompute rolling windows by brute force on 2,000 random rows.

Two properties of the synthetic data shape everything that follows. Fraud comes in bursts: 90% of frauds are within 24 hours of the card's previous fraud, so a short history already carries most of the signal, and card-level discovery is easier than transaction-level recall. And the data is easy: fraud amounts have a median of 372 dollars against 47 for normal transactions, 75% of frauds are above 200 dollars against 4% of normals, merchant locations are placed at random, and the 145 test rows of cards that first appear in the test period are all fraud. Headline numbers will not transfer to real data; the differences at a fixed alert rate and the ablation trends are the transferable part.

## 3. Tokenizer

Each transaction becomes seven tokens in a fixed order. Everything learned from data is fit on train rows only; the time-gap and hour bins are fixed by definition.

| token | source | bins or values | fit on data |
|---|---|---|---|
| TX | separator | 1 | no |
| DT | minutes since the card's previous transaction | 12 log-spaced bins from 1 minute to 30 days, plus DT_NA for a card's first row | no |
| HOUR | hour of day | 6 periods of 4 hours | no |
| CAT | merchant category | 14 | list from train |
| MER | merchant | top 500 by train frequency, else MER_OTHER (17% of rows in every split) | yes |
| AMT | amount | 20 bins, log-spaced between the train 1st and 99th percentile (1.26 to 546 dollars); each interior bin spans a 1.36x range | edges from train |
| DIST | home to merchant distance | 8 log-spaced bins | edges from train |

Vocabulary 567. Worked example, one row: "TX DT_02 HOUR_LATE_NIGHT CAT_grocery_pos MER_fraud_Kilback_LLC AMT_17 DIST_06" for a grocery purchase of 372 dollars, 15 minutes after the previous transaction, at 1 a.m., 61 km from home. The validation and test periods encode entirely with the train vocabulary (no unseen category; 1% of rows in the open-ended amount and distance bins, as designed).

Binning trades precision inside a bin for an unchanged GPT-2 stack and trainer. The cost is explicit: 47 and 51 dollars can be the same token, and everything above 546 dollars is one token. Equal-ratio bins were chosen over equal-width because amounts are heavy-tailed; with equal-width bins 39% of rows fall into one bin against 15% for log bins, and the ablation in section 5 shows that the equal-width tokenizer produces a better perplexity and no better fraud detection. The amount cap is the limitation the error analysis later finds, and the hybrid head in section 7 addresses it.

## 4. Model and training

The pretrained model is a GPT-2 decoder with 6 layers, 384 hidden units, 6 heads of 64, 512 positions and learned absolute positions, 11.06M parameters (0.41M embeddings, 10.65M in the blocks). Windows of 512 tokens (73 transactions) with stride 256 are cut per card within each split; 31,856 train windows, 16.3M tokens per epoch. Pretraining is causal next-token prediction for 8 epochs (7,968 steps), batch 32, learning rate 5e-4 with cosine decay and 5% warmup, mixed precision on the T4, evaluation and checkpoint every 500 steps, best checkpoint by validation loss. The best step was 7,500 of 7,968 with validation loss 1.6237 (perplexity 5.07); the last step was within 0.0002 of it, so the run neither peaked early nor overfit, and train and validation loss agree to 0.01. Two later retrainings of the same recipe on a different machine landed at 1.6253 and 1.6246. Pretraining took 37 minutes on a Colab T4 and 78 minutes on an Apple M4 Pro.

Per-field next-token accuracy against the prior (always guessing the field's most common train token) says what was learned: HOUR 0.69 vs 0.20, AMT 0.27 vs 0.15, CAT 0.22 vs 0.10, DT 0.28 vs 0.22, MER 0.18 vs 0.17, DIST 0.36 vs 0.36. The model learned time of day, amount and category habits, a little rhythm, almost nothing about merchants, and nothing about distance, which is random in this data.

Fine-tuning examples are the card's previous twelve transactions followed by the transaction to judge (92 tokens), labelled with that transaction's fraud flag. Train examples keep every fraud and sample 5% of normals (about 67,000 examples, 7,017 frauds, resampled per seed); validation and test are not sampled. All parameters train, with a new 384 x 2 head on the last real token, at learning rate 5e-5 for 3 epochs, batch 64, checkpoint chosen by validation PR-AUC. One fine-tuning run takes 10 minutes on the T4 and 23 on the M4 Pro. Every configuration is run with three seeds and reported as mean and sample standard deviation. The control for pretraining is the same architecture fine-tuned from random weights.

The XGBoost baseline uses 29 features in four groups (current transaction, time since previous transaction overall, same category and same merchant, rolling counts and amounts over 1 hour to 30 days, and deviations from the card's history), computed in 5 seconds for the full table, with a 12-point grid on validation (best: depth 6, learning rate 0.05, scale_pos_weight 20, 819 trees) and three seeds; a fit takes 20 seconds. A current-transaction-only XGBoost is the history control.

Steps 1 to 4 ran on the Colab T4; steps 5 and 6 ran on the M4 Pro after the Colab quota ended. Before any training on the laptop, a T4 checkpoint was re-scored there (PR-AUC agreed to 6e-6) and the reference arm was retrained; it landed 0.0012 PR-AUC below the T4 mean, inside the laptop's own seed spread but eight T4 standard deviations away. Fine-tuning is not reproducible across machines at the 1e-3 level, so every ablation below compares arms trained on one machine, each run records its machine, and no laptop number is placed next to a T4 number.

## 5. Results

Main comparison, test period, three seeds, alert rate 0.6% (3,334 alerts for 2,145 frauds).

| model | PR-AUC | ROC-AUC | recall at 0.6% | card discovery | FP ratio |
|---|---|---|---|---|---|
| transformer, pretrained, H12 (T4) | 0.9857 +- 0.0003 | 0.9995 | 0.989 | 1.000 | 0.57 |
| transformer, from scratch, H12 (T4) | 0.9415 +- 0.0009 | 0.9975 | 0.950 | 0.980 | 0.64 |
| XGBoost, 29 features | 0.9751 +- 0.0004 | 0.9994 | 0.978 | 1.000 | 0.59 |
| XGBoost, current transaction only | 0.7818 +- 0.0006 | 0.9962 | 0.809 | 0.986 | 0.92 |

Transformer minus XGBoost, paired bootstrap over 1,000 resamples of cards (rows of one card are not independent): PR-AUC +0.0106 [+0.0059, +0.0156], recall at 0.6% +0.0103 [+0.0039, +0.0163], FP ratio -0.0165 [-0.0265, -0.0065]; all three intervals exclude zero, and the three seed-pair point estimates lie inside them. The PR curves coincide below recall 0.9; the whole difference is in the last decile. ROC-AUC is reported because others report it; at 0.4% fraud it hides every difference above.

Ablations (laptop, each arm compared with a control trained on the same machine; H12 last-token reference 0.9845 +- 0.0021 for the fine-tuning ablations, retrained controls for the pretraining ablations).

| ablation | arm | PR-AUC | difference | reading |
|---|---|---|---|---|
| history length | H = 0 | 0.7405 +- 0.0055 | | below XGBoost current-only: one binned row carries less than six raw numbers |
| | H = 5 | 0.9788 +- 0.0019 | +0.238 from 0 | already above the 29-feature XGBoost with its 30-day windows |
| | H = 12 | 0.9845 +- 0.0021 | +0.006 from 5 | about 3 sd |
| | H = 32 | 0.9843 +- 0.0008 | 0.000 from 12 | four times the training time |
| pooling at H12 | mean instead of last token | 0.8973 +- 0.0028 | -0.087 | the judged transaction is the last token; averaging dilutes it |
| tokenizer | linear amount bins | 0.9814 +- 0.0021 | -0.0022, 1.6 sd | a tie, although its perplexity is lower (4.49 vs 5.08) |
| | no time-gap token | 0.9771 +- 0.0021 | -0.0065, 4.6 sd | the only tokenizer switch that matters |
| objective | MLM encoder, same size | 0.9507 +- 0.0017 | -0.033, 17 sd | still well above scratch; last-token head on both, CLS pooling untested |

In the guide's phrasing: on this data pretraining raises PR-AUC by 0.044 over scratch and 0.011 over a tuned XGBoost; the gain from zero to five transactions of history is larger than any other change, and from twelve to thirty-two there is none. The direction matches Wells Fargo; the emphasis does not, since the talk places the big step between short and long history and this data places it at the first few transactions, consistent with fraud arriving in bursts.

## 6. Validation findings

Intrinsic vs extrinsic. Five snapshots of one pretraining run (steps 500 to 7,500, perplexity 5.31 to 5.08) were each fine-tuned once. Spearman's rho between perplexity and test PR-AUC is -0.89 over six points, but the shape matters more than the coefficient: after step 2,000 perplexity falls from 5.15 to 5.08 while PR-AUC moves from 0.981 to 0.984, within 1.5 seed standard deviations, and the step-1,000 snapshot scores below the step-500 one. Downstream quality flattens while the pretraining metric still improves, and across tokenizers the two decouple completely (the linear-bin tokenizer has the better perplexity and the worse fraud detection). Checkpoint and tokenizer selection should read the downstream metric.

Time stability. With one cutoff fixed on the whole test period, monthly recall stays between 0.975 and 0.995 for the transformer (0.960 to 0.985 for XGBoost); no month degrades. The alert rate drifts from 0.32% in December to 0.83% in October because it follows the fraud rate, not the transaction volume (December has 139,000 rows and a 0.18% fraud rate). This is desirable behaviour and an operational fact a fixed threshold must plan for. December is the hardest month for both models (FP ratio 0.75 transformer, 1.11 XGBoost). Population stability indices of every token field between the train and test periods are below 0.03; the synthetic generator does not drift, so this check is reported as done and clean rather than skipped.

Subgroups. By category the transformer matches or beats XGBoost in 12 of 14, with the largest recall gaps in the low-fraud categories where XGBoost is weakest (travel +0.10, home +0.075); both models are weaker where frauds are few. By gender both models score female cardholders lower (transformer PR-AUC 0.979 vs 0.993; XGBoost 0.964 vs 0.988) with the same direction and size, which points at the data rather than either model, and the transformer's alert rate is also higher for that group (0.65% vs 0.54%); this would need a fairness review on real data. Age and city size are flat. The history-length cut exposes a generator artefact: the 145 test rows of cards with fewer than twelve earlier transactions are all fraud and both models flag all of them, so the "new card" risk that a sequence model faces in production cannot be measured here.

Error analysis. Of the frauds flagged by only one model at its own cutoff, the 20 transformer-only cases have a median amount of 21.5 dollars and an amount z-score of -0.3 against the card's history: normal by amount, caught because the decoded history shows earlier frauds in the same burst. The 8 XGBoost-only cases have a median of 840 dollars and a z-score of 7.3 with an 18-hour gap: isolated large amounts, which XGBoost reads from the raw value and the transformer cannot distinguish once the amount exceeds the top bin. The two models are complementary, and the second kind is a tokenizer limitation with a known fix.

Ceiling. The transformer scores 95.7% of test frauds above 0.99 and the 0.6% cutoff sits at 0.047; the alert budget exceeds the fraud count, so recall near 1 is partly by construction. This is why the report leans on the fixed-rate differences and the ablation trends rather than the absolute numbers.

## 7. Limitations and next steps

Scale and data. 11M parameters, 1.85 million synthetic transactions, 999 cards. Real data has more merchants than any top-N keeps, new cards that are mostly legitimate, actual drift, and labels that arrive late. No serving path was built; scoring is batch.

Fairness of the comparison. The XGBoost baseline was tuned on a grid and the transformer was not; a learning-rate sweep, a linear probe (frozen body, head only, pretrained vs random body), a sampling-rate check, a seed ensemble and a calibration study are running as follow-ups (notebook 06d), together with a hybrid head that gives the classifier the raw amount, its z-score and ratio against the card's history, and the time gap next to the sequence representation (notebook 06e), the experiment the error analysis points at. Their results will be added here; the transformer side of the comparison can only improve from them.

If scaled to real data and a proper GPU budget, in order: run on real transactions first, since every conclusion above about ceilings and new cards is a property of the generator; size the model to the token count rather than copying the 1B figure, and use the perplexity scatter only within one tokenizer; replace learned positions with RoPE and raise the context before raising depth, since history saturates at twelve here but real card histories are longer and less bursty; fine-tune several heads from one body (fraud, default, churn) to amortise pretraining, which is the actual economic argument for a foundation model; and reserve FSDP, Flash attention and LoRA for when the body is large enough to need them.

What to monitor after deployment, from the checks above: population stability of every token field against the training period; alert rate and recall at the fixed threshold by month; recall by category and by cardholder group with a minimum-count rule; the share of rows on which the fine-tuning seeds disagree; and the new-card segment separately, because it is the one segment this study could not test.
