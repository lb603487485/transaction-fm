# Should we invest in a transaction foundation model to replace XGBoost? Evidence from a small end-to-end build

Bo Lin, November 2026. Code, notebooks and every number below: github.com/lb603487485/transaction-fm (results/metrics/*.json).

## 1. Summary and recommendation

An 11M-parameter GPT-style decoder was pretrained on 1.2 million card transactions with next-token prediction and fine-tuned to flag fraud on the transaction being judged, given the card's previous twelve. On an out-of-time test period it reaches PR-AUC 0.9857 (3 seeds, sd 0.0003), against 0.9751 for an XGBoost model with 29 hand-built features tuned on the same validation period. The difference is real but small: a paired bootstrap over cards gives +0.0106 with a 95% interval of +0.0059 to +0.0156, and at a fixed 0.6% alert rate it is 1.1 points more recall (0.989 vs 0.978) and 0.017 fewer false positives per flagged fraud (0.57 against 0.59). Pretraining itself is worth +0.044 PR-AUC over the same model fine-tuned from random weights, fifty times the seed noise.

The ablations agree with the Wells Fargo talk on direction and disagree on emphasis. History is the largest driver, last-token pooling beats mean pooling, the decoder beats a same-size MLM encoder, and the time-gap token matters. But the history gain sits almost entirely in the first five transactions (PR-AUC 0.74 to 0.98); twelve add 0.006 and thirty-two add nothing at four times the compute. Perplexity stops predicting downstream quality before pretraining stops improving it, and it cannot be compared across tokenizers at all, which supports the Wealthsimple position. The transformer's fine-tuning was then tuned as the baseline was. The one caveat the reader should weigh: the learning rate selected on the validation month tests 3 sd below the rate fixed in advance, and the validation month, with 489 frauds, cannot tell them apart; with the selected rate the margin over XGBoost shrinks to 0.003, inside noise, and a paired bootstrap of that model against XGBoost confirms it: two of its three seeds have intervals that include zero. A linear probe on the frozen pretrained body alone matches a model trained from scratch end to end, feeding the raw amount to the classifier head changes nothing, and the only change that raised the score was averaging three seeds.

Recommendation: do not replace XGBoost on the strength of a one-point gain on synthetic data. The case for investing is narrower. The sequence model's edge is concentrated in frauds that look normal by amount but sit inside a burst, which hand-built features miss; the representation is learned without labels and could serve other tasks; and the gap in the other direction (isolated large amounts, where XGBoost wins) is eight frauds that a score-level combination with XGBoost would likely recover, which this study did not test. The sensible next step is a pilot on real transactions with the pretrained model as a challenger, an ensemble of fine-tuning seeds, the checks in section 6 as release gates, and XGBoost kept as champion until an out-of-time real-data test is passed.

## 2. Data and split

Sparkov synthetic credit card data (Kaggle): 1,852,394 transactions, 999 cards, 2019 to 2020, 14 categories, 693 merchants, 0.52% fraud. Identifying columns are dropped at load and the card number becomes a dense id. Gender, age, state and city size are kept in a separate card table that is never a model input and is read only for the subgroup check.

The split is by time: train before 2020-05-22, validation the next 30 days, test from 2020-06-21 (the Kaggle test file): 1,212,704 / 83,971 / 555,719 rows, 2,145 frauds in test (0.39%). All splits live in one table so a test transaction can see its card's earlier history, as a production scorer would. The forbidden direction is blocked in code: every function that learns from data (bin edges, vocabulary, merchant list, category codes, models) raises if a validation or test row is present, history windows are built backward only, and model selection reads validation only. The XGBoost features are covered by tests that edit a later row and assert no earlier feature changes, flip every label and assert the feature matrix is unchanged, and recompute the rolling windows by brute force on 2,000 rows.

Two properties of the generator shape the results. Fraud comes in bursts (90% of frauds are within 24 hours of the card's previous fraud), so a short history carries most of the signal. And the data is easy: fraud amounts have a median of 372 dollars against 47 for normal rows, merchant locations are random, and the 163 test rows of cards that first appear in the test period are all fraud. Absolute numbers will not transfer to real data; the differences at a fixed alert rate and the ablation trends are the transferable part.

## 3. Tokenizer

Each transaction becomes seven tokens in a fixed order. Everything learned from data is fit on train rows only.

| token | source | bins or values | fit on data |
|---|---|---|---|
| TX | separator | 1 | no |
| DT | minutes since the card's previous transaction | 12 log-spaced bins, 1 minute to 30 days; DT_NA on a card's first row | no |
| HOUR | hour of day | 6 periods of 4 hours | no |
| CAT | merchant category | 14 | list from train |
| MER | merchant | top 500 by train count, else MER_OTHER (17% of rows in every split) | yes |
| AMT | amount | 20 bins, log-spaced between the train 1st and 99th percentile (1.26 to 546 dollars), each spanning a 1.36x range | edges from train |
| DIST | home to merchant distance | 8 log-spaced bins | edges from train |

Vocabulary 567. Worked example: "TX DT_02 HOUR_LATE_NIGHT CAT_grocery_pos MER_fraud_Kilback_LLC AMT_17 DIST_06" is a 372-dollar grocery purchase 15 minutes after the previous transaction, at 1 a.m., 61 km from home. Validation and test encode entirely with the train vocabulary.

Binning trades precision inside a bin for an unchanged GPT-2 stack and trainer: 47 and 51 dollars can be the same token, and everything above 546 dollars is one token. Equal-ratio bins were chosen over equal-width because amounts are heavy-tailed; with equal-width bins 39% of rows land in one bin against 15% for log bins, and section 5 shows the equal-width tokenizer gets a better perplexity and no better fraud detection. The amount cap is the first explanation the error analysis later offers for the frauds only XGBoost catches; section 5 tests it at the classifier head and finds no gain, and a finer amount vocabulary for the body was not tried.

## 4. Model and training

GPT-2 decoder, 6 layers, 384 hidden units, 6 heads, 512 positions, 11.06M parameters (0.41M embeddings). Windows of 512 tokens (73 transactions) with stride 256 per card and split: 31,856 train windows, 16.3M tokens per epoch. Pretraining: causal next-token prediction, 8 epochs (7,968 steps), batch 32, learning rate 5e-4 with cosine decay and 5% warmup, mixed precision on a Colab T4, checkpoint every 500 steps, best by validation loss. Best step 7,500: validation loss 1.6237, perplexity 5.07; the last step was within 0.0002 of it and train loss agrees with validation to 0.01, so the run neither peaked early nor overfit. Two retrainings of the same recipe reached 1.6253 and 1.6246. Wall time 37 minutes on the T4, 78 on an Apple M4 Pro.

Per-field next-token accuracy against the prior (always guessing the field's most common train token): HOUR 0.69 vs 0.20, AMT 0.27 vs 0.15, CAT 0.22 vs 0.10, DT 0.28 vs 0.22, MER 0.18 vs 0.17, DIST 0.36 vs 0.36. The model learned time of day, amount and category habits, a little rhythm, almost nothing about merchants and nothing about distance.

Fine-tuning examples are the card's previous twelve transactions plus the one to judge (92 tokens), labelled with that transaction's fraud flag. Train examples keep every fraud and sample 5% of normals (about 67,000 examples, 7,017 frauds); validation and test are not sampled. All parameters train, with a new 384 x 2 head on the last real token, at learning rate 5e-5 for 3 epochs, batch 64, checkpoint by validation PR-AUC; 10 minutes per run on the T4, 23 on the M4 Pro. Every configuration runs three seeds, reported as mean and sample sd. The pretraining control is the same architecture fine-tuned from random weights.

XGBoost: 29 features in four groups (current transaction; time since the previous transaction overall, in the same category and at the same merchant; rolling counts and amounts over 1 hour to 30 days; deviations from the card's history), computed in 5 seconds for the whole table, 12-point grid on validation (depth 6, learning rate 0.05, scale_pos_weight 20, 819 trees), 20 seconds per fit. A current-transaction-only XGBoost is the history control.

Steps 1 to 4 ran on the T4; steps 5 and 6 on the M4 Pro after the Colab quota ended. A T4 checkpoint re-scored on the laptop agreed to 6e-6 in PR-AUC, but the reference arm retrained there landed 0.0012 below the T4 mean, inside the laptop's own seed spread yet eight T4 standard deviations away. Fine-tuning is not reproducible across machines at the 1e-3 level, so every ablation compares arms trained on one machine and no laptop number is placed next to a T4 number.

## 5. Results

Main comparison, test period, three seeds, alert rate 0.6% (3,334 alerts for 2,145 frauds).

| model | PR-AUC | ROC-AUC | recall at 0.6% | card discovery | FP ratio |
|---|---|---|---|---|---|
| transformer, pretrained, H12 (T4) | 0.9857 +- 0.0003 | 0.9995 | 0.989 | 1.000 | 0.57 |
| transformer, from scratch, H12 (T4) | 0.9415 +- 0.0009 | 0.9975 | 0.950 | 0.980 | 0.64 |
| XGBoost, 29 features | 0.9751 +- 0.0004 | 0.9994 | 0.978 | 1.000 | 0.59 |
| XGBoost, current transaction only | 0.7818 +- 0.0006 | 0.9962 | 0.809 | 0.986 | 0.92 |

Transformer minus XGBoost, paired bootstrap over 1,000 resamples of cards: PR-AUC +0.0106 [+0.0059, +0.0156], recall at 0.6% +0.0103 [+0.0039, +0.0163], FP ratio -0.0165 [-0.0265, -0.0065]; all three intervals exclude zero. The same bootstrap on the laptop reference (06b, three seeds, each against the XGBoost seed with the same number) gives PR-AUC +0.0075 to +0.0122, every interval excluding zero. On the learning rate selected on validation (2e-4) it gives +0.0075 [+0.0030, +0.0120], -0.0009 [-0.0066, +0.0038] and +0.0024 [-0.0035, +0.0077]: one seed beats XGBoost, two tie with it (06f). The PR curves coincide below recall 0.9; the whole difference is in the last decile. ROC-AUC hides every difference at 0.4% fraud.

Ablations on the laptop, each arm against a control trained on the same machine (H12 last-token reference 0.9845 +- 0.0021 for the fine-tuning ablations; retrained controls for the pretraining ablations).

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

On this data pretraining raises PR-AUC by 0.044 over scratch and 0.011 over a tuned XGBoost; the gain from zero to five transactions of history is larger than any other change, and from twelve to thirty-two there is none. The direction matches Wells Fargo; the emphasis does not, since the talk's largest gain is from five to twelve transactions and this data places it in the first five, with five to twelve worth 0.006, consistent with fraud arriving in bursts.

Fine-tuning variants, all from the same pretrained body on the laptop, against the same H12 reference (0.9845 +- 0.0021). This closes the tuning asymmetry noted in section 4: the XGBoost baseline had a grid search, the transformer had defaults.

| variant | trainable parameters | PR-AUC | difference | reading |
|---|---|---|---|---|
| learning rate selected on validation (2e-4, from 1e-5 to 2e-4) | all 11.06M | 0.9781 +- 0.0043 | -0.006, 3.0 sd | taken literally, the validation-selected model is 0.003 above XGBoost, inside noise; the headline keeps the rate fixed before any sweep, which the test period prefers by 3 sd, but that preference was seen on the test period, so the honest margin over XGBoost is between 0.003 and 0.011; bootstrapped on its own against XGBoost, this model's interval excludes zero for one seed of three and includes it for the other two, so the bottom of the range is a tie |
| linear probe, frozen pretrained body | 768 | 0.9452 +- 0.0005 | -0.039 | level with the from-scratch model that trains every weight (0.9415) |
| linear probe, frozen random body | 768 | 0.4260 +- 0.0283 | -0.56 | the representation alone is worth +0.52; retraining it adds +0.039 |
| negative sampling 2% and 10% instead of 5% | all | 0.9823 +- 0.0023 and 0.9856 +- 0.0015 | -1.0 and +0.5 sd | ranking does not depend on the sampling rate; probabilities do (section 6) |
| hybrid head: last token plus raw amount, z-score, ratio and gap | 11.09M | 0.9842 +- 0.0014 | -0.1 sd | binning the amount loses nothing the head can use; recall by amount bucket matches the reference in every bucket with history |
| mean of the three reference seeds | 3 models | 0.9867 | +0.002, 1.0 sd | recall at 0.6% 0.989; 477 of 555,719 test rows move by more than 0.5 between seeds |
| mean of the three XGBoost seeds, for symmetry | 3 models | 0.9756 | +0.0005 over its own seeds | the tree's seeds agree (sd 0.0004), so averaging buys it little; ensemble against ensemble the gap is 0.011, the same as the single-model headline |

The default configuration is the best or tied with the best on every axis tried (all 25 rows, including the single-seed learning rates, are in results/metrics/06f_compare_finetunes.json); the hybrid head does not recover the frauds only XGBoost catches by giving the head the raw amount, so the gap is not information the head lacks; whether a finer amount vocabulary for the body would close it was not tried. The exception is the 32 frauds whose card has fewer than two earlier transactions, so no amount z-score, where the raw amount lifts recall from 0.94 to 1.00.

## 6. Validation findings

Intrinsic vs extrinsic. Five snapshots of one pretraining run (steps 500 to 7,500, perplexity 5.31 to 5.08) were each fine-tuned once. Spearman's rho between perplexity and test PR-AUC is -0.89, but the shape matters more: after step 2,000 perplexity falls from 5.15 to 5.08 while PR-AUC moves from 0.981 to 0.984, within 1.5 seed sd, and the step-1,000 snapshot scores below the step-500 one. Across tokenizers the two decouple completely. Checkpoint and tokenizer selection should read the downstream metric.

Time stability. With one cutoff fixed on the whole test period, monthly recall stays between 0.975 and 0.995 (XGBoost 0.960 to 0.985). The alert rate drifts from 0.32% in December to 0.83% in October because it follows the fraud rate, not the volume (December has 139,000 rows and a 0.18% fraud rate): desirable behaviour, and an operational fact a fixed threshold must plan for. Population stability indices of every token field between train and test are below 0.03; the generator does not drift, so the check is reported as done and clean.

Subgroups. By category the transformer matches or beats XGBoost in 12 of 14, with the largest recall gaps in the low-fraud categories where XGBoost is weakest (travel +0.10, home +0.075). Both models score female cardholders lower (transformer PR-AUC 0.979 vs 0.993; XGBoost 0.964 vs 0.988) with the same direction and size, which points at the data, and the transformer's alert rate is higher for that group (0.65% vs 0.54%); on real data this needs a fairness review. Age and city size are flat. The 161 test rows of cards with fewer than twelve earlier transactions (16 with none) are all fraud; XGBoost flags all of them and the transformer 160, so the new-card risk a sequence model faces in production cannot be measured here.

Error analysis. Agreement between the two models on the 2,145 test frauds, seed 0, both at the 0.6% alert rate:

| | XGBoost flags | XGBoost misses |
|---|---|---|
| Transformer flags | 2,090 | 30 |
| Transformer misses | 8 | 17 |

The two models agree on 2,107 frauds; the 38 they disagree on split 30 to 8 in the transformer's favour, which is the 22-fraud recall gap of section 5. Under equal performance the split would be even; an exact McNemar test on the 38 discordant frauds gives p = 0.0005, optimistic because frauds in one burst are not independent, which is why the card-level bootstrap in section 5 is the primary evidence. The 30 frauds flagged only by the transformer have a median amount of 22 dollars and an amount z-score of -0.2 against the card's history: normal by amount, caught because the decoded history shows earlier frauds in the same burst. The 8 flagged only by XGBoost have a median of 840 dollars and a z-score of 7.3 with an 18-hour gap: isolated large amounts, which XGBoost reads from the raw value and the transformer cannot distinguish above the top bin. The 17 that escape both are small isolated amounts with no burst around them, 27 hours from the card's previous transaction. The models are complementary, and the most a combination of the two scores could add at this alert rate is those 8 frauds. The hybrid head in section 5 gives the raw amount to the head and flags 4 of the 8 against 2 for the reference with the score unchanged overall, so the head is not what is missing; the body still sees the capped amount token, and a finer amount vocabulary was not tried.

Calibration. The classifier is trained on 5% of the normal rows, which multiplies the fraud prior by about 20, so raw scores are overconfident at the top: rows scored 0.99 are fraud 93.0% of the time. Correcting the logit for the sampling rate brings that to 97.3%, and a temperature of 1.31 fitted on re-scored validation rows to 97.9%; expected calibration error falls from 0.0009 to 0.0002, a small number either way because 99.5% of rows sit in the lowest bin. Ranking metrics are unchanged by both steps. Any probability shown to an analyst needs the correction; the threshold does not.

Ceiling. The transformer scores 95.7% of test frauds above 0.99 and the 0.6% cutoff sits at 0.047; the alert budget exceeds the fraud count, so recall near 1 is partly by construction. Hence the weight on fixed-rate differences and trends rather than absolute numbers.

## 7. Limitations and next steps

11M parameters, 1.85 million synthetic transactions, 999 cards, batch scoring only. Real data has more merchants than any top-N keeps, new cards that are mostly legitimate, actual drift and late labels. The learning-rate sweep exposed a design limit: the one-month validation set with 489 frauds ranks two learning rates within noise that the test period separates by three standard deviations, and the decision to keep the pre-set rate was made with the test numbers in view. The defensible statement is the range, 0.003 to 0.011 over XGBoost, not the top of it; bootstrapped on its own, the validation-selected model beats XGBoost with one seed and ties with two. On real data, select with several seeds on a longer validation period and touch the test period once. The seed ensemble is the one change that raised the score and is the configuration a pilot should ship.

If scaled to real data and a real GPU budget, in order: run on real transactions first, since every conclusion about ceilings and new cards is a property of the generator; size the model to the token count rather than copying the 1B figure, and use perplexity only within one tokenizer; replace learned positions with RoPE and raise the context before raising depth, since real card histories are longer and less bursty than these; fine-tune several heads from one body (fraud, default, churn) to amortise pretraining, which is the actual economic argument for a foundation model; and reserve FSDP, Flash attention and LoRA for when the body is large enough to need them.

To monitor after deployment: population stability of every token field against the training period; alert rate and recall at the fixed threshold by month; recall by category and cardholder group with a minimum-count rule; the share of rows on which the fine-tuning seeds disagree; and the new-card segment separately, because it is the one segment this study could not test.
