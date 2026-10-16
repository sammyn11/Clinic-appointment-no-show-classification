# Speaker script (about 7 minutes)

Record this yourself, with your face visible, while you share the report PDF or the notebook. The course asks for a 5–10 minute video and for you to be able to answer follow-up questions about your own code. This script is a prompt, not a recording.

Have these figures ready: lead time, SMS by lead time, logistic learning curve, forest learning curve, neural epoch curves, calibration, both confusion matrices, and the slice chart.

## 0:00–1:00 Problem

My mission is to improve how African clinics book and manage appointments, so that waiting rooms are less crowded and scarce clinician time is not left idle. A no-show wastes a slot. Filling every risky slot by double-booking creates the crowd I am trying to avoid. So the classification task is not “predict no-show and overbook.” It is “rank visits, then choose a threshold for the action.” A reminder can tolerate false alarms. An extra booking cannot.

## 1:00–2:10 Data

I could not find an open African facility extract with at least 20,000 labeled visits. National EMR warehouses are not public. I used the Vitória, Brazil appointment file released by Joni Hoppen: 110,527 rows, about 20.2 percent no-shows after cleaning. The label `No-show = Yes` means the patient did not attend. The appointment window is only 29 April to 8 June 2016, one city, 81 neighbourhoods. Dantas and colleagues put the average African no-show rate near 43 percent, so I do not treat these thresholds as transferable. What I do take from the reviews is that lead time and prior misses show up across settings.

## 2:10–3:20 Method

Patients are kept entirely inside train, validation, or test. A row-wise split would leak, because a later training visit could use an earlier test visit’s outcome as “history.” Prior no-show rate uses only earlier visits of the same patient. Neighbourhood target encoding is fit out of fold on training patients. SMS was never sent for same-day visits, so I do not read the crude SMS gap as an effect of the message. Same-day no-show is about 4.5 percent; two weeks out it is about 32 percent.

I ran sixteen experiments: a majority baseline, a lead-time rule, logistic regression as features and the penalty C were added, three forests including one that was allowed to overfit, gradient boosting with and without class weights, a linear SVM, an RBF SVM on 8,000 rows, and three Keras networks fed by tf.data. Thresholds and hyperparameters were locked on validation. Test was scored after that.

## 3:20–5:10 Results

The majority classifier is 80 percent accurate and catches nobody. Demographics alone are almost chance, ROC-AUC about 0.54. Lead time alone is about 0.70. The full logistic model is about 0.73, and changing C did nothing visible. The unrestricted forest hit a training ROC-AUC of 0.999 and a gap of 0.25, then did worse on test than the shallow forest. That is the variance side of the bias–variance picture. The logistic learning curve is flat and overlapped, which is bias: more rows from these six weeks will not fix a linear model.

The selected traditional model is unweighted histogram gradient boosting, test ROC-AUC 0.748 and average precision 0.411. The best network, with a neighbourhood embedding, is 0.743. The epoch curves flatten. The network is not blowing up, and it is not beating the booster. Class-weighted models sit below the diagonal on the calibration plot. The unweighted booster follows it. I would not overbook using a reweighted score.

## 5:10–6:20 Errors and the decision

At the F1 threshold the booster catches about 73 percent of test no-shows and is wrong on about two thirds of the people it flags. That is a possible reminder list. It is a bad overbooking rule. If a false extra patient is five times as costly as an empty slot, the threshold jumps to about 0.78, precision is about 0.91, and recall is under 2 percent. The practical rule I would pilot is to contact the top 20 percent of scores: precision about 0.40, recall about 0.41, roughly twice a random list of the same length.

The slice chart matters. Older no-shows are recalled at about 0.36, younger adults at about 0.89. A single threshold spends reminders on the easy group. Same-day misses are rarely flagged because they are rare. Shuffling lead time collapses the model. SMS does not, once lead time is known.

## 6:20–7:00 Close

I would not deploy this score in a clinic that has not logged its own outcomes. I would deploy the data design: store booking time, store whether the patient came, do not leak history across patients, and do not use one threshold for reminders and for double-booking. Offering more same-day slots changes the base rate more than any model I trained. The saved models are the booster (joblib) and the embedding network (Keras), chosen by validation ROC-AUC. Weights and Biases was run offline; the run folders are in the repository.
