# Layer 1 threshold: the empirical midpoint between the highest benign
# score (0.296) and the lowest Direct-Overrides attack score (0.344)
# observed in the labeled dataset. Picked as the max-margin point between
# the two clusters rather than skewing toward either edge, since both
# edges are themselves small-sample estimates (20 benign, 10 Direct
# Overrides) that could shift with more data.
LAYER1_THRESHOLD = 0.32

# Layer 2 (logistic regression) regularization strength. Picked via a
# cross-validated sweep over C in [0.001 .. 1000]: below C=1 the model
# degenerates to blocking everything (100% FPR, meaningless), and
# performance plateaus at C=30 with no further gain at higher C. 30 is
# the smallest — most conservative — value that reaches the plateau.
LOGREG_C = 30.0

# Layer 2's reported accuracy comes from LEAVE-ONE-OUT cross-validation
# (see run_experiment.py, Phase 2), not from fitting and testing on the
# same 70 examples. With 70 examples and 384-dim embeddings, testing on
# the training data would only show what the classifier memorized, not
# what it generalizes to. The classifier actually shipped is still fit
# on all 70 examples — only the *reported* metrics use leave-one-out.
