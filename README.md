# PRENEU
PRobabilistic Evaluation of genomic NEUral networks

## Usage
PRENEU simulates sequence-phenotype data following a variance-components model studied widely across statistical genetics. That is, we assume the variance in the phenotype can be explained by different features in the sequence data, as specified here as motif effects, and non-motif, sequence-based effects. PRENEU can be used to probe methodlogies and for evaluation purposes in sequence-to-function studies.

There are a number of parameters to explore using PRENEU, these can make the relationship between sequence and phenotype simplistic or more complex. We explain the parameters as follows:

* `N` &mdash; the number of sequences to generate (sample size)
* `seqlen` &mdash; the length of the sequences. Set to 200 in the paper.
* `h2` &mdash; the amount of variability explained by the _sequence_, must be within [0,1]. In the absence of sequence-based effects, this is the motif explainability (i.e. heritability in statistical genetics). Set to 0.80 in the paper.
* `alpha` &mdash; Proportion of the motif explainability that is due to simple additive effects. 1 is purely additive, 0 is purely interaction effects
* `interactgap` &mdash; the maximum number of nucleotides between the two motifs responsible for the distance-based interaction effect. Set to 3 in the paper.
* `pi` &mdash; sparsity parameter for motif additive effects. Set to 0.6 in paper (6 of 10 motifs will have additive effects)
* `delta` &mdash; positional preference of motif. This scales the additive effects of the motifs based on preferred position. When higher, motifs will mostly only have an effect in their preferred position (randomly sampled per simulation), and their additive effect decays as a function of distance from this preferred position. Set to 0 in the paper.
* `Iw` &mdash; How to weight the (1-$alpha$)$h^2$ amongst interaction effects. In other words, how to share the (1-$alpha$)$h^2$ explainability amongst interaction effects, 0.25, 0.25, 0.25, 0.25 (default in the paper) gives equal weight to all interactions, whereas (1,0,0,0) would make it such that only one interactive effect influences the phenotype.
* `psi` &mdash; Fraction of $h^2$ explained by non-motif sequence effects.
* `curargument` &mdash; parameter for naming simulation runs. Enables running large-scale experiments by varying the names accordingly.


We include a script to show an example of how to run PRENEU under `example_run.py`. PRENEU can very simply be added to pipelines following the provided structure. We include all models with their evaluated hyperparameters under `PRENEU_models.py`, where researchers can add and evaluate their on models and seamlessly integrate them into the pipeline.



Scripts to replicate the figures will be provided shortly.
