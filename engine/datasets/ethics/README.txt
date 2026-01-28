There are four tasks separated in four folders (``commonsense``, ``deontology``, ``justice``, and ``virtue``). Each folder contains ``train``, ``test``, and ``test_hard`` files. Commonsense morality also has a set of ambiguous examples which are useful for disagreement detection.

We took many efforts to ensure that labels have high agreement rates. However, even some of the authors of this paper do not agree with some commonsense morality examples about meat-eating,
but the point of this dataset is not to model our own moral convictions but instead more widely held moral beliefs.
We hope that future work will include more belief systems than the four currently provided.

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}