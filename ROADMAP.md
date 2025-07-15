# roadmap

we are looking to evaluate diffusion models (and perhaps autoregressive models down the line) on novel perceptual / reasoning tasks. here is the rough order of (low-level) operations i've been thinking about:
1. build an inference pipeline that can accomodate a wide range of models, cache their activations, and probe those activations *(in progress)*
2. use that pipeline to "reproduce" existing results (à la geirhos, bhattad, & serre lab benchmarks incl. VPT), and compare this approach to the ELBO-based approach
3. if this works, proceed to more novel experiments (which we still need to specify), and if not, reevaluate our approach and possibly circle back to the ELBO/conditional approach
4. ...
