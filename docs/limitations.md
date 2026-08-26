# Project Scope Limitations & Future Work

## Documented Limitations

1.  **Planetary Dataset Access**: ISSDC and Chandrayaan-2 planetary data require permissions/approval. Substituting with public LRO NAC and other open-source lunar imagery for initial testing.
2.  **Domain Adaptation for Deep Matcher**: Pre-trained LoFTR model uses weights trained on terrestrial outdoor scenes (MegaDepth). Direct transfer to lunar environments is not optimized, but will serve as a strong base.
3.  **Hyperspectral IIRS Bands**: True hyperspectral correspondence for IIRS involves dozens of bands. The prototype will focus on single-band gray levels or subset selection.
4.  **Hardware Performance**: Deep learning matching (LoFTR) might require GPU resources for real-time operation. CPU fallbacks are implemented but will run slower.

## Future Work

1.  **Model Fine-tuning**: Fine-tuning LoFTR on lunar craters dataset (LRO NAC / TMC-2).
2.  **Sub-pixel Accuracy Validation**: Benchmarking with synthetic warping to establish a true ground truth for sub-pixel accuracy.
3.  **Dynamic Scale Pyramid**: Auto-estimating massive scale factors before applying matchers.
