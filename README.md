# DPRMemSAM

```latex
\section{Short Description}

DPRMemSAM extends DAM4SAM~\cite{videnovic2025distractor} with a long-term memory, DPR (Diverse Prototypical Representations), that collects diverse, prototypical views of the target. While tracking, each frame is summarized by two compact descriptors from SAM2~\cite{ravi2025sam}: a global object descriptor and a summary of the features inside the predicted mask. Frames are compared by fusing a cosine similarity on the descriptors with a 2-Wasserstein distance similarity on the masked features. Of seven memory slots, one goes to RAM, one to DRM, and five to DPR, each holding the most prototypical view of an appearance cluster ($k$-medoids), such that each memorized representation is at once a prototypical representation of it's cluster and ensures diversity among each memory slot. Our tracker is available at~\url{https://github.com/Vujas-Eteph/DPRMemSAM}.



\section{Long Description}


DPRMemSAM is a training-free extension of DAM4SAM~\cite{videnovic2025distractor} that adds a third memory, \emph{Diverse Prototypical Representations} (DPR).
Its objective is to accumulate diverse views of the target during tracking to better handle long-term challenges, akin to~\cite{cheng2022xmem, vujasinovic2023readmem, zhou2024rmem}.
For each frame, DPR forms two condensed descriptors based on SAM2~\cite{ravi2025sam}'s prediction:
(i) the global object descriptor (the decoder's object-pointer token), which captures the object's global state such as presence and localization, and
(ii) a multivariate Gaussian $\mathcal{N}(\mu,\Sigma)$, summarizing the memory-encoder features within the predicted object mask, providing finer appearance detail than the global descriptor to tell similar "views" apart.

We compute the global similarity between two object representations by combining two complementary measures into a single value: the cosine similarity of the global descriptors and a bounded similarity derived from the 2-Wasserstein (Bures) distance between the mask-memory Gaussians (both in $[0,1]$), which we combine by taking their product.
From this global similarity, DPR selects, through $k$-medoids, the most prototypical (most central) frame of each appearance cluster.
Since the clusters partition the space, the retained medoids are at once the most prototypical appearance within each cluster and the most diverse across clusters.
Note that the initialization frame is permanently retained and anchors the clustering.
Crucially, these two descriptors are condensed summaries, negligible in size next to the full memory-encoder features; hence, we can maintain their representation throughout the sequence without increasing computational cost. They are used exclusively to decide whether the current frame better represents its cluster.
Essentially, while tracking, DPR replaces a memorized frame representation whenever the current frame's condensed representation is closer to its cluster's optimal center than the one currently stored to represent this cluster.
Note that a full object representation can only be accessed while the tracker is processing a given frame or has memorized it.
Thus, each stored frame in DPR is the best accessible proxy for its cluster's center, and not the actual center of the cluster.

As in DAM4SAM~\cite{videnovic2025distractor}, DPRMemSAM uses seven memory slots, split into one for RAM, one for DRM, and five for DPR. We use SAM2.1's~\cite{ravi2025sam} Large checkpoint with native post-processing disabled, keep DAM4SAM's default RAM/DRM settings, and admit a frame to DPR only when SAM2 is confident in it \ie, predicted mask $\text{IoU}\geq 0.5$ and object-presence score $\geq 0.8$.
```

# Setup

0. Setting up workspace with <vots2026/main> stack [more stacks](https://github.com/votchallenge/toolkit/tree/master/vot/stack)
```bash
pixi run vot initialize vots2026/main --workspace workspace
```

1. Test tracker integration with default NVCC tracker (visualize results with the `-g` flag)
```bash
cd .. && \
git clone https://github.com/votchallenge/integration && \
mv integration vot_ncc_test_integration && \
cd ../vot_worskpace_2026/workspace && \
pixi run vot test NCCPython
```


2. Test integration and see preview on a small validation sequence.
```bash
pixi run vot initialize tests/multiobject --workspace test2026 && \
cd test2026/ && \
pixi run vot evaluate DPRMemSAM && \
pixi run vot analysis --format=json DPRMemSAM && \
pixi run vot report --format=latex DPRMemSAM && \
cd ..
```

3. Run DRPMemSAM on VOT2026
```bash
cd workspace
pixi run vot evaluate DRPMemSAM
pixi run vot pack DPRMemSAM
```


# Add. Ext. Docs:
- [VOT Support](https://www.votchallenge.net/howto/)
- [Google Groupe Technical Support](https://groups.google.com/g/votchallenge-help)



### Credits

```bibtex
@inproceedings{ravi2025sam,
	title		= {Sam 2: Segment anything in images and videos},
	author		= {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
	booktitle	= {International Conference on Learning Representations},
	year		= {2025}
}

@inproceedings{videnovic2025distractor,
	title		= {A distractor-aware memory for visual object tracking with sam2},
	author		= {Videnovic, Jovana and Lukezic, Alan and Kristan, Matej},
	booktitle	= {Computer Vision and Pattern Recognition Conference},
	year		= {2025}
}
```



# First Time Setup with [Pixi](https://pixi.prefix.dev/latest/installation/)

0. Install pixi or get latest update
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

1. Init workspace
```bash
pixi init vot_workspace_2026
```

2. Adding dependencies
```bash
pixi add python
pixi add --pypi vot-toolkit
```