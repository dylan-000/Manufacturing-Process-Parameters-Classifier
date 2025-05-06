# Multi-Output MLP Process Parameter Classifier
The source code here is the code developed over the period of my 2024-2025 undergraduate research, which aimed at using MLP architectures to classify Lathe Process Paramaters by using a dataset of recorded audio data.

The model classifies the following parameters:
* **Depth of Cut** - The depth at which the lathe will cut into the metal, measured in inches.
* **Spindle Speed** - The speed at which the lathe is running, measured in RPM.
* **Feed Rate** -  The rate at which the metal is fed into the bit, measured in inches/revolution.

In this repository, you will find the files used to train the model, as well as those used to construct the datasets and process the raw data.

All raw data is publicly available at [this kaggle dataset repository.](https://www.kaggle.com/datasets/fisher200/lathe-process-parameter-acoustic-signal-dataset/data?select=top32_absolute_value_sum_ranked_mic_scores_per_test_new.csv)