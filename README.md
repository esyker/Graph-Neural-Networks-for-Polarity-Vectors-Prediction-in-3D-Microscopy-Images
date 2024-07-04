# Graph-Neural-Networks-for-Polarity-Vectors-Prediction-in-3D-Microscopy-Images
Paper submited at 2024 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, to be held in Orlando, Florida, USA, July 15-19, 2024 ([https://embc.embs.org/2024/](https://embc.embs.org/2024/)).

## Requirements

***conda_requirements.txt*** was created using ```console conda list -e > conda_requirements.txt``` and can be used to install the requirements to a conda enviroment using ```console conda create --name <env> --file conda_requirements.txt```.

***pip_requirements.txt*** was created using ```console pip list --format=freeze > pip_requirements.txt``` and can be used to install the requirements to a pip enviroment using ```console python3 -m venv env  
source env/bin/activate  
pip install -r pip_requirements.txt```.

Install from "conda_requirements.txt" to install for a conda environment or "pip_requirements.txt" to install to a pip environment.

## Data

The ***data*** folder contains the input data in .csv format:
- ***data/vectors***, contains the data for the manually annotated graphs.
- ***data/vectors_automatic_csv***, contains the data for the graphs with centroids detected by the CNN centroid detection model.

## Results

The ***results*** folder contais the results saved in the same .csv format as the input data.

## Source

The ***src*** folder contains the source code for the link prediction, evaluation and visualizations:
- ***1_generate_artificial_data.ipynb***, contains the source code for generating artificial graphs.
- ***2_train_eval_classical_bipartite_matching.ipynb***, contains the source code for training and evaluating classical bipartite matching algorithms, such as the Hopcroft-Karp and Modified Jonker-Volgenant algorithms.
- ***2_train_eval_gnn.ipynb***, contains the source code for training and evaluating deep learning models, the proposed GNN and MLP models.
- ***3_analyze_results.ipynb***, contains the source code for generating getting evaluation metrics and figures using the .csv files stored in the results folder.
- ***3_eval_vectors_automatic.ipynb***, contains the source code for getting evaluation metrics for the results that are unsupervised, namely for the data that is not manually annotated and for which ground-truth links do not exist and therefore require a different set of metrics for evaluating the TPR and FPR, as defined in *H. Narotamo, M. Ouarn ́e, C. A. Franco, and M. Silveira, “Joint segmentation and pairing of nuclei and Golgi in 3D microscopy images,” in 43rd IEEE/EMBC, pp. 3017–3020, 2021*.

## Figures

The figures folders contains visualizations that allow to interpret the results qualitatively. These figures are genereated using the source code in **3_analyze_results.ipynb**, which takes as input the results contained in the ***results*** folder in order to generate the figures. 

- Figure for manually annotated data:

![Polarity Vectors Prediction Manually Annotated](https://github.com/esyker/Graph-Neural-Networks-for-Polarity-Vectors-Prediction-in-3D-Microscopy-Images/blob/main/figures/real_annotated_normalized/mpnn_recurrent_constraints_with_threshold_with_angular_feats.png)

- Figure for data with centroids detected automatically by a CNN model:

![Polarity Vectors Prediction CNN detection](https://github.com/esyker/Graph-Neural-Networks-for-Polarity-Vectors-Prediction-in-3D-Microscopy-Images/blob/main/figures/real_automatic_normalized/merged-crops-5-6-7-8.jpg)
