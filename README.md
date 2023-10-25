# Targeted_extraction

* Targeted_extraction is an automatic tool for achieving targeted extraction of metabolite EIC peaks in untargeted LC-HRMS data.

![Fig 4](https://github.com/JunZeng1999/Targeted_extraction/assets/109707707/8dcd83c8-c201-4c21-b518-7d7249fa4283)


# Cite

* Combination of in silico Prediction and Convolutional Neural Network framework for Targeted Screening of Metabolites from LC-HRMS Fingerprints: A case study with “Pericarpium Citri Reticulatae - Fructus Aurantii”

# Requirements for Peak_CF operation
* Python, version 3.7 or greater
* Pytorch 1.9.0
* Windows 10
* Install additional libraries, listed in the requirements.txt

# Usage
* Before running Targeted_extraction_main.py, please run Database_construction.py first
* Download the model (/Targeted_extraction/save_weights/Classifier99.pth), then unzip it into the save_weights folder.
* Run Targeted_extraction_main.py
* Select blank group, administration group, and m/z list for Input
* Select the corresponding mode
* Click Run in the Output (usually it takes a while to wait)
* Click Export in the Output and select the output path to get the list of results
* The more detailed instruction on how to use Targeted_extraction is available via the link(https://pan.baidu.com/s/1ibcdX58qYPo1r4dcCyQagg?pwd=o9m0).
