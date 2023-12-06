# Targeted_extraction

* Targeted_extraction is an automatic tool for achieving targeted extraction of metabolite EIC peaks in untargeted LC-HRMS data.

![Fig 4](https://github.com/JunZeng1999/Targeted_extraction/assets/109707707/8d2111c2-7913-4508-bc95-be8e8672f4bf)



# Cite

* Combination of in silico prediction and convolutional neural network framework for targeted screening of metabolites from LC-HRMS fingerprints: A case study with “Pericarpium Citri Reticulatae - Fructus Aurantii”

# Requirements for Peak_CF operation
* Python, version 3.7 or greater
* Pytorch 1.9.0
* Windows 10
* Install additional libraries, listed in the requirements.txt

# Usage
* Before running “Targeted_extraction_main.py”, please run “Database_construction.py” first
* Open “Targeted_extraction” file and configure the environment required for running the code
* Download the model (Classifier99.pth), then unzip it into the save_weights folder
* Run “Targeted_extraction_main.py”
* Select “blank group”data, “administration group” data, and m/z list as inputs
* Select the desired mode (positive/negative ion mode)
* Click on “Run” under the “Output” option
* Click on “Export” under the “Output” option and select the path for the result list
* The more detailed instruction on how to use Targeted_extraction is available via the link (https://pan.baidu.com/s/1l0iNyqvAiYRQraBrZY3Daw?pwd=w8qb)
