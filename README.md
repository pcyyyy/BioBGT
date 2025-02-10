
This is a code source for paper "Biologically Plausible Brain Graph Transformer" (ICLR 2025).


1. Implementation Requirements:

    PyTorch Geometric v2.0.4
  
    PyTorch v1.9.1. 
  
    NVIDIA A6000 GPU with 48GB of memory

2. Dataset:

   ADHD-200 dataset can be downloaded from: https://fcon_1000.projects.nitrc.org/indi/adhd200/
   
   ABIDE can be downloaded from: https://fcon_1000.projects.nitrc.org/indi/abide/
   
   ADNI can be downloaded from: https://adni.loni.usc.edu/

   Since the dataset files are too large to upload here, if you need the preprocessed data please contact: ciyuan.p@outlook.com
   
4. Hyperparameter settings: see Table 4 in the paper

    <img width="567" alt="image" src="https://github.com/pcyyyy/Biologically-Plausible-Brain-Graph-Transformer/assets/43360332/9766bbe0-356c-47b0-94e2-f0d22c630cff">


5. Running
   
    a. Node importance encoding: node_perturbation.py
   
    b. Functional module extractor: FunctionalModuleExtractor.py
   
    c. Training: train_adhd.py (for ADHD-200 dataset)
   

   
