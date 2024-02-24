# Darcy Flow Non intrusive ROM Analysis
In this repo, some common used models are compared on a [Navier](https://github.com/guglielmopadula/PeriodicNavierStokes) [Stokes](https://github.com/guglielmopadula/PeriodicNavierStokes) [Dataset](https://github.com/guglielmopadula/PeriodicNavierStokes).


The models, with their main characteristics and 
performances, are summed up here.


|Model                                 |rel u train error|rel u test error super| 
|--------------------------------------|-----------------|----------------------|
|POD+Tree                              |1.8e-15          |0.08                  |
|POD DeepOnet (SpacetimeCoupled)       |0.73             |0.52                  |
|DeepOnet (SpacetimeCoupled)           |0.31             |0.39                  |
|FNO                                   |0.09             |0.08                  |
|AVNO                                  |0.20             |0.16                  |