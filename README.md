# TIMBRE
### ***T***racking ***I***nformative ***M***ultivariate ***B***rain ***R***hythms ***E***fficiently
<img src="Block%20Diagram.svg" width="500" alt="TIMBRE Block Diagram">

TIMBRE is a complex-valued neural network that can recover patterns from multi-channel oscillations carrying information about a variable of interest. It was primarily developed and tested using multi-electrode recordings of local field potentials (LFPs) measured in the hippocampus of rats. 

## Files
1. **TIMBRE.py** - neural network implementation of carrier-free decoding.
2. **carrier_based.py** - carrier-based decoding of LFPs (see **References** for more detailed description)
3. Data preprocessing
4. Running TIMBRE - 0-n hidden nodes
5. plotting performance and receptive fields
6. recipes

## References
Agarwal, B Lustig, S Akera, E Pastalkova, AK Lee, FT Sommer. [News without the buzz: reading out weak theta rhythms in the hippocampus.](https://www.biorxiv.org/content/10.1101/2023.12.22.573160v1) bioRxiv, 2023.12. 22.573160
