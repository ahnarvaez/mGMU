# A Minimal Gated Multi-Modal Unit for Sensor Fusion in Insurance Telematics

This is a companion repository for our paper titled ["A Minimal Gated Multi-Modal Unit for Sensor Fusion in Insurance Telematics"](https://ieeexplore.ieee.org/document/10234395).

## Minimal Gated Multi-Modal Unit (mGMU) diagram.

![mGMU diagram](mgmu.jpg "")

## Equations that governs the mGMU

$$
	h_{ij} = \tanh\left ( W_{i} \cdot X_{ij}\right),\;\text{para }\;i = 1,\ldots,n,\;\text{y }\;j = 1,\ldots,C
$$

$$
	z_{i} = \sigma \left (\left [ X_{1},\dots, X_{n} \right ] \cdot 
  \left [
  \underbrace{        
   Wz_{i},\dots,Wz_{i} 
     }_{n \text{ times}}
  \right ]
 \right ),\;\text{para }\;i = 1,...,n
$$

$$
    O_{T} = \sum_{i=1}^{n}  \sum_{j=1}^{C} \left ( z_{i} \ast h_{ij} \right )
$$

$$
\Theta = \lbrace W_{11},..., W_{nC}, Wz_{1},...,Wz_{n} \rbrace
$$

Where $X_{i} \in \mathbb{R}^{S \times  C}$ is represents the information from modality $i$. $X_{ij} \in \mathbb{R}^{S \times  1}$ is the channel $j$ of the modality $i$. $W_{i} \in \mathbb{R}^{S \times S}$ and $Wz_{i} \in \mathbb{R}^{C \times 1}$ are the weighted matrices.  Further, $n$ is the number of modalities, $S$ the number of samples, $C$ the number of channels, $\theta$ the parameters to be learned, $\left [ .,. \right ]$ the concatenation operator, and $O_{T}$ the output of the gate.

## Folder Content

_BOFM  (Implements the Bag of Features algorithm for a multivariate time series)
- BoFM.py    
- BoFM_extraccion.py:  

_Gates (mGMU and GMU keras layers)
- Gates.py

_UAH_timeSeries (time series dataset from [UAH-Driveset](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/))
- UAH.mat (Matlab binary file)

_root
- BoFM-mGMU-FCL-UAH.ipynb (Example of BOFM as feature extractor stage)
- mgmu.jpg  (mGMU diagram)
- modelos.py (Function to build the keras models used in the article)
- Models_test.ipynb (modelos.py use example (2Ires_mGMU_GAP_FCL model))
- README.md (this file)
