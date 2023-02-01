# PrivPkt
**Privacy Preserving Collaborative Encrypted Network Traffic Classification**

Interconnecting the following works: 
* Differential Privacy
* Federated Learning (We plan to add split learning) 
* Membership Inference Attacks
* Encrypted Traffic Classification
___
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) 
[![License](https://img.shields.io/pypi/l/mia.svg)]() 
<a href="https://https://github.com/PrivPkt/PrivPkt/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/PrivPkt/PrivPkt"></a>
<a href="https://github.com/kaiiyer/PrivPkt/PrivPkt"><img alt="GitHub forks" src="https://img.shields.io/github/forks/PrivPkt/PrivPkt"></a>
<a href="https://github.com/PrivPkt/PrivPkt/graphs/contributors" alt="Contributors">
<img src="https://img.shields.io/github/contributors/PrivPkt/PrivPkt" /></a>
<a href="https://github.com/PrivPkt/PrivPkt/graphs/stars" alt="Stars">
<img src="https://img.shields.io/github/stars/PrivPkt/PrivPkt" /></a>
[![Open Source Love svg1](https://badges.frapsoft.com/os/v3/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

PAPER: https://www.researchgate.net/profile/Ezzeldin-Tahoun/publication/345974499_PrivPkt_Privacy_Preserving_Collaborative_Encrypted_Traffic_Classification/links/5fb378d592851cf24cd85891/PrivPkt-Privacy-Preserving-Collaborative-Encrypted-Traffic-Classification.pdf


<br><br><br>
### Federated Learning

![Our Collaborative Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/Images/privpkt.PNG)

We utilize Federated Averaging to enable the collaborative learning setting. 

Ref: https://arxiv.org/abs/1602.05629



<br><br><br>
### Differential Privacy
We make use of DPSGD to ensure a ceratin level of privacy.  

![DPSGD Algorithm](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/Images/dpsgd.PNG)

Ref:https://arxiv.org/abs/1602.05629



<br><br><br>
### Membership Inference Attacks 
We make use of Shokri et al. Membership Inference Attacks to evaluate our mitigations. 

![Membership Inference Attack Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/Images/mia.PNG)

Ref: https://arxiv.org/abs/1610.05820



<br><br><br>
### Encrypted Traffic Classification
We tackle the problem of Encrypted Traffic Classification. 
We utilize the work of DeepPacket and use the ISCX Vpn 2016 Dataset to evaluate our work. 

![DeepPacket Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/Images/deeppacket.PNG)

Ref: https://arxiv.org/abs/1709.02656

Ref: https://www.unb.ca/cic/datasets/vpn.html
