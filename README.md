# PrivPkt
**Privacy Preserving Collaborative Encrypted Network Traffic Classification**

Interconnecting the following works: 
* Differential Privacy
* Federated Learning
* Membership Inference Attacks
* Encrypted Traffic Classification
___
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) [![HitCount](http://hits.dwyl.io/PrivPkt/PrivPkt.svg)](http://hits.dwyl.io/PrivPkt/PrivPkt)
[![License](https://img.shields.io/pypi/l/mia.svg)]() 




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
