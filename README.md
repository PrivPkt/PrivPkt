# PrivPkt
Privacy Preserving Collaborative Encrypted Network Traffic Classification 
<br> 
Interconnecting the following works: Differential Privacy, Federated Learning, Membership Inference Attacks, Encrypted Traffic Classification
<br> 
<br> 
<br> 
<h3>Federated Learning</h3>
The architecture for our Federated learning Setting. 
![Our Collaborative Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/privpkt.PNG)
We utilize Federated Averaging to enable the collaborative learning setting. 
Ref: https://arxiv.org/abs/1602.05629
<br> 
<br> 
<br> 
<h3>Differential Privacy</h3>
We make use of DPSGD to ensure a ceratin level of privacy.  
![DPSGD Algorithm](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/dpsgd.PNG)
Ref:https://arxiv.org/abs/1602.05629
<br> 
<br> 
<br> 
<h3>Membership Inference Attacks </h3>
We make use of Shokri et al. Membership Inference Attacks to evaluate our mitigations. 
![Membership Inference Attack Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/mia.PNG)
Ref: https://arxiv.org/abs/1610.05820
<br> 
<br> 
<br> 
<h3> Encrypted Traffic Classification</h3>
We tackle the problem of Encrypted Traffic Classification. 
We utilize the work of DeepPacket and use the ISCX Vpn 2016 Dataset to evaluate our work. 
![DeepPacket Architecture](https://raw.githubusercontent.com/PrivPkt/PrivPkt/master/deeppacket.PNG)
Ref: https://arxiv.org/abs/1709.02656
Ref: https://www.unb.ca/cic/datasets/vpn.html

