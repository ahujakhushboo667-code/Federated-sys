# FusionNet: Hackathon Q&A Prep

This document contains anticipated questions from judges regarding the technical trade-offs of the FusionNet architecture, along with robust answers to defend the design.

## Q1: If you prune 40% of the model for weak edge devices, doesn't that cause unacceptable loss degradation?

**Answer:** 
Yes, aggressive pruning does increase the local loss for that specific edge device. It becomes a much "weaker learner." However, this does not ruin the global model due to two specific architectural safeguards:

1. **Federated Dropout (Sub-Model Training):** 
   Even though the weak device is missing 40% of the model, the 60% it *does* have is still calculating gradients based on its unique, local, isolated data. The gradients it generates are "noisy" but point in the generally correct mathematical direction.
   
2. **Weighted Aggregation (The Safety Net):** 
   When the central server averages all the LoRA weights together using FedAvg, it does not treat them equally. The system uses a **Contribution Weight** multiplier. An MI300X Cloud GPU gets a `5.0x` multiplier, while a sliced CPU-only PC gets a `0.1x` multiplier. 

**Conclusion:** 
Because the pruned model's weight is discounted so heavily, its higher loss doesn't "pollute" the high-quality learning done by the big GPUs. It acts as a tiny, helpful nudge to the global model.

## Q2: If the pruned models are so weak, why bother including them in the network at all?

**Answer:**
Because in industries with extreme data gravity and privacy laws (like Healthcare or Legal), that weak, CPU-only laptop might hold the *only* copy of a rare patient disease record or a specific legal precedent. 

Even a highly degraded, `0.1x` weighted mathematical nudge derived from that rare, inaccessible data is incredibly valuable for the global model to learn from. Excluding those devices means excluding the data entirely, which defeats the purpose of Federated Learning.
