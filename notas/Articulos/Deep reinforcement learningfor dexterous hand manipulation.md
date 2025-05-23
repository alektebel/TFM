**ID:** `202504120140ART`


#deepreinforcementlearning 
#inversereinforcementlearning 
#handdynamics 
#simulation 
#dexterousmanipulation

### TITLE:
Deep reinforcement learning
for dexterous hand manipulation
- **CONTEXTO**: 
Tesis de máster para aprender mediante técnicas de aprendizaje por refuerzo los movimientos de una mano para movimientos concretos. Aplica técnicas del estado del arte para controlar las recompensas, utilizando aprendizaje profundo inverso, para estimar las recompensas en función de políticas de ejemplo mostradas. Presentan una mejora consistente respecto al estado del arte, además de incluir técnicas de GANs para la creación de políticas.

- **RESUMEN**: 
	- Uses **Inverse Reinforcement Learning (IRL)** to infer rewards directly from human demonstrations.
    
	- Implements **Adversarial IRL (AIRL)** to improve robustness by disentangling rewards from environment dynamics.
    
	- Introduces **Particle Swarm Optimization (PSO)** to refine imperfect human demonstrations for better training data.
    
	- Proposes **state-space masking** and **reward normalization** to stabilize learning in high-dimensional spaces.

* ##### INNOVACIONES
- **ransferability**: The method generalizes to new tasks (e.g., in-hand manipulation, tool use) without task-specific reward engineering.
    
- **Sample Efficiency**: Achieves higher success rates with fewer samples compared to standard IRL.
    
- **Robustness**: The learned reward functions are less prone to exploitation by the policy during training.

###### RESULTADOS:
- Outperforms state-of-the-art IRL methods (e.g., GAIL, GCL) in simulation tasks like object relocation.
    
- Demonstrates successful transfer to novel tasks (e.g., hammering a nail) using the same framework.
    
- Quantitative evaluation shows improved reward function robustness (Table 5.1).

##### LIMITACIONES

- Evaluated only in simulation (MuJoCo); real-world transfer is future work.
    
- Combining all proposed techniques (masking, normalization, etc.) did not yield additive improvements, suggesting trade-offs.

- **APLICACION**: 
	* Simulación de la prótesis
	* Grado de libertad
	* Aprendizaje por refuerzo inverso
	* Generación de políticas 
	* 
- **ENLACE**: https://mediatum.ub.tum.de/doc/1553993/file.pdf