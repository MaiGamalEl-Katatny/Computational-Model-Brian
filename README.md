# Major Depressive Disorder Computational Model using Brian2 Simulator
In this project, a computational spiking network model of two cortical brain regions is implemented using Brian2 simulator for studying and better
understanding Major Depressive Disorder (MDD). This project is based on the model presented in: Ramirez-Mahaluf, Juan P., et al. "A computational model of Major Depression: the role of glutamate dysfunction on cingulo-frontal network dynamics." Cerebral cortex 27.1 (2017): 660-679.

The project was implemented and presented at the eigtht edition of the Latin American School on Computational Neuroscience – LASCON VIII held at the University of São Paulo in the city of São Paulo, Brazil (LASCON 2020: http://sisne.org/lascon). Presentation of the project can be found at: https://www.youtube.com/watch?v=piW5kmoQGR4

Project Description and Main Results:

Major Depressive Disorder (MDD) is a common psychiatric disorder worldwide which can lead to
suicide. It is usually accompanied by negative thoughts, sadness, and lack of energy. Despite being
the leading cause of disability worldwide affecting more than 250 million people, there is a lack
of mechanistic models that could provide explanations for its symptoms. Studies suggest that the
ventral anterior cingulate cortex (vACC) is the main hub connecting regions that drive alterations in
system dynamics in MDD. It has been reported that MDD causes hyperactivity in vACC, and proper
treatment can suppress this activity. In this project, we reproduced some of the results obtained by
(Ramirez-Mahaluf, Juan P., et al. 2017) where we implemented a biophysical computational model
of vACC and the dorso-lateral prefrontal cortex (dlPFC) and their interactions using Brian2 simulator.
We used this model to process emotional and cognitive tasks. To simulate MDD in the model, we
slowed glutamate decay in vACC. This simulation led to hyperactivity and sustained
activation in vACC that is not suppressed by dlPFC activation. Moreover, dlPFC was not persistently 
activated in response to a cognitive signal because of the hyperactive vACC network. We have also stimulated two
types of treatments of MDD which are Selective Serotonin Reuptake Inhibitor (SSRI) and NMDA Receptor Antagonist.
Our results show that the model was able to reproduce the MDD symptoms caused by glutamate dysfunction. It also 
showed the success of the two simulated treatments. 
Afterward, we studied if MDD is related to synchronization in these
networks, since synchronization is a mechanism for neuronal information processing within a brain
area as well as for communication between different brain areas (Pikovsky, Arkady, et al. 2003). This
model could help in understanding the dynamics and symptoms of MDD, and it could also help in
predicting the outcome of treatment such as SSRI treatments and others.
