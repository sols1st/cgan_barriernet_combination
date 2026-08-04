# Safety Certification in the Latent space using Control Barrier Functions and World Models

Mehul Anand<sup>1</sup>, and Shishir Kolathaya<sup>2</sup> 

Abstract— Synthesising safe controllers from visual data typically requires extensive supervised labelling of safety-critical data, which is often impractical in real-world settings. Recent advances in world models enable reliable prediction in latent spaces, opening new avenues for scalable and data-efficient safe control. In this work, we introduce a semi-supervised framework that leverages control barrier certificates (CBCs) learned in the latent space of a world model to synthesise safe visuomotor policies. Our approach jointly learns a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive power of modern vision transformers for latent dynamics modelling. 

## I. INTRODUCTION

As autonomous systems become increasingly prevalent, ensuring their safety remains a critical challenge—particularly when using learning-based controllers, which lack inherent guarantees. Various paradigms have been developed to address safety in control systems. Constrained Reinforcement Learning (CRL) [1]–[3] encodes safety specifications as constraints during policy optimization, allowing data-driven learning but often lacking formal guarantees and risking unsafe behavior during exploration. 

In contrast, optimal control approaches such as Hamilton-Jacobi (HJ) reachability [4]–[6] provide rigorous safety analysis by characterizing safe sets through the solution of partial differential equations. However, their computational complexity scales poorly with state-space dimensionality, limiting their applicability to high-dimensional systems. 

Control Barrier Functions (CBFs) have emerged as a scalable and effective tool for safety-critical control [7]–[10]. CBF-based methods often formulate the control synthesis as a Quadratic Program (QP), enabling real-time safety filtering [7], [8], [11], [12]. However, traditional QP-based formulations are typically limited to control-affine systems and do not naturally handle input constraints. Certificatebased formulations [9], [10] extend the utility of CBFs by accommodating general nonlinear dynamics and bounded control inputs, making Control Barrier Certificates (CBCs) a versatile and scalable alternative. 

A common limitation of these approaches is the reliance on known or approximate system dynamics. While approximate models may be available in state-feedback settings [13], applying CBFs directly to visual observations remains nontrivial due to the absence of predictive models in the visual domain. Furthermore, synthesizing valid CBFs for arbitrary systems is itself a challenging task. Recent advances in neural CBFs [14], [15] leverage the expressive power of neural networks to learn barrier certificates directly from data, enabling applicability to a broader range of systems and inputs, including vision. 

Several recent efforts have explored the use of CBFs for safe control from visual observations [16]–[18], but most of them rely on the assumption of control-affine dynamics, which limits generalization. Methods integrating Neural Radiance Fields (NeRFs) with CBFs [16] show promise for visuomotor control but incur significant computational overhead, impeding real-time deployment. Others use Generative Adversarial Networks (GANs) [17] to infer 3D obstacle geometry for geometric CBF computation. Latent-spacebased approaches [19], [20] generate CBFs from encoded visuomotor representations. However, these methods often employ autoencoders or GANs, which lack the capacity to model action-conditioned temporal dynamics, making them unsuitable for control and planning. 

World models [21] address this gap by learning structured, action-aware latent representations that capture system dynamics, enabling predictive planning and safe decisionmaking. In this work, we exploit the strengths of world models to synthesize safe visuomotor policies using learned control barrier certificates in the latent space. 

The main contributions of this work are: 

• We propose a semi-supervised framework that integrates Control Barrier Certificates (CBCs) with a latent world model to enable safe visuomotor control using limited labeled data. 

• We demonstrate the effectiveness of our framework on two diverse systems—the Inverted Pendulum and Dubins Car—showing that safe policies can be synthesized without expert demonstrations. 

Section II reviews relevant background and problem formulation. Section III outlines the proposed framework, detailing the world model architecture, neural barrier synthesis, and safe controller design. Section IV presents simulation results, and Section V concludes the paper with a discussion of limitations and future directions. 

## II. PRELIMINARIES

In this section, we will formally introduce Control Barrier Certificates (CBCs), and the problem formulation for the paper. 

## A. System Description

A discrete-time control system is a tuple $S = ( X , U , f )$ where $X \subseteq \mathbb { R } ^ { n }$ is the state set of the system, $U \subseteq \mathbb { R } ^ { n _ { u } }$ is the input set of the system, and $f : X \times U  X$ describes the state evolution of the system via the following difference equation: 

$$
x (t + 1) = f (x (t), u (t)), \quad \forall t \in \mathbb {N},\tag{1}
$$

where $x ( t ) \in X$ and $u ( t ) \in U$ , denote the state and input of the system, respectively. 

Consider a set S defined as the sub-zero level set of a continuous function $B : X \subseteq \mathbb { R } ^ { n } \to$ R yielding, 

$$
S = \{x \in X \subset \mathbb {R} ^ {n}: B (x) \leq 0 \}\tag{2}
$$

$$
X - S = \{x \in X \subset \mathbb {R} ^ {n}: B (x) > 0 \}.\tag{3}
$$

We further restrict the class of S where its interior and boundary are precisely the sets given by Int $( S ) = \{ x \in X \subset$ $\mathbb { R } ^ { n } : B ( x ) < 0 \}$ and $\partial S = \{ x \in X \subset \mathbb { R } ^ { n } : B ( x ) = 0 \}$ respectively. 

## B. Control Barrier Certificates (CBCs)

In this section, we introduce the notion of a control barrier certificate, which provides sufficient conditions together with controllers for the satisfaction of safety constraints. 

Definition 1: A function $B : X \to \mathbb { R } _ { 0 } ^ { + }$ is a control barrier certificate for a discrete-time control system $S = ( X , U , f )$ if for any state, $x \in X$ there exists an input $u \in U$ , such that 

$$
B (f (x, u)) \leq B (x),\tag{4}
$$

The following lemma allows us to synthesise controllers for the discrete-time control system S, ensuring the satisfaction of safety properties. 

Lemma ${ \bf 1 } ( { \bf \Gamma } ( 8 ] ) $ : For a discrete-time control system $S =$ $( X , U , f )$ , safe set $X _ { s } \subseteq S _ { }$ , and unsafe set $X _ { u } \subseteq X - S$ the existence of a control barrier certificate, B, as defined in Definition 1, under a control policy $\pi : X \to U$ implies that the sequence state in S starting from $x _ { s } \in X _ { s }$ under the policy π do not reach any unsafe states in $X _ { u }$ 

The zero-level set of the CBC $B ( x ) = 0$ separates the unsafe regions from the safe ones. For an initial state $x _ { 0 }$ within the safe region $( x _ { 0 } \in X _ { s } ) , B ( x _ { 0 } ) \leq 0$ by condition (2). According to equation (4), which ensures $B ( x )$ it remains non-increasing, the level set is not crossed, preventing access to unsafe regions. Therefore, ensuring system safety requires computing appropriate control barrier certificates and corresponding control policies. 

## C. Problem Formulation

Given a discrete-time robotic system S as defined in (1). Let S represent samples of visuomotor observations from the safe region, U represent samples from the unsafe region, and D denote the complete set of samples (both labelled and unlabeled). 

The objective is to devise an algorithm to jointly synthesise a provably correct parameterised barrier certificate $B _ { \theta }$ and a safe parameterised policy $\pi _ { \theta }$ , with parameter θ such that it satisfies the condition (4) over the entire latent space defined by a world model, using a finite number of samples. 

In the following section, we propose an algorithmic approach to solve the above problem. 

## III. METHODOLOGY

We introduce a semi-supervised policy learning framework that employs control barrier certificates (CBCs) to jointly learn both the barrier certificate and a safe policy. This framework leverages the forward invariance properties of CBCs to ensure the learned policy remains within the safe set when initial conditions are safe. Since no barrier certificate exists initially, we start with initial safe and unsafe sets $\mathcal { X } _ { s } \subseteq { \mathcal { S } }$ and $\mathcal { X } _ { u } \subseteq \mathcal { X } - \mathcal { S }$ such that trajectories starting in $\mathcal { X } _ { s }$ never enter $\mathcal { X } _ { u }$ , making the framework semi-supervised. 

We collect datasets ${ \mathcal { C } } , { \mathcal { U } } .$ , and D corresponding to N visuomotor data points sampled from $\mathcal { X } _ { s } , \mathcal { X } _ { u } ,$ , and the state set $x ,$ respectively. The barrier function $B _ { \theta }$ and policy $\pi _ { \theta }$ are represented as neural networks learned in a latent space z by a world model. 

## A. Latent World Models

The surge of excitement in large-scale foundation models has spurred the development of extensive video generation world models that are conditioned on agent actions, particularly in areas such as self-driving, control, and general purpose video generation. These models are designed to produce video predictions based on textual descriptions or high-level action sequences. Although they have proven valuable for downstream applications such as data augmentation, their dependence on language-based conditioning restricts their effectiveness in scenarios requiring the achievement of visually specific goals. Furthermore, the reliance on diffusion models for video generation introduces significant computational overhead, limiting their suitability for testtime optimisation. Therefore, it makes sense to model the environment in a latent space rather than pixel space, as it will reduce computational requirements while retaining information about the environment. 

Recent advances in world models for reinforcement learning, such as Dreamer v3 [24], have demonstrated remarkable generality and performance across a wide range of tasks by learning environment dynamics in latent space and enabling agents to plan through imagined trajectories. Dreamer $\mathbf { v } 3$ employs a Recurrent State-Space Model (RSSM) that combines recurrent neural networks with stochastic latent variables to capture temporal dependencies and uncertainty in the environment. While this approach has led to impressive results, several shortcomings remain. Notably, the reliance on pixel-space reconstruction losses during training introduces significant computational overhead, especially when dealing with high-dimensional visual observations. This can hinder scalability and efficiency, particularly for long rollouts or large batch sizes. Additionally, the latent representations learned by RSSM, while expressive, may not always align with visually or semantically meaningful features required for visually grounded tasks, potentially limiting their effectiveness in scenarios demanding fine-grained visual understanding. 

The world model we are using consists of two parts, an observation model and a vision transformer to model the dynamics in latent space. The zero-shot encoder being used is a DINO-v2-small model [25], which gives us patch embeddings (16x16) and an embedded dimension of 384. Because DINO-v2 is a pre-trained model on a selfdistillation loss, it allows the model to learn representations effectively, capturing semantic layouts and improving spatial understanding within images. The embeddings would have better separation of features than a custom-trained encoder. Hence, we learn the latent dynamics $d _ { \theta }$ on a transition model, a transformer with a causal attention mask that takes a context length and proprioceptive information from the actual environment. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-04/40ec4849-7c3a-4760-85e0-9f41bd59e68e/ff1b038061e1aa3614fe8ba3356b9c56076e06ce94c6f65c1e4b1058ee8f7609.jpg)



Fig. 1: The Transition model is trained via autoregressive training on a latent consistency loss with a context length of 3 and a prediction horizon of 3; the actions provided are uniformly generated from an action space for generalisability of the dynamics


More specifically, at each time step t, our world model consists of the following components: 

$$
\text { Observation   model: } z _ {t} \sim \mathbf {e n c} (z _ {t} | o _ {t})\tag{5}
$$

$$
\text { Transition   model: } z _ {t + 1} \sim \mathbf {d} _ {\theta} (z _ {t + 1} | z _ {t - H: t}, a _ {t - H: t}, p _ {t - H: t})\tag{6}
$$

where the observation model encodes image observations to latent states $z _ { t } ,$ and the transition model takes in a history of past latent states of length H. $a _ { t }$ is the action taken at time step t and $p _ { t }$ is the proprioceptive information at time step t. [21] 

$$
L _ {\mathrm{pred}} = \left\| d _ {\theta} (\mathsf {e n c} _ {\theta} (o _ {t - H: t}), a _ {t - H: t}, p _ {t - H: t}) - \mathsf {e n c} _ {\theta} (o _ {t + 1}) \right\|\tag{7}
$$

## B. Barrier Network

Since we are working with a latent space, the only way to formulate a barrier function is via a neural network with tanh activation functions and an unbounded final layer. Because of how well DINO-v2 captures spatial information, the states are already well separated in the latent dimension. This makes it easier to learn the difference between safe and unsafe states; hence, it can be done with a small network. The separability of safe/unsafe regions in feature space is crucial for barrier certificate accuracy. We use a loss that learns a latent representation where safe/unsafe regions are distinctly separable. 

The loss function integrates safety segregation: 

$$
\begin{array}{c} L _ {\text { barrier }} (\theta) = \xi_ {1} \sum_ {z _ {i} \in \mathcal {C}} \max (0, B _ {\theta} (z _ {i})) \\ + \xi_ {2} \sum_ {z _ {i} \in \mathcal {U}} \max (0, - B _ {\theta} (z _ {i})) \end{array}\tag{8}
$$

where $\xi _ { 1 } , \xi _ { 2 } > 0$ are the weighting coefficients. This design ensures that safe states are mapped to non-positive barrier values, while unsafe states are mapped to non-negative values, thereby enforcing a clear separation in the latent space. 

To further regularise the barrier and encourage smooth transitions between states, we introduce the Lie loss: 

$$
\begin{array}{l} L _ {\text {lie}} (\theta) = \sum_ {z _ {i + 1} \in \mathcal {C}} \max (0, B _ {\theta} (z _ {i + 1}) - \alpha B _ {\theta} (z _ {i})) \\ \quad + \sum_ {z _ {i + 1} \in \mathcal {U}} \max (0, B _ {\theta} (z _ {i}) - \alpha B _ {\theta} (z _ {i + 1})) \end{array}\tag{9}
$$

where $\alpha > 0$ 

The Lie loss not only smoothens the barrier function but also encodes the principle that lower barrier values correspond to safer regions. This encourages monotonicity in the barrier function along safe trajectories and penalises decreases along unsafe ones, thereby improving the robustness of safety guarantees. 

The barrier network takes the current state $z _ { t }$ as input and outputs a scalar value representing the safety certificate. While it is possible to augment the input with proprioceptive information, this would require the transition function to predict proprioceptive values for the next state, which is unnecessary for our simplified environments. 

## C. Controller Synthesis

Similar to the barrier network, the controller network is also a feed-forward neural network, but its outputs are bounded by a sigmoid function to ensure valid action ranges. The controller receives as input the current latent state $z _ { t }$ and current proprioceptive information $p _ { t } .$ , and outputs the action $a _ { t } .$ . All hidden layers employ tanh activation functions to preserve the ability to represent both positive and negative values. 

Based on Lemma 1, satisfying condition (4) ensures trajectories starting in $\mathcal { X } _ { s }$ avoid $\mathcal { X } _ { u }$ . The synthesis loss enforces this condition even in the presence of latent dynamics errors: 

$$
L _ {\mathrm{syn}} (\theta) = \sum_ {x _ {i} \in \mathcal {D}} \max \left(0, B _ {\theta} (d _ {\theta} (z _ {t}, \pi_ {\theta} (z _ {t}), p _ {t})) - B _ {\theta} (z _ {t})\right)\tag{10}
$$

This loss, similar in spirit to the Lie loss in equation (9), encourages the controller to select actions that do not increase the barrier function, thereby maintaining safety over time. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-04/40ec4849-7c3a-4760-85e0-9f41bd59e68e/8e4f375f3f408eba87e2ebc1d35b33e4b2fe42b603b9531ed2eac145d6455bfb.jpg)



Fig. 2: The figure encapsulates the flow of our entire framework, the environment with a reference controller generates the image at time t, and proprioceptive information. The image is then encoded by Dino $\mathbf { v } 2 ,$ , which is further stacked onto a context of length H. This is used to predict the next state by the action given by our controller U, which is then further fed into the barrier function to check if it satisfies the lie condition


To further enhance performance while maintaining safety, we introduce an MSE loss to a user-defined policy $\pi _ { \mathrm { u s e r } } \colon$ 

$$
L _ {\pi} (\theta) = \left\| \pi_ {\theta} - \pi_ {\mathrm{user}} \right\| ^ {2}\tag{11}
$$

This term ensures that the learned controller remains close to a desired behaviour, which can encode preferences or prior knowledge. Importantly, our approach does not require expert demonstrations. This flexibility allows for scalable and data-efficient synthesis of safe controllers in complex, highdimensional environments. 

## D. Training Scheme

The training procedure is structured into two distinct stages: 

1) World Model Training: We begin by training the world model using a dataset of 50,000 state transitions generated from random actions across both environments. The model is trained with a context window of length 3 and a prediction horizon of 3, enabling it to capture short-term temporal dependencies and accurately forecast future states. 

2) Barrier and Controller Training: After establishing the world model, we collect approximately 250 trajectories with random initialisation using a reference policy to provide diverse coverage of the state space. Both the barrier network and the controller are trained simultaneously during this stage. To promote generalisation and prevent overfitting, training of the barrier network is halted once its loss converges. The controller, however, continues to be optimised until its own convergence criterion is met. This staged approach ensures that the barrier remains a robust safety certificate while allowing the controller to fully exploit the learned safe regions for effective and safe policy execution. 

```txt
Algorithm 1 Safe Controller Synthesis
Require: Datasets C, U, D, dθ, enc, Bθ, πθ, πuser
1: Initialise θ
2: for i = 1 to maxiter do
3:    Lpred ← (dθ, enc, Ot-H:t+1, at-H:t, pt-H:t)
4:    θ ← Learn θ
5:    dθ ← θ
6: end for
7: Initialise θ
8: while until Convergence do
9:    Lbarrier ← (Bθ, C, U, D)
10:    Llie ← (Bθ, C, U)
11:    Lsyn ← (Bθ, πθ, dθ, D)
12:    Lπ ← (πθ, πuser)
13:    Ltotal = Lbarrier + Llie + Lsyn + Lπ
14:    θ ← Learn θ
15:    Bθ, πθ ← θ
16: end while 
```

## IV. SIMULATIONS

In this section, we assess the efficacy of our proposed framework through two distinct case studies: the inverted pendulum system and the obstacle avoidance of a Dubin’s car. Both case studies are conducted on a computing platform equipped with an Intel i9-11900K CPU, 32GB RAM, and NVIDIA GeForce RTX 4090 GPU. 

## A. INVERTED PENDULUM

We analyse an inverted pendulum system characterised by the state vector $\mathbf { x } = [ \Theta , \dot { \Theta } ] \in [ - \pi , \pi ] \times [ - 3 . 5 , 3 . 5 ]$ , where Θ represents the angular position, and $\dot { \Theta }$ denotes the angular velocity. The control input to the system is the applied torque $\tau \in [ - 6 , 6 ]$ Nm, as illustrated in Fig. 2a. The discrete-time dynamics governing this system are given by: 


(b)


$$
\left[ \begin{array}{c} \Theta_ {t + 1} \\ \dot {\Theta} _ {t + 1} \end{array} \right] = \left[ \begin{array}{c} \Theta_ {t} \\ \dot {\Theta} _ {t} \end{array} \right] + \left(\left[ \begin{array}{c} \dot {\Theta} _ {t} \\ \frac {g}{l} \sin (\Theta_ {t}) \end{array} \right] + \left[ \begin{array}{c} 0 \\ \frac {1}{m l ^ {2}} \end{array} \right] u _ {t}\right) \Delta t\tag{12}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-04/40ec4849-7c3a-4760-85e0-9f41bd59e68e/4fe095f7703ec5b9425e1d665d65966a054d330002ae15fc6956fcfa73eb0e84.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-04/40ec4849-7c3a-4760-85e0-9f41bd59e68e/36cde933b9ad4515a6b7be1406b74fa3fe41cf41fe509f74989557f7d2efe532.jpg)



Fig. 3: a. represents the image of the inverted pendulum in the OpenAI gymnasium environment. Fig b The embedding, when projected onto the first two principal components, enables a clear visualisation of the decision boundary that partitions safe and unsafe regions. This separation is achieved with a low rate of misclassified points.


where $m$ and l represent the mass and length of the pendulum, respectively. 

The datasets, ${ \mathcal { C } } , { \mathcal { U } } ,$ and ${ \mathcal { D } } ,$ are sampled as follows: 

$$
\begin{array}{l} \mathcal {C} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in \left[ - \frac {\pi}{1 2}, \frac {\pi}{1 2} \right] \times [ - 0. 2 5, 0. 2 5 ] \right\} \\ \mathcal {U} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in [ - \pi , \pi ] \times [ - 3. 5, 3. 5 ] \setminus \left[ - \frac {\pi}{2}, \frac {\pi}{2} \right] \times [ - 1. 5, 1. 5 ] \right\} \\ \mathcal {D} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in [ - \pi , \pi ] \times [ - 3. 5, 3. 5 ] \right\} \end{array}\tag{13}
$$

Here $O _ { t }$ is a stack of the latent state representation of the RGB frames of the environment and $\dot { \Theta }$ at time t. 

A reference policy, a simple PD controller, is used to roll out data for imitation $\pi _ { \mathrm { r e f } }$ 

Visualisations of the trained barrier function are presented in both the latent space with sample points (Fig. 2b) and in the plane (Fig. 2c). These visualisations demonstrate the successful separation of the safe region from the unsafe region. As shown in Fig. 2d, the trajectories generated by the learned policy $\pi _ { \theta }$ start in the unsafe region and successfully reach the safe region, validating our approach 

## B. OBSTACLE AVOIDANCE USING DUBINS’ CAR

We set up an environment with an obstacle of fixed radius at the centre and an end goal that the agent has to reach. The agent needs to learn a controller to reach the end goal while avoiding the obstacle. The reference controller is a simple P controller. 

The state vector is defined as $[ x _ { t } , y _ { t } , \Theta _ { t } ] ^ { T } \in [ - 1 . 5 , 1 . 5 ] ^ { 2 } \times$ $[ - \pi , \pi ]$ , where $( x _ { t } , y _ { t } )$ denotes the robot’s position and $\Theta _ { t }$ represents its orientation. The agent moves at a constant speed $v { = } 1$ , while the control input u corresponds to $\dot { \Theta }$ turning the agent about its axis. It follows the discrete-time dynamics given by: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-04/40ec4849-7c3a-4760-85e0-9f41bd59e68e/c6dc48d0b2b8d54b87135f0bab50982b769808d7c30975a258940a2db2a216cf.jpg)



Fig. 4: a This figure shows the difference between the trajectories generated by our controller (green) and the reference controller(red). Fig b represents the value of barrier function over the state space of the env


$$
\left[ \begin{array}{c} \mathbf {x} _ {t + 1} \\ \mathbf {y} _ {t + 1} \\ \Theta_ {t + 1} \end{array} \right] = \left[ \begin{array}{c} \mathbf {x} _ {t} \\ \mathbf {y} _ {t} \\ \Theta_ {t} \end{array} \right] + \left(\left[ \begin{array}{c} v \cos \Theta_ {t} \\ v \sin \Theta_ {t} \\ 0 \end{array} \right] + \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right] \mathbf {u} _ {t}\right) d t\tag{14}
$$

The datasets, C, U, and D, are sampled as 

$$
\begin{array}{l} \mathcal {C} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in [ - 1. 5, 1. 5 ] ^ {2} \times [ - \pi , \pi ] \setminus [ - 0. 9, 0. 9 ] ^ {2} \times [ - \pi , \pi ] \right\} \\ \mathcal {U} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in [ - 0. 7, 0. 7 ] ^ {2} \times [ - \pi , \pi ] \right\} \\ \mathcal {D} = \left\{O _ {t} \mid \mathbf {x} _ {t} \in [ - 1. 5, 1. 5 ] ^ {2} \times [ - \pi , \pi ] \right\} \end{array}\tag{15}
$$

Here $O _ { t }$ is a stack of the latent state representation of the RGB frames of the environment and the heading angle at time t. 

## V. CONCLUSIONS

In this work, we presented a semi-supervised framework for synthesising safe visuomotor policies by leveraging control barrier certificates (CBCs) in the latent space of a world model. Our approach enables the joint learning of both a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive capabilities of modern vision transformers for latent dynamics modelling. Through case studies on the inverted pendulum and Dubins car environments, we demonstrated that our method can effectively separate safe and unsafe regions in the latent space and synthesise policies that promote safe behaviour. 

Limitations and Future Work: While our results are promising, the current framework does not provide formal safety guarantees; it instead encourages safer behaviour via the learned barrier. Future work will focus on extending this framework to provide formal verification of safety properties, scaling to higher-dimensional and more complex environments, and exploring integration with more expressive world models. We believe that combining neural CBCs with powerful world models paves the way for scalable, data-efficient, and provably safe visuomotor control in real-world robotics applications. 



[1] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in International conference on machine learning. PMLR, 2017, pp. 22–31. 





[2] A. K. Jayant and S. Bhatnagar, “Model-based safe deep reinforcement learning via a constrained proximal policy optimization algorithm,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 24 432–24 445. 





[3] F. Berkenkamp, M. Turchetta, A. Schoellig, and A. Krause, “Safe model-based reinforcement learning with stability guarantees,” in Advances in Neural Information Processing Systems, vol. 30, 2017. 





[4] J. Lygeros, “On reachability and minimum cost optimal control,” Automatica, vol. 40, no. 6, pp. 917–927, 2004. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0005109804000263 





[5] S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-jacobi reachability: A brief overview and recent advances,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC), 2017, pp. 2242–2253. 





[6] M. Tayal, A. Singh, S. Kolathaya, and S. Bansal, “A physicsinformed machine learning framework for safe and optimal control of autonomous systems,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=SrfwiloGQF 





[7] A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, “Control barrier function based quadratic programs for safety critical systems,” IEEE Transactions on Automatic Control, vol. 62, no. 8, pp. 3861–3876, 2017. 





[8] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, “Control barrier functions: Theory and applications,” in 18th European control conference (ECC). IEEE, 2019, pp. 3420– 3431. 





[9] S. Prajna, “Barrier certificates for nonlinear model validation,” Automatica, vol. 42, no. 1, pp. 117–126, 2006. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0005109805002839 





[10] P. Jagtap, S. Soudjani, and M. Zamani, “Formal synthesis of stochastic systems via control barrier certificates,” IEEE Transactions on Automatic Control, vol. 66, no. 7, pp. 3097–3110, 2020. 





[11] A. Taylor, A. Singletary, Y. Yue, and A. Ames, “Learning for safety-critical control with control barrier functions,” in Proceedings of the 2nd Conference on Learning for Dynamics and Control, ser. Proceedings of Machine Learning Research, A. M. Bayen, A. Jadbabaie, G. Pappas, P. A. Parrilo, B. Recht, C. Tomlin, and M. Zeilinger, Eds., vol. 120. PMLR, 10–11 Jun 2020, pp. 708–717. [Online]. Available: https://proceedings.mlr.press/v120/taylor20a.html 





[12] M. Tayal and S. Kolathaya, “Control barrier functions in dynamic uavs for kinematic obstacle avoidance: a collision cone approach,” arXiv preprint arXiv:2303.15871, 2023. 





[13] P. Jagtap, G. J. Pappas, and M. Zamani, “Control barrier functions for unknown nonlinear systems using gaussian processes,” in 2020 59th IEEE Conference on Decision and Control (CDC), 2020, pp. 3699– 3704. 





[14] Z. Qin, D. Sun, and C. Fan, “Sablas: Learning safe control for black-box dynamical systems,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1928–1935, 2022. 





[15] M. Tayal, H. Zhang, P. Jagtap, A. Clark, and S. Kolathaya, “Learning a formally verified control barrier function in stochastic environment,” arXiv preprint arXiv:2403.19332, 2024. 





[16] M. Tong, C. Dawson, and C. Fan, “Enforcing safety for vision-based controllers via control barrier functions and neural radiance fields,” 2023. [Online]. Available: https://arxiv.org/abs/2209.12266 





[17] H. Abdi, G. Raja, and R. Ghabcheloo, “Safe control using visionbased control barrier function (v-cbf),” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 782–788. 





[18] M. De Sa, P. Kotaru, and K. Sreenath, “Point cloud-based control barrier function regression for safe and efficient vision-based control,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 366–372. 





[19] M. Harms, M. Kulkarni, N. Khedekar, M. Jacquet, and K. Alexis, “Neural control barrier functions for safe navigation,” 2024. [Online]. Available: https://arxiv.org/abs/2407.19907 





[20] M. Tayal, A. Singh, P. Jagtap, and S. Kolathaya, “Semi-supervised safe visuomotor policy synthesis using barrier certificates,” 2025 IEEE 64th Conference on Decision and Control (CDC), 2024. 





[21] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “Dino-wm: World models on pre-trained visual features enable zero-shot planning,” 2025. [Online]. Available: https://arxiv.org/abs/2411.04983 





[22] K. Nakamura, L. Peters, and A. Bajcsy, “Generalizing safety beyond collision-avoidance via latent-space reachability analysis,” 2025. [Online]. Available: https://arxiv.org/abs/2502.00935 





[23] M. Tayal, A. Singh, P. Jagtap, and S. Kolathaya, “Cp-ncbf: A conformal prediction-based approach to synthesize verified neural control barrier functions,” 2025. [Online]. Available: https: //arxiv.org/abs/2503.17395 





[24] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” arXiv preprint arXiv:2301.04104, 2023. 





[25] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” 2024. [Online]. Available: https://arxiv.org/abs/2304.07193 

