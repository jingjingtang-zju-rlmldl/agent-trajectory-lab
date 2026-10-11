# What the reference calculations mean

For G rewards from one coherent rollout group, this implementation defines

$$
\bar r = G^{-1}\sum_i r_i,\qquad
s=\sqrt{G^{-1}\sum_i(r_i-\bar r)^2}.
$$

Three estimators are available:

$$
A_i^{\mathrm{GRPO}}=(r_i-\bar r)/(s+\epsilon),\quad
A_i^{\mathrm{centered}}=r_i-\bar r,\quad
A_i^{\mathrm{RLOO}}=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j
=\frac G{G-1}(r_i-\bar r).
$$

We use **population** standard deviation and put epsilon **outside** the square root. Frameworks may use other conventions. RLOO above is an algebraic baseline calculation; reproducing the full [REINFORCE-style method](https://arxiv.org/abs/2402.14740) additionally requires its sampling and optimization setup.

All-equal rewards give zero relative advantages, including all-correct and all-incorrect groups. This says nothing about KL or entropy gradients. Group size one is rejected because it does not define the intended comparison.

Let $q_{it}=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$, and let the token objective be

$$
\ell_{it}=-\min(q_{it}A_i,\operatorname{clip}(q_{it},1-\epsilon_c,1+\epsilon_c)A_i).
$$

With action mask $m_{it}$, active length $L_i=\sum_t m_{it}>0$, batch size B and fixed horizon H:

$$
L_{seq}=\frac1B\sum_i\frac{\sum_t m_{it}\ell_{it}}{L_i},\quad
L_{tok}=\frac{\sum_{it}m_{it}\ell_{it}}{\sum_iL_i},\quad
L_{fixed}=\frac{\sum_{it}m_{it}\ell_{it}}{BH}.
$$

At ratio 1, away from the clipping boundary, the derivative with respect to one active token's new log probability is respectively $-A_i/(BL_i)$, $-A_i/\sum_j L_j$, or $-A_i/(BH)$. Setting B=2, lengths (2,8), advantages (+1,−1) gives (-.25,+.0625), (-.1,+.1), and (-.0625,+.0625) for H=8.

This derivation predicts [the measured finite differences](../results/mechanisms.md). It also explains why the sequence-mean scalar loss can be zero while individual gradients are nonzero. It does **not** establish that one objective is universally better, nor that equal token coefficients produce equal neural parameter updates.

## Interview questions worth answering with code

1. Why does PPO clipping use the minimum even for negative advantages?
2. Why does inserting masked padding leave the objective invariant?
3. Why is a zero-reward-variance group different from a zero scalar loss?
4. What changes if a tool observation receives an active loss mask?
5. When does group centering fail to compare like-for-like rollouts?
6. Which denominators change under gradient accumulation or distributed workers?

Question 6 remains an integration task: this reference has no distributed trainer. Keep that boundary explicit.
