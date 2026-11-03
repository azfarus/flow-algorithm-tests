# Weak-Form MeanFlow: Derivations Step by Step

*Companion to `Weak-Form-MeanFlow.md` · 21 October 2026*

This note derives every result in the proposal from first principles. It explains the reason behind each step as well as the step itself. The derivations use only the chain rule, integration by parts, and the tower property of conditional expectation.

---

## 0. The whole argument in one paragraph

MeanFlow learns a **displacement**. It does not learn a velocity. Let $\Delta = (t-r)\,u$ be the distance a particle travels between time $r$ and time $t$. Then the MeanFlow identity says only this: *as you ride along with a particle, its displacement grows at the particle's velocity*, $D_t\Delta = v$. Checking this pointwise means differentiating $\Delta$, and therefore the network, which is where the JVP comes from. The alternative is to check the identity *on average* against many smooth test fields $\varphi$. Averaging under the moving distribution $\mu_t$ allows integration by parts along the flow, which moves the derivative from $\Delta$ onto $\varphi$. Because $\varphi$ is a function we chose, its derivative is free. Every remaining appearance of $v$ is linear and is multiplied by a function of $z_t$. So $v$ can be replaced by the per-sample velocity $e-x$ without bias, and no target network is needed. The squared averages form a quadratic in $u$, so the optimality condition is linear. The only solution with finite energy is the true $u^*$.

The rest of the note makes each sentence of that paragraph precise.

| Step | Result | Main tool |
| --- | --- | --- |
| 1 | Transport lemma: $\frac{d}{dt}\mathbb E[h(z_t,t)]=\mathbb E[D_t h]$ | Chain rule + tower property |
| 2 | MeanFlow residual $R(u)=D_t\Delta-v$ | Fundamental theorem of calculus along a trajectory |
| 3 | Weak identity (Proposition 1) | Transport lemma used as integration by parts |
| 4 | Per-sample velocity is unbiased (Proposition 2) | Tower property |
| 5 | Monte Carlo integrand $g$ | Change of measure to uniform sampling |
| 6 | Convex loss, linear stationarity (Proposition 3) | Affine map composed with a square |
| 7 | Unique zero, diagonal implied, no uniform stability (Proposition 4) | Method of characteristics + $L^2$ |
| 8 | U-statistic, decoupled forcing, Rao–Blackwell | Independence, Gaussian conditioning |
| 9 | Closed-form readout | Normal equations |
| 10 | Fourier test fields | Bochner's theorem |
| 11 | Inference | Definition of the flow map |
| 12 | 1-D Gaussian check of everything | Closed forms |
| A | Why the dropped components fail | Counterexamples |

---

## 1. Paths, velocities and the transport lemma

### Step 1.1 — The per-sample path

Draw a data point $x$ and independent noise $e\sim\mathcal N(0,I)$. Connect them by a straight line:

$$
z_t=(1-t)\,x+t\,e,\qquad t\in[0,1].
$$

Differentiating in $t$ with $x,e$ fixed gives the **per-sample (conditional) velocity**:

$$
\frac{d z_t}{dt}=e-x=:v_c .
$$

> **Intuition.** Each particle moves at constant speed in a straight line. Data sits at $t=0$, noise at $t=1$. The velocity $v_c$ is known exactly for every sample, but it depends on the hidden pair $(x,e)$. At a given location $z$, many different pairs pass through, each with a different $v_c$.

### Step 1.2 — The marginal velocity

Let $\mu_t$ be the distribution of $z_t$. A single point $z$ at time $t$ is reached by many lines, so the natural velocity field to assign to $z$ is the average over all lines through it:

$$
v(z,t)=\mathbb E\bigl[v_c\mid z_t=z\bigr].
$$

This is the **marginal velocity**. It is a deterministic field that can be evaluated at any point, which $v_c$ is not.

> **Intuition.** Picture traffic at an intersection. Individual cars ($v_c$) go in many directions. The *flow* of traffic ($v$) is their average. A deterministic ODE can only follow the average.

### Step 1.3 — The transport lemma (the workhorse)

Define the **material derivative** along $v$:

$$
D_t h=\partial_t h+v\cdot\nabla_z h .
$$

**Lemma 1 (transport).** For any smooth, bounded $h(z,t)$ with bounded derivatives,

$$
\frac{d}{dt}\,\mathbb E\bigl[h(z_t,t)\bigr]=\mathbb E\bigl[(D_t h)(z_t,t)\bigr].
$$

*Proof.* Use the chain rule along each sampled line, then the tower property:

$$
\begin{aligned}
\frac{d}{dt}\,\mathbb E\bigl[h(z_t,t)\bigr]
&=\mathbb E\Bigl[\partial_t h(z_t,t)+\nabla_z h(z_t,t)\cdot\frac{dz_t}{dt}\Bigr]
&&\text{(chain rule; boundedness allows swapping } d/dt \text{ and } \mathbb E)\\
&=\mathbb E\bigl[\partial_t h(z_t,t)+\nabla_z h(z_t,t)\cdot v_c\bigr]
&&\text{(Step 1.1)}\\
&=\mathbb E\Bigl[\partial_t h(z_t,t)+\nabla_z h(z_t,t)\cdot\mathbb E[v_c\mid z_t]\Bigr]
&&(\nabla_z h(z_t,t)\text{ is a function of } z_t)\\
&=\mathbb E\bigl[(D_t h)(z_t,t)\bigr].
&&\blacksquare
\end{aligned}
$$

> **Intuition.** To track how the average of $h$ changes, you could follow every individual particle with its own velocity $v_c$. But $\nabla h$ only sees *where* a particle is, not which line it came from, so averaging $v_c$ first gives the same answer. Two consequences follow: the marginal velocity $v$ moves the distribution exactly as the real particles do, and **$v$ and $v_c$ are interchangeable whenever they multiply a function of $z_t$ alone.** That second fact is the reason no target network is needed (Section 4).

### Step 1.4 — Lemma 1 is the continuity equation

Write $\mathbb E[h(z_t,t)]=\int h\,d\mu_t$. With densities $\rho_t$, Lemma 1 says that for all test functions $h$,

$$
\int h\,\partial_t\rho_t\,dz=\int (v\cdot\nabla h)\,\rho_t\,dz=-\int h\,\nabla\!\cdot(v\rho_t)\,dz,
$$

which is the weak form of

$$
\partial_t\mu_t+\nabla\!\cdot(v\,\mu_t)=0 .
$$

Lemma 1 is therefore the continuity equation in a form that needs no densities. Every proof below uses the lemma directly.

### Step 1.5 — The product rule for $D_t$

Because $D_t$ is a first-order differential operator, it obeys the product rule. For vector fields $f,g$:

$$
D_t(f\cdot g)=(D_t f)\cdot g+f\cdot(D_t g).
$$

Here $D_t$ acts on each component. Combined with Lemma 1, this gives integration by parts *along the flow*:

$$
\boxed{\;\int_a^b\mathbb E\bigl[f\cdot D_t g\bigr]\,dt=\Bigl[\mathbb E[f\cdot g]\Bigr]_{t=a}^{t=b}-\int_a^b\mathbb E\bigl[g\cdot D_t f\bigr]\,dt\;}
$$

where every expectation is over $z_t$. To prove it, apply Lemma 1 to $h=f\cdot g$, use the product rule, and integrate over $t\in[a,b]$.

> **Intuition.** This is ordinary integration by parts, $\int f g' = [fg]-\int f' g$, carried out in the frame of a particle riding the flow. It works because $\mu_t$ is carried by $v$: nothing enters or leaves the frame, so no extra divergence term appears.

---

## 2. The average velocity and the MeanFlow identity

### Step 2.1 — The flow map

Let $\Phi_{s,t}(z)$ be the solution of the ODE $\dot y=v(y,\tau)$ that starts at $y=z$ at time $t$, evaluated at time $s$. It maps "where a particle is at time $t$" to "where it was or will be at time $s$". It has two properties used below:

- **Composition:** $\Phi_{r,s}\circ\Phi_{s,t}=\Phi_{r,t}$.
- **Pushforward:** if $z\sim\mu_t$, then $\Phi_{r,t}(z)\sim\mu_r$. This holds because $v$ transports $\mu_t$ (Step 1.4).

Note that $\Phi_{r,t}(z_t)$ has the *law* of $z_r$ but is **not** the same random variable as $z_r=(1-r)x+re$. The marginal flow does not follow the straight lines. Appendix A.1 shows why this distinction matters.

### Step 2.2 — Average velocity and displacement

The **average velocity** over $[r,t]$ is total displacement divided by elapsed time:

$$
u^*(z,r,t)=\frac{1}{t-r}\int_r^t v\bigl(\Phi_{\tau,t}(z),\tau\bigr)\,d\tau=\frac{z-\Phi_{r,t}(z)}{t-r}.
$$

The second equality is the fundamental theorem of calculus along the trajectory. Define the **displacement**

$$
\Delta^*(z,r,t)=(t-r)\,u^*(z,r,t)=z-\Phi_{r,t}(z).
$$

> **Intuition.** $u^*$ is the single velocity that would take you from $\Phi_{r,t}(z)$ to $z$ in one straight jump. One-step generation uses exactly this jump.

### Step 2.3 — Differentiating along a trajectory

Fix $r$ and follow a particle $y(t)=\Phi_{t,t_0}(z_0)$ of the marginal flow. Its starting point $\Phi_{r,t}(y(t))=y(r)$ does not depend on $t$, by the composition property. So

$$
\frac{d}{dt}\,\Delta^*\bigl(y(t),r,t\bigr)=\frac{d}{dt}\bigl[y(t)-y(r)\bigr]=v\bigl(y(t),t\bigr).
$$

The left side is a derivative along the flow, so it equals $D_t\Delta^*$. This gives

$$
D_t\Delta^*=v .
$$

Expanding $D_t[(t-r)u^*]=u^*+(t-r)D_t u^*$, since $D_t(t-r)=1$, gives the **MeanFlow identity**:

$$
u^*=v-(t-r)\,D_t u^* .
$$

### Step 2.4 — The residual is "displacement rate minus velocity"

For any candidate field $u$, let $\Delta=(t-r)u$ and define the residual

$$
R(u)=u-v+(t-r)\,D_t u=D_t\Delta-v .
$$

The second form is the key rewrite. It shows that $R$ measures how far the candidate displacement is from growing at the true velocity.

### Step 2.5 — Where the JVP and the moving target come from

Evaluating $R(u_\theta)$ requires $D_t u_\theta=\partial_t u_\theta+v\cdot\nabla_z u_\theta$. That is a Jacobian–vector product of the network in the direction $(v,1)$. MeanFlow regresses $u_\theta$ onto the stop-gradient target $v_c-(t-r)D_t u_\theta$. That target contains the network itself, so it moves as the network trains. Both problems come from **evaluating $D_t$ on $u$**. The weak form never does this.

---

## 3. The weak identity (Proposition 1)

### Step 3.1 — Test the residual

Take a smooth vector field $\varphi(z,r,t)$, bounded with bounded first derivatives. Fix $r$ and integrate the residual against $\varphi$ over $t\in[r,1]$ and over $z\sim\mu_t$:

$$
W_r(\varphi;u)=\int_r^1\mathbb E\bigl[\varphi\cdot R(u)\bigr]\,dt=\int_r^1\mathbb E\bigl[\varphi\cdot D_t\Delta\bigr]\,dt-\int_r^1\mathbb E\bigl[\varphi\cdot v\bigr]\,dt .
$$

> **Intuition.** Instead of requiring $R=0$ at every point, we require it to have zero overlap with every smooth "probe" $\varphi$. If enough probes report zero, $R$ must be zero (Section 7). Each probe sees only a weighted average, and averages can be computed without differentiating $u$.

### Step 3.2 — Move $D_t$ onto the test field

Apply the boxed rule of Step 1.5 with $f=\varphi$, $g=\Delta$, $a=r$, $b=1$:

$$
\int_r^1\mathbb E\bigl[\varphi\cdot D_t\Delta\bigr]\,dt
=\mathbb E\bigl[\varphi\cdot\Delta\bigr]_{t=1}-\mathbb E\bigl[\varphi\cdot\Delta\bigr]_{t=r}-\int_r^1\mathbb E\bigl[\Delta\cdot D_t\varphi\bigr]\,dt .
$$

### Step 3.3 — Evaluate the boundaries

- **At $t=r$:** $\Delta=(t-r)u=0$. This boundary vanishes for any bounded $u$. *The factor $t-r$ was designed for this.*
- **At $t=1$:** $z_1\sim\mu_1=\mathcal N(0,I)$ exactly, because $z_1=e$, and $\Delta=(1-r)\,u(z_1,r,1)$:

$$
\mathbb E\bigl[\varphi\cdot\Delta\bigr]_{t=1}=(1-r)\,\mathbb E_{z_1\sim\mathcal N(0,I)}\bigl[\varphi(z_1,r,1)\cdot u(z_1,r,1)\bigr].
$$

### Step 3.4 — Assemble

Substituting $\Delta=(t-r)u$ in the remaining integral:

$$
W_r(\varphi;u)=-\int_r^1\mathbb E\bigl[\varphi\cdot v\bigr]dt\;-\;\int_r^1\mathbb E\bigl[(t-r)\,u\cdot D_t\varphi\bigr]dt\;+\;(1-r)\,\mathbb E\bigl[\varphi(z_1,r,1)\cdot u(z_1,r,1)\bigr].
$$

Integrating over $r\in[0,1]$ gives **Proposition 1**:

$$
\boxed{
\begin{aligned}
W(\varphi;u)={}&-\int_0^1\!\!\int_r^1\mathbb E\bigl[\varphi\cdot v\bigr]\,dt\,dr
\;-\int_0^1\!\!\int_r^1\mathbb E\bigl[(t-r)\,u\cdot D_t\varphi\bigr]\,dt\,dr\\
&+\int_0^1(1-r)\,\mathbb E\bigl[\varphi(z_1,r,1)\cdot u(z_1,r,1)\bigr]\,dr
\end{aligned}}
$$

The three terms are the **forcing**, the **transport**, and the **boundary** terms. No derivative of $u$ appears.

### Step 3.5 — Where did the "$+u$" in the residual go?

In the original form, $R(u)=u-v+(t-r)D_tu$ has a bare $+u$ term. In the derivation above, that term was absorbed by the rewrite $R=D_t\Delta-v$, because $D_t(t-r)=1$. If you instead integrate $(t-r)\,\varphi\cdot D_tu$ by parts directly, the derivative falls on $(t-r)\varphi$ and produces

$$
D_t\bigl[(t-r)\varphi\bigr]=\varphi+(t-r)\,D_t\varphi .
$$

Its first piece gives $-\mathbb E[\varphi\cdot u]$, which cancels the $+\mathbb E[\varphi\cdot u]$ exactly. Both routes give the same result. The cancellation is the identity $D_t(t-r)=1$ in disguise.

> **Intuition.** $u$ appears twice in $R$: once directly, and once inside the rate of change of the displacement. Integration by parts turns the second appearance into $-u$ plus a term involving $D_t\varphi$, and the two bare $u$'s cancel. What remains is "displacement, weighted by how fast the probe changes along the flow".

### Step 3.6 — The right side is a definition

The derivation assumed $u$ smooth. The right-hand side, however, makes sense for any $u$ that is square-integrable under $\mu_t\,dt\,dr$. We take it as the **definition** of $W(\varphi;u)$ for all such $u$, including neural networks. This is the standard move in the theory of weak solutions of PDEs.

### Step 3.7 — Regularity assumptions, stated once

The steps above need:

- $\varphi$, $\partial_t\varphi$ and $\nabla_z\varphi$ bounded (true for the Fourier fields of Section 10);
- $\mathbb E|v_c|^2<\infty$ and $u$ square-integrable, so that every product above is integrable by Cauchy–Schwarz;
- $v$ regular enough for its flow to exist (Lipschitz in $z$ on $t>0$ suffices) where flow maps are used.

Steps 3.1–3.4 use only Lemma 1, so they need no flow map at all.

---

## 4. Replacing $v$ by $e-x$ without bias (Proposition 2)

### Step 4.1 — Every $v$ multiplies a function of $z_t$

Look at where $v$ appears in the boxed formula of Step 3.4:

- the forcing term $\varphi(z_t,r,t)\cdot v(z_t,t)$;
- the transport term, through $D_t\varphi=\partial_t\varphi+(v\cdot\nabla_z)\varphi$, as $(t-r)\,u(z_t,r,t)\cdot\bigl[(v\cdot\nabla_z)\varphi\bigr](z_t,r,t)$.

In both terms, $v$ enters **linearly**, multiplied by quantities that depend only on $(z_t,r,t)$. The boundary term has no $v$.

### Step 4.2 — Tower property

For any function $F(z_t)$ (vector- or matrix-valued) with $\mathbb E|F(z_t)|^2<\infty$:

$$
\mathbb E\bigl[F(z_t)\,v_c\bigr]=\mathbb E\bigl[F(z_t)\,\mathbb E[v_c\mid z_t]\bigr]=\mathbb E\bigl[F(z_t)\,v(z_t,t)\bigr].
$$

Apply this to each term from Step 4.1. Replacing $v$ by $v_c=e-x$ everywhere leaves $W(\varphi;u)$ **exactly unchanged**. This is Proposition 2.

### Step 4.3 — What changes and what does not

| Quantity | Swap $v\to v_c$ allowed? | Reason |
| --- | --- | --- |
| $W(\varphi;u)$ | Yes, exactly | Linear in $v$, coefficients depend on $z_t$ only |
| $W(\varphi;u)^2$ inside one expectation | No | The square would involve $v_c$ twice; Section 8 handles this |
| The strong residual $R(u)$ | Yes, also linear in $v$ | But evaluating it needs $\nabla_z u$, the JVP |
| A learned or EMA field in place of $v$ | No | It is not $\mathbb E[v_c\mid z_t]$, so the tower step fails |

> **Intuition.** Linearity is what makes the swap legal. You may replace a quantity by a noisy unbiased copy anywhere it appears linearly, multiplied by things that don't depend on the noise. This is the same reason conditional flow matching works. A learned field is not an unbiased copy of $v$, which is why using one would reintroduce a teacher and a bias.

---

## 5. Turning integrals into a sampleable integrand

### Step 5.1 — Uniform sampling on the triangle

The region $\{0\le r<t\le1\}$ has area $1/2$. If $(r,t)$ is uniform on it, with density $2$, then

$$
\int_0^1\!\!\int_r^1 f(r,t)\,dt\,dr=\tfrac12\,\mathbb E_{(r,t)\sim U(\triangle)}\bigl[f(r,t)\bigr].
$$

For a general sampling density $\pi(r,t)$ on the triangle, the factor $\tfrac12$ becomes $1/\pi(r,t)$ (importance weighting).

### Step 5.2 — The boundary term

With $r'\sim U[0,1]$, which has density $1$:

$$
\int_0^1(1-r)\,\mathbb E\bigl[\varphi(z_1,r,1)\cdot u(z_1,r,1)\bigr]dr=\mathbb E_{r',z_1}\bigl[(1-r')\,\varphi(z_1,r',1)\cdot u(z_1,r',1)\bigr].
$$

### Step 5.3 — The per-sample integrand

Collect one sample $\xi=(x,e,r,t,z_1,r')$, all components independent, and use Section 4 to swap $v\to v_c$:

$$
\begin{aligned}
g(\xi;\varphi,u)={}&\tfrac12\Bigl[-\varphi\cdot v_c-(t-r)\,u(z_t,r,t)\cdot\bigl(\partial_t\varphi+(v_c\cdot\nabla_z)\varphi\bigr)\Bigr]_{(z_t,r,t)}\\
&+(1-r')\,\varphi(z_1,r',1)\cdot u(z_1,r',1),
\end{aligned}
$$

so that $\mathbb E[g]=W(\varphi;u)$ exactly.

> **Why a separate $z_1$?** The boundary term needs a sample from $\mu_1=\mathcal N(0,I)$ at its own random $r'$. Reusing the path's $(r,t)$ would bias the triangle term. A fresh standard Gaussian costs nothing and is an exact sample.

---

## 6. The loss: convex, with a linear optimality condition (Proposition 3)

### Step 6.1 — Affine decomposition

From Step 3.4, $W(\varphi;\cdot)$ is a linear function of $u$ minus a constant:

$$
W(\varphi;u)=\ell_\varphi(u)-b_\varphi,
$$

$$
\ell_\varphi(u)=-\iint\mathbb E\bigl[(t-r)\,u\cdot D_t\varphi\bigr]\,dt\,dr+\int(1-r)\,\mathbb E\bigl[\varphi\cdot u\bigr]_{t=1}dr,
\qquad
b_\varphi=\iint\mathbb E\bigl[\varphi\cdot v_c\bigr]\,dt\,dr .
$$

The integrals run over the triangle. Under uniform sampling, $b_\varphi=\tfrac12\,\mathbb E_{U(\triangle)}[\varphi\cdot v_c]$.

This holds for **every** test field $\varphi$, linear or not, because $u$ never passes through $\varphi$. Section A.2 contrasts this with moment matching, where $u$ sits *inside* $\varphi$.

### Step 6.2 — The population loss

Let $\nu$ be a probability distribution over test fields:

$$
\mathcal L(u)=\mathbb E_{\varphi\sim\nu}\bigl[W(\varphi;u)^2\bigr]=\mathbb E_{\varphi\sim\nu}\bigl[(\ell_\varphi(u)-b_\varphi)^2\bigr].
$$

**Convexity.** For each $\varphi$, the map $u\mapsto(\ell_\varphi(u)-b_\varphi)^2$ is a convex function (a square) composed with an affine map, so it is convex. An average of convex functions is convex.

### Step 6.3 — The kernel view

Pretend $R(u)$ is an ordinary function, which it is when $u$ is smooth. Then $W(\varphi;u)=\langle\varphi,R\rangle$ and

$$
\mathcal L(u)=\mathbb E_\nu\bigl[\langle\varphi,R\rangle^2\bigr]=\iint R(\xi)^{\top}\,k(\xi,\xi')\,R(\xi')\,d\xi\,d\xi',\qquad k(\xi,\xi')=\mathbb E_\nu\bigl[\varphi(\xi)\,\varphi(\xi')^{\top}\bigr],
$$

where $\xi=(z,r,t)$ and $d\xi$ stands for $\mu_t(dz)\,dt\,dr$. So $\mathcal L$ is a **kernel-weighted squared norm of the residual**. If $k$ is strictly positive definite, $\mathcal L(u)=0$ forces $R=0$.

> **Intuition.** Squaring a single probe's reading would tell you about one direction of $R$. Averaging the squared readings of many random probes measures $R$ in every direction at once, weighted by how often the probe family looks there. That weighting is $k$.

### Step 6.4 — First variation is linear in $u$

Perturb $u$ to $u+\varepsilon h$. Because $\ell_\varphi$ is linear, $W(\varphi;u+\varepsilon h)=W(\varphi;u)+\varepsilon\,\ell_\varphi(h)$. So

$$
\frac{d}{d\varepsilon}\mathcal L(u+\varepsilon h)\Big|_{\varepsilon=0}=2\,\mathbb E_\nu\bigl[W(\varphi;u)\,\ell_\varphi(h)\bigr].
$$

Setting this to zero for every direction $h$ gives the **stationarity condition**

$$
\mathbb E_\nu\bigl[(\ell_\varphi(u)-b_\varphi)\,\ell_\varphi(h)\bigr]=0\quad\text{for all }h,
$$

which is **linear in $u$**. The loss itself is quadratic. "Linear loss" would be wrong, because a nonzero linear functional has no minimum.

> **Intuition.** This is least squares in function space. The residual is an affine function of the unknown, so the normal equations are linear. For a neural network the loss is non-convex in the weights $\theta$ but convex in the function $u_\theta$ the weights produce. Flow matching is in exactly the same position.

---

## 7. Identifiability (Proposition 4)

The question is: if $\mathcal L(u)=0$, must $u=u^*$?

### Step 7.1 — Zero loss ⇒ zero weak residual

If $k$ is strictly positive definite on the open triangle $\times\,\mathbb R^d$ (Section 10 gives a family where it is), Step 6.3 shows $\mathcal L(u)=0$ implies $W(\varphi;u)=0$ for a dense set of $\varphi$ supported in the interior. For such $\varphi$ the boundary term vanishes, so

$$
\iint\mathbb E\bigl[\varphi\cdot(D_t\Delta-v)\bigr]\,dt\,dr=0\quad\text{for all interior }\varphi .
$$

This says $D_t\Delta=v$ in the sense of distributions.

### Step 7.2 — Solve along characteristics

Fix $r$ and follow a particle $y(t)$ of the marginal flow. $D_t\Delta=v$ says

$$
\frac{d}{dt}\Bigl[\Delta\bigl(y(t),r,t\bigr)-y(t)\Bigr]=v-v=0 .
$$

So $\Delta-y$ is **constant along each trajectory**. A trajectory is labelled by where it is at time $r$, namely $\Phi_{r,t}(z)$. So there is some function $c(\cdot,r)$ with

$$
\Delta(z,r,t)=z-\Phi_{r,t}(z)+c\bigl(\Phi_{r,t}(z),r\bigr)=\Delta^*(z,r,t)+c\bigl(\Phi_{r,t}(z),r\bigr).
$$

Dividing by $t-r$:

$$
u=u^*+\frac{c\bigl(\Phi_{r,t}(z),r\bigr)}{t-r}.
$$

### Step 7.3 — The constant of integration is the diagonal condition

The equation $D_t\Delta=v$ fixes how fast the displacement grows, but not its starting value at $t=r$. That starting value is $c$. The correct starting value is $\Delta(z,r,r)=0$, since a particle has not moved yet. Equivalently, $u(z,t,t)=v(z,t)$. So **$c\equiv0$ is exactly the diagonal condition**, and the homogeneous mode $c/(t-r)$ is what a wrong diagonal looks like.

> **Intuition.** A first-order ODE needs an initial condition. Here the "initial condition" sits on the diagonal $t=r$. The weak residual cannot see it directly, because the factor $t-r$ silences the $t=r$ boundary. Something else has to enforce it.

### Step 7.4 — Square-integrability enforces $c=0$

Compute the $L^2$ energy of the homogeneous mode for fixed $r$. Use the pushforward property: if $z\sim\mu_t$, then $\Phi_{r,t}(z)\sim\mu_r$. So

$$
\int_r^1\mathbb E_{z\sim\mu_t}\Bigl|\frac{c(\Phi_{r,t}(z),r)}{t-r}\Bigr|^2dt
=\mathbb E_{y\sim\mu_r}\bigl|c(y,r)\bigr|^2\int_r^1\frac{dt}{(t-r)^2}.
$$

The time integral diverges, since $\int_0\tau^{-2}d\tau=\infty$. So the mode has infinite energy unless $\mathbb E|c(\cdot,r)|^2=0$. If $u$ and $u^*$ are both square-integrable, then so is their difference, which is the mode. Hence $c=0$ and $u=u^*$.

> **Intuition.** A wrong diagonal means the candidate claims a particle has already moved a finite distance $c$ in zero time. Spreading that over an interval of length $t-r$ needs a velocity $c/(t-r)$, which blows up at the diagonal. It blows up fast enough that its energy is infinite. Requiring finite energy rules it out. That is why the diagonal condition is *implied* in the population limit, not imposed.

### Step 7.5 — The true field has finite energy

By Jensen's inequality applied to the time average in Step 2.2,

$$
|u^*(z,r,t)|^2\le\frac{1}{t-r}\int_r^t\bigl|v(\Phi_{\tau,t}(z),\tau)\bigr|^2d\tau .
$$

Take $z\sim\mu_t$. By the pushforward property, $\Phi_{\tau,t}(z)\sim\mu_\tau$. By Jensen for conditional expectation, $\mathbb E|v(z_\tau,\tau)|^2\le\mathbb E|v_c|^2$. So

$$
\mathbb E_{\mu_t}|u^*(\cdot,r,t)|^2\le\mathbb E|e-x|^2<\infty .
$$

The candidate set therefore contains $u^*$, and by Step 7.4 nothing else. This completes the sketch of Proposition 4. $\blacksquare$

### Step 7.6 — Identification without stability

Uniqueness does not mean that small loss forces small error. Here is a sequence where the loss goes to zero but the error does not.

Fix a smooth bounded $c_0(y,r)$ with $\mathbb E|c_0|^2=1$. For $\varepsilon\in(0,1)$, let $\chi_\varepsilon(\tau)=\min(\tau/\varepsilon,1)$ and define

$$
u_\varepsilon=u^*+\sqrt{\varepsilon}\;c_0\bigl(\Phi_{r,t}(z),r\bigr)\,\frac{\chi_\varepsilon(t-r)}{t-r}.
$$

- **It is a legal candidate.** The extra displacement $\sqrt\varepsilon\,c_0\,\chi_\varepsilon(t-r)$ vanishes at $t=r$, and the extra velocity is bounded by $c_0/\sqrt\varepsilon$.
- **Its residual is concentrated near the diagonal.** $c_0(\Phi_{r,t}(z),r)$ is constant along trajectories, so $D_t$ only hits $\chi_\varepsilon$, giving $R(u_\varepsilon)=\sqrt\varepsilon\,c_0\cdot\frac{1}{\varepsilon}\mathbf 1\{t-r<\varepsilon\}$. Testing against a bounded $\varphi$ integrates $1/\sqrt\varepsilon$ over a strip of width $\varepsilon$, so $|W(\varphi;u_\varepsilon)|=O(\sqrt\varepsilon)$ and $\mathcal L(u_\varepsilon)=O(\varepsilon)\to0$.
- **Its error does not shrink.** By the pushforward computation of Step 7.4,

$$
\|u_\varepsilon-u^*\|^2=\varepsilon\int_0^1\!\!\int_r^1\frac{\chi_\varepsilon(t-r)^2}{(t-r)^2}\,dt\,dr\approx\varepsilon\cdot\frac{2}{\varepsilon}=2 .
$$

So no constant $C$ gives $\|u-u^*\|\le C\sqrt{\mathcal L(u)}$ for all $u$. The bad directions are sharp ramps at the diagonal. The proposal's two guards target exactly these:

- the **diagonal anchor** $\mathbb E|u(z_t,t,t)-v_c|^2$, which penalizes $u_\varepsilon$ at $t=r$, where $u_\varepsilon-u^*=c_0/\sqrt\varepsilon$, by order $1/\varepsilon$;
- the **$L^2$ penalty**, together with the smoothness of the network, which disfavours ramps of width $\varepsilon$.

---

## 8. Estimators

### Step 8.1 — Why the naive estimate is biased

Fix one test field $\varphi$ and draw $n$ independent samples $g_1,\dots,g_n$ with mean $W$ and variance $\sigma^2$. Then

$$
\mathbb E\Bigl[\Bigl(\frac1n\sum_i g_i\Bigr)^2\Bigr]=W^2+\frac{\sigma^2}{n}.
$$

The extra $\sigma^2/n$ depends on $u$, through the transport and boundary terms. So minimizing the naive estimate minimizes $W^2+\sigma^2(u)/n$. At small $n$, that pulls $u$ toward fields with low per-sample noise rather than toward $u^*$.

> **Intuition.** A square of an average is not an average of squares. When the batch is small, the noise in the average adds its own positive contribution, and the optimizer learns to shrink the noise instead of the residual.

### Step 8.2 — The U-statistic

For $i\ne j$, independence gives $\mathbb E[g_ig_j]=\mathbb E[g_i]\,\mathbb E[g_j]=W^2$. Averaging over the $n(n-1)$ ordered pairs:

$$
\frac{1}{n(n-1)}\sum_{i\ne j}g_ig_j=\frac{\bigl(\sum_i g_i\bigr)^2-\sum_i g_i^2}{n(n-1)},\qquad\mathbb E[\,\cdot\,]=W^2 .
$$

With $K$ test fields shared across the batch, condition on each $\varphi_k$, apply the above, then average over $k$. The result is

$$
\mathbb E\,\widehat{\mathcal L}=\mathbb E_\nu\bigl[W(\varphi;u)^2\bigr]=\mathcal L(u).
$$

The estimate is linear in the products $g_ig_j$, so differentiation commutes with expectation and $\mathbb E\nabla\widehat{\mathcal L}=\nabla\mathcal L$. On a given batch the estimate can be negative, which is fine for gradient descent.

### Step 8.3 — Decoupled forcing

Split $W_k=A_k(u)-b_k$, where $A_k=\ell_{\varphi_k}(u)$. Then $W_k^2=A_k^2-2A_kb_k+b_k^2$. Estimate each piece:

- $A_k^2$ by the U-statistic on the main batch: unbiased.
- $A_kb_k$ by $\widehat A_k\,\hat b_k$, with $\hat b_k$ computed from an **independent** auxiliary batch of size $N$: $\mathbb E[\widehat A_k\hat b_k]=A_kb_k$ by independence.
- $b_k^2$ by $\hat b_k^{\,2}$: biased by $\operatorname{Var}(\hat b_k)$, but constant in $u$, so the gradient is unaffected.

The forcing estimate needs only $\varphi_k$ and $v_c$ at sampled points, with no network evaluation. So $N$ can be much larger than $n$, and the forcing noise in the gradient falls by about $n/N$.

### Step 8.4 — Rao–Blackwellization when $v$ is computable

Rao–Blackwell: replacing a random quantity by its conditional expectation never increases variance and keeps the mean. Here that means replacing $v_c$ by $v=\mathbb E[v_c\mid z_t]$ wherever $v_c$ multiplies a function of $z_t$.

**Derivation for Gaussian data.** Let $x\sim\mathcal N(m,\Sigma)$ and $z=(1-t)x+te$.

1. Solve the path for $e$: $e=\bigl(z-(1-t)x\bigr)/t$. Then $v_c=e-x=(z-x)/t$. Given $z$, the only unknown left is $x$, so

$$
v(z,t)=\frac{z-\mathbb E[x\mid z]}{t}.
$$

2. The pair $(x,z)$ is jointly Gaussian, with

$$
\operatorname{Cov}(x,z)=(1-t)\,\Sigma,\qquad \Sigma_z=\operatorname{Cov}(z)=(1-t)^2\Sigma+t^2I,\qquad \mathbb E z=(1-t)\,m .
$$

3. Gaussian conditioning gives

$$
\mathbb E[x\mid z]=m+(1-t)\,\Sigma\,\Sigma_z^{-1}\bigl(z-(1-t)m\bigr).
$$

For a Gaussian mixture, compute this per component and weight by the posterior responsibilities. For a finite training set of size $M$, the "mixture" has $M$ point masses, which costs $O(M)$ per query.

---

## 9. The closed-form linear readout

Let $u_w(z,r,t)=\Psi(z,r,t)\,w$, with fixed features $\Psi\in\mathbb R^{d\times p}$ and weights $w\in\mathbb R^p$.

### Step 9.1 — Each sample gives a linear row

The $u$-dependent part of $g$ is linear in $u$, hence in $w$. Write it as $\alpha_{ik}^{\top}w$:

$$
\alpha_{ik}=-\tfrac12(t-r)\,\Psi(z_t,r,t)^{\top}\bigl(\partial_t\varphi_k+(v_c\cdot\nabla_z)\varphi_k\bigr)+(1-r')\,\Psi(z_1,r',1)^{\top}\varphi_k(z_1,r',1).
$$

Then $A_k(w)=\mathbb E[\alpha_k]^{\top}w$, and $W_k=\mathbb E[\alpha_k]^{\top}w-b_k$.

### Step 9.2 — Normal equations

The weak-residual part of the objective is $\frac1K\sum_k\bigl(\bar\alpha_k^{\top}w-b_k\bigr)^2$. Its gradient is

$$
\frac2K\sum_k\bar\alpha_k\bigl(\bar\alpha_k^{\top}w-b_k\bigr).
$$

Replace $\bar\alpha_k\bar\alpha_k^{\top}$ by its U-statistic, which removes the self-pairs exactly as in Step 8.2, to get $G$. Replace $b_k$ by the independent estimate $\hat b_k$ to get $h$:

$$
G=\frac1K\sum_k\frac{\bigl(\sum_i\alpha_{ik}\bigr)\bigl(\sum_i\alpha_{ik}\bigr)^{\top}-\sum_i\alpha_{ik}\alpha_{ik}^{\top}}{n(n-1)},\qquad h=\frac1K\sum_k\bar\alpha_k\,\hat b_k .
$$

The anchor term $\lambda_a\,\mathbb E|\Psi_\Delta w-v_c|^2$, where $\Psi_\Delta=\Psi(z_t,t,t)$, has gradient $2\lambda_a(Hw-c)$, with $H=\mathbb E[\Psi_\Delta^{\top}\Psi_\Delta]$ and $c=\mathbb E[\Psi_\Delta^{\top}v_c]$. A ridge term $\lambda_2\|w\|^2$ has gradient $2\lambda_2 w$. Setting the total gradient to zero and dividing by 2:

$$
\boxed{(G+\lambda_aH+\lambda_2I)\,w=h+\lambda_a\,c}
$$

> **A note on the penalty.** $\lambda_2 I$ penalizes $\|w\|^2$. To penalize $\|u_w\|^2_{L^2}$ exactly as written in the proposal's $\mathcal J$, replace $I$ by $\mathbb E[\Psi^{\top}\Psi]$ over the triangle. Either choice makes the system uniquely solvable once $\lambda_2>0$ and $G+\lambda_aH$ is positive semidefinite.

### Step 9.3 — What it converges to

As the sample sizes grow, $G\to\mathbb E_\nu[\bar\alpha\bar\alpha^{\top}]$, which is positive semidefinite. The solution then converges to the minimizer of the regularized objective over the span of $\Psi$. If $u^*$ lies in the span, then $\mathcal L$ and the anchor are both zero there, so the minimizer is $u^*$ as $\lambda_2\to0$. Otherwise it is the best approximation measured in the kernel norm of Step 6.3 plus the anchor, which in general is not the $L^2$ projection.

---

## 10. Fourier test fields

### Step 10.1 — The family and its derivatives

$$
\varphi(z,r,t)=a\,\eta(r,t)\cos(\omega\cdot z+\beta),
$$

with a random direction $a\in\mathbb R^d$, frequency $\omega$, phase $\beta$, and a smooth time factor $\eta$. Its gradient is $\nabla_z\varphi_j=-a_j\,\eta\,\sin(\omega\cdot z+\beta)\,\omega$, so

$$
(v_c\cdot\nabla_z)\varphi=-a\,\eta(r,t)\sin(\omega\cdot z+\beta)\,(\omega\cdot v_c),\qquad
\partial_t\varphi=a\,\partial_t\eta(r,t)\cos(\omega\cdot z+\beta).
$$

Each costs one dot product, $O(d)$. These are exact derivatives of a known function, not of the network.

### Step 10.2 — Why random frequencies give a positive-definite kernel

Average over a uniform phase. Using $\cos A\cos B=\tfrac12[\cos(A-B)+\cos(A+B)]$, the $A+B$ term averages to zero:

$$
\mathbb E_\beta\bigl[\cos(\omega\cdot z+\beta)\cos(\omega\cdot z'+\beta)\bigr]=\tfrac12\cos\bigl(\omega\cdot(z-z')\bigr).
$$

Averaging over $\omega\sim p$ gives $\tfrac12\,\mathrm{Re}\,\hat p(z-z')$, where $\hat p$ is the characteristic function of $p$. By **Bochner's theorem**, this is a positive-definite kernel. It is *strictly* positive definite when $p$ has full support. Multiplying by a positive-definite kernel in $(r,t)$, built from the time basis $\eta$, and by $\mathbb E[aa^{\top}]=I/d$ for the output direction, keeps it positive definite. This is the condition Step 7.1 needs.

> **Intuition.** Each probe asks one question: how much does the residual oscillate at frequency $\omega$? If the frequency distribution has full support, every frequency gets asked eventually, and a function with zero response at every frequency is zero. In high dimension, almost all directions $\omega$ are uninformative for a given dataset, which is why test-field design at image scale is the main open risk.

---

## 11. Inference

By Step 2.2, $\Phi_{r,t}(z)=z-(t-r)\,u^*(z,r,t)$ exactly. So one step from noise to data is

$$
z_0=\Phi_{0,1}(z_1)=z_1-u^*(z_1,0,1).
$$

The composition property $\Phi_{t_{j-1},t_j}\circ\cdots=\Phi_{0,1}$ gives the multi-step sampler:

$$
z_{t_{j-1}}=z_{t_j}-(t_j-t_{j-1})\,u_\theta\bigl(z_{t_j},t_{j-1},t_j\bigr),\qquad j=k,\dots,1 .
$$

With the exact $u^*$, every grid gives the same output. With a learned $u_\theta$, more steps trade compute for robustness to error in any single long jump. The boundary term of Step 3.3 tests $u$ at exactly the inputs $(z_1,r,1)$ that the first jump uses.

---

## 12. Everything in closed form: the 1-D Gaussian

Take $d=1$, $x\sim\mathcal N(0,s^2)$, $e\sim\mathcal N(0,1)$.

### Step 12.1 — Marginals and velocity

$$
z_t\sim\mathcal N(0,\sigma_t^2),\qquad\sigma_t^2=(1-t)^2s^2+t^2 .
$$

The flow that preserves a centred Gaussian while changing its scale is a pure rescaling, $\Phi_{s',t}(z)=(\sigma_{s'}/\sigma_t)\,z$. Its velocity is

$$
v(z,t)=\frac{\dot\sigma_t}{\sigma_t}\,z=\frac{t-(1-t)s^2}{\sigma_t^2}\,z .
$$

**Check against Step 8.4.** $\mathbb E[x\mid z]=\frac{(1-t)s^2}{\sigma_t^2}z$, so

$$
\frac{z-\mathbb E[x\mid z]}{t}=\frac{\sigma_t^2-(1-t)s^2}{t\,\sigma_t^2}\,z=\frac{t^2-t(1-t)s^2}{t\,\sigma_t^2}\,z=\frac{t-(1-t)s^2}{\sigma_t^2}\,z=v .\;\checkmark
$$

### Step 12.2 — Average velocity and the MeanFlow identity

$$
u^*(z,r,t)=\frac{z-\Phi_{r,t}(z)}{t-r}=\Bigl(1-\frac{\sigma_r}{\sigma_t}\Bigr)\frac{z}{t-r}.
$$

**Check Step 2.3.** Along a trajectory $y(t)=\sigma_t\,\zeta$ with fixed $\zeta$, the displacement is $\Delta^*=(\sigma_t-\sigma_r)\,\zeta$. Its derivative is $\dot\sigma_t\,\zeta=(\dot\sigma_t/\sigma_t)\,y(t)=v$. ✓

### Step 12.3 — Numerical checks already run

| Check | Result |
| --- | --- |
| Proposition 1: both sides by quadrature, nonlinear $u$, $\varphi(\cdot,1)\ne0$ | 0.051340 on both sides |
| Proposition 2: Monte Carlo with $v_c$ against exact $v$ | $0.05152\pm0.0002$ against $0.05159$ |
| Step 8.4: Gaussian posterior velocity against exact $v$ | Equal to 15 digits |
| Appendix A.1: CRN paired variance at $u^*$, $(r,t)=(0,1)$ | 0.395 (independent pairing: 0.395) |
| Appendix A.3: corrected ambiguity formula | Pushforward variance 0.250 (target 0.25); the old formula gives 6.25 |

---

## Appendix A — Why the dropped components fail

### A.1 Common-random-number moment matching

CRN pairs $\phi\bigl(z_t-(t-r)u(z_t)\bigr)$ with $\phi(z_r)$, using the same $(x,e)$. The hope was that the two terms cancel exactly at $u=u^*$. But

$$
z_t-(t-r)\,u^*(z_t,r,t)=\Phi_{r,t}(z_t)\qquad\text{while}\qquad z_r=z_t-(t-r)\,v_c .
$$

These agree only if $u^*(z_t,r,t)=v_c=e-x$, which no function of $z_t$ alone can satisfy (Step 2.1). The paired difference is therefore nonzero at the optimum, with variance bounded by $L^2(t-r)^2\,\mathbb E|u-v_c|^2/n$. That bound does not vanish at $u=u^*$. In the 1-D test with $x\perp e$, the pairing gave no reduction at all.

### A.2 Linear test functions in moment matching

With $\phi(y)=a^{\top}y$, the moment residual is $a^{\top}\bigl(\mathbb E[z_t-z_r]-(t-r)\,\mathbb E[u(z_t,r,t)]\bigr)$. It constrains only the **mean** of $u$, which gives at most $d$ numbers per $(r,t)$. Nonlinear $\phi$ would constrain more, but then $u$ sits inside $\phi$ and the residual is no longer affine in $u$. The weak form avoids this trade-off. There $u$ multiplies $\varphi$ and its derivatives rather than being composed with them (Step 6.1), so the residual is affine for any $\varphi$.

### A.3 The reflection ambiguity, derived correctly

Suppose $M=\mathrm{id}-(t-r)\,u$ pushes $\mu_t$ to $\mu_r$, and $S$ preserves $\mu_r$ (for a symmetric distribution, $S(y)=-y$). Then $S\circ M$ also pushes $\mu_t$ to $\mu_r$, so it matches every moment. Write it in the same form, $S\circ M=\mathrm{id}-(t-r)\,u'$, and solve for $u'$:

$$
u'=\frac{\mathrm{id}-S\circ M}{t-r}=u+\frac{M-S\circ M}{t-r}=u+\frac{(\mathrm{id}-S)\circ M}{t-r}.
$$

**Check.** Take $\mu_t=\mathcal N(0,1)$, $\mu_r=\mathcal N(0,\tfrac14)$, $t-r=1$, $u(z)=z/2$. Then $M(z)=z/2$ and $u'(z)=z/2+(z/2+z/2)=3z/2$. And $z-3z/2=-z/2\sim\mathcal N(0,\tfrac14)$. ✓ The weak form is immune to this, because it identifies $u$ through the equation $D_t\Delta=v$, not through the pushforward (Step 7.2).

### A.4 Moment-level semigroup consistency

The proposed condition was a fixed linear combination of $W_k(r,t)$, $W_k(r,s)$ and $W_k(s,t)$. At any $u$ with zero weak loss, every $W_k$ is already zero, so any linear combination of them is zero too. The condition therefore lies in the span of constraints already imposed, and adds nothing. The genuine semigroup property, $\Phi_{r,s}\circ\Phi_{s,t}=\Phi_{r,t}$, involves **composing** $u$ with itself. Composition is nonlinear in $u$, so no condition that evaluates $u$ only at $z_t$ can express it. Step 7.4 shows it is not needed: within $L^2$, the weak equation alone already pins down $u^*$, and $u^*$ satisfies the semigroup property automatically.
