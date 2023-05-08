---
title: ODE常微分方程总结
date: 2023-02-24 18:54:11
categories:
- 📝学习
tags:
- 数学
---

<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/0BRcn1RKWTKqF8OpdQmJXc?utm_source=generator" width="100%" height="352" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>

> 教材：[DLI-BSc-Mathematics-Documents/Ordinary Differential Equatrions.pdf at main · JavaZeroo/DLI-BSc-Mathematics-Documents (github.com)](https://github.com/JavaZeroo/DLI-BSc-Mathematics-Documents/blob/main/y3s2/常微分方程/Ordinary Differential Equatrions.pdf)

## Explicit First Order Equations

这种形式的
$$
y' = f(x, y)
$$
称为 'Explicit First Order Equations' 。

### $y'=f(x)$

### Equations with Separated Variables 

$$
y' = f(x)g(y) 
$$

 这种可以直接变成 $\frac{dy}{g(y)}=f(x)dx$

积分完后
$$
\int\frac{dy}{g(y)}=\int f(x)dx
$$
IVP: $y(\xi)=\eta$
$$
\int^y_{\eta}\frac{dy}{g(y)}=\int^x_{\xi} f(x)dx
$$
这里需要注意的是如果$g(y(\xi))=g(\eta)=0$ 那么，直接就有 $y'=0$ 因此$y=\eta$ ;

### 普通的替换

$$
y'=f(ax+by+c)
$$

这种情况用$u(x) = ax+by+c$ 去替换掉变量$x$ 。原理是$u'=a+by'(x)=a+bf(u)$。很关键的一点是，$u$ 和$y$ 都是一次的。求导后刚好是线性关系。

因此 最后得出的$u(x)$后可以直接利用$u(x)=ax+by+c$得到$y$。

### 普通的Homogeneous Differential Equation 

$$
y'=f\left(\frac{y}{x}\right)
$$

同样的道理用$u(x) = \frac{y(x)}{x}, (x\neq0)$替换掉变量$y$。有$y'=u+xu'=f(\frac{y}{x})$ ，可得$u'=\frac{f(u)-u}{x}$

因此 最后得出的$u(x)$后可以直接利用$u(x)=\frac{y(x)}{x}$得到$y$。

### 高级的 Homogeneous Differential Equation 

$$
y'=f\left(\frac{ax+by+c}{\alpha x+\beta y+ \gamma}\right)
$$

这个的核心思想是转换成“普通的Homogeneous Differential Equation”。

首先分析行列式
$$
\left | \begin{matrix}
a		&b		\\
\alpha &\beta    \\
\end{matrix} \right | 
$$

1. 行列式为零时，说明$a=\lambda_a \alpha, b=\lambda_b \beta$ 此时 可以直接转换成“普通的Homogeneous Differential Equation”
2. 行列式不为零的时候，说明方程组有唯一解。

首先解方程组
$$
\left\{\begin{align}
  ax+by+c&=0\\
  \alpha x+\beta y+ \gamma&=0\\
\end{align}\right.
$$
可以解出一组$(x_0, y_0)$ 利用这组解将**“高级的 Homogeneous Differential Equation”转换成“普通的Homogeneous Differential Equation”**。

原理是新建坐标系得$\bar{x}:=x-x_0, \bar{y}:=y-y_0$ 那么在这个坐标系下面原方程就变成了$\bar{y}(\bar{x}):=y(\bar{x}+x_0)-y_0$。对这个方程求导可以将$y$消掉，将原问题变$\bar{y}$和$x$的关系。这样做的目的就是将**“高级的 Homogeneous Differential Equation”转换成“普通的Homogeneous Differential Equation”**。

对$\bar{y}$求导后可以发现（利用方程$\bar{y}(\bar{x}):=y(\bar{x}+x_0)-y_0, ax_0+by_0+c=0, \alpha x_0+\beta y_0+ \gamma=0$）
$$
\frac{\bar{y}(x)}{d\bar{x}} = y'(\bar{x}+x_0)=f\left(\frac{a\bar{x}+b\bar{y}(\bar{x})}{\alpha\bar{x}+\beta\bar{y}(\bar{x})}\right)
$$
这里就可以像刚才的**普通的Homogeneous Differential Equation**一样去做了。值得注意的是，这样解出来的$\bar{y}$需要转换成原来的$y$。这里可以用
$$
y(x):=y_0+\bar{y}(x-x_0)
$$
得到最后的$y$。

## The Linear Differential Equation

这种形式的
$$
y' + g(x)y=h(x)
$$
称为 'The Linear Differential Equation' 。

这时有两种情况：$h(x)=0$和$h(x)\neq0$， 分别称为"homogeneous" 和"nonhomogeneous"

事实上当$h(x)=0$也就是"homogeneous"时，就是上面的"Explicit First Order Equations"，这里就不在赘述了。

对于$h(x)\neq0$的情况，也就是"nonhomogeneous"时，我们需要用到"Method of variation of constants"。

**Method of variation of constants：**这个方法首先计算出齐次的时候的通解。对于方程$y'+g(x)y=h(x)$他的齐次方程的通解是通过解$y'+g(x)y=0$，可得
$$
y=C\cdot e^{-\int g(x)dx}
$$
此时我们将常数$C$作为一个与$x$的函数$C(x)$。这个时候，只需要解出一个$C(x)$就是**非齐次情况下的答案**。

那么现在的问题就变成了，如何得到一个$C(x)$使得$y' + g(x)y=h(x)$？

首先先计算$C'(x)$
$$
\begin{align}
y &=C(x)\cdot e^{-\int g(x)dx}\\
y'&=\left( C(x) \right)'\cdot e^{-\int g(x)}+C(x)\cdot \left( e^{-\int g(x)dx }\right)' \\
y'&=C'(x)\cdot e^{-\int g(x)}+C(x)\cdot \left( e^{-\int g(x)dx }\cdot\left(-\int g(x)\right)'\right) \\
y'&=C'(x)\cdot e^{-\int g(x)}+C(x)\cdot \left( e^{-\int g(x)dx }\cdot -g(x)\right) \\
&\Downarrow \\
y'&=C'\cdot e^{-\int g(x)}-gC\cdot e^{-\int g(x)dx}
\end{align}
$$
然后代入$y' + g(x)y=h(x)$
$$
\begin{align}
L_y \equiv y' + g(x)y&=h(x)\\
L_y&=C'\cdot e^{-\int g(x)dx}-gC\cdot e^{-\int g(x)dx}+gC\cdot e^{-\int g(x)dx}\\
L_y&=C'\cdot e^{-\int g(x)dx}=h(x) \\
&\Downarrow \\
C'(x)&=h(x)\cdot e^{\int g(x)dx} \\
&\Downarrow \\
C(x)&=\int h(x)\cdot e^{\int g(x)dx}dx + C_0
\end{align}
$$
现在我们可以知道，当$y=C(x)\cdot e^{-\int g(x)dx}$且$C(x)=\int h(x)\cdot e^{\int g(x)dx} + C_0$时。，有$y' + g(x)y=h(x)$。这里有个二级结论

>If $y, \bar{y}$, yare two solutions to the nonhomogeneous equation $L_y = h$, then $L(y - y) = L_y - L_{\bar{y}} = 0, i.e., z(x) = y - \bar{y}$ is a solution of the homogeneous equation $L_y = 0$. Thus all solutions $y(x)$ of the nonhomogeneous equation can be written in the form
>$$
>y(x)=\bar{y}+z(x)
>$$

这里面$z(x)$就是刚刚的$y=C\cdot e^{-\int g(x)dx}$，$\bar{y}$就是$C(x)=\int h(x)\cdot e^{\int g(x)dx} + C_0$和$y=C\cdot e^{-\int g(x)dx}$的结合中的$y$，也就是$\bar{y}=\left(\int h(x)\cdot e^{\int g(x)dx}dx + C_0\right)\cdot e^{-\int g(x)dx}$

### Bernoulli's Equation

这种形式的
$$
y'+g(x)y+h(x)y^{\alpha}=0.\alpha \neq 1
$$
非常的简单，只需要把$y^{\alpha}$解决了就可以了。等式去除$y^{\alpha}$有
$$
y'y^{-\alpha}+g(x)y^{(1-\alpha)}+h(x)=0
$$
利用$z=y^{(1-\alpha)} \implies z'=(1-\alpha)y^{-\alpha}\cdot y'$替换原式得
$$
\frac{1}{1-\alpha}z'+g(x)z+h(x)=0
$$
现在，就变成了nonhomogeneous的"The Linear Differential Equation"。最后解出$z$，别忘了替换回$y$。

### Exact differential equations

这种形式的
$$
M(x,y)dx+N(x,y)dy=0,\\ \exists\ U(x, y)\ s.t.\ U_x(x,y)=M(x,y),U_y(x,y)=N(x,y)
$$

> $xdx+ydy=0$ is an exact equation, and $U(x,y)=1/2 (x^2+y^2 )$is a potential function.

### Integrating Factors

Integrating Factors是用来让非 Exact 变成Exact differential equations。

> **E.g**. $ydx + 2xdy = 0$ is not exact. However, it can easily be made an exact differential equation (in the domain $x > 0$) by multiplying the equation by $1/\sqrt{x}$. The resulting differential equation
> $$
> \frac{y}{\sqrt{x}}dx+2\sqrt{x}dy=0
> $$
> is exact, and a potential function is given by
> $$
> F(x,y)=2y\sqrt{x}=0\ (x>0)
> $$

对于一个not excat differential equation我们需要找到一个Factor $U(x,y)$ 使得$U(x,y)\cdot M(x,y)dx+U(x,y)\cdot N(x,y)dy=0$变成一个Exact differential equations。

现在的问题就是，如何去找？首先令$M' = U\cdot M,N'=U\cdot N$如果$F_x=M',F_y=N'$则有$M'_y=N'_x$。利用这个关系可以知道
{% note info %}
事实上就是$F_{xy}=F_{yx}$ (Jacobian Matrix)
{% endnote %}
$$
\begin{align}
&(U\cdot M)_y=(U\cdot N)_x\\
\implies &U_y\cdot M+U\cdot M_y=U_x\cdot N+U\cdot N_x \\
\end{align}
$$
此时需要考虑，Integrating Factors是只与$x$有关还是只与$y$有关（只需要选一个）

1. 假如只与$x$有关则$M_y=0$则有

$$
\begin{align}
U_y\cdot M+U\cdot M_y&=U_x\cdot N+U\cdot N_x \\
U\cdot M_y &= U' \cdot N+U \cdot N_x \\
\frac{1}{U}U'&=\frac{M_y-N_x}{N} \\
(\ln U)'&=\frac{M_y-N_x}{N} \\
U&=e^{\int \frac{M_y-N_x}{N}dx}
\end{align}
$$
{% note info %}
 这里的答案不是$U=C\cdot e^{\int \frac{M_y-N_x}{N}dx}$的原因是，我们只需要找到一个$U$，因此你可以认为我们选择$C=1$作为答案。下面的情况同理。
{% endnote %}

2. 假如只与$y$有关则$M_x=0$则有

$$
\begin{align}
U_y\cdot M+U\cdot M_y&=U_x\cdot N+U\cdot N_x \\
U'\cdot M+U\cdot M_y &= U \cdot N_x \\
\frac{1}{U}U'&=\frac{N_x-M_y}{M} \\
(\ln U)'&=\frac{N_x-M_y}{M} \\
U &=e^{\int \frac{N_x-M_y}{M}}
\end{align}
$$

## An Existence and Uniqueness Theorem

中文名是**存在唯一性定理**。首先我们介绍*Lipschitz condition*:

> We consider the following initial value problem
> $$
> y' = f(x,y),\ \text{for}\ \xi \leq x\leq \xi+a,\ y(\xi)=\eta
> $$
> The main assumptions in the following theorem are that $f$ is continuous in the strip $S=J\times\mathbb{R}$ with $J=[\xi,\xi+a]$ and satisfies a *Lipschitz condition with respect to $y$* in $S$
> $$
> |f(x,y)-f(x, \bar{y})|\leq L|y-\bar{y}|
> $$
> No restrictions are placed on the value of the *Lipschitz constant* $L\geq 0$

然后引出存在唯一性定理：

> Let $f \in C(S)$ satisfy the *Lipschitz condition*. Then the IVP has exactly one solution $y(x)$. The solution exists in the interval $J: \xi\leq x \leq\xi +a$

## **The extension of solutions.**

以下三个定理是用于 **The extension of solutions.**

> 1️⃣**Local Lipschitz condition.** The function $f(x,y)$ is said to satisfy a local Lipschitz condition with respect to $y$ in $D\subset R^2$ if for every $(x_0,y_0 )\in D$ there exists a neighborhood $U=U(x_0,y_0 )$ and an $L=L(x_0,y_0 )$ such that in $U\cup D$ the function $f$ satisfies the Lipschitz condition $|f(x,y)-f(x,\bar{y})|\leq L|y-\bar{y}|$.

{% note info %}

注意！我们一般通过连续性来判断Local Lipschitz condition

>If $D$ is open and if $f \in C(D)$ has a continuous derivative $f_y$ in $D$, then f satisfies a **local Lipschitz condition** in this set.

{% endnote %}

> 2️⃣**Theorem on local solvability** If $D$ is open and $f\in C(D)$ satisfies a local Lipschitz condition in $D$, then the IVP is locally uniquely solvable for$ (x_0,y_0 )∈D$; i.e., there is a neighborhood $I$ of$ x_0$ such that exactly one solution exists in $I$.



> 3️⃣**Theorem on the extension of solutions** Let $f \in C (D)$ satisfy a local Lipschitz condition with respect to $y$ in $D$. Then for every $(x_0, y_0)\in D$ the initial value problem $y' = f (x, y), y(x_0) = y_0$ has a solution that can be extended to the left and to the right comes arbitrarily close to the boundary of $D$.

最后我们有：

>  **The Peano existence theorem.**  If $f(x,y)$ is continuous in a domain $D$ and $(\xi,\eta)$ is a point in $D$, then at least one solution of the differential equation $y′=f(x,y)$ goes through $(\xi,\eta)$. Every solution can be extended to the left and to the right up to the boundary of $D$.

## Linear System

这里开始就是在讨论，常微分方程组。

### Systems of n Linear Differential Equations

给出常微分方程组的形式：
$$
\begin{align}
y_1' &= a_{11}(t)y_1+\cdots+ a_{1n}(t)y_n+b_1(t) \\
&\ \ \vdots \\
y_1' &= a_{11}(t)y_1+\cdots+a_{1n}(t)y_n+b_1(t) \\

\end{align}
$$
或者：
$$
\mathbf{y}'=A(t)\mathbf{y}+\mathbf{b}(t)\\
\text{where} \hfill \\
A(t)=(a_{ij}(t)),\ \mathbf{b}(t)=(b_1(t), \dots, b_n(t))^\top
$$
值得一提的是，存在唯一性定理在常微分方程组也同样适用。

### Homogeneous Linear Systems

对于齐次的形式，常微分方程组就变成了：
$$
\mathbf{y}'=A(x)\mathbf{y}
$$
此时，根据存在唯一性定理有：
$$
\exist \text{ exactly one solution } \mathbf{y}=\mathbf{y}(t;\tau,\boldsymbol{\eta})\ \forall \tau \in J, \boldsymbol{\eta}\in \R^n\text{ or }\C^n
$$
当然，齐次常微分方程组有一些重要的性质：

1. $ \mathbf{y} \equiv 0$ in $ J$ is a solution of the **homogeneous linear systems**.
2. There exist $n$ linearly independent solutions $_1,\dots,\mathbf{y}_n$. Every such set of $n$ linearly independent solutions is called a **fundamental system of solutions**. If $\mathbf{y}_1,\dots,\mathbf{y}_n$ is a fundamental system, then every solution $\mathbf{y}$ can be written in a unique way as a linear combination $\mathbf{y}=C_1 \mathbf{y}_1+\dots+C_n \mathbf{y}_n$.
3. A system of $n$ solutions $\mathbf{y}_1,…,\mathbf{y}_n$ can be assembled into an $n\times n$ solution matrix $\Phi(x)=(\mathbf{y}_1,\dots,\mathbf{y}_n )$. If $n$ solutions $\mathbf{y}_1,\dots,\mathbf{y}_n$  are linearly independent, then $\Phi(x)$ is a system of $n$ solutions $\mathbf{y}_1,\dots,\mathbf{y}_n$ can be assembled into an $n\times n$ solution matrix $\Phi(x)=(\mathbf{y}_1,\dots,\mathbf{y}_n )$. If $n$ solutions $\mathbf{y}_1,\dots,\mathbf{y}_n$  are linearly independent, then $\Phi(x)$is a **Fundamental Matrix.**

#### **The Wronskian**

现在讨论一下，齐次常微分方程的解，是线性无关还是线性相关。

- **The Wronskian.** If $\Phi(x)=(\mathbf{y}_1,\dots,\mathbf{y}_n )$ is a solution matrix of $\mathbf{y}^′=A(x)\mathbf{y}$, then its determinant $W(x)=|\Phi(x)|$is called the Wronskian determinant.  

- **Theorem** If $\mathbf{y}_1,\dots,\mathbf{y}_n$ are linearly dependent in $J$, then the Wronskian $W(x)\equiv0$.

- **Theorem** If $\mathbf{y}_1,…,\mathbf{y}_n$ is a fundamental system of equation $\mathbf{y}'=A(x)\mathbf{y}$, then the Wronskian $W(x)\neq0$ in $J$.

- **Theorem.** There exists a fundamental system of solutions for equation $\mathbf{y}'=A(x)\mathbf{y}$.


因此，我们先求出$n$个解，然后再去判断这$n$个解的**Fundamental Matrix**的行列式，也就是**The Wronskian**，是否为零。

### **Inhomogeneous Systems**

对于非齐次的常微分方程组，就是最开始样子：
$$
\mathbf{y}'=A(t)\mathbf{y}+\mathbf{b}(t)
$$
下面这个定理类似线性代数，非齐次方程组的通解是，齐次方程组的通解+非齐次方程组的特解：

- **Theorem.** Let $\tilde{\mathbf{y}}(x)$be a fixed solution of the inhomogeneous equation (1). If $\mathbf{y}_0 (x)$ is an arbitrary solution of the homogeneous equation, then $\mathbf{y}(x)=\tilde{\mathbf{y}}(x)+\mathbf{y}_0 (x)$is a solution of the inhomogeneous equation, and all solutions of the inhomogeneous equation are obtained in this way.
