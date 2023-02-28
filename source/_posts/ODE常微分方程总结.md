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
M(x,y)dx+N(x,y)dy=0,\\ \exist\ U(x, y)\ s.t.\ U_x(x,y)=M(x,y),U_y(x,y)=N(x,y)
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
U_y\cdot M+U\cdot M_y&=U_x\cdot N+U\cdot h_x \\
U\cdot M_y &= U' \cdot N+U \cdot N_x \\
\frac{1}{U}U'&=\frac{M_y-N_x}{h} \\
(\ln U)'&=\frac{M_y-N_x}{h} \\
U&=e^{\int \frac{M_y-N_x}{N}dx}
\end{align}
$$
{% note info %}
 这里的答案不是$U=C\cdot e^{\int \frac{g_y-h_x}{h}dx}$的原因是，我们只需要找到一个$U$，因此你可以认为我们选择$C=1$作为答案。下面的情况同理。
{% endnote %}

1. 假如只与$y$有关则$M_x=0$则有

$$
\begin{align}
U_y\cdot M+U\cdot M_y&=U_x\cdot N+U\cdot N_x \\
U'\cdot M+U\cdot M_y &= U \cdot N_x \\
\frac{1}{U}U'&=\frac{N_x-M_y}{M} \\
(\ln U)'&=\frac{N_x-M_y}{M} \\
U &=e^{\int \frac{N_x-M_y}{M}}
\end{align}
$$



