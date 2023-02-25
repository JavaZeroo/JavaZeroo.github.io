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