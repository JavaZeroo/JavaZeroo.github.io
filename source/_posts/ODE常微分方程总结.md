---
title: ODE常微分方程总结
date: 2023-02-24 18:54:11
categories:
- 📝学习
tags:
- 数学
---

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
y'=f(\frac{y}{x})
$$

同样的道理用$u(x) = \frac{y(x)}{x}, (x\neq0)$替换掉变量$y$。有$y'=u+xu'=f(\frac{y}{x})$ ，可得$u'=\frac{f(u)-u}{x}$

因此 最后得出的$u(x)$后可以直接利用$u(x)=\frac{y(x)}{x}$得到$y$。

### 高级的 Homogeneous Differential Equation 

$$
y'=f(\frac{ax+by+c}{\alpha x+\beta y+ \gamma})
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
可以解出一组$(x_0, y_0)$ 利用这组解将“高级的 Homogeneous Differential Equation”转换成“普通的Homogeneous Differential Equation”。

原理是新建坐标系得$\bar{x}:=x-x_0, \bar{y}:=y-y_0$ 那么在这个坐标系下面原方程就变成了$\bar{y}(\bar{x}):=y(\bar{x}+x_0)-y_0$。对这个方程求导可以将$y$消掉，将原问题