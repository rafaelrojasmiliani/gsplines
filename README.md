# General Splines Library
This is a library to compute generalized splines passing through a series of points given a sequence of intervals.

# Scope
Generalized splines appear naturaly in problems of trajectory optimization when waypoint constraints are added.
In other words, if we desire to optimize a motion which is required to pass by a sequence of positions we will meet with generalized splines.

However, generalized splines trajectories are uniquely characterized by the waypoints that it attains, the time intervals between waypoints and the boundary conditions.
Such characterization may vary depending on how we formalize the problem, but the idea remains the same.
The important fact is that se can identify a generlized spline with a point in Rn and such a characterization allows to formulate optimization problems on a space of curves as optimization problems in real variables.

# Background

This library is aimed to find a trajectory passing trough a sequence of waypoints <img src="https://render.githubusercontent.com/render/math?math=\{\mathbf{w}_0, ...,\mathbf{w}_{N+1}\}"> such that the following integral is minized
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\Large \int_0^T \alpha_1\left\|\frac{\mathsf{d}\mathbf{q}}{\mathsf{d} t }\right\|^2 %2B \alpha_2 \left\|\frac{\mathsf{d}^2\mathbf{q}}{\mathsf{d} t^2 }\right\|^2 %2B \alpha_3\left\|\frac{\mathsf{d}^3\mathbf{q}}{\mathsf{d} t^3 }\right\|^2 %2B  \alpha_4\left\|\frac{\mathsf{d}^4\mathbf{q}}{\mathsf{d} t^4 }\right\|^2 \mathsf{d} t \ \ \ \ \ (1)">
</p>
It may be proven that such a problem can be subdivided in two steps

 1. Find the family of optimal curves that joint waypoints
 2. Compute time instants <img src="https://render.githubusercontent.com/render/math?math=\{t_0, t_1,  ...,t_N, t_{N+1}\}"> where the optimal curves must be attached

The step 1. is done by solving a linear ordinary differential equation. One method to achieve 2. is to formulate an optimization problem (e.g. a gradient based one).

## Optimal curves
We underline that this library leverages on the [general theory of linear ODEs](https://en.wikipedia.org/wiki/Linear_differential_equation).
It may be proven that any optimal of (1) solves the following linear ODE, which turn out to be the Euler-Lagrange equations at each interval <img src="https://render.githubusercontent.com/render/math?math=[t_i, t_{i%2B1}]">
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=-\alpha_1\frac{\mathsf{d}^2\mathbf{q}}{\mathsf{d} t^2 } %2B \alpha_2 \frac{\mathsf{d}^4\mathbf{q}}{\mathsf{d} t^4 } - \alpha_3\frac{\mathsf{d}^6\mathbf{q}}{\mathsf{d} t^6 } %2B  \alpha_4 \frac{\mathsf{d}^8\mathbf{q}}{\mathsf{d} t^8 } = 0\ \ \ \ \ (2)">
</p>
with the following boundary conditions
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{q}(t_i) = \mathbf{w}_i\ \ \ \ \ \ \ \mathbf{q}(t_{i%2B1}) = \mathbf{w}_{i%2B1}\ \ \ \ \ \ \ \ \ \ \ \ \ \ (3)">
</p>
Because the ODE (2) is linear, we can compute its general suction depending on the value of the coefficients <img src="https://render.githubusercontent.com/render/math?math=\alpha_i">.

In fact, the general solution of (2) may be written as
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{q} = \sum_{i=1}^{n_b} \mathbf{a}_i B_i(t) (4)">
</p>
where <img src="https://render.githubusercontent.com/render/math?math=n_b"> and <img src="https://render.githubusercontent.com/render/math?math=B_i(t)"> depend on the coefficients <img src="https://render.githubusercontent.com/render/math?math=\alpha_i">.


# Requirements

- numpy
- scipy
- matplotlib
