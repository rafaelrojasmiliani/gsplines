# General Splines Library
This is a library to compute generalized splines passing through a series of points given a sequence of intervals.

# Scope
Generalized splines appear naturaly in problems of trajectory optimization when waypoint constraints are added.
In other words, if we desire to optimize a motion which is required to pass by a sequence of positions we will meet with generalized splines.

However, generalized splines trajectories are uniquely characterized by the waypoints that it attains, the time intervals between waypoints and the boundary conditions.
Such characterization may vary depending on how we formalize the problem, but the idea remains the same.
The important fact is that se can identify a generlized spline with a point in Rn and such a characterization allows to formulate optimization problems on a space of curves as optimization problems in real variables.

# Background
 ```math
 SE = \frac{\sigma}{\sqrt{n}}
 ```
<img src="https://render.githubusercontent.com/render/math?math=\Large \int_0^T \left\|\frac{\mathsf{d}\mathbf{q}}{\mathsf{d} t }\right\|^2+ \left\|\frac{\mathsf{d}^2\mathbf{q}}{\mathsf{d} t^2 }\right\| + \left\|\frac{\mathsf{d}^3\mathbf{q}}{\mathsf{d} t^3 }\right\| + \left\|\frac{\mathsf{d}^4\mathbf{q}}{\mathsf{d} t^4 }\right\| ">
