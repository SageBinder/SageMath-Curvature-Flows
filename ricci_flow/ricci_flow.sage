import numpy as np
import os

pi = RR.pi()

plot_initial_rev_curve = True
plot_initial_m = False
plot_initial_h = False
plot_initial_K = False
plot_initial_ricci = False

folder_name = "./curve4"
print(f"Using folder: {folder_name}")
if not os.path.exists(folder_name):
    print("Folder did not exist. Creating...")
    os.mkdir(folder_name)

def path(name):
    return os.path.join(folder_name, name)


def revolve(x, y):
    # x/y are reversed to rotate around z axis
    return (lambda u, v: y(v)*cos(u), lambda u, v: y(v)*sin(u), lambda u, v: x(v))


def xy_splines_from_hm(h, m, srange=(0, pi), step_size=0.1, eps=0.01):
    def y(rho):
        if m(rho) < 0:
            print(f"\tnegative: {rho}, {m(rho)}")
            return 0
        return sqrt(m(rho))
    
    rho_space = np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size))
    y_spline = spline([(rho, y(rho)) for rho in rho_space])
    
    def x(rho):
        # TODO: Don't re-integrate every time. This is O(n^2)
        def integrand(s):
            d = h(s) - (y_spline.derivative(s))**2
            return sqrt(d) if d >= 0 else 0
        return numerical_integral(integrand, eps, rho)[0]

    x_spline = spline([(rho, x(rho)) for rho in rho_space])
    
    return x_spline, y_spline


def sqrt_spline(f):
    return spline([(x, sqrt(y)) for x, y in f])


def to_spline(f, srange=(0, pi), step_size=0.01):
    return spline([(s, f(s)) for s in np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size))])


def hm_to_ricci_tensor(h, m, return_K=False, eps=0.1):
    sqrt_m = sqrt_spline(m)
    
    def K(rho):
        return -sqrt_m.derivative(rho, order=2) / sqrt_m(rho)

    def R(rho):
        if rho <= eps or rho >= pi - eps:
            rho = eps
        R11 = K(rho) + ((m.derivative(rho) * h.derivative(rho)) / (4 * m(rho) * h(rho)))
        R22 = (R11 / h(rho)) * m(rho)
        return matrix([[R11, 0], [0, R22]])

    return R if not return_K else (R, K)


c3 = 0.766
c5 = -0.091
h(rho) = 1
m(rho) = ((sin(rho) + c3*sin(3*rho) + c5*sin(5*rho))/(1 + 3*c3 + 5*c5))**2

srange=(0, pi)
h = to_spline(h, srange)
m = to_spline(m, srange)
x, y = xy_splines_from_hm(h, m, srange, step_size=0.1, eps=0)

R, K = hm_to_ricci_tensor(h, m, return_K=True)
if plot_initial_rev_curve:
    xy_plot = parametric_plot((x, y), (x.list()[0][0], x.list()[-1][0]))
    xy_plot.save(path("initial_curve_of_revolution.png"))
if plot_initial_m:
    plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)").save(path("initial_sqrt_m.gif"))
if plot_initial_h:
    plot(h, srange, title="h").save(path("initial_h.gif"))
if plot_initial_K:
    plot(K, srange, marker=",", linestyle="", title="K").save(path("initial_K.gif"))
if plot_initial_ricci:
    def R11(rho): return R(rho)[0][0]
    def R22(rho): return R(rho)[1][1]
    plot(R11, srange).save(path("initial_R11.gif"), title="R11")
    plot(R22, srange).save(path("initial_R22.gif"), title="R22")


# Ricci flow
dt = 0.00001
N = 10001
plot_gap = 100
space, dt = np.linspace(0, dt*(N-1), N, retstep=True)
eps = 0.1

print("Running ricci flow...")
print(f"c3 = {c3}")
print(f"c5 = {c5}")
print(f"dt = {dt}")
print(f"N = {N}")
print(f"plot_gap = {plot_gap}")

plots = []
revolved_plots = []
for i in range(N):
    R = hm_to_ricci_tensor(h, m)
    def R11(rho): return R(rho)[0][0]
    def R22(rho): return R(rho)[1][1]

    print(f"Iteration {i}/{N-1}, t = {dt*i}")
    
    h = spline([(rho, h_rho_ - 2*R11(rho)*dt) for rho, h_rho_ in filter(lambda z: z[0] >= eps and z[0] <= pi-eps, h.list())])
    m = spline([(rho, m_rho_ - 2*R22(rho)*dt) for rho, m_rho_ in filter(lambda z: z[0] >= eps and z[0] <= pi-eps, m.list())])

    m.append((0, 0))
    m.append((pi, 0))

    sqrt_m = sqrt_spline(m)

    h.append((0, sqrt_m.derivative(0)**2))
    h.append((pi, sqrt_m.derivative(pi)**2))

    x, y = xy_splines_from_hm(h, m, srange, eps=0)
    
    if i % plot_gap == 0:
        plots.append(parametric_plot((x, y), (0, pi)))
        revolved_plots.append(parametric_plot3d(revolve(x, y), (0, 2*pi), srange))

print("Animating...")
a_curve = animate(plots)
a_surf = animate(revolved_plots)

print("Saving...")
a_curve.save(path("curve_flow.gif"), show_path=True)
a_surf.save(path("surf_flow.html"), online=True, show_path=True)
