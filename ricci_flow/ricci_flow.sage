import numpy as np
import os

pi = RR.pi()

plot_initial_curve = True
plot_initial_m = True
plot_initial_h = True
plot_initial_R = True
plot_initial_K = True

animate_curve = True
animate_m = True
animate_h = True
animate_R = True
animate_K = True

folder_name = "./curve7_anim_all_10000_steps_eps=0.01"
print(f"Using folder: {folder_name}")
if not os.path.exists(folder_name):
    print("Folder did not exist. Creating...")
    os.mkdir(folder_name)

def path(name):
    return os.path.join(folder_name, name)


def revolve(x, y):
    # x/y are reversed to rotate around z axis
    return (lambda u, v: y(v)*cos(u), lambda u, v: y(v)*sin(u), lambda u, v: x(v))


def xy_splines_from_hm(h, m, srange=(0, pi), step_size=0.1):
    def y(rho):
        if m(rho) < 0:
            print(f"\tnegative: {rho}, {m(rho)}")
            return 0
        return sqrt(m(rho))
    
    rho_space, drho = np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size), retstep=True)
    y_spline = spline([(rho, y(rho)) for rho in rho_space])
    
    def x_integral(rho1, rho2):
        def integrand(s):
            d = h(s) - (y_spline.derivative(s))**2
            return sqrt(d) if d > 0 else 0
        return numerical_integral(integrand, rho1, rho2)[0]

    x_spline_list = [(rho_space[0], x_integral(0, rho_space[0]))]
    for rho in rho_space[1:]:
        last_rho = x_spline_list[-1][0]
        last_x = x_spline_list[-1][1]
        x_spline_list.append((rho, last_x + x_integral(last_rho, rho)))
    x_spline = spline(x_spline_list)
    
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

srange = (0, pi)
h = to_spline(h, srange)
m = to_spline(m, srange)
x, y = xy_splines_from_hm(h, m, srange, step_size=0.1)
R, K = hm_to_ricci_tensor(h, m, return_K=True)

if plot_initial_curve:
    xy_plot = parametric_plot((x, y), (x.list()[0][0], x.list()[-1][0]))
    xy_plot.save(path("initial_curve_of_revolution.png"))
if plot_initial_m:
    plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)").save(path("initial_sqrt_m.png"))
if plot_initial_h:
    plot(h, srange, title="h").save(path("initial_h.png"))
if plot_initial_R:
    def R11(rho): return R(rho)[0][0]
    def R22(rho): return R(rho)[1][1]
    plot(R11, srange, title="R11").save(path("initial_R11.png"))
    plot(R22, srange, title="R22").save(path("initial_R22.png"))
if plot_initial_K:
    plot(K, srange, marker=",", linestyle="", title="K").save(path("initial_K.png"))


# Ricci flow
dt = 0.00001
N = 10001
plot_gap = 100
space, dt = np.linspace(0, dt*(N-1), N, retstep=True)
eps = 0.01

print("Running ricci flow...")
print(f"c3 = {c3}")
print(f"c5 = {c5}")
print(f"dt = {dt}")
print(f"N = {N}")
print(f"plot_gap = {plot_gap}")

curve_plots = []
revolved_plots = []
m_plots = []
h_plots = []
R11_plots = []
R22_plots = []
K_plots = []

for i in range(N):
    print(f"Iteration {i}/{N-1}, t = {dt*i}")
    print("\tGetting Ricci tensor")
    R, K = hm_to_ricci_tensor(h, m, return_K=True, eps=eps)
    def R11(rho): return R(rho)[0][0]
    def R22(rho): return R(rho)[1][1]

    print("\tComputing h spline")
    h = spline([(rho, h_rho_ - 2*R11(rho)*dt) for rho, h_rho_ in filter(lambda z: z[0] >= eps and z[0] <= pi-eps, h.list())])
    print("\tComputing m spline")
    m = spline([(rho, m_rho_ - 2*R22(rho)*dt) for rho, m_rho_ in filter(lambda z: z[0] >= eps and z[0] <= pi-eps, m.list())])

    m.append((0, 0))
    m.append((pi, 0))

    sqrt_m = sqrt_spline(m)

    h.append((0, sqrt_m.derivative(0)**2))
    h.append((pi, sqrt_m.derivative(pi)**2))

    print("\tGetting xy functions from h and m splines")
    x, y = xy_splines_from_hm(h, m, srange)
    
    if i % plot_gap == 0:
        print("\tAppending plots")
        revolved_plots.append(parametric_plot3d(revolve(x, y), (0, 2*pi), srange))
        if animate_curve:
            curve_plots.append(parametric_plot((x, y), srange))
        if animate_m:
            m_plots.append(plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)"))
        if animate_h:
            h_plots.append(plot(h, srange, title="h"))
        if animate_R:
            R11_plots.append(plot(R11, srange, title="R11"))
            R22_plots.append(plot(R22, srange, title="R22"))
        if animate_K:
            K_plots.append(plot(K, srange, title="K"))

if animate_curve:
    print("Animating curve...")
    curve_anim = animate(curve_plots)
    print("Saving curve animation...")
    curve_anim.save(path("curve_flow.gif"), show_path=True)
if animate_m:
    print("Animating sqrt(m)...")
    m_anim = animate(m_plots)
    print("Saving sqrt(m) animation...")
    m_anim.save(path("sqrt_m_anim.gif"), show_path=True)
if animate_h:
    print("Animating h...")
    h_anim = animate(h_plots)
    print("Saving h animation...")
    h_anim.save(path("h_anim.gif"), show_path=True)
if animate_R:
    print("Animating R...")
    R11_anim = animate(R11_plots)
    R22_anim = animate(R22_plots)
    print("Saving R animation...")
    R11_anim.save(path("R11_anim.gif"), show_path=True)
    R22_anim.save(path("R22_anim.gif"), show_path=True)
if animate_K:
    print("Animating K...")
    K_anim = animate(K_plots)
    print("Saving K animation...")
    K_anim.save(path("K_anim.gif"), show_path=True)

print("Animating surface flow...")
a_surf = animate(revolved_plots)
print("Saving surface flow...")
a_surf.save(path("surf_flow.html"), online=True, show_path=True)
