# This code for visualizing Ricci flow in the special case of surfaces of revolution of genus 0
# was written based on the work in following paper by Rubinstein, J. Hyam and Sinclair, Robert:

# Rubinstein, J. Hyam and Sinclair, Robert. (2004). "Visualization Ricci Flow of Manifolds of Revolution".
#     10.48550/ARXIV.MATH/0406189. https://arxiv.org/abs/math/0406189.

import numpy as np
import os
import logging
import time

# logging.disable(logging.INFO)

pi = RR.pi()

plot_initial_curve = False
plot_initial_m = False
plot_initial_h = False
plot_initial_R = False
plot_initial_K = False
plot_initial_tissot = False

animate_curve = False
animate_m = False
animate_h = False
animate_R = False
animate_K = False
animate_tissot = False
animate_gauss_colored_surface = True

center_surface_anim = True

cm = colormaps.RdYlGn # Color map for coloring the surface by Gauss curvature

# The following variables and function is for coloring the surface by Gauss curvature.
# It's disgusting.
# This code is necessary because when animating a series of frames with a color function,
# SageMath calls the color function repeatedly for each frame in the animation.
# The problem is, it doesn't seem possible to give each frame in the animation a different color function;
# when you try to do so, SageMath seemingly just applies the last frame's color function to every frame in the animation.
# Thus, this nonsense is required. To check if SageMath is finished plotting a frame, and hence that we
# should move to the next y spline, we check if theta was reset to 0.
gauss_color_scale = 1/3
y_splines = []
color_iter = 0
last_theta = 0
def c(theta, rho, eps=0.1):
    global color_iter

    rho = clamp(rho, eps, pi-eps)
    y = y_splines[color_iter % len(y_splines)]
    K = - y.derivative(rho, order=2) / y(rho)
    sigmoid = 1 / (1 + exp(-K * gauss_color_scale))

    global last_theta
    if last_theta != 0 and theta == 0:
        color_iter += 1
    last_theta = theta

    return 1 - sigmoid

# Folder in which all output will be saved.
# WARNING: The program will overwrite previously saved output.
folder_name = "./atcm-gauss-color"
print(f"Using folder: {folder_name}")
if not os.path.exists(folder_name):
    print("Folder did not exist. Creating...")
    os.mkdir(folder_name)

def path(name):
    """Join a desired filename with folder_name."""
    return os.path.join(folder_name, name)


def clamp(x, low, high):
    """Clamp x between low and high."""
    return low if x < low else high if x > high else x


def tissot(g, urange=(0, 2*pi), vrange=(0, pi), ucount=5, vcount=5, sq_len=0.2):
    """Generate a Tissot indicatrix visualization for the metric g.
    
    Args:
        g (func : urange × vrange → M_2(ℝ)): A function which takes in two inputs, u and v,
            and returns a 2×2 matrix with real entries which represents the metric at (u, v)
        urange ((float, float), optional): Range of u across which to take samples of g.
        vrange ((float, float), optional): Range of v across which to take samples of g.
        ucount (int, optional): Number of samples of g in the u direction.
        vcount (int, optional): Numer of samples of g in the v direction.
        sq_len (float, optional): Size of the initial square to which g is applied.
    
    Returns:
        (ndarray, ndarray, ndarray): A tuple containing, respectively, the coordinates for the
            initial squares, the coordinates for the transformed squares, and the coordinates
            for the ellipses.
    """
    def angle(a, b):
        # Compute the angle between vectors a and b.
        return arccos(a.inner_product(b) / (a.norm() * b.norm())) if a.norm() * b.norm() != 0 else 0
    
    squares = []
    transformed_squares = []
    ellipses = []

    uspace = np.linspace(urange[0], urange[1], ucount)
    vspace = np.linspace(vrange[0], vrange[1], vcount)
    space = cartesian_product_iterator([uspace, vspace])
    for u, v in space:
        g_curr = g(u, v)
        eigenvectors = g_curr.eigenvectors_right()
        eigvec = eigenvectors[0][1][0]
        k1 = eigenvectors[0][0]
        if eigenvectors[0][2] == 2:
            k2 = k1
        else:
            k2 = eigenvectors[1][0]
        
        square = matrix([[0, -sq_len/2, 0, sq_len/2], [-sq_len/2, 0, sq_len/2, 0]])
        cross = vector([eigvec[0], eigvec[1], 0]).cross_product(vector([1, 0, 0]))
        theta = arctan2(cross[2], eigvec.dot_product(vector([1, 0])))
        R = matrix([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        
        translate = matrix([[u, u, u, u], [v, v, v, v]])
        transformed = g_curr * R * square
        
        squares.append((R * square) + translate)
        transformed_squares.append(transformed + translate)
        ellipses.append((u, v, abs(k1*sq_len/2), abs(k2*sq_len/2), -theta))

    return squares, transformed_squares, ellipses


def make_g(h, m):
    """Create a metric based on splines h and m, as in the Rubinstein and Sinclair paper.
    
    Args:
        h (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
    
    Returns:
        func : [0, 2*pi] × [0, pi] → M_2(ℝ): Given values for theta and rho, return the metric at that point.
    """
    return lambda theta, rho: matrix([[h(rho), 0], [0, m(rho)]])


def revolve(x, y):
    """Create a 3-tuple which can be given to SageMath's ParametricPlot3D function to plot the revolution of
    the given xy curve.
    
    Args:
        x (spline): x values of the curve of revolution
        y (spline): y values of the curve of revolution
    
    Returns:
        (func, func, func): Tuple which can be given to ParametricPlot3D.
    """
    return (lambda u, v: y(v)*cos(u), lambda u, v: y(v)*sin(u), lambda u, v: x(v))


def xy_splines_from_hm(h, m, srange=(0, pi), step_size=0.1):
    """Obtain splines for x and y given h and m, based on the equation 3—2 in the Rubinstein and Sinclair paper.
    
    Args:
        h (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
        srange ((float, float), optional): Domain on which to compute the x and y splines
        step_size (float, optional): Approximate step size to take along srange
    
    Returns:
        (spline, spline): A tuple containing, respectively, the spline approximation for x,
            and the spline approximation for y. The domain for both splines is srange.
    """
    def y(rho):
        if m(rho) < 0:
            logging.debug(f"\tMaking xy splines from hm, encountered negative: ({rho}, {m(rho)}). Returning 0.")
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
    """Compute the square root of a spline."""
    return spline([(x, sqrt(y)) for x, y in f])


def to_spline(f, srange=(0, pi), step_size=0.01):
    """Take a function and return a spline by sampling the function along srange."""
    return spline([(s, f(s)) for s in np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size))])


def hm_to_ricci_tensor(h, m, return_K=False, eps=0.1):
    """Compute the Ricci tensor given h and m according to equation (3—3) in the Rubinstein and Sinclair paper.
    
    Args:
        h (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
        return_K (bool, optional): If true, return a function for Gauss curvature K along with the Ricci tensor.
        eps (float, optional): We cannot compute the Ricci tensor or the Gauss curvature too close to the poles (rho=0, rho=pi)
            because m→0 as we approach either pole, which causes numerical instability when dividing by m. Hence, when computing
            the Ricci tensor, we clamp rho to be between eps and pi-eps.
    
    Returns:
        func : [0, pi] → M_2(ℝ): Function which takes rho as input and returns the Ricci tensor at rho.
    """
    sqrt_m = sqrt_spline(m)
    
    def K(rho):
        return -sqrt_m.derivative(rho, order=2) / sqrt_m(rho)

    def R(rho):
        rho = clamp(rho, eps, pi-eps)
        R11 = K(rho) + ((m.derivative(rho) * h.derivative(rho)) / (4 * m(rho) * h(rho)))
        R22 = (R11 / h(rho)) * m(rho)
        return matrix([[R11, 0], [0, R22]])

    return R if not return_K else (R, K)


def reparam(h, m, ds=0.1):
    """Reparametrizes h and m by arclength according to the reparametrization given in section 3.1 of the
    Rubinstein and Sinclair paper.
    
    Args:
        h (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
        ds (float, optional): Approximate step size to take when computing l along [0, pi].
    
    Returns:
        (spline, spline): A tuple containing, respectively, the reparametrized h spline and the reparametrized m spline.
    """
    l_spline_list = [(0, 0)]
    def l_integrand(s): return sqrt(h(s)) if h(s) > 0 else 0
    s_space = np.linspace(0, pi, round(pi / ds))

    curr_int = 0
    for i, curr_s in enumerate(s_space[1:]):
        curr_int += numerical_integral(l_integrand, s_space[i], curr_s)[0]
        l_spline_list.append((curr_s, curr_int))

    l = spline(l_spline_list)
    l_inv = spline([(b, a) for a, b in l.list()])

    tot_s = l(pi)
    h_reparam = to_spline(lambda z: (tot_s / pi)^2)
    m_reparam = spline([(rho, max(0, m(l_inv(rho * tot_s / pi)))) for rho, _ in m.list()])

    return h_reparam, m_reparam


def add_cap(h, m):
    """Since the Ricci tensor cannot be computed near the poles due to numerical instability,
    we must avoid some neighborhood around the poles when running Ricci flow. Hence, to define
    h and m on the full domain [0, pi], we append endpoints to h and m at rho=0 and rho=pi
    according to the boundary conditions given by equation (3—1) in the Rubinstein and Sinclair paper.
    
    Args:
        h (spline): h spline without pole (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline without pole (numeric approximation for m in the Rubinstein and Sinclair paper)
    
    Returns:
        (spline, spline): A tuple containing, respectively, the h spline with endpoints, and the m spline with endpoints.
    """
    h_capped = spline(h.list())
    m_capped = spline(m.list())

    m_capped.append((0, 0))
    m_capped.append((pi, 0))

    sqrt_m = sqrt_spline(m_capped)

    h_capped.append((0, sqrt_m.derivative(0)**2))
    h_capped.append((pi, sqrt_m.derivative(pi)**2))

    return h_capped, m_capped


def euler_step(h, m, dt, rho_space, eps, k_space=None, cap=True, return_R=False):
    """Run one Euler step. If k_space is None, we compute the Ricci tensor and run the Euler
    step according to the differential equation for Ricci flow as usual. If k_space is given,
    then instead of computing the Ricci tensor, we use k_space for the derivative. This k_space
    parameter is used in the implementation of rk4.
    
    Args:
        h (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
        dt (float): Timestep.
        rho_space (ndarray): 1D array which contains the rho values for which h and m will be sampled
        eps (float): Given to hm_to_ricci_tensor
        k_space (ndarray | None, optional): If None, the function computes the Ricci tensor for the derivative step.
            If given as a 1D array, the function uses k_space as the derivative instead of computing the Ricci tensor.
            This functionality is necessary for the implementation of rk4.
        cap (bool, optional): If True, calls add_cap(h_next, m_next) before returning h_next and m_next.
        return_R (bool, optional): If True, returns the computed Ricci tensor R along with h_next, m_next, and k_space.
            If return_R is True, then k_space must be None (since if k_space is not None, the Ricci tensor will not be computed).
    
    Returns:
        (spline, spline, ndarray): A tuple containing, respectively, the h spline after the Euler step,
            the m spline after the Euler step, and k_space.
    """
    if k_space is None:
        logging.info("\tComputing Ricci tensor")
        R = hm_to_ricci_tensor(h, m, eps=eps)
        def k(rho): return -2*R(rho)
        k_space = [k(rho) for rho in rho_space]
    rho_k_space = list(zip(rho_space, k_space))

    logging.info("\tComputing h spline")
    h_next = spline([(rho, h(rho) + k[0][0]*dt) for rho, k in rho_k_space])
    logging.info("\tComputing m spline")
    m_next = spline([(rho, m(rho) + k[1][1]*dt) for rho, k in rho_k_space])

    if cap:
        h_next, m_next = add_cap(h_next, m_next)

    return (h_next, m_next, k_space, R) if return_R else (h_next, m_next, k_space)


def rk4_step(h1, m1, dt, eps=0.01, drho=0.01):
    """Run one rk4 step of Ricci flow.
    
    Args:
        h1 (spline): h spline (numeric approximation for h in the Rubinstein and Sinclair paper)
        m1 (spline): m spline (numeric approximation for m in the Rubinstein and Sinclair paper)
        dt (float): Timestep.
        eps (float, optional): How close we get to the poles. Also given to euler_step.
        drho (float, optional): Approximate step size for taking samples of [eps, pi-eps] for computing h and m during the Euler step.
    
    Returns:
        (spline, spline, func): A tuple containing, respectively, the h spline after the rk4 step, the m spline after the rk4 step,
            and the Ricci tensor function.
    """
    rho_space = np.linspace(eps, pi-eps, round((pi-2*eps) / drho))

    logging.info("\tRunning rk4 step 1")
    h2, m2, k1_space, R = euler_step(h1, m1, dt/2, rho_space, eps, return_R=True)

    logging.info("\n\tRunning rk4 step 2")
    _, _, k2_space = euler_step(h2, m2, dt/2, rho_space, eps)
    h3, m3, _ = euler_step(h1, m1, dt/2, rho_space, eps, k_space=k2_space)

    logging.info("\n\tRunning rk4 step 3")
    _, _, k3_space = euler_step(h3, m3, dt/2, rho_space, eps)
    h4, m4, _ = euler_step(h1, m1, dt, rho_space, eps, k_space=k3_space)
    
    logging.info("\n\tRunning rk4 step 4")
    _, _, k4_space = euler_step(h4, m4, dt, rho_space, eps)

    k_space = [(k1 + 2*k2 + 2*k3 + k4)/6 for k1, k2, k3, k4 in zip(k1_space, k2_space, k3_space, k4_space)]
    h_next, m_next, _ = euler_step(h1, m1, dt, rho_space, eps, k_space=k_space)

    logging.info("\n")

    return h_next, m_next, R


# Constants for the first curve from the Rubinstein and Sinclair paper.
c3 = 0.766
c5 = -0.091

# Constants for the second curve from the Rubinstein and Sinclair paper.
# c3 = 0.021
# c5 = 0.598

# Initial metric as given in the Rubinstein and Sinclair paper.
h(rho) = 1
m(rho) = ((sin(rho) + c3*sin(3*rho) + c5*sin(5*rho))/(1 + 3*c3 + 5*c5))**2

# Plotting initial values
srange = (0, pi)
h = to_spline(h, srange)
m = to_spline(m, srange)
x, y = xy_splines_from_hm(h, m, srange, step_size=0.1)
R, K = hm_to_ricci_tensor(h, m, return_K=True)

# Since the principle curvatures approach 0 and infinity as we approach the poles,
# we need to remain some epsilon away from the poles when taking samples for the Tissot indicatrix.
tissot_eps = 0.5

# Since the Tissot ellipses will be larger than the space along which they are sampled,
# we need to view a larger range than the sample space when making the animation.
tissot_theta_padding = 4*pi
tissot_rho_padding = 0.25

# In the Tissot visualization, the Tissot ellipses are rescaled at each step so that
# the ellipses at rho=pi/2 have a constant height (i.e., constant diameter in the rho-direction).
# tissot_const sets that constant diameter.
tissot_const = 0.25

tissot_theta_placement_scale = 8
tissot_rho_placement_scale = 12

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
if plot_initial_tissot:
    tissot_scale = tissot_const / m(pi/2)
    _, _, ellipses = tissot(make_g(h, m), vrange=(tissot_eps, pi-tissot_eps), sq_len=1, ucount=3, vcount=7)
    tissot_plot = Graphics()
    tissot_plot += sum([ellipse((x*tissot_theta_placement_scale, y*tissot_rho_placement_scale), k1*tissot_scale, k2*tissot_scale, theta, axes=False) for x, y, k1, k2, theta in ellipses])
    tissot_plot.set_axes_range(xmin=-tissot_theta_padding, xmax=tissot_theta_placement_scale*2*pi + tissot_theta_padding, ymin=-tissot_rho_padding, ymax=tissot_rho_placement_scale*pi + tissot_rho_padding)
    tissot_plot.save(path("initial_tissot.png"), xmin=-tissot_theta_padding, xmax=2*pi + tissot_theta_padding, ymin=-tissot_rho_padding, ymax=pi + tissot_rho_padding)


# Running Ricci flow
dt = 0.0001

# The simulation will run for at most N timesteps. If it encounters a numerical error earlier, it will terminate and save all animations up until that timestep.
# N = 1000 + 5000 + 1
N = 2001
plot_gap = 400
reparam_gap = 4
space, dt = np.linspace(0, dt*(N-1), N, retstep=True)
eps = 0.1
drho = 0.1

shift_params_after = 0
new_dt = 0.00001
new_plot_gap = 100
new_eps = 0.1
new_drho = 0.05


print("Running ricci flow...")
print(f"c3 = {c3}")
print(f"c5 = {c5}")
print(f"dt = {dt}")
print(f"N = {N}")
print(f"plot_gap = {plot_gap}")

curve_plots = []
revolved_plots = []
revolved_gauss_colored_plots = []
m_plots = []
h_plots = []
R11_plots = []
R22_plots = []
K_plots = []
K_sigmoid_plots = []
tissot_plots = []

for i in range(N):
    try:
        # Switch to smaller timestep (and update other params accordingly) after a certain number of iterations
        if shift_params_after > 0 and i == shift_params_after:
            dt = new_dt 
            plot_gap = new_plot_gap
            eps = new_eps 
            drho = new_drho 

        # Run RK4 step:
        print(f"\nRK4: Iteration {i}/{N-1}, " + "t = {:.6f} (dt = {:.6f})".format(dt*i, dt))
        h, m, R = rk4_step(h, m, dt, eps=eps, drho=drho)
        
        # Reparametrize:
        if i % reparam_gap == 0:
            print("\tReparametrizing...")
            h, m = reparam(h, m)

        # Save plots:
        if i % plot_gap == 0:
            logging.info("\tGetting x and y splines from h and m splines")
            x, y = xy_splines_from_hm(h, m, srange, step_size=0.1)
            y_splines.append(y)
            
            print("\tAppending plots")
            
            x_to_anim = spline([(rho, x_val - x(pi/2)) for rho, x_val in x.list()]) if center_surface_anim else x

            revolved_plots.append(parametric_plot3d(revolve(x_to_anim, y), (0, 2*pi), srange, frame=False))
            if animate_gauss_colored_surface:
                revolved_gauss_colored_plots.append(parametric_plot3d(revolve(x_to_anim, y), (0, 2*pi), srange, plot_points=[20, 80], color=(c, cm), frame=False))
            if animate_curve:
                curve_plots.append(parametric_plot((x, y), srange))
            if animate_m:
                m_plots.append(plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)"))
            if animate_h:
                h_plots.append(plot(h, srange, title="h"))
            if animate_R:
                R11_plots.append(plot(lambda rho: R(rho)[0][0], srange, title="R11"))
                R22_plots.append(plot(lambda rho: R(rho)[1][1], srange, title="R22"))
            if animate_K:
                K_plots.append(plot(lambda rho: y.derivative(rho, order=2) / y(rho), (eps, pi-eps), title="K"))
                K_sigmoid_plots.append(plot(lambda rho: 1 / (1 + exp(-(y.derivative(rho, order=2) / y(rho)))), (eps, pi-eps), title="sigmoid(K)"))
            if animate_tissot:
                tissot_scale = tissot_const / m(pi/2)
                _, _, ellipses = tissot(make_g(h, m), vrange=(tissot_eps, pi-tissot_eps), sq_len=1, ucount=3, vcount=7)
                tissot_plot = Graphics()
                tissot_plot += sum([ellipse((x*tissot_theta_placement_scale, y*tissot_rho_placement_scale), k1*tissot_scale, k2*tissot_scale, theta, axes=False) for x, y, k1, k2, theta in ellipses])
                tissot_plot.set_axes_range(xmin=-tissot_theta_padding, xmax=tissot_theta_placement_scale*2*pi + tissot_theta_padding, ymin=-tissot_rho_padding, ymax=tissot_rho_placement_scale*pi + tissot_rho_padding)
                tissot_plots.append(tissot_plot)
    except Exception as e:
        print(f"Encountered exception on iteration {i}/{N-1}, t = {dt*i}:")
        print(repr(e))
        print("Terminating flow.\n")
        break


def save_animation(plots, label, filename, show_path=True, online=None):
    print(f"Animating {label}...")
    start = time.time()
    anim = animate(plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print(f"Saving {label} animation...")
    start = time.time()
    if online is not None:
        anim.save(path(filename), show_path=show_path, online=online)
    else:
        anim.save(path(filename), show_path=show_path)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.\n")

    return end - start


total_time = 0
total_time += save_animation(revolved_plots, "surface flow", "surf_flow.html", online=True)

if animate_gauss_colored_surface:
    total_time += save_animation(revolved_gauss_colored_plots, "Gauss-colored surface flow", "gauss_colored_surf_flow.html", online=True)
if animate_curve:
    total_time += save_animation(curve_plots, "curve", "curve_flow.gif")
if animate_m:
    total_time += save_animation(m_plots, "sqrt(m)", "sqrt_m_anim.gif")
if animate_h:
    total_time += save_animation(h_plots, "h", "h_anim.gif")
if animate_R:
    total_time += save_animation(R11_plots, "R11", "R11_anim.gif")
    total_time += save_animation(R22_plots, "R22", "R22_anim.gif")
if animate_K:
    total_time += save_animation(K_plots, "K", "K_anim.gif")
    total_time += save_animation(K_sigmoid_plots, "sigmoid(K)", "K_sigmoid_anim.gif")
if animate_tissot:
    total_time += save_animation(tissot_plots, "Tissot", "tissot_anim.gif")

print(f"Done saving plots in {total_time} seconds.")