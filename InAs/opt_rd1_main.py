import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from solver_NFTPV import sel_em_TPV

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Bounds
N_l, wp_l, gam_l = 16, 1e14, 1e13
N_u, wp_u, gam_u = 19, 4e16, 1e15
bounds = torch.tensor([[N_l, N_l, wp_l, gam_l, wp_l, gam_l],
                       [N_u, N_u, wp_u, gam_u, wp_u, gam_u]], device=device)

# Objective
def example_obj(ind):
    z = []
    for x in ind:
        eff, _ = sel_em_TPV(x[0], x[1], x[2], x[3], x[4], x[5])
        z.append(eff)
    return torch.tensor(z, device=device)

# Initial data
def generate_initial_data(n):
    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, 6, device=device, dtype=torch.double)
    train_obj = example_obj(train_x).unsqueeze(-1)
    best_value = train_obj.max().item()
    return train_x, train_obj, best_value

# Acquisition optimization
def get_next_points_analytic(init_x, init_y, best_init_y, bounds, n_points):
    model = SingleTaskGP(init_x, init_y,
                         input_transform=Normalize(d=6),
                         outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    EI = qLogExpectedImprovement(model=model, best_f=best_init_y)

    new_point_analytic, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=200,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 400}
    )
    return new_point_analytic, model, EI

# ---------------- RUN ---------------- #
init_x, init_y, best_init_y = generate_initial_data(200)

np.savetxt("Init_X.txt", init_x.cpu().numpy())
np.savetxt("Init_Y.txt", init_y.cpu().numpy())

evaluated_x, evaluated_y = [], []
best_x_list, best_y_list = [], []
std_ev_list = []
std_pr_list = []
acq_list = []
mean_ev_list = []
mean_pr_list = []

ctr = 0
best_old = 0

for i in range(2000):
    new_candidates, model, EI = get_next_points_analytic(init_x, init_y, best_init_y, bounds, 1)
    new_results = example_obj(new_candidates).unsqueeze(-1)
    acq_val = EI(new_candidates).item()

    # update training data
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
    index = torch.argmax(init_y)
    best_init_x = init_x[index].unsqueeze(0)

    # uncertainty estimation
    posterior_evaluated = model.posterior(best_init_x)
    mean_evaluated = posterior_evaluated.mean.item()
    std_evaluated = posterior_evaluated.variance.sqrt().item()

    posterior_predicted = model.posterior(new_candidates)
    mean_predicted = posterior_predicted.mean.item()
    std_predicted = posterior_predicted.variance.sqrt().item()

    # convergence check
    if best_old == best_init_y:
        ctr = ctr + 1
    else:
        ctr = 0

    print(f"mean: {mean_predicted:.4f}, SD: {std_predicted:.4f}, objective: {best_init_y:.4f}, counter: {ctr},  iteration:{i}")

    best_old = best_init_y

    if std_predicted < 1e-4 and ctr > 99:
        print(f"Converged at iteration {i}, acquisition value = {acq_val:.3e}")
        break

    # save
    evaluated_x.append(new_candidates.cpu().numpy())
    evaluated_y.append(new_results.cpu().numpy())
    best_x_list.append(best_init_x.cpu().numpy())
    best_y_list.append(best_init_y)
    std_ev_list.append(std_evaluated)
    std_pr_list.append(std_predicted)
    mean_ev_list.append(mean_evaluated)
    mean_pr_list.append(mean_predicted)
    acq_list.append(acq_val)

# save results
np.savetxt("X_list.txt", np.vstack(evaluated_x))
np.savetxt("Y_list.txt", np.vstack(evaluated_y))
np.savetxt("Best_X_list.txt", np.vstack(best_x_list))
np.savetxt("Best_Y_list.txt", np.array(best_y_list))
np.savetxt("Std_ev.txt", np.array(std_ev_list))
np.savetxt("Std_pr.txt", np.array(std_pr_list))
np.savetxt("Mean_ev.txt", np.array(mean_ev_list))
np.savetxt("Mean_pr.txt", np.array(mean_pr_list))
np.savetxt("Acq_fun.txt", np.array(acq_list))