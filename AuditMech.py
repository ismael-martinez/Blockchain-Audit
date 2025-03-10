import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(25)
np.random.seed(100)

# Global variables
kappa = 2
partial = 0.1
time_window = 6 # hours
audits = []
fog_nodes = []
system_time = 0
tau = 0.7
base_audit, target = 0.05, 0.05
penalty, reward = 0.2, 0.1
penalty_min = 0.2
penalty_partial = 0.15
deposit_penalty = 0.5
time = 0
iota = 2
IVs = dict()
avg_rep = dict()

class Audit:
    def __init__(self, fog_id, pass_audit, full, timestamp):
        self.fog_id = fog_id
        self.pf = pass_audit # pass / fail
        self.full = full # full 1, partial 0
        self.time = timestamp

        global fog_nodes
        if not self.pf:
            if self.full:
                fog_nodes[fog_id].reputation -= penalty
                if fog_nodes[fog_id].reputation <= 0:
                    fog_nodes[fog_id].alive = 0
                fog_nodes[fog_id].collateral -= deposit_penalty
                if fog_nodes[fog_id].collateral <= 0:
                    fog_nodes[fog_id].alive = 0
            else:
                fog_nodes[fog_id].reputation -= penalty_partial
                if fog_nodes[fog_id].reputation <= 0:
                    fog_nodes[fog_id].alive = 0
        else:
            fog_nodes[fog_id].reputation += reward
            if fog_nodes[fog_id].reputation > 1:
                fog_nodes[fog_id].reputation = 1

class FogNode:
    def __init__(self, fog_id, intra_honesty, inter_honesty, rep_score):
        self.fog_id = fog_id
        self.beta = intra_honesty
        self.alpha = inter_honesty
        self.reputation = rep_score
        self.collateral = random.random()*5 + 5 # random between [5,10]
        self.alive = 1
        self.busy = 0
        self.last_request = 0
        self.lambda_r = 0
        self.mu_r = 0

def audit_arrival(rep_score, thresh, base_audit):
    audit_ar = base_audit # audit_arrival_rate
    if(rep_score < thresh):
        sensitivity_term = ((thresh - rep_score)/thresh)**kappa
        audit_ar += (1-base_audit)*sensitivity_term
    return audit_ar

def prob_passing(honesty_rate_beta, partial_verification): # both \in (0,1)
    num = (honesty_rate_beta - partial_verification)**2
    denom = honesty_rate_beta * (1-partial_verification)
    return num / denom

def opt_base_mu(rep_scores, thresh, target):
    F = len(rep_scores) # number of fog nodes
    num = F*target
    denom = F
    for r in rep_scores:
        if r < thresh:
            thresh_r_term = ((thresh - r)/thresh)**kappa
            num -= thresh_r_term
            denom -= thresh_r_term
    return num / denom

def integrity_value(t):
    global audits
    t_prev = t - time_window
    if t_prev < 0:
        return None
    # Remove all audits older than t_prev
    index_keep = 0
    for i,a in enumerate(audits):
        if a.time >= t_prev:
            index_keep = i
            break
    audits = audits[i:]
    total = len(audits)
    passed = 0
    for a in audits:
        if a.pf:
            passed += 1
    if total > 0:
        return passed / total
    return 1

def audit(fog_id):
    alpha = fog_nodes[f].alpha
    a = random.random()
    global audits
    if a < alpha:
        # Pass audit
        audits.append(Audit(fog_id, 1, 1, time))
    else:
        audits.append(Audit(fog_id, 0, 1, time))
        if random.random() < 0.5:
            fog_nodes[fog_id].alpha += 0.1
            if fog_nodes[fog_id].alpha > 1:
                fog_nodes[fog_id].alpha = 1

def partial_verification(fog_id):
    alpha = fog_nodes[f].alpha
    beta = fog_nodes[f].beta
    a = random.random()
    global audits
    if a < alpha:
        # Pass audit
        audits.append(Audit(fog_id, 1, 0, time))
    else:
        v = prob_passing(beta, partial)
        if random.random() < v:
            audits.append(Audit(fog_id, 1, 0, time))
        else:
            audits.append(Audit(fog_id, 0, 0, time))

            if random.random() < 0.5:
                fog_nodes[f].beta += 0.1
                if fog_nodes[f].beta > 1:
                    fog_nodes[f].beta = 1
                    fog_nodes[f].alpha = 1

# Update lambda arrival for all fog nodes
def lambda_r():
    global iota, fog_nodes
    sum_rep = 0
    for f in fog_nodes:
        if f.alive:
            sum_rep += f.reputation**iota
    for f in fog_nodes:
        if f.alive:
            f.lambda_r = f.reputation**iota / sum_rep

def mu_r():
    global kappa, base_audit, tau
    for f in fog_nodes:
        if f.alive:
            f.mu_r = base_audit
            if f.reputation < tau:
                lf_term = ((tau - f.reputation)/tau)**kappa
                f.mu_r += (1-base_audit)*lf_term

def update_audit_parameters():
    lambda_r() # update request arrival rates
    mu_r() # update audit arrival rates
    IV = integrity_value(time)
    if IV is not None:
        global IVs
        IVs[time] = IV
        global penalty
        penalty = penalty_min + 0.2 * (1 - IV)
    global avg_rep
    reps = [f.reputation for f in fog_nodes if f.alive]
    avg_rep[time] = np.average(reps)

    rep_scores = [f.reputation for f in fog_nodes if f.alive > 0]
    global base_audit
    base_audit = opt_base_mu(rep_scores, tau, target)


# Setup
rep_scores = []
for r in range(100):
    samp = 0
    while samp <= 0 or samp > 1:
        samp = np.random.normal(0.5, 1)
    rep_scores.append(samp)

for f, r in enumerate(rep_scores):
    alpha = random.random() #inter
    beta = random.random() #intra
    fog_nodes.append(FogNode(f, beta, alpha, r))


while time <= 550:
    update_audit_parameters()
    arrival_rates = [f.mu_r + f.lambda_r for f in fog_nodes if f.alive]
    arrival_rates_cumul = arrival_rates
    for i, ar in enumerate(arrival_rates):
        if i > 0:
            arrival_rates_cumul[i] += arrival_rates_cumul[i-1]
    max_ar = arrival_rates_cumul[-1]
    rand = random.random()*max_ar
    fog_id = 0
    alive_fog_ids = [f.fog_id for f in fog_nodes if f.alive]
    for f, arc in enumerate(arrival_rates_cumul):
        if rand < arc:
            fog_id = alive_fog_ids[f]
            break
    fog_node = fog_nodes[fog_id]
    next_full_audit = 0
    arrival_rates_f = [fog_node.mu_r, fog_node.mu_r + fog_node.lambda_r]
    rand = random.random()*arrival_rates_f[-1]
    if rand < arrival_rates_f[0]:
        next_full_audit = 1

    # Schedule audit or partial verification
    interarrival_time_next_audit = np.random.exponential(1./max_ar)
    time += interarrival_time_next_audit
    if next_full_audit:
        audit(fog_id)
    else:
        partial_verification(fog_id)


IVs_vals = sorted(IVs.items())
x,y = zip(*IVs_vals)
plt.plot(x,y)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Integrity $\Phi_t$')
plt.title(r'System Integrity $\Phi_t$ over time')
plt.savefig("ivs_sim.png")
plt.show()

rep_vals = sorted(avg_rep.items())
x,y = zip(*rep_vals)
plt.plot(x,y)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average fog node reputation score $r$')
plt.title(r'Average reputation scores over time')
plt.savefig("rep_sim.png")
plt.show()