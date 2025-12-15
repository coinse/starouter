import numpy as np
from data_utils import load_keys, retrieve_corresponding_result_loader
from config import ALGORITHMS

def compute_random_performance_for_given_ratio(strong_result, weak_result, strong_call_ratio, n_samples=101):
    np.random.seed(42)
    strong_result = np.asarray(strong_result)
    weak_result = np.asarray(weak_result)
    
    call_count = len(strong_result)
    strong_model_call_count = int(call_count * strong_call_ratio)
    
    random_perf = np.empty(n_samples)
    
    for i in range(n_samples):
        mask = np.zeros(call_count, dtype=bool)
        mask[np.random.choice(call_count, strong_model_call_count, replace=False)] = True
        aggregated_sum = np.sum(strong_result[mask]) + np.sum(weak_result[~mask])
        random_perf[i] = aggregated_sum / call_count
    
    return np.median(random_perf)

def compute_thresholds_for_quantiles(probs, num_steps=100):
    quantiles = np.linspace(0, 1, num_steps)
    return np.quantile(probs.tolist(), quantiles).tolist()

def compute_routed_performance(probs, strong_result, weak_result):
    proportion_of_calls_to_strong_model = list()
    routing_results = list()
    for threshold in [-100] + compute_thresholds_for_quantiles(probs) + [+100]:
        routed_performance = 0
        call_count = 0
        for i, p in enumerate(probs):
            if p >= threshold:
                routed_performance += weak_result[i]
            else:
                routed_performance += strong_result[i]
                call_count += 1
        proportion_of_calls_to_strong_model.append(call_count / len(strong_result))
        routing_results.append(routed_performance / len(strong_result))

    return proportion_of_calls_to_strong_model, routing_results

def compute_area_under_curve(xs, ys):
    area = 0.0
    for i in range(len(xs) - 1):
        del_x = xs[i + 1] - xs[i]
        mean_y = (ys[i] + ys[i + 1]) / 2
        area += del_x * mean_y
    return area

def compute_best_and_worst_routing_performance(weak_result, strong_result):
    performance_gain = [strong_result[i] - weak_result[i] for i in range(len(weak_result))]
    sorted_gain = sorted(performance_gain, reverse=True)
    
    x_indices = [i / len(weak_result) for i in range(0, len(weak_result) + 1)]
    weak_only = sum(weak_result)

    best_performances = [weak_only / len(weak_result)]
    performance = weak_only
    for gain in sorted_gain:
        performance += gain
        best_performances.append(performance / len(weak_result))
    
    worst_performances = [weak_only / len(weak_result)]
    performance = weak_only
    for gain in reversed(sorted_gain):
        performance += gain
        worst_performances.append(performance / len(weak_result))

    return x_indices, best_performances, worst_performances

def compute_router_optimality(benchmark, strong_model_name, weak_model_name, strong_model_result_path, weak_model_result_path, probs_df):
    keys = load_keys(benchmark, strong_model_name, weak_model_name)
    loader = retrieve_corresponding_result_loader(benchmark) 
    strong_result = loader(strong_model_result_path, keys)
    weak_result = loader(weak_model_result_path, keys)

    ro_dict = dict()
    x_indices, best_performances, worst_performances = compute_best_and_worst_routing_performance(weak_result, strong_result)
    area_under_best = compute_area_under_curve(x_indices, best_performances)
    area_under_worst = compute_area_under_curve(x_indices, worst_performances)

    for algorithm in ALGORITHMS:
        weak_win_probs = probs_df[probs_df['Model'] == algorithm]["CV Probabilities"].iloc[0][:, 1]
        ratios, routed_perf = compute_routed_performance(weak_win_probs, strong_result, weak_result)
        area_under_routed_curve = compute_area_under_curve(ratios, routed_perf)
        ro_dict[algorithm] = (area_under_routed_curve - area_under_worst) / (area_under_best - area_under_worst)
    
    return ro_dict

def compute_apgr(benchmark, strong_model_name, weak_model_name, strong_model_result_path, weak_model_result_path, probs_df):
    keys = load_keys(benchmark, strong_model_name, weak_model_name)
    loader = retrieve_corresponding_result_loader(benchmark) 
    strong_result = loader(strong_model_result_path, keys)
    weak_result = loader(weak_model_result_path, keys)
    apgr_dict = dict()
    for algorithm in ALGORITHMS:
        weak_win_probs = probs_df[probs_df['Model'] == algorithm]["CV Probabilities"].iloc[0][:, 1]
        ratios, routed_perf = compute_routed_performance(weak_win_probs, strong_result, weak_result)
        weak_only_perf, strong_only_perf = routed_perf[0], routed_perf[-1]
        area_under_routed_curve = compute_area_under_curve(ratios, routed_perf)
        apgr_dict[algorithm] = (area_under_routed_curve - weak_only_perf) / (strong_only_perf - weak_only_perf)
    return apgr_dict
        
def compute_cpt(benchmark, strong_model_name, weak_model_name, strong_model_result_path, weak_model_result_path, probs_df, target_pgr=0.5):
    keys = load_keys(benchmark, strong_model_name, weak_model_name)
    loader = retrieve_corresponding_result_loader(benchmark) 
    strong_result = loader(strong_model_result_path, keys)
    weak_result = loader(weak_model_result_path, keys)
    cpt_dict = dict()
    for algorithm in ALGORITHMS:
        weak_win_probs = probs_df[probs_df['Model'] == algorithm]["CV Probabilities"].iloc[0][:, 1]
        ratios, routed_perf = compute_routed_performance(weak_win_probs, strong_result, weak_result)
        weak_only_perf, strong_only_perf = routed_perf[0], routed_perf[-1]
        if strong_only_perf <= weak_only_perf:
            cpt_dict[algorithm] = -1.0
            continue
        for i in range(len(ratios)):
            current_pgr = (routed_perf[i] - weak_only_perf) / (strong_only_perf - weak_only_perf)
            if current_pgr >= target_pgr:
                cpt_dict[algorithm] = ratios[i]
                break
    return cpt_dict