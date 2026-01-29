import json
import matplotlib.pyplot as plt

# Carica risultati
with open('tuning_results/xgboost/bayesian_trials593_cv5_2026-01-29_07.31.json') as f:
    data = json.load(f)

# Estrai score per trial
trials = data['all_results']
scores = [t['score'] for t in trials]

# Plot convergenza
plt.figure(figsize=(12, 6))
plt.plot(scores, marker='o', alpha=0.6)
plt.axhline(max(scores), color='r', linestyle='--', label=f'Best: {max(scores):.4f}')
plt.xlabel('Trial #')
plt.ylabel('Composite Score')
plt.title('Optuna Convergence - XGBoost')
plt.legend()
plt.grid(True, alpha=0.3)

# Quando ha trovato il best?
best_idx = scores.index(max(scores))
print(f"Best trovato al trial: {best_idx+1}/593")
print(f"Trials dopo il best: {593 - best_idx}")