import numpy as np
import pandas as pd


class MarketSimulation:
    def __init__(self, N=14, a=10, b=1, s=2, T=100, num_simulations=100, m=5, beta=5,
                 tau=0.5, C=0.5, lambda_exp=0.1, f=0.1, price_control=None, p_max=8, p_min=2):
        self.N = N
        self.a = a
        self.b = b
        self.s = s
        self.T = T
        self.num_simulations = num_simulations
        self.m = m
        self.beta = beta
        self.tau = tau
        self.C = C
        self.lambda_exp = lambda_exp
        self.f = f
        self.price_control = price_control
        self.p_max = p_max
        self.p_min = p_min

    def run_simulation(self, model="baseline"):
        results = []
        for sim in range(self.num_simulations):
            np.random.seed(sim)
            price_taker_ratio = np.round(1 - 1 / self.N, 2)
            num_price_takers = int(self.N * price_taker_ratio)
            num_price_makers = self.N - num_price_takers
            firm_types = np.array(['PT'] * num_price_takers + ['PM'] * num_price_makers)
            np.random.shuffle(firm_types)

            U_PT = np.full(self.N, 100.0)
            U_PM = np.full(self.N, 100.0)
            p_t_minus_1 = np.random.uniform(0, self.a)
            profits = np.zeros(self.N)
            previous_profits = np.zeros(self.N)
            K = np.full(self.N, 5.0) if model == "excess_capacity" else None

            market_compositions = []
            increase_counts = {n: 0 for n in range(self.N + 1)}
            remain_counts = {n: 0 for n in range(self.N + 1)}
            decrease_counts = {n: 0 for n in range(self.N + 1)}
            total_counts = {n: 0 for n in range(self.N + 1)}

            for t in range(1, self.T + 1):
                n_t = np.sum(firm_types == 'PT')
                n_pm = self.N - n_t

                if n_t > 0:
                    m_PT = p_t_minus_1 / self.s
                else:
                    m_PT = 0
                if n_pm > 0:
                    numerator = self.a - self.b * n_t * m_PT / self.s
                    denominator = self.s + self.b * (n_pm + 1)
                    m_PM = max(numerator / denominator, 0)
                else:
                    m_PM = 0

                if model == "excess_capacity":
                    m_PT = min(max(m_PT, 0), np.min(K[firm_types == 'PT'])) if n_t > 0 else 0
                    m_PM = min(max(m_PM, 0), np.min(K[firm_types == 'PM'])) if n_pm > 0 else 0
                else:
                    m_PT = max(m_PT, 0)
                    m_PM = max(m_PM, 0)

                total_quantity = n_t * m_PT + n_pm * m_PM
                if self.price_control == "maximum":
                    p_t = min(self.a - self.b * total_quantity, self.p_max)
                elif self.price_control == "minimum":
                    p_t = max(self.a - self.b * total_quantity, self.p_min)
                else:
                    p_t = self.a - self.b * total_quantity
                p_t = max(p_t, 0)

                previous_profits = profits.copy()
                for i in range(self.N):
                    if firm_types[i] == 'PT':
                        m_i = m_PT
                    else:
                        m_i = m_PM
                    if model == "excess_capacity":
                        m_i = min(m_i, K[i])
                        profits[i] = p_t * m_i - (self.s * m_i ** 2) / 2 - self.f * K[i]
                    else:
                        profits[i] = p_t * m_i - (self.s * m_i ** 2) / 2

                if t % self.m == 0:
                    for i in range(self.N):
                        current_type = firm_types[i]
                        if np.random.rand() < self.lambda_exp:
                            new_type = np.random.choice(['PT', 'PM'])
                            if model == "excess_capacity":
                                K[i] = max(K[i] + np.random.uniform(-1, 1), 0.1)
                        else:
                            if current_type == 'PT':
                                U_PT[i] = (1 - self.tau) * profits[i] + self.tau * U_PT[i]
                            else:
                                U_PM[i] = (1 - self.tau) * (profits[i] - self.C) + self.tau * U_PM[i]
                            exp_PT = np.exp(self.beta * U_PT[i])
                            exp_PM = np.exp(self.beta * U_PM[i])
                            P_PT = exp_PT / (exp_PT + exp_PM)
                            new_type = 'PT' if np.random.rand() < P_PT else 'PM'
                            if model == "excess_capacity":
                                K[i] = max(K[i] + 0.5 * np.sign(profits[i] - previous_profits[i]), 0.1)
                        changed_strategy = new_type != firm_types[i]
                        profit_change = profits[i] - previous_profits[i]
                        if changed_strategy:
                            if profit_change > 0:
                                increase_counts[n_t] += 1
                            elif profit_change < 0:
                                decrease_counts[n_t] += 1
                        else:
                            if profit_change >= 0:
                                remain_counts[n_t] += 1
                        firm_types[i] = new_type
                        total_counts[n_t] += 1

                p_t_minus_1 = p_t
                market_compositions.append(n_t)

            unique_n = np.unique(market_compositions)
            for n in unique_n:
                indices = [i for i, x in enumerate(market_compositions) if x == n]
                if indices:
                    avg_profit_PT = np.mean([profits[i] for i in range(self.N) if firm_types[i] == 'PT']) if n > 0 else 0
                    avg_profit_PM = np.mean([profits[i] for i in range(self.N) if firm_types[i] == 'PM']) if n < self.N else 0
                    total = total_counts[n] if total_counts[n] > 0 else 1
                    increase_pct = (increase_counts[n] / total) * 100
                    remain_pct = (remain_counts[n] / total) * 100
                    decrease_pct = (decrease_counts[n] / total) * 100
                    results.append({
                        'n': n,
                        'avg_profit_PT': avg_profit_PT,
                        'avg_profit_PM': avg_profit_PM,
                        'Increase (%)': increase_pct,
                        'Remain (%)': remain_pct,
                        'Decrease (%)': decrease_pct
                    })
        return pd.DataFrame(results).groupby('n').mean().reset_index()


# Usage Example
baseline_model = MarketSimulation()
baseline_results = baseline_model.run_simulation(model="baseline")

excess_capacity_model = MarketSimulation()
excess_capacity_results = excess_capacity_model.run_simulation(model="excess_capacity")

price_control_model = MarketSimulation(price_control="maximum")
price_control_results = price_control_model.run_simulation(model="price_control")

print("Baseline Results")
print(baseline_results)

print("Excess Capacity Results")
print(excess_capacity_results)

print("Price Control Results")
print(price_control_results)
