import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from statsmodels.stats.proportion import proportions_ztest
matplotlib.use('TKAgg')

df = pd.read_csv('../../Data/raw/cookie_cats.csv')

print(df.head())
print(df.info())
print(df['version'].count())

grouped = df.groupby('version').agg({
    'userid': 'nunique',
    'retention_1': 'mean',
    'retention_7': 'mean',
    'sum_gamerounds': 'mean'
}).reset_index()


grouped.columns = ['version', 'num_users', 'retention_1_rate', 'retention_7_rate', 'avg_gamerounds']

print("\n=== A/B Test Summary by version ===")
print(grouped)

# --- Statistical test: Two-proportion z-test for retention_1 ---

# Separate groups
group_A = df[df['version'] == 'gate_30']
group_B = df[df['version'] == 'gate_40']

# number of successes (users who returned next day)
successes = [
    group_A['retention_1'].sum(),
    group_B['retention_1'].sum()
]

# number of observations in each group
n_obs = [
    group_A['retention_1'].count(),
    group_B['retention_1'].count()
]

z_stat, p_value = proportions_ztest(count=successes, nobs=n_obs)
print("\n=== Two-proportion z-test for retention_1 ===")
print(f"z-stat: {z_stat:.3f}")
print(f"p-value: {p_value:.5f}")

retention_summary = (
    df.groupby('version')[['retention_1', 'retention_7']]
      .mean()
      .reset_index()
)

retention_melted = retention_summary.melt(
    id_vars='version',
    value_vars=['retention_1', 'retention_7'],
    var_name='metric',
    value_name='rate'
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=retention_melted,
    x='metric',
    y='rate',
    hue='version'
)
plt.title('Retention Day 1 & Day 7 by Version')
plt.ylabel('Retention Rate')
plt.xlabel('')
plt.ylim(0, 0.6)
plt.legend(title='Version')
plt.tight_layout()
plt.show()

ci_df = (
    df.groupby('version')
      .agg(
          n=('retention_1', 'size'),
          p=('retention_1', 'mean')
      )
      .reset_index()
)

ci_df['se'] = np.sqrt(ci_df['p'] * (1 - ci_df['p']) / ci_df['n'])
ci_df['ci_low'] = ci_df['p'] - 1.96 * ci_df['se']
ci_df['ci_high'] = ci_df['p'] + 1.96 * ci_df['se']

plt.figure(figsize=(6, 5))
plt.errorbar(
    x=ci_df['version'],
    y=ci_df['p'],
    yerr=1.96 * ci_df['se'],
    fmt='o',
    capsize=5
)
plt.title('Retention Day 1 with 95% Confidence Intervals')
plt.ylabel('Retention Rate')
plt.xlabel('Version')
plt.ylim(0, 0.6)
plt.tight_layout()
plt.show()

df_plot = df.copy()
df_plot = df_plot[df_plot['sum_gamerounds'] <= 200]  # برای خوانایی

plt.figure(figsize=(9, 5))
sns.histplot(
    data=df_plot,
    x='sum_gamerounds',
    hue='version',
    bins=50,
    stat='density',
    common_norm=False,
    alpha=0.4
)
plt.title('Distribution of Game Rounds by Version (<= 200 rounds)')
plt.xlabel('Total Game Rounds')
plt.ylabel('Density')
plt.tight_layout()
plt.show()


max_round = 100
thresholds = np.arange(1, max_round + 1)

surv_curves = {}

for version in df['version'].unique():
    sub = df[df['version'] == version]
    surv = []
    for t in thresholds:
        surv.append((sub['sum_gamerounds'] >= t).mean())
    surv_curves[version] = surv

plt.figure(figsize=(9, 5))
for version, surv in surv_curves.items():
    plt.plot(thresholds, surv, label=version)

plt.title('Share of Users with at Least N Rounds (by Version)')
plt.xlabel('N (Number of Rounds)')
plt.ylabel('Proportion of Users')
plt.legend(title='Version')
plt.tight_layout()
plt.show()
