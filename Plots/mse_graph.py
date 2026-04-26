import pandas as pd
import matplotlib.pyplot as plt

# 1. Create the data with shortened labels
data = {
    'model': ['Baseline', 'CNN', 'DinoV2', 'FT', 'Aug + FT'],
    'MSE': [141.11, 287.43, 220.82, 134.64, 127.84]
}
df = pd.DataFrame(data)

# 2. Determine colors: Lowest MSE gets bright red, others pink
min_mse = df['MSE'].min()
colors = ['#FF0000' if mse == min_mse else 'pink' for mse in df['MSE']]

# 3. Set up the plot (less wide: 7x6 instead of 10x6)
plt.figure(figsize=(7, 6))

# 4. Plot bars with skinnier width (0.5 instead of default 0.8)
bars = plt.bar(df['model'], df['MSE'], color=colors, edgecolor='black', width=0.5)

# 5. Add labels and title
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Model')
plt.title('Model Comparison by MSE')

# Rotate the x-axis labels so they don't overlap
plt.xticks(rotation=45, ha='right')

# 6. Add the exact MSE numbers on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}', ha='center', va='bottom')

# 7. Add padding to the top of the chart so numbers aren't cut off (15% headroom)
plt.ylim(0, df['MSE'].max() * 1.15)

# 8. Adjust layout and save the plot
plt.tight_layout()
plt.savefig('mse_comparison_updated.png')
plt.show()