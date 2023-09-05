import matplotlib.pyplot as plt
import numpy as np

# import matplotlib as mpl 
# mpl.rcParams['text.usetex'] = True


# Set the font family to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

bar_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']

whisker_colors = ['blue', 'green', 'red', 'salmon']
categories = ['DAMM', 'PC-GMM', 'GMM (position)', 'GMM (position+velocity)']

# categories_empty= ['DAMM', 'PC-GMM', 'GMM (position)', 'GMM (position+velocity)']

fig, axs = plt.subplots(3, 2, figsize=(12, 5))


'''
Time
'''
# Calculate summary statistics
means = [1.2, 95, 13, 18]
q3 = [3, 340, 25, 30]

ax11 = axs[0, 0]

# Create a bar graph with specified bar colors
bars = ax11.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)
ax11.set_xticklabels([])

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax11.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax11.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax11.set_xlabel('') 
ax11.set_ylabel('Time (sec)', size=13)
ax11.set_yscale('log')
ax11.grid(axis='y', linestyle='--', alpha=0.7)



'''
BIC
'''
means = [5854, 6023, 6361, 7600]
q3 = [6135, 6340, 8825, 9213]

ax21 = axs[1, 0]

# Create a bar graph with specified bar colors
bars = ax21.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)
ax21.set_xticklabels([])

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax21.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax21.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax21.set_ylabel('BIC', size=13)
ax21.grid(axis='y', linestyle='--', alpha=0.7)

'''
AIC
'''

means = [5635, 5874, 6213, 7498]
q3 = [5923, 6482, 8255, 9130]

ax31 = axs[2, 0]

# Create a bar graph with specified bar colors
bars = ax31.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax31.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax31.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax31.set_ylabel('AIC', size=13)
ax31.grid(axis='y', linestyle='--', alpha=0.7)




'''
RMSE
'''
# Calculate summary statistics
means = [0.81, 0.958, 1.28, 1.38]
q3 = [1.03, 1.35, 1.73, 1.64]

ax12 = axs[0, 1]

# Create a bar graph with specified bar colors
bars = ax12.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)
ax12.set_xticklabels([])

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax12.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax12.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax12.set_xlabel('') 
ax12.set_ylabel('RMSE', size=13)
ax12.grid(axis='y', linestyle='--', alpha=0.7)



'''
e_dot
'''
means = [0.065, 0.089, 0.357, 0.481]
q3 = [0.089, 0.129, 0.542, 0.681]

ax21 = axs[1, 1]

# Create a bar graph with specified bar colors
bars = ax21.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)
ax21.set_xticklabels([])

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax21.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax21.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax21.set_ylabel(r'$\dot{e}$', size=13)
ax21.grid(axis='y', linestyle='--', alpha=0.7)

'''
DTWD
'''

means = [280, 431, 581, 690]
q3 = [300, 530, 620, 781]

ax31 = axs[2, 1]

# Create a bar graph with specified bar colors
bars = ax31.bar(categories, means, capsize=5, edgecolor=whisker_colors, color=bar_colors)

# Manually draw upper whiskers as vertical lines with specified whisker colors
whisker_height = 0.05  # Adjust this value to control the length of the whiskers
for k, bar in enumerate(bars):
    x = bar.get_x() + bar.get_width() / 2
    ax31.plot([x, x], [means[k], q3[k]], color=whisker_colors[k], linewidth=1)
    ax31.plot([x - 0.05, x + 0.05], [q3[k], q3[k]], color=whisker_colors[k], linewidth=1)

ax31.set_ylabel('DTWD', size=13)
ax31.grid(axis='y', linestyle='--', alpha=0.7)













# Adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Show the plots
plt.savefig('metric.png', dpi=300, bbox_inches='tight')

plt.show()


