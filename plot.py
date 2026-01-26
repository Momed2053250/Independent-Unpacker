#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('validation_plots', exist_ok=True)

# Read CSV
df = pd.read_csv('validation_results.csv')

print(f"Loaded {len(df)} matched clusters from validation_results.csv")
print(f"Events: {sorted(df['event'].unique())}")
print(f"\nSaving plots to: validation_plots/")

# 1. X position: Expected vs Unpacked
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['expected_x'], df['unpacked_x'], alpha=0.5, s=20)
max_x = max(df['expected_x'].max(), df['unpacked_x'].max())
ax.plot([0, max_x], [0, max_x], 'r--', label='Perfect match', linewidth=2)
ax.set_xlabel('Expected X', fontsize=12)
ax.set_ylabel('Unpacked X', fontsize=12)
ax.set_title('X Position Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/x_position_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ x_position_comparison.png")

# 2. Y position: Expected vs Unpacked
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['expected_y'], df['unpacked_y'], alpha=0.5, s=20)
max_y = max(df['expected_y'].max(), df['unpacked_y'].max())
ax.plot([0, max_y], [0, max_y], 'r--', label='Perfect match', linewidth=2)
ax.set_xlabel('Expected Y', fontsize=12)
ax.set_ylabel('Unpacked Y', fontsize=12)
ax.set_title('Y Position Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/y_position_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ y_position_comparison.png")

# 3. Width: Expected vs Unpacked
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['expected_width'], df['unpacked_width'], alpha=0.5, s=20)
max_w = max(df['expected_width'].max(), df['unpacked_width'].max())
ax.plot([0, max_w], [0, max_w], 'r--', label='Perfect match', linewidth=2)
ax.set_xlabel('Expected Width', fontsize=12)
ax.set_ylabel('Unpacked Width', fontsize=12)
ax.set_title('Cluster Width Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/width_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ width_comparison.png")

# 4. X position residuals
residuals_x = df['unpacked_x'] - df['expected_x']
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(residuals_x, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero residual')
ax.set_xlabel('X Residual (Unpacked - Expected)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'X Position Residuals\nMean: {residuals_x.mean():.2f}, Std: {residuals_x.std():.2f}', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/x_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ x_residuals.png")

# 5. Y position residuals
residuals_y = df['unpacked_y'] - df['expected_y']
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(residuals_y, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero residual')
ax.set_xlabel('Y Residual (Unpacked - Expected)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Y Position Residuals\nMean: {residuals_y.mean():.2f}, Std: {residuals_y.std():.2f}', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/y_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ y_residuals.png")

# 6. Width residuals
residuals_w = df['unpacked_width'] - df['expected_width']
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(residuals_w, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero residual')
ax.set_xlabel('Width Residual (Unpacked - Expected)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Width Residuals\nMean: {residuals_w.mean():.2f}, Std: {residuals_w.std():.2f}', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_plots/width_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ width_residuals.png")

# Print statistics
print("\n========================================")
print("Statistical Summary")
print("========================================")
print(f"X position - Mean residual: {residuals_x.mean():.3f}, RMS: {residuals_x.std():.3f}")
print(f"Y position - Mean residual: {residuals_y.mean():.3f}, RMS: {residuals_y.std():.3f}")
print(f"Width      - Mean residual: {residuals_w.mean():.3f}, RMS: {residuals_w.std():.3f}")
print(f"\nPerfect matches (all residuals = 0): {((residuals_x == 0) & (residuals_y == 0) & (residuals_w == 0)).sum()} / {len(df)}")
print("\nAll plots saved to: validation_plots/")