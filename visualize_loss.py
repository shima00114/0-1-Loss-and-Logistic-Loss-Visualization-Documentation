```python
# -*- coding: utf-8 -*-
"""
Visualization of 0-1 Loss and Logistic Loss Functions
0-1損失とロジスティック損失の視覚化スクリプト
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Define margin range z = y * f(x)
    z = np.linspace(-3, 3, 600)

    # 0-1 Loss
    loss_01 = np.where(z < 0, 1.0, 0.0)

    # Logistic Loss (Base 2)
    loss_log2 = np.log2(1 + np.exp(-z))

    # Logistic Loss (Natural Log)
    loss_loge = np.log(1 + np.exp(-z))

    # Plot setup
    plt.figure(figsize=(9, 6))

    # Step plot for 0-1 loss
    plt.step(z, loss_01, where='post', label='0-1 Loss $L_{0-1}(z)$', color='black', linewidth=2)
    plt.plot(z, loss_log2, label='Logistic Loss (base 2) $\log_2(1 + e^{-z})$', color='blue', linewidth=2)
    plt.plot(z, loss_loge, label='Logistic Loss (natural log) $\ln(1 + e^{-z})$', color='red', linestyle='--', linewidth=2)

    plt.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    plt.title('Comparison of 0-1 Loss and Logistic Loss', fontsize=14)
    plt.xlabel('Margin $z = y \cdot f(x)$', fontsize=12)
    plt.ylabel('Loss $L(z)$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.ylim(-0.2, 3.5)
    plt.tight_layout()

    # Save output plot
    output_filename = 'loss_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully as '{output_filename}'.")


if __name__ == '__main__':
    main()
