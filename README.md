# 0-1 Loss vs Logistic Loss Comparison 
# 0-1損失とロジスティック損失の比較

This repository provides mathematical definitions, explanations, and visualization scripts for 0-1 loss and logistic loss functions.

本リポジトリでは、二値分類における0-1損失とロジスティック損失の数学的定義、解説、および視覚化スクリプトを提供します。

---

## 1. Mathematical Definitions / 数学的定義

Let $y \in \{-1, +1\}$ be the true label and $f(x) \in \mathbb{R}$ be the model output. 
The margin is defined as $z = y \cdot f(x)$.

真のラベルを $y \in \{-1, +1\}$、モデルの予測出力を $f(x) \in \mathbb{R}$ とします。
マージンは $z = y \cdot f(x)$ と定義されます。

### 0-1 Loss / 0-1損失
$$L_{0-1}(z) = \begin{cases} 1 & \text{if } z < 0 \\ 0 & \text{if } z \ge 0 \end{cases}$$

### Logistic Loss (Base 2) / ロジスティック損失（底2）
$$L_{\text{logistic-2}}(z) = \log_2(1 + e^{-z})$$

### Logistic Loss (Natural Log) / ロジスティック損失（自然対数）
$$L_{\text{logistic-e}}(z) = \ln(1 + e^{-z})$$

---

## 2. Surrogate Loss Function / サロゲート損失としての性質

- **0-1 Loss**: Non-convex and non-differentiable, making direct optimization computationally intractable (NP-hard).
- **Logistic Loss**: Convex and smooth continuous upper bound of 0-1 loss, allowing optimization via gradient descent.

- **0-1損失**: 非凸かつ微分不可能であるため、直接の勾配法による最適化が困難（NP困難）です。
- **ロジスティック損失**: 0-1損失の滑らかな凸代理関数（Surrogate Loss）であり、勾配降下法による効率的な最適化が可能です。

---

## 3. How to Run / 実行方法

```bash
pip install numpy matplotlib
python visualize_loss.py
