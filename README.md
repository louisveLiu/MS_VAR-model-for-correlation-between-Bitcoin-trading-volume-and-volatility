# MS-VAR and VAR Models for Bitcoin Trading Volume and Volatility

## Overview
This repository explores the **relationship between Bitcoin’s trading volume and volatility** using both **Markov Switching Vector Autoregression (MS-VAR)** and **single Vector Autoregression (VAR)** models. The original MS-VAR approach draws on methodology from a study conducted in the Chinese stock market context. You can find that reference paper here: [A Flexible Framework for Non-Linear Modeling](https://www.sciencedirect.com/science/article/pii/S2666764923000413).

We extended their approach to see if regime-switching dynamics or a simpler VAR method could capture **Bitcoin’s dynamic behavior** more effectively, given the crypto market’s unique volatility profile.

---

## Key Files

### 1. MS-VAR Approach
- **`lag_p.py`**  
  Uses the **BIC model** to determine the best lag order for the MS-VAR model.  

- **`ms_var.py`**  
  Implements the **EM algorithm** to estimate the final transition matrix, correlation matrix, and covariance matrix for the MS-VAR model.  

- **`spe_relation.py`** and **`relationship.py`**  
  Perform **posterior analysis** on the MS-VAR model outputs, helping interpret regime shifts and correlation patterns.

### 2. Single VAR Approach
- **`var_order.py`**  
  Determines the **optimal lag order** for the single VAR model.  

- **`bullish_initial.py`**  
  Builds the **coefficient matrix** for the VAR model under a **strong bull market** segment of historical Bitcoin data.  

- **`bullish_bestmodel.py`**  
  Evaluates whether the single VAR model can capture the **trading volume–volatility correlation** during a strong bull market phase.

---

## Approach and Findings

### Why MS-VAR?
MS-VAR can **switch** between different regimes (e.g., high-volatility vs. low-volatility states), potentially adapting better than a single linear model to the **highly fluctuating** nature of crypto markets.

### Why Also Test Single VAR?
After testing MS-VAR, we wanted to see if a simpler **linear VAR** model could capture significant correlations in a more homogeneous market segment (e.g., a pronounced bull market).

### Limitations Discovered
- **MS-VAR**: While regime-switching is promising, its success in the Chinese stock market context does not necessarily translate seamlessly to Bitcoin, which experiences **sharper, more frequent** volatility swings.
- **Single VAR**: Focusing on a **strong bull market** subset of data showed that a linear VAR model still **failed to capture** the trading volume–volatility relationship effectively.

These findings suggest that **both** MS-VAR and standard VAR approaches have **limitations** when modeling Bitcoin’s volatility and trading volume across different market conditions.

---

## Future Directions
Given the challenges with both MS-VAR and single VAR models, we are exploring:

- **Deep Learning Methods**:
  - **LSTM** for long-range dependency modeling.
  - **Attention-based** architectures for dynamic weighting of different time steps.

- **Hybrid Approaches**:  
  Combining the **regime-switching** concept with **deep learning** features to potentially capture complex crypto market structures.

- **Expanded Feature Set**:  
  Incorporating **on-chain metrics**, macroeconomic indicators, and other external data to improve the robustness of the correlation analysis.

---

## How to Use

### 1. Data Collection
Compile Bitcoin historical data, including daily or intraday **prices, trading volumes, and volatility** metrics.

### 2. MS-VAR Analysis
1. Run **`lag_p.py`** to find the best MS-VAR lag order.  
2. Use **`ms_var.py`** to estimate model parameters via the **EM algorithm**.  
3. Inspect results with **`spe_relation.py`** or **`relationship.py`** to analyze posterior distributions.

### 3. Single VAR Analysis
1. Use **`var_order.py`** to determine the VAR model’s **optimal lag order**.  
2. If analyzing a **bull market** subset, adjust **`bullish_initial.py`** to build the model’s coefficient matrix.  
3. Evaluate the model’s performance with **`bullish_bestmodel.py`** to see if any correlation is captured.

### 4. Compare & Conclude
- Compare **MS-VAR** vs. **single VAR** outcomes.  
- Note where each model fails or succeeds in capturing volume–volatility relationships.

### 5. Explore Alternatives
Experiment with **deep learning models** to address the limitations found in MS-VAR and single VAR approaches.

---

## Contributing
We welcome contributions and discussions on alternative approaches or improvements to existing ones.

- **Issues & Feedback**: Submit a GitHub issue for suggestions or bug reports.  
- **Pull Requests**: Feel free to create PRs for new features or optimized approaches.

---

## License
This project is licensed under the **[MIT License]**. Please see the `LICENSE` file for more details.

---

### Disclaimer
The analysis and models provided in this repository should **not** be taken as financial advice. They are purely for **research and exploration** into the dynamic relationships between crypto trading volume and volatility.
