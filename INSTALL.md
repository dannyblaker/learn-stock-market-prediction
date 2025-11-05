# Installation & Setup Guide

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package installer (comes with Python)
- **Git**: (optional, for version control)

## Step-by-Step Installation

### 1. Choose and Set Up Your Python Environment

You can use either a standard Python virtual environment or Conda. Pick whichever you're comfortable with:

**Option A: Python Virtual Environment** (simpler, recommended)
```bash
# Create environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR on Windows:
# venv\Scripts\activate
```

**Option B: Conda Environment** (alternative)
```bash
# Create environment
conda create -n stock_market python=3.10 -y

# Activate it
conda activate stock_market
```

### 2. Verify Python Version

```bash
python --version
```

Should show Python 3.8 or higher.

### 3. Install All Dependencies

**Simple Method** (recommended - installs everything at once):
```bash
pip install -r requirements.txt
```

**Alternative: Install Step-by-Step** (if you want to understand what's being installed):

#### Essential Data Science Libraries
```bash
pip install numpy pandas scipy matplotlib seaborn plotly
```

#### Machine Learning Frameworks
```bash
pip install scikit-learn xgboost lightgbm
```

#### Deep Learning (PyTorch)
```bash
# For CPU only
pip install torch torchvision torchaudio

# For CUDA (if you have NVIDIA GPU) - visit pytorch.org for latest command
# Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Financial Data & Time Series
```bash
pip install yfinance pandas-datareader statsmodels arch pmdarima
```

#### Visualization
```bash
pip install plotly ipywidgets
```

#### Reinforcement Learning
```bash
pip install gym stable-baselines3
```

#### Technical Analysis
```bash
pip install ta
# Note: ta-lib requires additional system dependencies
# For now, we'll use the 'ta' library which is pure Python
```

#### Jupyter & Development Tools
```bash
conda install jupyter notebook -y
pip install pytest black
```

### 4. Install All at Once (Alternative)

Instead of the above, you can install everything from requirements.txt:

```bash
pip install -r requirements.txt
```

**Note**: Some packages (especially ta-lib) may have installation issues. If you encounter problems, skip ta-lib for now - the repository will work without it.

### 5. Verify Installation

Test that key libraries are installed:

```python
python -c "
import numpy, pandas, sklearn, torch, yfinance, statsmodels
print('✓ All core libraries installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 6. Set Up the Package

From the repository root:

```bash
# Development installation (recommended for learning/modifying)
pip install -e .

# Or regular installation
pip install .
```

This makes the `src` module importable from anywhere.

### 7. Download Sample Data

Start Python and run:

```python
from src.utils.data_loader import download_sample_data
download_sample_data()
```

Or from command line:

```bash
python -c "from src.utils.data_loader import download_sample_data; download_sample_data()"
```

This downloads sample data for AAPL, MSFT, GOOGL, etc. to `data/sample/`.

## Verify Everything Works

### Quick Test

```bash
python demo.py
```

This should:
1. Download Apple stock data
2. Add technical indicators
3. Train a Random Forest model
4. Show predictions and evaluation metrics

### Test in Python

```python
# Test data loading
from src.utils.data_loader import load_stock_data
data = load_stock_data('AAPL', start='2023-01-01', end='2023-12-31')
print(f"Loaded {len(data)} rows")

# Test preprocessing
from src.utils.preprocessing import calculate_technical_indicators
data = calculate_technical_indicators(data)
print(f"Added indicators: {data.shape[1]} columns")

# Test model
from src.classical_ml.random_forest import RandomForestPredictor
model = RandomForestPredictor(task='classification')
print("✓ Model initialized")

print("\n✓ All tests passed!")
```

### Test Jupyter Notebooks

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

Run through the cells to ensure everything works.

## Troubleshooting

### Issue: Import Errors

**Solution 1**: Make sure you're in the repository root
```bash
cd /home/d/Documents/stock_market
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Solution 2**: Install package in development mode
```bash
pip install -e .
```

### Issue: yfinance Not Downloading Data

**Symptoms**: Network errors, empty dataframes

**Solutions**:
1. Check internet connection
2. Try different date ranges
3. Use sample data: `load_sample_data('AAPL')`
4. Update yfinance: `pip install --upgrade yfinance`

### Issue: PyTorch CUDA Not Working

**Check CUDA availability**:
```python
import torch
print(torch.cuda.is_available())
```

If False but you have NVIDIA GPU:
1. Check CUDA installation: `nvidia-smi`
2. Install correct PyTorch version for your CUDA
3. For learning, CPU version works fine (just slower)

### Issue: ta-lib Installation Fails

ta-lib has C dependencies that can be tricky:

**Linux**:
```bash
# Install system dependencies first
sudo apt-get install ta-lib

# Then install Python wrapper
pip install ta-lib
```

**macOS**:
```bash
brew install ta-lib
pip install ta-lib
```

**Windows**:
Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

**Workaround**: The repository uses the `ta` library (pure Python) as alternative, which should work fine for learning.

### Issue: Out of Memory (GPU)

When training deep learning models:

**Solutions**:
1. Reduce batch size: `batch_size=16` instead of `32`
2. Reduce sequence length: `sequence_length=30` instead of `60`
3. Use fewer LSTM layers: `num_layers=1` instead of `2`
4. Use CPU instead: `device='cpu'`

### Issue: Matplotlib Not Displaying Plots

In Jupyter notebooks:
```python
%matplotlib inline
```

In Python scripts, add:
```python
plt.show()
```

### Issue: Slow Training

**Tips**:
1. Use fewer data samples initially
2. Reduce model complexity
3. Use GPU if available
4. Reduce number of estimators/epochs

## Environment Management

### Save Your Environment

Once everything works:

```bash
# Conda environment
conda env export > environment.yml

# pip requirements
pip freeze > requirements_frozen.txt
```

### Recreate Environment Later

```bash
# From conda environment file
conda env create -f environment.yml

# From pip requirements
pip install -r requirements_frozen.txt
```

### Update Packages

```bash
# Update specific package
pip install --upgrade yfinance

# Update all packages (careful!)
pip install --upgrade -r requirements.txt
```

## Performance Tips

### For Faster Development

1. **Use smaller datasets**: Reduce date range during development
2. **Cache processed data**: Save preprocessed features
3. **Use sample data**: Faster than downloading repeatedly
4. **Reduce model complexity**: Start simple, add complexity gradually

### For Better Results

1. **More data**: Longer time periods (5-10 years)
2. **More features**: Add custom technical indicators
3. **Hyperparameter tuning**: Use GridSearchCV or Optuna
4. **Ensemble methods**: Combine multiple models
5. **Regular retraining**: Markets change over time

## Next Steps After Setup

1. ✅ Run `python demo.py`
2. ✅ Open `notebooks/01_getting_started.ipynb`
3. ✅ Read through `README.md` and `GLOSSARY.md`
4. ✅ Try different stocks and time periods
5. ✅ Experiment with features and models
6. ✅ Build your own models!

## Getting Help

- **Documentation**: Check docstrings in code
- **Examples**: See `if __name__ == "__main__":` blocks
- **Issues**: Open GitHub issue
- **Community**: Join discussions

## Final Checklist

- [ ] Conda environment activated
- [ ] Dependencies installed
- [ ] Package installed (`pip install -e .`)
- [ ] Sample data downloaded
- [ ] `demo.py` runs successfully
- [ ] Jupyter notebooks open and run
- [ ] No import errors

If all checked, you're ready to go! 🚀

Happy learning!
