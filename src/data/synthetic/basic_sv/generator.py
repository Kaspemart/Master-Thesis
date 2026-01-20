"""
--------------------------------------------------------------------------------------------------------------------------------------
- Synthetic Data Generator for Basic Stochastic Volatility Model
--------------------------------------------------------------------------------------------------------------------------------------
PURPOSE:
This module generates synthetic training data for neural network-based parameter estimation.
It creates thousands of return sequences with known parameters, which are then used to train
neural networks to learn the inverse mapping from returns → parameters.
--------------------------------------------------------------------------------------------------------------------------------------
WORKFLOW:
1. Sample random parameters (φ, σ_v) from specified distributions
2. For each parameter pair, simulate a return sequence using BasicSVModel
3. Store returns (observed) and parameters (ground truth) together
4. Split into train/validation/test sets
5. Save to disk for efficient loading during training
--------------------------------------------------------------------------------------------------------------------------------------
KEY DESIGN CHOICES:
- Parameter ranges based on empirical literature on financial data
- Sequence length T=252 (one trading year) as default
- Large datasets (50k+ sequences) needed for neural network training
- Storage in .npz format for efficiency and NumPy compatibility
--------------------------------------------------------------------------------------------------------------------------------------
"""

# IMPORTS:
import numpy as np
from pathlib import Path
from src.models.sv_basic import BasicSVModel
import logging
# ---------------------------------------------------------------------------------------------------------------------------

# Setting up module logger
logger = logging.getLogger(__name__)


class BasicSVDataGenerator:
    """
    Synthetic data generator for training neural networks on basic SV model.
    
    This class generates large datasets of simulated return sequences with known
    parameters. The data is used to train neural networks to perform amortized
    inference: learning to estimate parameters directly from observed returns.
    
    The generator:
    1. Samples random parameters from specified distributions
    2. Simulates return sequences using BasicSVModel
    3. Stores both returns (input) and parameters (output labels)
    4. Supports train/validation/test splits
    5. Saves data efficiently to disk
    
    This approach allows objective evaluation since true parameters are known.
    
    Parameters
    ----------
    phi_range : tuple[float, float], default=(0.85, 0.98)
        Range for sampling persistence parameter φ.
        Literature suggests φ ∈ [0.85, 0.99] for daily financial data.
        Higher values indicate stronger volatility clustering.
        
    sigma_v_range : tuple[float, float], default=(0.10, 0.40)
        Range for sampling volatility-of-volatility parameter σ_v.
        Typical values are between 0.1 and 0.5.
        Higher values produce more jumpy/unstable volatility.
        
    T : int, default=252
        Length of each return sequence (time steps).
        Default is 252 (one trading year of daily data).
        Longer sequences provide more information but increase computation.
        
    h0 : float, default=0.0
        Initial log-volatility for all simulations.
        Zero corresponds to unit volatility (exp(0/2) = 1).
        
    seed : int | None, default=None
        Random seed for reproducibility.
        If None, results will vary each run.
        Set to fixed value (e.g., 42) for reproducible experiments.
    
    Attributes
    ----------
    phi_range : tuple[float, float]
        Parameter sampling range for φ
    sigma_v_range : tuple[float, float]
        Parameter sampling range for σ_v
    T : int
        Sequence length
    h0 : float
        Initial log-volatility
    seed : int | None
        Random seed
    rng : np.random.Generator
        NumPy random number generator
    
    Examples
    --------
    >>> # Create generator with default settings
    >>> generator = BasicSVDataGenerator(seed=42)
    >>> 
    >>> # Generate small dataset for testing
    >>> data = generator.generate(n_sequences=1000)
    >>> data['returns'].shape
    (1000, 252)
    >>> data['parameters'].shape
    (1000, 2)
    >>> 
    >>> # Generate full training dataset with splits
    >>> train, val, test = generator.generate_splits(
    ...     n_train=50000,
    ...     n_val=10000,
    ...     n_test=10000
    ... )
    >>> 
    >>> # Save to disk
    >>> generator.save_dataset(train, 'data/train.npz')
    >>> 
    >>> # Load later
    >>> loaded = generator.load_dataset('data/train.npz')
    
    Notes
    -----
    The generated data has the following structure:
    - returns: (N, T) array of observed returns (input to neural network)
    - parameters: (N, 2) array of [φ, σ_v] (target outputs for neural network)
    - log_volatility: (N, T) array of latent states (for analysis, not used in training)
    
    Parameter ranges are chosen based on:
    - Kim, Shephard & Chib (1998) - typical values in econometrics literature
    - Empirical estimates from real financial data
    - Ensuring stationarity (φ < 1) and positivity (σ_v > 0)
    """
    
    def __init__(
        self,
        phi_range: tuple[float, float] = (0.85, 0.98),
        sigma_v_range: tuple[float, float] = (0.10, 0.40),
        T: int = 252,
        h0: float = 0.0,
        seed: int | None = None
    ) -> None:
        """Initialize the data generator."""
        self.phi_range = phi_range
        self.sigma_v_range = sigma_v_range
        self.T = T
        self.h0 = h0
        self.seed = seed
        
        # Creating random number generator
        self.rng = np.random.default_rng(seed)
        
        # Validating parameter ranges
        if not (0 < phi_range[0] < phi_range[1] < 1):
            raise ValueError(f"phi_range must be in (0, 1), got {phi_range}")
        if not (0 < sigma_v_range[0] < sigma_v_range[1]):
            raise ValueError(f"sigma_v_range must be positive, got {sigma_v_range}")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        
        logger.info(f"Initialized BasicSVDataGenerator: phi∈{phi_range}, σ_v∈{sigma_v_range}, T={T}")
    

    def sample_parameters(self, n: int) -> np.ndarray:
        """
        Sample random parameters from uniform distributions.
        
        This method generates parameter pairs (φ, σ_v) by sampling uniformly
        from the specified ranges. Uniform sampling ensures good coverage of
        the parameter space during neural network training.
        
        Parameters
        ----------
        n : int
            Number of parameter pairs to sample.
        
        Returns
        -------
        parameters : np.ndarray
            Array of shape (n, 2) where each row is [φ, σ_v].
        
        Examples
        --------
        >>> generator = BasicSVDataGenerator(seed=42)
        >>> params = generator.sample_parameters(5)
        >>> params.shape
        (5, 2)
        >>> params[:, 0]  # phi values
        array([0.87, 0.92, 0.89, 0.95, 0.88])
        """
        phi = self.rng.uniform(self.phi_range[0], self.phi_range[1], size=n)
        sigma_v = self.rng.uniform(self.sigma_v_range[0], self.sigma_v_range[1], size=n)
        
        # Stacking into (n, 2) array: [phi, sigma_v]
        parameters = np.column_stack([phi, sigma_v])
        
        logger.debug(f"Sampled {n} parameter pairs")
        return parameters
    

    def generate(self, n_sequences: int, show_progress: bool = True) -> dict[str, np.ndarray]:
        """
        Generate synthetic dataset of return sequences with known parameters.
        
        This is the main data generation method. It:
        1. Samples n_sequences random parameter pairs
        2. For each parameter pair, simulates a return sequence
        3. Stores returns, parameters, and (optionally) latent volatility
        
        The generated data is ready to use for training neural networks.
        
        Parameters
        ----------
        n_sequences : int
            Number of sequences to generate.
            Typical values: 50,000 for training, 10,000 for validation/test.
            
        show_progress : bool, default=True
            Whether to print progress updates during generation.
            Useful for large datasets that take time to generate.
        
        Returns
        -------
        dataset : dict[str, np.ndarray]
            Dictionary containing:
            - 'returns': (n_sequences, T) - observed return sequences
            - 'parameters': (n_sequences, 2) - true [φ, σ_v] for each sequence
            - 'log_volatility': (n_sequences, T) - latent log-volatility (for analysis)
        
        Examples
        --------
        >>> generator = BasicSVDataGenerator(T=100, seed=42)
        >>> data = generator.generate(n_sequences=1000)
        >>> 
        >>> # Check shapes
        >>> data['returns'].shape
        (1000, 100)
        >>> data['parameters'].shape
        (1000, 2)
        >>> 
        >>> # Inspect first sequence
        >>> print(f"First sequence parameters: phi={data['parameters'][0, 0]:.3f}, sigma_v={data['parameters'][0, 1]:.3f}")
        >>> print(f"Return statistics: mean={data['returns'][0].mean():.4f}, std={data['returns'][0].std():.4f}")
        
        Notes
        -----
        - Generation time is approximately linear in n_sequences
        - Each sequence is generated independently (can be parallelized in future)
        - Memory usage: roughly 8 bytes * n_sequences * T * 3 (returns + params + volatility)
        - For 50k sequences of length 252: ~300 MB memory
        """
        logger.info(f"Generating {n_sequences} sequences of length {self.T}...")
        
        # Sampling all parameters at once
        parameters = self.sample_parameters(n_sequences)
        
        # Initializing storage arrays
        returns = np.zeros((n_sequences, self.T))
        log_volatility = np.zeros((n_sequences, self.T))
        
        # Generating sequences
        for i in range(n_sequences):
            phi, sigma_v = parameters[i]
            
            # Creating model with these parameters
            model = BasicSVModel(phi=phi, sigma_v=sigma_v)
            
            # Simulating sequence (no fixed seed - we want different sequences)
            r, h = model.simulate(T=self.T, h0=self.h0, seed=None)
            
            returns[i] = r
            log_volatility[i] = h
            
            # Showing progress
            if show_progress and (i + 1) % max(1, n_sequences // 10) == 0:
                progress = (i + 1) / n_sequences * 100
                logger.info(f"  Progress: {i+1}/{n_sequences} ({progress:.1f}%)")
        
        logger.info(f"Successfully generated {n_sequences} sequences")
        
        return {
            'returns': returns,
            'parameters': parameters,
            'log_volatility': log_volatility
        }
    

    def generate_splits(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        show_progress: bool = True
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Generate train, validation, and test datasets with different random seeds.
        
        This method ensures that train/val/test sets are generated with independent
        random states, preventing any data leakage between sets.
        
        Parameters
        ----------
        n_train : int
            Number of training sequences.
            Typical: 50,000 or more.
            
        n_val : int
            Number of validation sequences.
            Typical: 10,000 (about 20% of training size).
            
        n_test : int
            Number of test sequences.
            Typical: 10,000 (same as validation).
            
        show_progress : bool, default=True
            Whether to show progress during generation.
        
        Returns
        -------
        train_data : dict[str, np.ndarray]
            Training dataset
        val_data : dict[str, np.ndarray]
            Validation dataset
        test_data : dict[str, np.ndarray]
            Test dataset
        
        Examples
        --------
        >>> generator = BasicSVDataGenerator(seed=42)
        >>> train, val, test = generator.generate_splits(
        ...     n_train=50000,
        ...     n_val=10000,
        ...     n_test=10000
        ... )
        >>> 
        >>> # Verify no overlap (parameters should be different)
        >>> assert not np.array_equal(train['parameters'], val['parameters'])
        >>> assert not np.array_equal(val['parameters'], test['parameters'])
        """
        logger.info(f"Generating splits: train={n_train}, val={n_val}, test={n_test}")
        
        # Generating training data
        logger.info("Generating training set...")
        train_data = self.generate(n_train, show_progress=show_progress)
        
        # Generating validation data
        logger.info("Generating validation set...")
        val_data = self.generate(n_val, show_progress=show_progress)
        
        # Generating test data
        logger.info("Generating test set...")
        test_data = self.generate(n_test, show_progress=show_progress)
        
        logger.info("All splits generated successfully")
        return train_data, val_data, test_data
    

    def save_dataset(self, data: dict[str, np.ndarray], filepath: str | Path) -> None:
        """
        Save dataset to disk in compressed .npz format.
        
        This method saves the dataset efficiently using NumPy's compressed format.
        The file can be loaded later using load_dataset().
        
        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dataset dictionary (output from generate() or generate_splits()).
            
        filepath : str or Path
            Path where to save the file.
            Should have .npz extension.
        
        Examples
        --------
        >>> generator = BasicSVDataGenerator(seed=42)
        >>> data = generator.generate(1000)
        >>> generator.save_dataset(data, 'data/train.npz')
        >>> 
        >>> # Check file size
        >>> import os
        >>> size_mb = os.path.getsize('data/train.npz') / 1024 / 1024
        >>> print(f"File size: {size_mb:.2f} MB")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset to {filepath}...")
        np.savez_compressed(filepath, **data)  # type: ignore[arg-type]
        
        # Calculating file size
        size_mb = filepath.stat().st_size / 1024 / 1024
        logger.info(f"Saved {data['returns'].shape[0]} sequences ({size_mb:.2f} MB)")
    

    @staticmethod
    def load_dataset(filepath: str | Path) -> dict[str, np.ndarray]:
        """
        Load dataset from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to .npz file created by save_dataset().
        
        Returns
        -------
        data : dict[str, np.ndarray]
            Loaded dataset with 'returns', 'parameters', and 'log_volatility'.
        
        Examples
        --------
        >>> data = BasicSVDataGenerator.load_dataset('data/train.npz')
        >>> print(f"Loaded {data['returns'].shape[0]} sequences")
        """
        filepath = Path(filepath)
        logger.info(f"Loading dataset from {filepath}...")
        
        with np.load(filepath) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        
        logger.info(f"Loaded {data['returns'].shape[0]} sequences")
        return data
    

    def get_statistics(self, data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        """
        Compute summary statistics of generated dataset.
        
        Useful for verifying data quality and understanding parameter distributions.
        
        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dataset dictionary.
        
        Returns
        -------
        stats : dict[str, dict[str, float]]
            Nested dictionary with statistics for returns and parameters.
        
        Examples
        --------
        >>> generator = BasicSVDataGenerator(seed=42)
        >>> data = generator.generate(1000)
        >>> stats = generator.get_statistics(data)
        >>> print(f"Mean phi: {stats['parameters']['phi_mean']:.3f}")
        >>> print(f"Mean return: {stats['returns']['mean']:.4f}")
        """
        returns = data['returns']
        params = data['parameters']
        
        stats = {
            'returns': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
            },
            'parameters': {
                'phi_mean': float(np.mean(params[:, 0])),
                'phi_std': float(np.std(params[:, 0])),
                'phi_min': float(np.min(params[:, 0])),
                'phi_max': float(np.max(params[:, 0])),
                'sigma_v_mean': float(np.mean(params[:, 1])),
                'sigma_v_std': float(np.std(params[:, 1])),
                'sigma_v_min': float(np.min(params[:, 1])),
                'sigma_v_max': float(np.max(params[:, 1])),
            }
        }
        
        return stats
    
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"BasicSVDataGenerator(phi∈{self.phi_range}, σ_v∈{self.sigma_v_range}, "
            f"T={self.T}, seed={self.seed})"
        )

