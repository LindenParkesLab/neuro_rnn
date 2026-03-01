import os, random, time, sys, copy
from timeit import default_timer as timer
from datetime import timedelta
from typing import List, Tuple, Optional, overload, Union, Literal

import torch
from torch import Tensor
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import pickle
import h5py

from src.utils import fix_labels, get_n_gpu

import neurogym as ngym

class RNN(nn.Module):
    """
    Initialize a flexible RNN with customizable weight initialization and training control.
    
    This RNN implementation provides fine-grained control over all weight matrices and biases,
    supporting custom activation functions, spatial regularization, weight masking, and 
    continuous-time dynamics.
    
    Parameters
    ----------
    input_size : int
        Number of expected features in the input x.
        
    hidden_size : int  
        Number of hidden layer nodes. 
        
    num_classes : int
        Number of output classes/features.
        
    type : {'rnn-tanh', 'rnn-sigmoid', 'rnn-relu', 'rnn-retanh', 'rnn-postanh'}, default='rnn-tanh'
        RNN activation function type:
        - 'rnn-tanh': Standard hyperbolic tangent, torch.tanh(x)
        - 'rnn-sigmoid': Standard sigmoid, torch.sigmoid(x)
        - 'rnn-relu': Standard rectified linear unit, torch.relu(x)
        - 'rnn-retanh': Rectified tanh, torch.relu(torch.tanh(x))
        - 'rnn-postanh': Positive tanh, 0.5*(torch.tanh(2x)+1)
        
    regularization_kernel : array-like, optional
        Spatial regularization kernel for hidden-to-hidden weights. Can be:
        - None: No spatial regularization
        - 2D array (hidden_size, hidden_size): Static regularization matrix
        - 3D array (hidden_size, hidden_size, n_epochs): Time-varying regularization
        
    input_weight_mask : array-like, optional
        Binary mask for input-to-hidden weights. Can be:
        - None: No masking applied
        - 1D array (hidden_size,): Applied to all input dimensions
        - 2D array (hidden_size, input_size): Full input weight mask
        
    output_weight_mask : array-like, optional
        Binary mask for hidden-to-output weights. Can be:
        - None: No masking applied  
        - 1D array (hidden_size,): Applied to all output dimensions
        - 2D array (num_classes, hidden_size): Full output weight mask
        
    init_ih_w : None, float, tuple of float, or ndarray, optional
        Input-to-hidden weight initialization:
        - None: Use PyTorch default initialization
        - float: Constant initialization (all weights set to this value)
        - (min, max): Uniform random initialization in [min, max] 
        - ndarray: Explicit weight values (shape: hidden_size, input_size)
        
    train_ih_w : bool, default=True
        Whether input-to-hidden weights are trainable (require gradients).
        
    init_ih_b : None, float, tuple of float, or ndarray, optional  
        Input-to-hidden bias initialization:
        - None: Use PyTorch default initialization
        - float: Constant initialization (all biases set to this value)
        - (min, max): Uniform random initialization in [min, max]
        - ndarray: Explicit bias values (shape: hidden_size,)
        
    train_ih_b : bool, default=True
        Whether input-to-hidden biases are trainable (require gradients).
        
    init_hh_w : None, float, tuple of float, or ndarray, optional
        Hidden-to-hidden weight initialization:
        - None: Use PyTorch default initialization  
        - float: Constant initialization (all weights set to this value)
        - (min, max): Uniform random initialization in [min, max]
        - ndarray: Explicit weight values (shape: hidden_size, hidden_size)
        
    train_hh_w : bool, default=True
        Whether hidden-to-hidden weights are trainable (require gradients).
        
    init_hh_b : None, float, tuple of float, or ndarray, optional
        Hidden-to-hidden bias initialization:
        - None: Use PyTorch default initialization
        - float: Constant initialization (all biases set to this value)
        - (min, max): Uniform random initialization in [min, max] 
        - ndarray: Explicit bias values (shape: hidden_size,)
        
    train_hh_b : bool, default=True
        Whether hidden-to-hidden biases are trainable (require gradients).
        
    init_ho_w : None, float, tuple of float, or ndarray, optional
        Hidden-to-output weight initialization:
        - None: Use PyTorch default initialization
        - float: Constant initialization (all weights set to this value)
        - (min, max): Uniform random initialization in [min, max]
        - ndarray: Explicit weight values (shape: num_classes, hidden_size)
        
    train_ho_w : bool, default=True  
        Whether hidden-to-output weights are trainable (require gradients).
        
    init_ho_b : None, float, tuple of float, or ndarray, optional
        Hidden-to-output bias initialization:
        - None: Use PyTorch default initialization
        - float: Constant initialization (all biases set to this value)
        - (min, max): Uniform random initialization in [min, max]
        - ndarray: Explicit bias values (shape: num_classes,)
        
    train_ho_b : bool, default=True
        Whether hidden-to-output biases are trainable (require gradients).
        
    allow_self_connections : bool, default=True
        Whether to allow self-connections (diagonal elements) in hidden-to-hidden weights.
        If False, diagonal elements of the hh_w matrix are forced to zero during forward pass.
        This constraint is applied regardless of initialization or training status.
        
    alpha : float or 1-D np.ndarray, default=1.0
        Continuous-time parameter in the range [0,1]. Can be:
        - float: single α shared across all nodes
        - 1-D np.ndarray of length hidden_size: per-node α values
        - 1.0: Standard discrete RNN updates
        - (0, 1): Interpolation between new activation and previous state
        - Update equation: h(t+1) = (1-α) * h(t) + α * f(Wh(t) + Ux(t) + b)
        
    rec_noise : float, default = 0.0
        Scale, σ, of the Gaussian noise (centered at zero with unit variance) injected into 
        each node during the forward pass, range [0,1]. Noise is calculated using the 
        following equation: noise = √(2 * α^-1 * σ^2) * N(0,1), where N is a vector of 
        independent Gaussian noise processes (one value per node).
    
    Examples
    --------
    Standard (vanilla) RNN:
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2)
    
    RNN with frozen input weights randomly initialized in (0.1, 0.5):
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2, 
                  init_ih_w=(0.1, 0.5), train_ih_w=False)
    
    RNN with constant hidden weight (0.1) initialization and no self-connections:
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2,
                  init_hh_w=0.1, allow_self_connections=False)
    
    RNN with spatial regularization and custom initialization:
    >>> kernel = np.eye(20) * 0.5  # Diagonal regularization
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2,
                  regularization_kernel=kernel, init_hh_w=(-0.1, 0.1))
    
    Reservoir computer:
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2,
                  train_hh_w=False, train_hh_b=False)
    
    Continuous-time RNN with sigmoid activation:
    >>> rnn = RNN(input_size=10, hidden_size=20, num_classes=2,
                  type='rnn-sigmoid', alpha=0.2)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_classes: int, 
                 type: Literal['rnn-tanh','rnn-sigmoid','rnn-relu','rnn-retanh','rnn-postanh'] = 'rnn-tanh', 
                 regularization_kernel = None, 
                 input_weight_mask = None, 
                 output_weight_mask = None, 
                 init_ih_w: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_ih_w: bool = True, 
                 init_ih_b: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_ih_b: bool = True, 
                 init_hh_w: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_hh_w: bool = True, 
                 init_hh_b: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_hh_b: bool = True, 
                 init_ho_w: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_ho_w: bool = True, 
                 init_ho_b: Union[None, float, tuple[float,float], np.ndarray] = None, 
                 train_ho_b: bool = True, 
                 allow_self_connections: bool = True,
                 alpha: Union[float, np.ndarray] = 1.0,
                 rec_noise: float = 0.05):
        
        super(RNN, self).__init__()
        
        # Store basic parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn_type = type
        self.alpha = alpha
        self.rec_noise = rec_noise
        self.allow_self_connections = allow_self_connections
        
        # Create RNN layer based on type
        if type.replace('rnn-','') in ['tanh','relu','retanh','postanh','sigmoid']:
            self.rnn = CustomRNN(input_size, hidden_size, 1, 
                               nonlinearity=type.replace('rnn-',''), 
                               alpha=alpha, rec_noise=rec_noise)
            n_repeats = 1
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, 1)
            n_repeats = 4
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 1)
            n_repeats = 3
        else:
            raise ValueError(f"RNN type '{type}' not recognized.")
        
        # Create output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Process and store regularization kernel
        self._setup_regularization_kernel(regularization_kernel, n_repeats)
        
        # Process and store weight masks
        self._setup_weight_masks(input_weight_mask, output_weight_mask, n_repeats)
        
        # Initialize all weights and biases
        self._initialize_all_weights(init_ih_w, init_ih_b, init_hh_w, init_hh_b, 
                                   init_ho_w, init_ho_b)
        
        # Set trainable flags for all parameters
        self._set_all_trainable_flags(train_ih_w, train_ih_b, train_hh_w, train_hh_b,
                                    train_ho_w, train_ho_b)

    def _setup_regularization_kernel(self, regularization_kernel, n_repeats):
        """Process and register regularization kernel as buffer."""
        if regularization_kernel is not None:
            regularization_kernel = torch.from_numpy(regularization_kernel).type(torch.float)
            if regularization_kernel.ndim == 2:
                regularization_kernel = regularization_kernel.repeat(n_repeats, 1)
            elif regularization_kernel.ndim == 3:
                regularization_kernel = regularization_kernel.repeat(n_repeats, 1, 1)
            self.register_buffer('regularization_kernel', regularization_kernel)
        else:
            self.regularization_kernel = None

    def _setup_weight_masks(self, input_weight_mask, output_weight_mask, n_repeats):
        """Process and register weight masks as buffers."""
        # Process input weight mask
        if input_weight_mask is not None:
            if input_weight_mask.ndim == 1:
                input_weight_mask = np.repeat(input_weight_mask[:, np.newaxis], 
                                            repeats=self.input_size, axis=1)
            input_weight_mask = torch.from_numpy(input_weight_mask).type(torch.float)
            input_weight_mask = input_weight_mask.repeat(n_repeats, 1)
            self.register_buffer('input_weight_mask', input_weight_mask)
        else:
            self.input_weight_mask = None

        # Process output weight mask
        if output_weight_mask is not None:
            if output_weight_mask.ndim == 1:
                output_weight_mask = np.repeat(output_weight_mask[np.newaxis, :], 
                                             repeats=self.num_classes, axis=0)
            output_weight_mask = torch.from_numpy(output_weight_mask).type(torch.float)
            self.register_buffer('output_weight_mask', output_weight_mask)
        else:
            self.output_weight_mask = None

    def _initialize_all_weights(self, init_ih_w, init_ih_b, init_hh_w, init_hh_b, 
                              init_ho_w, init_ho_b):
        """Initialize all weights and biases according to specifications."""
        
        # Initialize input-to-hidden weights
        if init_ih_w is not None:
            self._apply_initialization(self.rnn.weight_ih_l0, init_ih_w)
        
        # Initialize input-to-hidden bias
        if init_ih_b is not None and hasattr(self.rnn, 'bias_ih_l0') and self.rnn.bias_ih_l0 is not None:
            self._apply_initialization(self.rnn.bias_ih_l0, init_ih_b)
        
        # Initialize hidden-to-hidden weights
        if init_hh_w is not None:
            self._apply_initialization(self.rnn.weight_hh_l0, init_hh_w)
        
        # Initialize hidden-to-hidden bias
        if init_hh_b is not None and hasattr(self.rnn, 'bias_hh_l0') and self.rnn.bias_hh_l0 is not None:
            self._apply_initialization(self.rnn.bias_hh_l0, init_hh_b)
        
        # Initialize hidden-to-output weights
        if init_ho_w is not None:
            self._apply_initialization(self.fc.weight, init_ho_w)
        
        # Initialize hidden-to-output bias
        if init_ho_b is not None and self.fc.bias is not None:
            self._apply_initialization(self.fc.bias, init_ho_b)

    def _apply_initialization(self, param, init_spec):
        """Apply initialization to a parameter based on specification type."""
        
        with torch.no_grad():
            if isinstance(init_spec, (int, float)):
                # Constant initialization
                nn.init.constant_(param, init_spec)
            elif isinstance(init_spec, (tuple, list)) and len(init_spec) == 2:
                # Uniform initialization
                nn.init.uniform_(param, init_spec[0], init_spec[1])
            elif isinstance(init_spec, np.ndarray):
                # Explicit values
                init_tensor = torch.from_numpy(init_spec).type(torch.float)
                if param.shape != init_tensor.shape:
                    raise ValueError(f"Shape mismatch: parameter shape {param.shape} "
                                   f"vs initialization shape {init_tensor.shape}")
                param.copy_(init_tensor)
            else:
                raise ValueError(f"Invalid initialization specification: {init_spec}. "
                               f"Must be None, float, tuple, or ndarray.")

    def _set_all_trainable_flags(self, train_ih_w, train_ih_b, train_hh_w, train_hh_b,
                               train_ho_w, train_ho_b):
        """Set requires_grad for all parameters based on trainable flags."""
        
        # Input-to-hidden weights and bias
        self.rnn.weight_ih_l0.requires_grad_(train_ih_w)
        if hasattr(self.rnn, 'bias_ih_l0') and self.rnn.bias_ih_l0 is not None:
            self.rnn.bias_ih_l0.requires_grad_(train_ih_b)
        
        # Hidden-to-hidden weights and bias
        self.rnn.weight_hh_l0.requires_grad_(train_hh_w)
        if hasattr(self.rnn, 'bias_hh_l0') and self.rnn.bias_hh_l0 is not None:
            self.rnn.bias_hh_l0.requires_grad_(train_hh_b)
        
        # Hidden-to-output weights and bias
        self.fc.weight.requires_grad_(train_ho_w)
        if self.fc.bias is not None:
            self.fc.bias.requires_grad_(train_ho_b)

    def regularization(self, w, type='l1', matrix=None):
        """
        Compute regularization term for weights.
        
        Parameters
        ----------
        w : torch.Tensor
            Weight tensor to regularize
        type : str, default='l1'
            Type of regularization ('l1' or 'l2')
        matrix : torch.Tensor, optional
            Spatial regularization matrix. If None, applies uniform regularization.
            
        Returns
        -------
        torch.Tensor
            Scalar regularization loss
        """
        if type == 'l1' and matrix is None:
            return torch.abs(w).sum()
        elif type == 'l1' and matrix is not None:
            return torch.mul(torch.abs(w), matrix).sum()
        elif type == 'l2' and matrix is None:
            return torch.square(w).sum()
        elif type == 'l2' and matrix is not None:
            return torch.mul(torch.square(w), matrix).sum()
        else:
            raise ValueError(f"Unknown regularization type: {type}")

    def forward(self, x):
        """
        Forward pass with mask and self-connection constraint application.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, input_size) or 
            (batch_size, seq_len, input_size) depending on batch_first
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - action_pred: Output predictions of shape (seq_len, batch_size, num_classes)
            - hidden: Hidden states of shape (seq_len, batch_size, hidden_size)
        """
        
        with torch.no_grad():
            # Apply input weight mask
            if self.input_weight_mask is not None:
                self.rnn.weight_ih_l0.mul_(self.input_weight_mask)
                if hasattr(self.rnn, 'bias_ih_l0') and self.rnn.bias_ih_l0 is not None:
                    # Mask bias for hidden units that have no input connections
                    input_bias_mask = (self.input_weight_mask.sum(dim=1) > 0).float()
                    self.rnn.bias_ih_l0.mul_(input_bias_mask)
            
            # Apply output weight mask
            if self.output_weight_mask is not None:
                self.fc.weight.mul_(self.output_weight_mask)
            
            # Apply self-connection constraint
            if not self.allow_self_connections:
                # Zero out diagonal elements of hidden-to-hidden weights
                diagonal_mask = torch.eye(self.hidden_size, device=self.rnn.weight_hh_l0.device)
                if self.rnn_type in ['lstm', 'gru']:
                    # For LSTM/GRU, handle the stacked weight matrix
                    n_gates = 4 if self.rnn_type == 'lstm' else 3
                    for i in range(n_gates):
                        start_idx = i * self.hidden_size
                        end_idx = (i + 1) * self.hidden_size
                        self.rnn.weight_hh_l0[start_idx:end_idx].mul_(1 - diagonal_mask)
                else:
                    # For custom RNN, direct application
                    self.rnn.weight_hh_l0.mul_(1 - diagonal_mask)
        
        # Forward pass through RNN
        hidden, h_n = self.rnn(x)
        
        # Forward pass through output layer
        action_pred = self.fc(hidden)

        return action_pred, hidden

    def get_weight_info(self):
        """
        Get summary information about current weight values and masks.
        
        Returns
        -------
        dict
            Dictionary containing weight statistics and mask information
        """
        info = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'rnn_type': self.rnn_type,
            'allow_self_connections': self.allow_self_connections,
            'alpha': self.alpha
        }
        
        # Weight statistics
        with torch.no_grad():
            info['ih_weight_stats'] = {
                'mean': self.rnn.weight_ih_l0.mean().item(),
                'std': self.rnn.weight_ih_l0.std().item(),
                'min': self.rnn.weight_ih_l0.min().item(),
                'max': self.rnn.weight_ih_l0.max().item()
            }
            
            info['hh_weight_stats'] = {
                'mean': self.rnn.weight_hh_l0.mean().item(),
                'std': self.rnn.weight_hh_l0.std().item(),
                'min': self.rnn.weight_hh_l0.min().item(),
                'max': self.rnn.weight_hh_l0.max().item()
            }
            
            info['ho_weight_stats'] = {
                'mean': self.fc.weight.mean().item(),
                'std': self.fc.weight.std().item(),
                'min': self.fc.weight.min().item(),
                'max': self.fc.weight.max().item()
            }
        
        # Mask information
        if self.input_weight_mask is not None:
            info['input_mask_stats'] = {
                'total_connections': self.input_weight_mask.numel(),
                'masked_connections': (self.input_weight_mask == 0).sum().item(),
                'unmasked_connections': (self.input_weight_mask == 1).sum().item(),
                'sparsity': (self.input_weight_mask == 0).float().mean().item()
            }
        
        if self.output_weight_mask is not None:
            info['output_mask_stats'] = {
                'total_connections': self.output_weight_mask.numel(),
                'masked_connections': (self.output_weight_mask == 0).sum().item(),
                'unmasked_connections': (self.output_weight_mask == 1).sum().item(),
                'sparsity': (self.output_weight_mask == 0).float().mean().item()
            }
        
        return info

    def freeze_weights(self, ih_w=None, ih_b=None, hh_w=None, hh_b=None, ho_w=None, ho_b=None):
        """
        Freeze or unfreeze specific weight/bias components after initialization.
        
        Parameters
        ----------
        ih_w, ih_b, hh_w, hh_b, ho_w, ho_b : bool, optional
            Whether to freeze (True) or unfreeze (False) each weight/bias component.
            If None (default), the current state is unchanged.
        """
        if ih_w is not None:
            self.rnn.weight_ih_l0.requires_grad_(not ih_w)
        if ih_b is not None and hasattr(self.rnn, 'bias_ih_l0') and self.rnn.bias_ih_l0 is not None:
            self.rnn.bias_ih_l0.requires_grad_(not ih_b)
        if hh_w is not None:
            self.rnn.weight_hh_l0.requires_grad_(not hh_w)
        if hh_b is not None and hasattr(self.rnn, 'bias_hh_l0') and self.rnn.bias_hh_l0 is not None:
            self.rnn.bias_hh_l0.requires_grad_(not hh_b)
        if ho_w is not None:
            self.fc.weight.requires_grad_(not ho_w)
        if ho_b is not None and self.fc.bias is not None:
            self.fc.bias.requires_grad_(not ho_b)


def postanh(x):
    return 0.5 * (torch.tanh(2 * x) + 1)


class CustomRNN(nn.RNNBase):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.,
                 bidirectional: bool = False,
                 nonlinearity: str = 'postanh',
                 alpha: Union[float, np.ndarray] = 1.0,
                 rec_noise: float = 0.0) -> None:
        """
        RNN module with custom activation functions and continuous time dynamics.
        """
        super().__init__(
            mode='RNN_TANH',  # RNNBase requirement, doesn't affect our implementation
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.nonlinearity = nonlinearity
        self.rec_noise = rec_noise
        # alpha can be a scalar float or a per-node 1-D array.
        # For the vector case, register as a buffer so .to(device) moves it automatically.
        if isinstance(alpha, np.ndarray):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = float(alpha)

    def forward(self, 
                input: Union[torch.Tensor, PackedSequence], 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with integrated RNN cell computation.
        """
        # Handle PackedSequence input
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        # Initialize hidden state if not provided
        if hx is None:
            hx = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                           max_batch_size, self.hidden_size,
                           dtype=input.dtype, device=input.device)
        else:
            # Handle hidden state reshaping if needed
            if hx.dim() == 2:
                hx = hx.unsqueeze(0)

        # Check forward arguments
        self.check_forward_args(input, hx, batch_sizes)
        
        # Extract weights and biases from stored parameters
        if self.bias:
            w_ih, w_hh, b_ih, b_hh = self.all_weights[0]
        else:
            w_ih, w_hh = self.all_weights[0]
            b_ih = b_hh = None

        # Convert to seq_len, batch, input_size if batch_first
        if self.batch_first and not is_packed:
            input = input.transpose(0, 1)

        # Get input data (handle PackedSequence)
        if is_packed:
            input_data = input.data
        else:
            input_data = input

        # Extract dimensions
        seq_len = input_data.size(0)
        batch_size = input_data.size(1)
        hidden_size = hx.size(-1)

        # Initialize output storage and hidden state
        output = []
        h_n = hx[0]  # Use first layer (single layer support)
        
        # Pre-compute noise
        if self.training and self.rec_noise > 0:
            if isinstance(self.alpha, torch.Tensor):
                # Per-node scale: shape [hidden_size], broadcasts over [seq, batch, hidden]
                noise_scale = torch.sqrt((2.0 / self.alpha) * self.rec_noise ** 2)
            else:
                noise_scale = float(np.sqrt((2 / self.alpha) * self.rec_noise ** 2))
            noise = torch.randn((seq_len, batch_size, hidden_size), device=input.device) * noise_scale

        # Main sequence processing loop
        for t in range(seq_len):
            x_t = input_data[t]
            
            # Map inputs and previous hidden to current time step
            preactivation = F.linear(x_t, w_ih, b_ih)
            preactivation += F.linear(h_n, w_hh, b_hh) 
            
            # Add recurrent noise during training
            if self.training and self.rec_noise > 0:
                preactivation += noise[t]
            
            # Apply nonlinearity with alpha interpolation
            # h_n = (1-alpha) * h_n + alpha * f(preactivation)
            if self.nonlinearity == "tanh":
                h_n = torch.lerp(h_n, torch.tanh(preactivation), self.alpha)
            elif self.nonlinearity == "relu":
                h_n = torch.lerp(h_n, torch.relu(preactivation), self.alpha)
            elif self.nonlinearity == "postanh":
                h_n = torch.lerp(h_n, postanh(preactivation), self.alpha)
            elif self.nonlinearity == "retanh":
                h_n = torch.lerp(h_n, torch.relu(torch.tanh(preactivation)), self.alpha)
            elif self.nonlinearity == "sigmoid":
                h_n = torch.lerp(h_n, torch.sigmoid(preactivation), self.alpha)
            
            # Ensure proper batch dimensions
            # h_n = h_n.view(batch_size, -1)
            
            # Store hidden state for this time step
            output.append(h_n)
        
        # Stack outputs along sequence dimension
        output = torch.stack(output, dim=0)  # Shape: [seq_len, batch, hidden]
        
        # Convert back to batch_first if needed
        if self.batch_first and not is_packed:
            output = output.transpose(0, 1)  # Shape: [batch, seq_len, hidden]
        
        # Ensure hidden state has correct dimensions for return
        hidden = h_n.unsqueeze(0)  # Add layer dimension
        
        return output, hidden

    
def run_training(dataset, model, optimizer, criterion, config, scheduler=None, return_models=False, epoch_log=10, run=None):
    microtiming = False
    t_overall = timer()
    model.train()
    device = config['device']
    dt = config['time_step']
    try:
        decision = config['env_kwargs']['timing']['decision']
        trim = int((decision - dt) / dt)
    except:
        pass
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']

    # output container
    training_loss = []
    training_loss_task = []
    training_loss_spatial = []
    running_loss = 0.0
    validation_loss = []
    running_loss_val = 0.0
    test_accuracy = []
    if return_models:
        model_state = dict()

    # Train the model
    for epoch in range(n_epochs):
        # get data
        if microtiming:
            t1 = time.time()
        dataset.env.reset(seed=int(n_epochs+epoch))
        inputs, labels = dataset()
        if microtiming:
            data_gen_time = time.time() - t1 
        try:
            labels = fix_labels(labels, decision=int(decision / dt), trim=trim)
        except:
            pass
        # split into train and validation
        if microtiming:
            t2 = time.time()
        inputs_tra = inputs[:, :int(batch_size/2), :]
        inputs_val = inputs[:, int(batch_size/2):, :]
        labels_tra = labels[:, :int(batch_size/2)]
        labels_val = labels[:, int(batch_size/2):]
        if microtiming:
            data_split_time = time.time() - t2 
        # convert to tensor
        if microtiming:
            t3 = time.time()
        inputs_tra = torch.from_numpy(inputs_tra).type(torch.float).to(device)
        inputs_val = torch.from_numpy(inputs_val).type(torch.float).to(device)
        labels_tra = torch.from_numpy(labels_tra.flatten()).type(torch.long).to(device)
        labels_val = torch.from_numpy(labels_val.flatten()).type(torch.long).to(device)
        if microtiming:
            data_transfer_time = time.time() - t3
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # get model outputs for training data
        if microtiming:
            t4 = time.time()
        outputs, _ = model(inputs_tra)
        
        # compute loss for training data
        task_loss = criterion(outputs.view(-1, model.num_classes), labels_tra)
        loss = task_loss.clone()

        # perform regularization
        if model.regularization_kernel is None:
            reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type)
        else:
            if model.regularization_kernel.ndim == 2:
                reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type,
                                                        matrix=model.regularization_kernel)
            elif model.regularization_kernel.ndim == 3:
                reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type,
                                                        matrix=model.regularization_kernel[:, :, epoch])
        # add regularization component
        loss += reg

        # perform backward pass
        loss.backward()
        # perform optimization
        optimizer.step()
        # update scheduler
        if scheduler is not None:
            scheduler.step()
        if microtiming:
            compute_time = time.time() - t4 
        # store loss
        training_loss.append(loss.item())
        training_loss_task.append(task_loss.item())
        training_loss_spatial.append(reg.item())

        if microtiming:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Data gen: {data_gen_time:.3f}s, Data split: {data_split_time:.3f}, Transfer: {data_transfer_time:.3f}s, Compute: {compute_time:.3f}s", flush=True)
        
        # validation
        with torch.no_grad():
            # get model outputs for validation data
            outputs_val, _ = model(inputs_val)
            # compute loss for validation data
            loss_val = criterion(outputs_val.view(-1, model.num_classes), labels_val)
            # store validation loss
            validation_loss.append(loss_val.item())

        # print statistics
        running_loss += loss.item()
        running_loss_val += loss_val.item()
        n_trials = 100
        if epoch == 0:
            epoch_log_timestamp_last = timer()
            model.eval()
            accuracy, _, _, _, _, _ = run_testing(dataset=dataset, model=model, n_trials=n_trials, verbose=False)
            test_accuracy.append(accuracy)
            if return_models:
                model_state[epoch] = copy.deepcopy(model.state_dict())
            model.train()
        elif epoch % epoch_log == int(epoch_log-1):
            epoch_log_timestamp_current = timer()
            epoch_log_time_elapsed = epoch_log_timestamp_current - epoch_log_timestamp_last
            epoch_log_timestamp_last = epoch_log_timestamp_current
            model.eval()
            accuracy, _, _, _, _, _ = run_testing(dataset=dataset, model=model, n_trials=n_trials, verbose=False)
            test_accuracy.append(accuracy)
            if return_models:
                model_state[epoch] = copy.deepcopy(model.state_dict())
            model.train()

            if run == None:
                print('epoch {:06d} | running training loss: {:0.5f} | running validation loss: {:0.5f} | test accuracy: {:06.2f}% | time since last update: {:0.2f}s'
                    .format(epoch, running_loss / epoch_log, running_loss_val / epoch_log, accuracy * 100, epoch_log_time_elapsed), flush=True)
            else:
                print('run {:03d} | epoch {:06d} | running training loss: {:0.5f} | running validation loss: {:0.5f} | test accuracy: {:06.2f}% | time since last update: {:0.2f}s'
                    .format(run, epoch, running_loss / epoch_log, running_loss_val / epoch_log, accuracy * 100, epoch_log_time_elapsed), flush=True)
            running_loss = 0.0
            running_loss_val = 0.0

    t_overall = timer() - t_overall
    print('Finished training in {0}'.format(timedelta(seconds=t_overall)), flush=True)

    training_loss = np.asarray(training_loss)
    training_loss_task = np.asarray(training_loss_task)
    training_loss_spatial = np.asarray(training_loss_spatial)
    validation_loss = np.asarray(validation_loss)
    test_accuracy = np.asarray(test_accuracy)

    if return_models:
        return training_loss, validation_loss, test_accuracy, model_state, training_loss_task, training_loss_spatial
    else:
        return training_loss, validation_loss, test_accuracy, training_loss_task, training_loss_spatial


def infer_test_timing(env_or_timing):
    """Infer timing of environment for testing."""
    if isinstance(env_or_timing, dict):
        timing = {}
        for period in env_or_timing.keys():
            period_times = env_or_timing[period]
            try:
                timing[period] = np.median(period_times)
            except:
                timing[period] = period_times
    else:
        timing = {}
        for period in env_or_timing.timing.keys():
            period_times = [env_or_timing.sample_time(period) for _ in range(100)]
            timing[period] = np.median(period_times)
    return timing


def run_testing(dataset, model, n_trials=1000, verbose=True):
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    elif next(model.parameters()).is_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Environment
    env = dataset.env
    env.timing = infer_test_timing(env)
    env.reset() #no_step=True)

    # compute test performance
    inputs = list()
    labels = list()
    hidden_activity = list()
    output_activity = list()
    info = pd.DataFrame()
    with torch.no_grad():
        for trial in range(n_trials):
            env.reset(seed=int(3*trial))
            env.new_trial()
            ob, gt = env.ob, env.gt
            # print(ob.shape, gt.shape)
            inputs.append(ob)
            labels.append(gt)
            ob = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, hidden = model(ob)

            # detach
            try:
                action_pred = action_pred.detach().cpu().numpy()
                hidden = hidden.detach().cpu().numpy()
            except:
                action_pred = action_pred.detach().numpy()
                hidden = hidden.detach().numpy()

            # Compute performance
            choice = np.argmax(action_pred[-1, 0, :])
            # correct = choice == gt[-1]
            correct = trial_accuracy(action_pred, gt)

            # Log stimulus period activity
            hidden_activity.append(np.array(hidden)[:, 0, :])
            output_activity.append(np.array(action_pred)[:, 0, :])

            # Log trial info
            trial_info = env.trial
            trial_info.update({'correct': correct, 'choice': choice})
            info = pd.concat((info, pd.DataFrame([trial_info])), ignore_index=True)

        accuracy = np.mean(info['correct'])
        if verbose:
            print('Average accuracy across {:} trials: {:.2f}%'.format(n_trials, accuracy*100), flush=True)

        # inputs = np.array(inputs)
        # labels = np.array(labels)
        # hidden_activity = np.array(hidden_activity)
        # output_activity = np.array(output_activity)

        return accuracy, inputs, labels, hidden_activity, output_activity, info


def trial_accuracy(outputs, labels):
    
    # get model choice
    choice = np.argmax(outputs[:,0,:], axis=1)
    
    # define response window and find where model responded
    can_respond = labels > 0
    has_responded = choice > 0
    
    # if no response expected, then any response is incorrect
    if not np.any(can_respond) and np.any(has_responded):
        return 0 
    
    # if response expected but no response, then incorrect
    if np.any(can_respond) and not np.any(has_responded):
        return 0
        
    # if responses outside allowed window, trial is incorrect
    if not np.all(can_respond[has_responded]):
        return 0
    
    # check that response magnitude matches label
    if np.all(choice[has_responded]==labels[has_responded]):
        return 1
    else:
        0


def run_testing_rest(model, n_steps=1000, noise_mean=0.5, noise_sd=0.3, smooth_noise=0, fix_input_channels=(0,), fix_input_value=1.0):
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    elif next(model.parameters()).is_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # create noise input
    n_inputs = model.input_size
    inputs = np.random.normal(noise_mean, noise_sd, (n_steps,n_inputs))
    if fix_input_channels is not None:
        for i in fix_input_channels:
            inputs[:,i] = np.ones((n_steps,)) * fix_input_value
    if smooth_noise > 0:
        inputs = uniform_filter1d(inputs, size=smooth_noise, axis=0)
    ob = torch.from_numpy(inputs[:, np.newaxis, :]).type(torch.float).to(device)
    
    # test this model on noise input (resting state)
    action_pred, hidden = model(ob)
    try:
        action_pred = action_pred.detach().cpu().numpy()
        hidden = hidden.detach().cpu().numpy()
    except:
        action_pred = action_pred.detach().numpy()
        hidden = hidden.detach().numpy()
    hidden_activity = np.array(hidden)[:, 0, :]
    output_activity = np.array(action_pred)[:, 0, :]
    
    return inputs, hidden_activity, output_activity


def train_helper(run, config):
    
    # create dataset
    dataset = ngym.Dataset(config['task_no_modifier'],
                           env_kwargs=config['env_kwargs'],
                           batch_size=config['batch_size'], 
                           seq_len=config['seq_len'])
    
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_classes = dataset.env.action_space.n
    hidden_size = config['hidden_size']
    n_trials = 100
    
    # setup weight masks
    if config['mask_weights']:
        input_weight_mask = config['masks']['input_weight_mask']
        output_weight_mask = config['masks']['output_weight_mask']
    else:
        input_weight_mask = None
        output_weight_mask = None
    
    # seed random seed for reproducibility across runs
    random.seed(int(run))
    np.random.seed(int(run))
    torch.manual_seed(int(run))
    if config['device'].type == 'cuda':
        torch.cuda.manual_seed(int(run))
        torch.cuda.manual_seed_all(int(run))
    if config['device'].type == 'mps':
        torch.mps.manual_seed(int(run))
    
    # initialize the model
    model = RNN(input_size=input_size,
                hidden_size=hidden_size,
                num_classes=n_classes,
                type=config['rnn_model'],
                regularization_kernel=config['regularization_kernel'],
                input_weight_mask=input_weight_mask,
                output_weight_mask=output_weight_mask,
                alpha=config['alpha'],
                rec_noise=config['rec_noise'],
                init_ih_w=config['init_ih_w'],
                train_ih_w=config['train_ih_w'],
                init_ih_b=config['init_ih_b'],
                train_ih_b=config['train_ih_b'],
                init_hh_w=config['init_hh_w'],
                train_hh_w=config['train_hh_w'],
                init_hh_b=config['init_hh_b'],
                train_hh_b=config['train_hh_b'],
                init_ho_w=config['init_ho_w'],
                train_ho_w=config['train_ho_w'],
                init_ho_b=config['init_ho_b'],
                train_ho_b=config['train_ho_b'],
                allow_self_connections=config['allow_self_connections'])
    # if config['n_gpu'] > 1:
    #     model = nn.DataParallel(model)
    model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    
    # train the model
    training_loss, validation_loss, test_accuracy, trained_models, training_loss_task, training_loss_spatial \
        = run_training(dataset=dataset,
                       model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       config=config,
                       scheduler=scheduler,
                       return_models=True,
                       epoch_log=config['epoch_log'],
                       run=run)
        
    # get all outputs for final model
    _, inputs, labels, hidden_activity, output_activity, info \
        = run_testing(dataset=dataset, 
                      model=model, 
                      n_trials=n_trials)
    
    # package all outputs into a dict
    outputs = {
                'training_loss': training_loss,
                'training_loss_task': training_loss_task,
                'training_loss_spatial': training_loss_spatial,
                'validation_loss': validation_loss,
                'test_accuracy': test_accuracy,
                'inputs': inputs,
                'labels': labels,
                'hidden_activity': hidden_activity,
                'output_activity': output_activity,
                'info': info,
                'run': run,
                'device': config['device']
                }
    
    return outputs, trained_models


def train_helper_with_gpu(run, config, gpu_id=None):
    """
    Enhanced train_helper that handles GPU assignment for multi-GPU training
    
    Parameters:
    -----------
    run : int
        Training run number
    config : dict
        Training configuration
    gpu_id : int, optional
        GPU device ID to use for this run. If None, uses config device.
        
    Returns:
    --------
    tuple
        (outputs, trained_models) - same as original train_helper
    """
    if gpu_id is not None:
        # Verify the GPU is accessible
        try:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            # Quick test to make sure GPU works
            test_tensor = torch.ones(10, device=device)
            
            # Create modified config with assigned GPU device
            config = config.copy()
            config['device'] = device
            print(f'Run {run+1} assigned to GPU {gpu_id}')
            
        except Exception as e:
            print(f'Warning: GPU {gpu_id} failed for run {run+1}: {e}')
            print(f'Falling back to original device for run {run+1}')
            # Keep original config device (fallback)
    
    # Call the original train_helper with potentially modified config
    return train_helper(run, config)


class ModelStateManager:
    """
    Description
    -
    A class to handle writing model states to, and reading them from, a file.
    
    It allows the user to access specific model states by specifying the run and/or epoch.
    
    First initialize a new manager for the file you'd like to interact with:
    >>> manager = ModelStateManager('path/to/file.h5')
    
    Then, use one of the methods below to write to or read from that file.
    
    Methods
    -
    >>> manager.get_info()
    >>> manager.save_model_states(data, run, epoch)
    >>> manager.load_model_states(run, epoch)
    >>> manager.load_key_across_runs(epoch, key)
    """
    
    def __init__(self, filename: str):
        self.filename = filename
    
    def get_info(self):
        """
        Returns the number of runs and the indices of epochs for which data exists.

        >>> num_runs, logged_epochs, keys_per_epoch = manager.get_info()
        """
        with h5py.File(self.filename, 'r') as models_file:
            runs = list(models_file.keys())
            num_runs = len(runs)
            logged_epochs = list(int(v.replace('epoch_','')) for v in models_file[runs[0]].keys())
            logged_epochs.sort()
            keys_per_epoch = list(models_file[runs[0]][f'epoch_{logged_epochs[0]}'].keys())
        return num_runs, logged_epochs, keys_per_epoch

    def save_model_states(self, data, run=None, epoch=None):
        """
        Save model states to an HDF5 file.
        
        * To save the state from a specific run and epoch, pass that data as a dictionary: 
        
        >>> manager.save_model_states(rnn.model_state(), run=my_run, epoch=my_epoch)
        
        * To save model state dicts for all runs and epochs after training, pass that data as a list where each item is a dictionary. 
        Each dictionary should have epochs as keys, and each key should have the corresponding model state as a value.
        
        >>> manager.save_model_states(list_of_states)
        """
        
        # Function that saves the model state of a specific run and epoch.
        def save_state_for_run_and_epoch(file_path, state, run, epoch):
            with h5py.File(file_path, 'a') as models_file:
                run_group = models_file.require_group(f'run_{run}')
                epoch_group = run_group.require_group(f'epoch_{epoch}')
                for key, value in state.items():
                    if key in epoch_group:
                        del epoch_group[key]  # Remove existing dataset to avoid conflicts
                    if torch.is_tensor(value):
                        value = value.cpu().numpy()
                    epoch_group.create_dataset(key, data=value)

        # Check that run and epoch are args are compatible.
        if (run == None and (not epoch == None)):
            raise TypeError('If argument ''run'' is None, then argument ''epoch'' must also be None.')
        
        # List input is only valid if run/epoch are not defined.
        if isinstance(data,list) and (not run == None):
            raise TypeError('To save data for all runs/epochs, data must be a list.')
        
        # Save a model state from a specific run and epoch.
        if epoch is not None:
            save_state_for_run_and_epoch(self.filename, data, run, epoch)
        # Save all model states from a specific run.
        elif run is not None:
            for epoch, epoch_data in data.items():
                save_state_for_run_and_epoch(self.filename, epoch_data, run, epoch)
        # Save all model states from all runs and epochs.
        else:
            run = 0
            for run_data in data:
                for epoch, epoch_data in run_data.items():
                    save_state_for_run_and_epoch(self.filename, epoch_data, run, epoch)
                run += 1

    def load_model_states(self, run=None, epoch=None):
        """
        Load model states from an HDF5 file.
        
        * To load the model state dict from a specific run and epoch:
        
        >>> state_dict = manager.load_model_states(run, epoch)
        
        * To load model state dicts from all logged epochs of a specific run:
        
        >>> states_dicts_from_run = manager.load_model_states(run)
        
        * To load model state dicts from all runs and epochs as a list, where each list item is a dict of states from one run: 
        
        >>> state_dicts_all = manager.load_model_states()
        """
        
        # Function that loads the model state of a specific run and epoch.
        def load_state_for_run_and_epoch(file_path, run, epoch):
            with h5py.File(file_path, 'r') as models_file:
                try:
                    epoch_group = models_file[f'run_{run}/epoch_{epoch}']
                    state_dict = {key: torch.tensor(value[()]) for key, value in epoch_group.items()}
                    return state_dict
                except KeyError as e:
                    raise ValueError(f"Specified run/epoch not found: {e}")
        
        # Check that run and epoch are args are compatible.
        if (run == None and (not epoch == None)):
            raise TypeError('If argument ''run'' is None, then argument ''epoch'' must also be None.')
        
        # Load the model state of a specific run and epoch.
        if epoch is not None:
            loaded_states = load_state_for_run_and_epoch(self.filename, run, epoch)
        else:
            n_runs, logged_epochs, _ = self.get_info()
            # Load the model states of all logged epochs from a specific run.
            if run is not None:
                loaded_states = {}
                for epoch in logged_epochs:
                    loaded_states[epoch] = load_state_for_run_and_epoch(self.filename, run, epoch)
            # Load all model states for all runs and epochs.
            else:
                loaded_states = []
                for run in range(n_runs):
                    run_states = {}
                    for epoch in logged_epochs:
                        run_states[epoch] = load_state_for_run_and_epoch(self.filename, run, epoch)
                    loaded_states.append(run_states)
        
        return loaded_states

    def load_key_across_runs(self, epoch, key):
        """
        Load a specific model state key from a specific epoch across all runs.
        Returns a list (one entry per run).
        
        E.g., to load the hidden weights from all runs at epoch 99:
        >>> hidden_weights_epoch99 = manager.load_key_across_runs(epoch=99, key='weight_hh_l0')
        """        
        
        data = []
        with h5py.File(self.filename, 'r') as models_file:
            runs = [run for run in models_file.keys() if run.startswith('run_')]
            for run in runs:
                try:
                    value = models_file[f'{run}/epoch_{epoch}/{key}'][()]
                    data.append(value)
                except KeyError:
                    continue  # Skip if the epoch or key is not found in this run
        if not data:
            raise ValueError(f"No data found for epoch {epoch} and key {key}")
        return data


class ModelDataManager:
    """
    Description
    -
    A class to handle writing model performance data to, and reading them from, a file.
    
    It allows the user to access the data from a specific model by specifying the training run.
    
    First initialize a new manager for the file you'd like to interact with:
    >>> manager = ModelDataManager('path/to/file.h5')
    
    Then, use one of the methods below to write to or read from that file.
    
    Methods
    -
    >>> manager.get_info()
    >>> manager.save_model_data(data, run)
    >>> manager.load_model_data(run)
    >>> manager.load_key_across_runs(key)
    """
    
    def __init__(self, filename: str):
        self.filename = filename
    
    def get_info(self):
        """
        Returns the number of runs and the keys of the data items stored per run.

        >>> num_runs, data_keys = manager.get_info()
        """
        with h5py.File(self.filename, 'r') as models_file:
            runs = list(models_file.keys())
            num_runs = len(runs)
            data_keys = list(models_file[runs[0]].keys())
        return num_runs, data_keys

    def save_model_data(self, data, run=None):
        """
        Save model performance data to an HDF5 file.
        
        * To save the data from a specific run, pass that data as a dictionary: 
        
        >>> manager.save_model_data(run_data_dict, run=my_run)
        
        * To save model data for all runs after training, pass that data as a list where each item is a dictionary. 
        
        >>> manager.save_model_data(list_of_dicts)
        """
        
        # Function that saves the model state of a specific run.
        def save_data_for_run(file_path, data, run):
            with h5py.File(file_path, 'a') as models_file:
                run_group = models_file.require_group(f'run_{run}')
                for key, value in data.items():
                    if key in run_group:
                        del run_group[key]  # Remove existing dataset to avoid conflicts
                    # Pickle the object and store as bytes
                    pickled_data = pickle.dumps(value)
                    run_group.create_dataset(key, data=np.void(pickled_data))
                    
                    # # Store the original type as an attribute
                    # run_group[key].attrs['original_type'] = str(type(value))
        
        # List input is only valid if run is not defined.
        if isinstance(data,list) and (not run == None):
            raise TypeError('To save data for all runs, data must be a list.')
        
        # Save model data from a specific run.
        if run is not None:
            save_data_for_run(self.filename, data, run)
        # Save model data from all run.
        else:
            run = 0
            for run, run_data in enumerate(data):
                save_data_for_run(self.filename, run_data, run)

    def load_model_data(self, run=None):
        """
        Load model performance data from an HDF5 file.
        
        * To load the model data dict from a specific run:
        
        >>> data_dict = manager.load_model_data(run)
        
        * To load model data from all runs as a list where each item is a dict of data from one run: 
        
        >>> data_dicts_all = manager.load_model_data()
        """
        
        # Function that loads the model data of a specific run.
        def load_data_for_run(file_path, run):
            with h5py.File(file_path, 'r') as models_file:
                try:
                    run_group = models_file[f'run_{run}']
                    return {key: pickle.loads(value[()].tobytes()) for key, value in run_group.items()}
                except KeyError as e:
                    raise ValueError(f"Specified run not found: {e}")
        
        n_runs, _ = self.get_info()
        # Load model data for a specific run.
        if run is not None:
            loaded_data = load_data_for_run(self.filename, run)
        # Load model data for all runs.
        else:
            loaded_data = []
            for run in range(n_runs):
                loaded_data.append(load_data_for_run(self.filename, run))
        
        return loaded_data

    def load_key_across_runs(self, key):
        """
        Load a specific model performance key across all runs.
        Returns a list (one entry per run).
        
        E.g., to load the test accuracy from all runs:
        >>> test_accuracy_all = manager.load_key_across_runs('test_accuracy')
        """        
        
        data = []
        with h5py.File(self.filename, 'r') as models_file:
            runs = [run for run in models_file.keys() if run.startswith('run_')]
            for run in runs:
                try:
                    value = pickle.loads(models_file[f'{run}/{key}'][()].tobytes())
                    data.append(value)
                except KeyError:
                    continue  # Skip if the run or key is not found 
        if not data:
            raise ValueError(f"No data found for run {run} and key {key}")
        return data


def create_rnn_and_env_for_model(model_info: pd.Series, run, epoch, data_dir: str, device: torch.device):
    
    if isinstance(model_info, pd.DataFrame):
        model_info = model_info.iloc[0]
    
    # create dataset
    dataset = ngym.Dataset(model_info.task_no_modifier, 
                            env_kwargs = model_info.env_kwargs, 
                            batch_size = model_info.batch_size, 
                            seq_len = model_info.seq_len)
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_classes = dataset.env.action_space.n
    
    # load model state
    file = os.path.join(data_dir, model_info.file_str_models)
    manager = ModelStateManager(file)
    if epoch == -1:
        _, epochs, _ = manager.get_info()
        epoch = epochs[-1]
    state = manager.load_model_states(run, epoch)
    
    # create rnn
    if 'regularization_kernel' in state.keys():
        regularization_kernel = np.zeros((model_info.hidden_size,model_info.hidden_size))
    else:
        regularization_kernel = None
    if 'input_weight_mask' in state.keys():
        input_weight_mask = np.zeros((model_info.hidden_size,input_size))
        output_weight_mask = np.zeros((n_classes,model_info.hidden_size))
    else:
        input_weight_mask = None
        output_weight_mask = None
    if 'alpha' in model_info.keys():
        alpha = model_info.alpha
    else:
        alpha = 0
    rnn = RNN(input_size = input_size, 
              hidden_size = model_info.hidden_size.item(), 
              num_classes = n_classes, 
              type = model_info.rnn_model, 
              regularization_kernel = regularization_kernel,
              input_weight_mask = input_weight_mask, 
              output_weight_mask = output_weight_mask,
              alpha=alpha).to(device)
    rnn.load_state_dict(state)
    rnn.eval()
    
    return dataset, rnn
