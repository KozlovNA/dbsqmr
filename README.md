# Restarted Seed Block SQMR

A high-performance Julia implementation of the **Block Symmetric Quasi-Minimal Residual (Block-SQMR)** method with a dynamic **Seed Deflation** mechanism. 

This solver is specifically designed to efficiently solve large, sparse, complex symmetric linear systems $AX = B$ with hundreds of right-hand sides, which commonly arise in computational electromagnetics (e.g., calculating radar cross sections for multiple incident angles).

## 🛠 Prerequisites & Installation

1. Install [Julia](https://julialang.org/downloads/).
2. Open the Julia REPL and install the required dependencies by typing `]` to enter the Pkg prompt, then run:

```julia
pkg> add LinearAlgebra FileIO JLD2 CSV DataFrames Plots Printf SparseArrays Formatting
```

## 🚀 Quickstart 

1. Place your serialized matrix and right-hand side data file alm.jld2 in the root directory of the project. The file must contain complex symmetric matrix and matrix of right-hand sides.
2. Execute from terminal: 

```bash
julia main.jl
```
