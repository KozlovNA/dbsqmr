using LinearAlgebra
using FileIO
using JLD2
using CSV
using DataFrames
using Plots
using Printf
using SparseArrays
using Formatting
using Profile

function norms2(A)
    return [norm(c) for c in eachcol(A)]
end

function get_thin_Q(QR, m)
    T = eltype(QR.R)
    I_thin = Matrix{T}(I, size(QR.Q, 1), m)
    return QR.Q * I_thin
end

function append_history!(hist_k, hist_max_res, hist_block_size, hist_norms, hist_states, k, block_size, curr_norms, states)
    push!(hist_k, k)
    push!(hist_max_res, maximum(curr_norms))
    push!(hist_block_size, block_size)
    push!(hist_norms, copy(curr_norms))
    push!(hist_states, copy(states))
end

function write_history_to_csv(filename, hist_k, hist_max_res, hist_block_size, hist_norms, hist_states)
    m_total = length(hist_norms[1])

    df = DataFrame(
        k=hist_k,
        real_residual=hist_max_res,
        quasi_residual=hist_max_res,
        block_size=hist_block_size
    )

    for i in 1:m_total
        df[!, "res_$i"] = [n[i] for n in hist_norms]
        df[!, "state_$i"] = [s[i] for s in hist_states]
    end

    CSV.write(filename, df)
end

# ==============================================================================
# 1. ORIGINAL ALGORITHM (Baseline reference)
# ==============================================================================
function bsqmr_original(A, B, tol, filename)
    mkpath(dirname(filename))
    m = size(B, 2)

    T = eltype(B)
    T_real = real(T)
    tol_T = T_real(tol)

    R = copy(B)
    X = zero(B)
    R0_norms = norms2(R)

    states = ones(Int, m)

    hist_k = Int[]
    hist_max_res = Float64[]
    hist_block_size = Int[]
    hist_norms = Vector{Float64}[]
    hist_states = Vector{Int}[]

    append_history!(hist_k, hist_max_res, hist_block_size, hist_norms, hist_states, 0, m, ones(T_real, m), states)

    # Init Lanczos
    Vₖ = copy(R)
    QR_init = qr!(Vₖ)
    Vₖ = get_thin_Q(QR_init, m)
    βₖ = QR_init.R
    γₖ = transpose(Vₖ) * Vₖ
    τ_tilde = copy(βₖ)

    # Init History
    Vₖ₋₁ = zero(Vₖ)
    Pₖ = zero(Vₖ)
    Pₖ₋₁ = zero(Vₖ)
    Pₖ₋₂ = zero(Vₖ)
    APₖ = zero(Vₖ)
    APₖ₋₁ = zero(Vₖ)
    APₖ₋₂ = zero(Vₖ)

    b_km2 = zeros(T, m, m)
    b_km1 = zeros(T, m, m)
    c_km1 = zeros(T, m, m)
    a_km1 = Matrix{T}(I, m, m)
    d_km2 = Matrix{T}(I, m, m)
    d_km1 = Matrix{T}(I, m, m)

    δ_km1 = zeros(T, m, m)
    γₖ₋₁ = Matrix{T}(I, m, m)

    k = 1
    while true
        AVₖ = A * Vₖ

        δ_km1 = γₖ₋₁ \ (transpose(βₖ) * γₖ)
        V_tilde = AVₖ - Vₖ₋₁ * δ_km1
        αₖ = γₖ \ (transpose(Vₖ) * V_tilde)
        V_tilde = V_tilde - Vₖ * αₖ

        QR1 = qr!(V_tilde)
        Vₖ₊₁ = get_thin_Q(QR1, m)
        βₖ₊₁ = QR1.R
        QR2 = qr!(Vₖ₊₁)
        Vₖ₊₁ = get_thin_Q(QR2, m)
        β_tmp = QR2.R
        βₖ₊₁ = β_tmp * βₖ₊₁

        α_tilde = γₖ \ (transpose(Vₖ) * Vₖ₊₁)
        αₖ = αₖ + α_tilde * βₖ₊₁
        Vₖ₊₁ = Vₖ₊₁ - Vₖ * α_tilde

        δ_tilde = γₖ₋₁ \ (transpose(Vₖ₋₁) * Vₖ₊₁)
        δ_km1 = δ_km1 + δ_tilde * βₖ₊₁
        Vₖ₊₁ = Vₖ₊₁ - Vₖ₋₁ * δ_tilde

        QR3 = qr!(Vₖ₊₁)
        Vₖ₊₁ = get_thin_Q(QR3, m)
        β_tmp = QR3.R
        βₖ₊₁ = β_tmp * βₖ₊₁

        γₖ₊₁ = transpose(Vₖ₊₁) * Vₖ₊₁

        θₖ = b_km2 * δ_km1
        ηₖ = a_km1 * d_km2 * δ_km1 + b_km1 * αₖ
        ζ_tilde = c_km1 * d_km2 * δ_km1 + d_km1 * αₖ

        mat_to_qr = [ζ_tilde; βₖ₊₁]
        QR_qmr = qr(mat_to_qr)
        Qₖ_full = Matrix{T}(I, 2m, 2m)
        lmul!(QR_qmr.Q, Qₖ_full)
        Qₖ_star = Qₖ_full'

        aₖ = @views Qₖ_star[1:m, 1:m]
        bₖ = @views Qₖ_star[1:m, m+1:2m]
        cₖ = @views Qₖ_star[m+1:2m, 1:m]
        dₖ = @views Qₖ_star[m+1:2m, m+1:2m]

        ζₖ = aₖ * ζ_tilde + bₖ * βₖ₊₁

        Pₖ = (Vₖ - Pₖ₋₁ * ηₖ - Pₖ₋₂ * θₖ) / ζₖ
        APₖ = (AVₖ - APₖ₋₁ * ηₖ - APₖ₋₂ * θₖ) / ζₖ

        τₖ = aₖ * τ_tilde
        τ_tilde = cₖ * τ_tilde

        X .+= Pₖ * τₖ
        R .-= APₖ * τₖ

        curr_norms = norms2(R) ./ R0_norms
        max_res = maximum(curr_norms)
        printfmt("k = {:5d} | max active res = {:7f}\n", k, max_res)

        if all(curr_norms .<= tol_T)
            states .= 2
        end

        append_history!(hist_k, hist_max_res, hist_block_size, hist_norms, hist_states, k, m, curr_norms, states)

        if all(curr_norms .<= tol_T)
            break
        end

        APₖ₋₂ = APₖ₋₁
        APₖ₋₁ = APₖ
        Pₖ₋₂ = Pₖ₋₁
        Pₖ₋₁ = Pₖ
        Vₖ₋₁ = Vₖ
        Vₖ = Vₖ₊₁
        βₖ = βₖ₊₁
        γₖ₋₁ = γₖ
        γₖ = γₖ₊₁
        b_km2 = b_km1
        b_km1 = bₖ
        c_km1 = cₖ
        a_km1 = aₖ
        d_km2 = d_km1
        d_km1 = dₖ

        k += 1
    end
    write_history_to_csv(filename, hist_k, hist_max_res, hist_block_size, hist_norms, hist_states)

    return X
end

# ==============================================================================
# 2. SEED ALGORITHM
# ==============================================================================
function bsqmr_seed_restarted(A, B, tol, filename; max_active=45, threshold_tau=0.0001)
    mkpath(dirname(filename))
    m_total = size(B, 2)

    T = eltype(B)
    T_real = real(T)
    tol_T = T_real(tol)
    tau_T = T_real(threshold_tau)

    X_full = zero(B)
    R_full = copy(B)
    global_R0_norms = norms2(B)

    global_norms = ones(T_real, m_total)
    global_states = zeros(Int, m_total)
    unconverged_idx = collect(1:m_total)

    global_k = 1
    is_first_save = true
    initial_idx_a = Int[]

    hist_k = Int[]
    hist_max_res = Float64[]
    hist_block_size = Int[]
    hist_norms = Vector{Float64}[]
    hist_states = Vector{Int}[]

    while !isempty(unconverged_idx)
        R_rem = @view R_full[:, unconverged_idx]
        norms_rem = norms2(R_rem)

        B_normalized = R_rem ./ transpose(norms_rem)
        qr_res = qr(B_normalized, ColumnNorm())

        R_diag = abs.(diag(qr_res.R))
        m_a = count(x -> x >= tau_T, R_diag)
        m_a = clamp(m_a, 1, min(max_active, length(unconverged_idx)))

        p = qr_res.p
        unconverged_idx = unconverged_idx[p]

        idx_a = unconverged_idx[1:m_a]
        idx_s = unconverged_idx[m_a+1:end]
        m_seed = length(idx_s)

        println("\n--- RESTART ---")
        println("Remaining RHS: $(length(unconverged_idx)). New Active: $m_a, Seed: $m_seed")

        global_states[idx_a] .= 1
        global_states[idx_s] .= 0

        if is_first_save
            initial_idx_a = copy(idx_a)
            append_history!(hist_k, hist_max_res, hist_block_size, hist_norms, hist_states, 0, m_a, global_norms, global_states)
            is_first_save = false
        end

        X_a = zeros(T, size(B, 1), m_a)
        R_a = R_full[:, idx_a]
        X_s = zeros(T, size(B, 1), m_seed)
        R_s = R_full[:, idx_s]
        qR_s = copy(R_s)

        Vₖ = copy(R_a)
        QR_init = qr!(Vₖ)
        Vₖ = get_thin_Q(QR_init, m_a)
        βₖ = QR_init.R
        γₖ = transpose(Vₖ) * Vₖ
        τ_tilde_a = copy(βₖ)

        if m_seed > 0
            ρ_s = γₖ \ (transpose(Vₖ) * qR_s)
            qR_s .-= Vₖ * ρ_s
            τ_tilde_s = copy(ρ_s)
        else
            τ_tilde_s = zeros(T, 0, 0)
        end

        Vₖ₋₁ = zero(Vₖ)
        Pₖ = zero(Vₖ)
        Pₖ₋₁ = zero(Vₖ)
        Pₖ₋₂ = zero(Vₖ)
        APₖ = zero(Vₖ)
        APₖ₋₁ = zero(Vₖ)
        APₖ₋₂ = zero(Vₖ)

        b_km2 = zeros(T, m_a, m_a)
        b_km1 = zeros(T, m_a, m_a)
        c_km1 = zeros(T, m_a, m_a)
        a_km1 = Matrix{T}(I, m_a, m_a)
        d_km2 = Matrix{T}(I, m_a, m_a)
        d_km1 = Matrix{T}(I, m_a, m_a)

        δ_km1 = zeros(T, m_a, m_a)
        γₖ₋₁ = Matrix{T}(I, m_a, m_a)

        # preallocations
        AVₖ = zeros(T, size(A, 1), m_a)
        # --- INNER LOOP ---
        while true
            mul!(AVₖ, A, Vₖ)

            δ_km1 = γₖ₋₁ \ (transpose(βₖ) * γₖ)
            V_tilde = copy(AVₖ) # Only allocates once
            mul!(V_tilde, Vₖ₋₁, δ_km1, -1.0, 1.0) # V_tilde = -1*(Vₖ₋₁*δ_km1) + 1*V_tilde
            αₖ = γₖ \ (transpose(Vₖ) * V_tilde)
            V_tilde = V_tilde - Vₖ * αₖ

            QR1 = qr!(V_tilde)
            Vₖ₊₁ = get_thin_Q(QR1, m_a)
            βₖ₊₁ = QR1.R
            QR2 = qr!(Vₖ₊₁)
            Vₖ₊₁ = get_thin_Q(QR2, m_a)
            β_tmp = QR2.R
            βₖ₊₁ = β_tmp * βₖ₊₁

            α_tilde = γₖ \ (transpose(Vₖ) * Vₖ₊₁)
            αₖ = αₖ + α_tilde * βₖ₊₁
            Vₖ₊₁ = Vₖ₊₁ - Vₖ * α_tilde

            δ_tilde = γₖ₋₁ \ (transpose(Vₖ₋₁) * Vₖ₊₁)
            δ_km1 = δ_km1 + δ_tilde * βₖ₊₁
            Vₖ₊₁ = Vₖ₊₁ - Vₖ₋₁ * δ_tilde

            QR3 = qr!(Vₖ₊₁)
            Vₖ₊₁ = get_thin_Q(QR3, m_a)
            β_tmp = QR3.R
            βₖ₊₁ = β_tmp * βₖ₊₁

            γₖ₊₁ = transpose(Vₖ₊₁) * Vₖ₊₁

            if m_seed > 0
                ρ_s_kp1 = γₖ₊₁ \ (transpose(Vₖ₊₁) * qR_s)
                qR_s .-= Vₖ₊₁ * ρ_s_kp1
            else
                ρ_s_kp1 = zeros(T, 0, 0)
            end

            θₖ = b_km2 * δ_km1
            ηₖ = a_km1 * d_km2 * δ_km1 + b_km1 * αₖ
            ζ_tilde = c_km1 * d_km2 * δ_km1 + d_km1 * αₖ

            mat_to_qr = [ζ_tilde; βₖ₊₁]
            QR_qmr = qr(mat_to_qr)
            Qₖ_full = Matrix{T}(I, 2m_a, 2m_a)
            lmul!(QR_qmr.Q, Qₖ_full)
            Qₖ_star = Qₖ_full'

            aₖ = @views Qₖ_star[1:m_a, 1:m_a]
            bₖ = @views Qₖ_star[1:m_a, m_a+1:2m_a]
            cₖ = @views Qₖ_star[m_a+1:2m_a, 1:m_a]
            dₖ = @views Qₖ_star[m_a+1:2m_a, m_a+1:2m_a]

            ζₖ = aₖ * ζ_tilde + bₖ * βₖ₊₁
            Pₖ = (Vₖ - Pₖ₋₁ * ηₖ - Pₖ₋₂ * θₖ) / ζₖ
            APₖ = (AVₖ - APₖ₋₁ * ηₖ - APₖ₋₂ * θₖ) / ζₖ

            τₖ_a = aₖ * τ_tilde_a
            τ_tilde_a = cₖ * τ_tilde_a
            mul!(X_a, Pₖ, τₖ_a, 1.0, 1.0)
            mul!(R_a, APₖ, τₖ_a, -1.0, 1.0)

            if m_seed > 0
                τₖ_s = aₖ * τ_tilde_s + bₖ * ρ_s_kp1
                τ_tilde_s = cₖ * τ_tilde_s + dₖ * ρ_s_kp1
                mul!(X_s, Pₖ, τₖ_s, 1.0, 1.0)
                mul!(R_s, APₖ, τₖ_s, -1.0, 1.0)
            end

            curr_norms_a = norms2(R_a) ./ global_R0_norms[idx_a]
            global_norms[idx_a] = curr_norms_a

            if m_seed > 0
                curr_norms_s = norms2(R_s) ./ global_R0_norms[idx_s]
                global_norms[idx_s] = curr_norms_s

                converged_mask = curr_norms_s .<= tol_T
                if any(converged_mask)
                    conv_local_idx = findall(converged_mask)
                    conv_global_idx = idx_s[conv_local_idx]

                    global_states[conv_global_idx] .= 2
                    @views X_full[:, conv_global_idx] .+= X_s[:, conv_local_idx]
                    @views R_full[:, conv_global_idx] .= R_s[:, conv_local_idx]

                    keep_mask = .!converged_mask
                    idx_s = idx_s[keep_mask]
                    X_s = X_s[:, keep_mask]
                    R_s = R_s[:, keep_mask]
                    qR_s = qR_s[:, keep_mask]
                    τ_tilde_s = τ_tilde_s[:, keep_mask]
                    m_seed = length(idx_s)
                end
            end

            max_res_active = maximum(curr_norms_a)
            printfmt("k = {:5d} | active_m = {:2d} | max ACTIVE res = {:7f} | seed remaining = {:d}\n", global_k, m_a, max_res_active, m_seed)

            if all(curr_norms_a .<= tol_T)
                global_states[idx_a] .= 2
            end

            append_history!(hist_k, hist_max_res, hist_block_size, hist_norms, hist_states, global_k, m_a, global_norms, global_states)
            global_k += 1

            if all(curr_norms_a .<= tol_T)
                @views X_full[:, idx_a] .+= X_a
                @views R_full[:, idx_a] .= R_a
                if m_seed > 0
                    @views X_full[:, idx_s] .+= X_s
                    @views R_full[:, idx_s] .= R_s
                end
                unconverged_idx = idx_s
                break
            end

            APₖ₋₂ = APₖ₋₁
            APₖ₋₁ = APₖ
            Pₖ₋₂ = Pₖ₋₁
            Pₖ₋₁ = Pₖ
            Vₖ₋₁ = Vₖ
            Vₖ = Vₖ₊₁
            βₖ = βₖ₊₁
            γₖ₋₁ = γₖ
            γₖ = γₖ₊₁
            b_km2 = b_km1
            b_km1 = bₖ
            c_km1 = cₖ
            a_km1 = aₖ
            d_km2 = d_km1
            d_km1 = dₖ
        end
    end

    println("Solution finished in $(global_k-1) iterations.")

    write_history_to_csv(filename, hist_k, hist_max_res, hist_block_size, hist_norms, hist_states)
    return X_full, initial_idx_a
end

# ==============================================================================
# 3. TEST HARNESS
# ==============================================================================
function test_bsqmrr2_vs_deflation()
    # Profile.init(n=10^8, delay=0.01)

    if !isfile("./alm.jld2")
        println("File ./alm.jld2 not found.")
        return
    end

    println("Loading alm.jld2...")
    f = FileIO.load("./alm.jld2")

    A = Array{ComplexF64}(f["A"])
    A = Symmetric((A + transpose(A)) ./ 2, :L)

    B_raw = Array{ComplexF64}(f["B"])
    B_raw = B_raw[:, 1:722]
    m_total = size(B_raw, 2)

    tol = 1e-3
    file_orig = "output/bsqmr_original.csv"
    file_defl = "output/bsqmr_seed.csv"

    threshold_tau = 0.01
    max_active = 722


    println("\n--- WARMUP (Compiling Functions) ---")
    bsqmr_seed_restarted(A, @views(B_raw[:, 1:5]), tol, "output/warmup.csv"; max_active=5, threshold_tau=threshold_tau)
    println("\n--- 1. Running Seed Algorithm with Profiler ---")
    # Profile.clear() # Clear any previous profiling data

    println("\n--- 1. Running Seed Algorithm (Restarts & Dynamic Tracking) ---")
    @time X_full, initial_idx_a = bsqmr_seed_restarted(A, B_raw, tol, file_defl; max_active=max_active, threshold_tau=threshold_tau)
    println("\n--- PROFILING RESULTS ---")
    # Profile.print(format=:tree, mincount=10) # mincount filters out tiny visual noise

    m_active = length(initial_idx_a)

    println("\n--- 2. Running original on EXACT SAME active block (s=$m_active) ---")
    @time bsqmr_original(A, B_raw, tol, file_orig)

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    println("Parsing DataFrames and preparing plots...")
    d1 = CSV.read(file_orig, DataFrame)
    d2 = CSV.read(file_defl, DataFrame)

    k_vec = d2.k
    n_iters = length(k_vec)

    res_cols = ["res_$i" for i in 1:m_total]
    state_cols = ["state_$i" for i in 1:m_total]

    res_matrix = Matrix(d2[!, res_cols])
    state_matrix = Matrix(d2[!, state_cols])

    mat_active = fill(NaN, n_iters, m_total)
    mat_seed = fill(NaN, n_iters, m_total)
    mat_conv = fill(NaN, n_iters, m_total)

    max_active_res = fill(NaN, n_iters)

    for t in 1:n_iters
        max_act = NaN
        for j in 1:m_total
            r_val = res_matrix[t, j]
            s_val = state_matrix[t, j]

            if s_val == 1.0 # Active
                mat_active[t, j] = r_val
                if t > 1
                    mat_active[t-1, j] = res_matrix[t-1, j]
                end
                max_act = isnan(max_act) ? r_val : max(max_act, r_val)

            elseif s_val == 0.0 # Seed
                mat_seed[t, j] = r_val
                if t > 1
                    mat_seed[t-1, j] = res_matrix[t-1, j]
                end

            elseif s_val == 2.0 # Converged
                mat_conv[t, j] = r_val
                if t > 1
                    mat_conv[t-1, j] = res_matrix[t-1, j]
                end
            end
        end
        max_active_res[t] = max_act
    end

    p1 = plot(title="States: Seed (Grn) → Active (Blu) → Converged (Gld)",
        yaxis=:log, xlabel="Iteration", ylabel="Relative Residual Norm", legend=:outertopright)

    plot!(p1, k_vec, mat_seed, color=:green, lw=1, alpha=0.15, label="")
    plot!(p1, [0], [NaN], color=:green, lw=2, label="Passive RHS")

    plot!(p1, k_vec, mat_active, color=:blue, lw=1, alpha=0.3, label="")
    plot!(p1, [0], [NaN], color=:blue, lw=2, label="Active RHS")

    plot!(p1, k_vec, mat_conv, color=:gold, lw=1, alpha=0.3, label="")
    plot!(p1, [0], [NaN], color=:gold, lw=2, label="Converged RHS")

    plot!(p1, k_vec, max_active_res, color=:red, lw=2, label="Max (CURRENT Active)")

    p2 = plot(title="Max Residual (Original vs Seed)", yaxis=:log, xlabel="Iteration", legend=:topright)
    plot!(p2, d1.k, d1.real_residual, label="Original (s=$m_active)", lw=4, color=:black, alpha=0.5)
    plot!(p2, k_vec, max_active_res, label="Restarted Seed-SQMR", lw=2, color=:red, linestyle=:dash)

    max_all_d2 = [maximum(r) for r in eachrow(res_matrix)]
    plot!(p2, k_vec, max_all_d2, label="Seed-SQMR (Max of ALL $m_total RHS)", lw=2, color=:magenta, linestyle=:dot)

    p3 = plot(title="Krylov Subspace Dimension Size (m_active per Step)", xlabel="Iteration", legend=:topright)
    plot!(p3, d1.k, d1.block_size, label="Original (s=$m_active constant)", lw=3, color=:black, alpha=0.5)
    plot!(p3, d2.k, d2.block_size, label="Restarted Seed-SQMR", lw=2, color=:red, linestyle=:dash, linetype=:steppost)

    display(plot(p1, p2, p3, layout=(3, 1), size=(800, 1000), margin=5Plots.mm))
    savefig("output/bsqmr_alm_compare_722.png")
    println("Saved comparison to output/bsqmr_alm_compare_722.png")
end

test_bsqmrr2_vs_deflation()