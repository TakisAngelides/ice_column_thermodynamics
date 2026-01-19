USE_GPU = true
using Plots
using BenchmarkTools
using ParallelStencil
ParallelStencil.@reset_parallel_stencil()
using ParallelStencil.FiniteDifferences1D

@static if USE_GPU
    @init_parallel_stencil(Metal, Float32, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end

# Define a handy alias for the data type (Float32 on GPU because Metal does not work with Float64, Float64 on CPU)
# Data.Number = The type of numbers used by @zeros, @ones, @rand and @fill and in all array types of module `Data` (selected with argument `numbertype` of [`@init_parallel_kernel`](@ref)).
const DATAT = Data.Number

@parallel function assign!(A::Data.Array, B::Data.Array)
    @all(A) = @all(B) 
    return
end

function setup_grid(spacing, ξ; spacing_order=2.0, spacing_factor=2.0)
    # Ensure parameters match the precision of ξ
    T = eltype(ξ)
    spacing_order = T(spacing_order)
    spacing_factor = T(spacing_factor)

    spacing == "even" && return ξ
    spacing == "polynomial" && return ξ .^ spacing_order
    spacing == "exponential" && return (exp.(spacing_factor .* ξ) .- 1) ./ (exp(spacing_factor) - 1)
    error("Unknown spacing")
end

@parallel_indices (i) function time_step_five_point!(θ_before, θ_now, h, ω, a_1, a_2, a_3, a_4, a_5, n, Δτ, Ω)
    if (i >= 3) && (i <= n-2)
        θ_now[i] = θ_before[i] + Δτ*(a_1[i]*θ_before[i-2] + a_2[i]*θ_before[i-1] + a_3[i]*θ_before[i] + a_4[i]*θ_before[i+1] + a_5[i]*θ_before[i+2] 
                                     - ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h[i] + h[i-1]) + Ω) 
    end
    return
end

@parallel_indices (i) function time_step_three_point!(θ_before, θ_now, h, ω, c_1, c_2, c_3, n, Δτ, Ω)
    if (i >= 3) && (i <= n-2)
        θ_now[i] = θ_before[i] + Δτ*(c_1[i]*θ_before[i+1] + c_2[i]*θ_before[i] + c_3[i]*θ_before[i-1] 
                                     - ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h[i] + h[i-1]) + Ω) 
    end
    return
end

@parallel_indices (i) function near_boundary_time_step!(θ_before, θ_now, h, ω, n, Δτ, Ω)
    if (i == 2) || (i == n-1)
        h_1 = h[i-1]
        h_2 = h[i]
        θ_now[i] = θ_before[i] + Δτ*(2*(h_1*θ_before[i+1] - (h_2+h_1)*θ_before[i] + h_2*θ_before[i-1]) / (h_2*h_1*(h_2+h_1)) 
                                     - ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h_2 + h_1) + Ω) 
    end
    return
end

@parallel_indices (i) function base_boundary!(θ_now, γ, b_1, b_2, b_3)
    if (i == 1)
        θ_now[1] = (γ - b_2*θ_now[2] - b_3*θ_now[3]) / b_1 
    end
    return
end

@parallel_indices (i) function surface_boundary!(θ_now, h, β, n)
    if (i == n)
        θ_now[end] = (h[end] + β*θ_now[end-1]) / (h[end] + β) 
    end
    return
end

# =========================
# Time evolution
# =========================
function time_evolution()

    # =========================
    # Physics constants (Cast to DATAT)
    # =========================
    Pe = DATAT(5.0)
    γ  = DATAT(0.35)
    β  = DATAT(0.5)
    Ω  = DATAT(0.0) 
    initial_θ = DATAT(1.0) 

    # =========================
    # Numerical parameters
    # =========================
    n = 30
    Δτ = DATAT(1e-5) # Must be DATAT for kernel multiplication
    tsteps = 100_000
    spacing = "polynomial"

    # =========================
    # Saving states
    # =========================
    save_every = 10 

    # =========================
    # Grid (Ensure LinRange uses DATAT)
    # =========================
    # LinRange defaults to Float64 endpoints, so we force DATAT
    ξ = LinRange(DATAT(0), DATAT(1), n) 
    
    # ζ and h setup
    ζ = setup_grid(spacing, ξ)
    h = diff(ζ) # This will inherit Float32 from ζ

    # ω calculation: Ensure literals are cast or allow broadcasting to handle it
    ω = Data.Array(- Pe .* ζ) 

    # =========================
    # Initialize state
    # =========================
    θ_before = Data.Array(fill(initial_θ, n)) 
    θ_now    = Data.Array(fill(initial_θ, n)) 

    # =========================
    # Storage of states (CPU Storage can remain Float64 for precision/plotting)
    # =========================
    num_saved_states = Int(floor(tsteps / save_every)) + 1 
    θ = zeros(Float64, n, num_saved_states) 
    time = zeros(Float64, num_saved_states)
    θ[:, 1] .= Array(θ_now) 
    time[1] = 0.0
    counter = 2

    # =========================
    # Coefficients calculation
    # =========================
    # Initialize on CPU with correct type
    a_1 = zeros(DATAT, n)
    a_2 = zeros(DATAT, n)
    a_3 = zeros(DATAT, n)
    a_4 = zeros(DATAT, n)
    a_5 = zeros(DATAT, n)
    c_1 = zeros(DATAT, n)
    c_2 = zeros(DATAT, n)
    c_3 = zeros(DATAT, n)
    
    # Note: h is currently a GPU array (Data.Array). 
    # We must pull h back to CPU to compute coefficients sequentially on CPU.
    h_cpu = Array(h)

    for i in 3:n-2

        h_1 = h_cpu[i-2]
        h_2 = h_cpu[i-1]
        h_3 = h_cpu[i]
        h_4 = h_cpu[i+1]
        H_2 = h_1 + h_2 + h_3 + h_4
        
        # Use DATAT(2.0) to prevent implicit promotion to Float64
        two = DATAT(2.0)
        
        a_1[i] = (- two * h_2 * (two * h_3 + h_4) + two * h_3 * (h_3 + h_4)) / (h_1 * (h_1 + h_2) * (h_1 + h_2 + h_3) * H_2)
        a_2[i] = (two * (h_1 + h_2) * (two * h_3 + h_4) - two * h_3 * (h_3 + h_4)) / (h_1 * h_2 * (h_1 + h_3) * (h_2 + h_3 + h_4))
        a_3[i] = (two * h_2 * (h_1 + h_2) - two * (h_1 + two * h_2) * (two * h_3 + h_4) + two * h_3 * (h_3 + h_4)) / ((h_1 + h_2) * h_2 * h_3 * (h_3 + h_4))
        a_4[i] = (two * (h_1 + two * h_2) * (h_3 + h_4) - two * h_2 * (h_1 + h_2)) / ((h_1 + h_2 + h_3) * (h_2 + h_3) * h_3 * h_4)
        a_5[i] = (two * (h_1 + h_2) * h_2 - two * (h_1 + two * h_2) * h_3) / (H_2 * (h_2 + h_3 + h_4) * (h_3 + h_4) * h_4)

        c_1[i] = two * h_2 / (h_3 * h_2 * (h_3 + h_2))
        c_2[i] = - two * (h_3 + h_2) / (h_3 * h_2 * (h_3 + h_2))
        c_3[i] = two * h_3 / (h_3 * h_2 * (h_3 + h_2))
    
    end

    # Boundary coefficients
    h_1 = h_cpu[1]
    h_2 = h_cpu[2]
    H_1 = h_1 + h_2
    
    # Ensure results are DATAT
    b_1 = DATAT((2.0 * h_1 + h_2) / (h_1 * H_1))
    b_2 = DATAT(- H_1 / (h_1 * h_2))
    b_3 = DATAT(h_1 / (h_2 * H_1))
    
    # Move coefficients to GPU
    a_1, a_2, a_3, a_4, a_5 = Data.Array(a_1), Data.Array(a_2), Data.Array(a_3), Data.Array(a_4), Data.Array(a_5)
    c_1, c_2, c_3 = Data.Array(c_1), Data.Array(c_2), Data.Array(c_3)
    
    # h is already on GPU, but we update it from our DATAT conversion just in case
    h = Data.Array(h_cpu)

    # =========================
    # Time loop
    # =========================
    for t in 1:tsteps
        # @parallel (3:n-2) time_step_five_point!(θ_before, θ_now, h, ω, a_1, a_2, a_3, a_4, a_5, n, Δτ, Ω)
        @parallel (3:n-2) time_step_three_point!(θ_before, θ_now, h, ω, c_1, c_2, c_3, n, Δτ, Ω)
        @parallel near_boundary_time_step!(θ_before, θ_now, h, ω, n, Δτ, Ω)
        @parallel base_boundary!(θ_now, γ, b_1, b_2, b_3)
        @parallel surface_boundary!(θ_now, h, β, n)
        
        if mod(t, save_every) == 0
            θ[:, counter] .= Array(θ_now)
            time[counter] = t * Δτ
            counter += 1
        end
        
        @parallel assign!(θ_before, θ_now)
    end

    return time, θ, ζ
end

# =========================
# Plotting
# =========================
function plot_heatmap(time, θ, ζ; path_to_save = nothing, display_flag = true, T_air = 223.15)
    p1 = heatmap(time, ζ, @.(T_air*θ-273.15), xlabel = "Time", ylabel = "Depth (ζ)", colorbar_title = "Temperature", colormap = :thermal)
    if display_flag display(p1) end
    if !isnothing(path_to_save) savefig(p1, path_to_save) end
end

function plot_lines(time, θ, ζ; path_to_save = nothing, display_flag = true, T_air = 223.15, n_lines = 10)
    p1 = plot(xlabel = "Temperature (°C)", ylabel = "Depth (ζ)", legend = false)
    num_saved_states = size(θ, 2)
    for idx in 1:Int(floor(num_saved_states / n_lines)):num_saved_states
        plot!(p1, T_air .* θ[:, idx] .- 273.15, ζ, label = "t=$(round(time[idx], digits=3)) s", color = :black, alpha = idx / num_saved_states)
    end
    if display_flag display(p1) end
    if !isnothing(path_to_save) savefig(p1, path_to_save) end
end

time, θ, ζ = time_evolution()
plot_lines(time, θ, ζ)
