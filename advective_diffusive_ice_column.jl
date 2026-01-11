using Plots
using BenchmarkTools

# =========================
# Physics constants
# =========================
const Pe = 5.0
const γ = 0.35
const β = 0.5
const Ω = 0.0 # 0.0, -1.0
const initial_θ = 1.0 # 1.0, 1.05
const T_air = 223.15 # 223.15, 248.15

# =========================
# Numerical parameters
# =========================
const n = 100
const Δτ = 1e-5
const tsteps = 100_000
const spacing = "even"

# =========================
# Plotting
# =========================
const plot_heatmap_flag = false
const plot_lines_flag = false
const n_lines = 10 # total number of line snapshots to plot
const save_every = 1000 # save every n time steps

# =========================
# Grid
# =========================
const ξ = LinRange(0, 1, n)
const ω = - Pe .* ξ # the minus sign implies ice is advecting heat downward because positive ξ points upwards and the ice is moving downward (ω₀ < 0)

function setup_grid(spacing; spacing_order=2.0, spacing_factor=2.0)
    spacing == "even" && return ξ
    spacing == "polynomial" && return ξ .^ spacing_order
    spacing == "exponential" &&
        return (exp.(spacing_factor .* ξ) .- 1) ./ (exp(spacing_factor) - 1)
    error("Unknown spacing")
end

const ζ = setup_grid(spacing)
const h = diff(ζ)

# =========================
# Initial condition for θ
# =========================
function compute_initial_θ()
    θ_before = fill(initial_θ, n)
    θ_now = fill(initial_θ, n)    
    return θ_before, θ_now
end

# =========================
# Time evolution
# =========================
function time_evolution_for_loop()

    # =========================
    # Initialize state
    # =========================
    θ_before, θ_now = compute_initial_θ()

    # =========================
    # Storage of states
    # =========================
    nt_plot = Int(floor(tsteps / save_every)) + 1 # + 1 for initial state
    θ = zeros(n, nt_plot)
    time = zeros(nt_plot)
    θ[:, 1] .= θ_now
    time[1] = 0.0
    counter = 2

    @inbounds begin

        # =========================
        # 5 point stencil coefficients from Eq 30 of https://www.researchgate.net/publication/229045683_Finite_Difference_Formulae_for_Unequal_Sub-Intervals_Using_Lagrange's_Interpolation_Formula
        # =========================
        a_1 = zeros(n)
        a_2 = zeros(n)
        a_3 = zeros(n)
        a_4 = zeros(n)
        a_5 = zeros(n)
        for i in 3:n-2
            h_1 = h[i-2]
            h_2 = h[i-1]
            h_3 = h[i]
            h_4 = h[i+1]
            H_2 = h_1 + h_2 + h_3 + h_4
            a_1[i] = (- 2.0 * h_2 * (2.0 * h_3 + h_4) + 2.0 * h_3 * (h_3 + h_4)) / (h_1 * (h_1 + h_2) * (h_1 + h_2 + h_3) * H_2)
            a_2[i] = (2.0 * (h_1 + h_2) * (2.0 * h_3 + h_4) - 2.0 * h_3 * (h_3 + h_4)) / (h_1 * h_2 * (h_1 + h_3) * (h_2 + h_3 + h_4))
            a_3[i] = (2.0 * h_2 * (h_1 + h_2) - 2.0 * (h_1 + 2.0 * h_2) * (2.0 * h_3 + h_4) + 2.0 * h_3 * (h_3 + h_4)) / ((h_1 + h_2) * h_2 * h_3 * (h_3 + h_4))
            a_4[i] = (2.0 * (h_1 + 2.0 * h_2) * (h_3 + h_4) - 2.0 * h_2 * (h_1 + h_2)) / ((h_1 + h_2 + h_3) * (h_2 + h_3) * h_3 * h_4)
            a_5[i] = (2.0 * (h_1 + h_2) * h_2 - 2.0 * (h_1 + 2.0 * h_2) * h_3) / (H_2 * (h_2 + h_3 + h_4) * (h_3 + h_4) * h_4)
        end

        # =========================
        # Coefficients for base boundary condition - 3 point forward first derivative
        # =========================
        h_1 = h[1]
        h_2 = h[2]
        H_1 = h_1 + h_2
        b_1 = (2.0 * h_1 + h_2) / (h_1 * H_1)
        b_2 = - H_1 / (h_1 * h_2)
        b_3 = h_1 / (h_2 * H_1)

        # =========================
        # Time loop
        # =========================
        for t in 1:tsteps

            # ========================
            # Integration in time for inner spatial loop avoiding 2 boundary points from each end
            # =======================
            for i in 3:n-2
                diffusion = a_1[i]*θ_before[i-2] + a_2[i]*θ_before[i-1] + a_3[i]*θ_before[i] + a_4[i]*θ_before[i+1] + a_5[i]*θ_before[i+2]
                advection = ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h[i] + h[i-1]) 
                θ_now[i] = θ_before[i] + Δτ*(diffusion - advection + Ω)
            end

            # ========================
            # Integrate in time points near the boundary that require a 3-point stencil
            # =======================
            for i in (2, n-1)
                h_1 = h[i-1]
                h_2 = h[i]
                diffusion = 2*(h_1*θ_before[i+1] - (h_2+h_1)*θ_before[i] + h_2*θ_before[i-1]) / (h_2*h_1*(h_2+h_1))
                advection = ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h_2 + h_1)
                θ_now[i] = θ_before[i] + Δτ*(diffusion - advection + Ω)
            end

            # ========================
            # Integrate base boundary point - 3 point forward first derivative
            # =======================
            θ_now[1] = (γ - b_2*θ_now[2] - b_3*θ_now[3]) / b_1 # 3 point forward first derivative

            # ========================
            # Integrate surface boundary point - backward first derivative
            # =======================
            θ_now[end] = (h[end] + β*θ_now[end-1]) / (h[end] + β) # backward first derivative

            # ========================
            # Store results 
            # =======================
            if mod(t, save_every) == 0
                θ[:, counter] .= θ_now
                time[counter] = t * Δτ
                counter += 1
            end
            
            # ========================
            # Update state for next time step
            # =======================
            θ_now, θ_before = θ_before, θ_now

        end

    end

    return time, θ

end

function time_evolution_no_for_loop()

    # =========================
    # Initialize state
    # =========================
    θ_before, θ_now = compute_initial_θ()
    diffusion = similar(θ_before, n-4)
    advection = similar(θ_before, n-4)

    # =========================
    # Storage of states
    # =========================
    nt_plot = Int(floor(tsteps / save_every)) + 1 # + 1 for initial state
    θ = zeros(n, nt_plot)
    time = zeros(nt_plot)
    θ[:, 1] .= θ_now
    time[1] = 0.0
    counter = 2

    @views begin

        # =========================
        # Coefficients
        # =========================
        h_1, h_2, h_3, h_4 = h[1:end-3], h[2:end-2], h[3:end-1], h[4:end]
        H_2 = @. h_1 + h_2 + h_3 + h_4

        a_1 = @. (-2h_2*(2h_3+h_4) + 2h_3*(h_3+h_4)) /
                (h_1*(h_1+h_2)*(h_1+h_2+h_3)*H_2)

        a_2 = @. (2(h_1+h_2)*(2h_3+h_4) - 2h_3*(h_3+h_4)) /
                (h_1*h_2*(h_1+h_3)*(h_2+h_3+h_4))

        a_3 = @. (2h_2*(h_1+h_2) - 2(h_1+2h_2)*(2h_3+h_4) + 2h_3*(h_3+h_4)) /
                ((h_1+h_2)*h_2*h_3*(h_3+h_4))

        a_4 = @. (2(h_1+2h_2)*(h_3+h_4) - 2h_2*(h_1+h_2)) /
                ((h_1+h_2+h_3)*(h_2+h_3)*h_3*h_4)

        a_5 = @. (2(h_1+h_2)*h_2 - 2(h_1+2h_2)*h_3) /
                (H_2*(h_2+h_3+h_4)*(h_3+h_4)*h_4)

        b_1 = @. (2h[1] + h[2]) / (h[1]*(h[1] + h[2]))
        b_2 = @. -(h[1] + h[2]) / (h[1]*h[2])
        b_3 = @. h[1] / (h[2]*(h[1] + h[2]))

        # =========================
        # Time loop
        # =========================
        for t in 1:tsteps

            # ========================
            # Integration in time for inner spatial loop avoiding 2 boundary points from each end
            # =======================
            @. diffusion = a_1*θ_before[1:end-4] + a_2*θ_before[2:end-3] + a_3*θ_before[3:end-2] + a_4*θ_before[4:end-1] + a_5*θ_before[5:end]
            @. advection = ω[3:end-2] * (θ_before[4:end-1] - θ_before[2:end-3]) / (h_3 + h_2)
            @. θ_now[3:end-2] = θ_before[3:end-2] + Δτ*(diffusion - advection + Ω)

            # ========================
            # Integrate in time points near the boundary that require a 3-point stencil
            # =======================
            for i in (2, n-1)
                h_1 = h[i-1]
                h_2 = h[i]
                θ_now[i] = θ_before[i] + Δτ*(2*(h_1*θ_before[i+1] - (h_2+h_1)*θ_before[i] + h_2*θ_before[i-1]) / (h_2*h_1*(h_2+h_1)) - ω[i]*(θ_before[i+1]-θ_before[i-1])/(h_2 + h_1) + Ω)
            end
            
            # ========================
            # Integrate base boundary point - 3 point forward first derivative
            # =======================
            θ_now[1] = (γ - b_2*θ_now[2] - b_3*θ_now[3]) / b_1

            # ========================
            # Integrate surface boundary point - backward first derivative
            # =======================
            θ_now[end] = (β*θ_now[end-1] + h[end]) / (β + h[end])

            if mod(t, save_every) == 0
                θ[:, counter] .= θ_now
                time[counter] = t * Δτ
                counter += 1
            end

            θ_before, θ_now = θ_now, θ_before

        end

    end

    return time, θ
end

function plot_heatmap(time, θ; path_to_save = nothing, display_flag = true)

    # =========================
    # Heatmap
    # =========================
    if plot_heatmap_flag
        p1 = heatmap(time, ζ, @.(T_air*θ-273.15), xlabel = "Time", ylabel = "Depth (ζ)", colorbar_title = "Temperature", colormap = :thermal)
        if display_flag 
            display(p1)
        end
        if !isnothing(path_to_save)
            savefig(p1, path_to_save)
        end
    end

end

function plot_lines(time, θ; path_to_save = nothing, display_flag = true)

    # =========================
    # Line plots
    # =========================
    if plot_lines_flag
        p1 = plot(xlabel = "Temperature (°C)", ylabel = "Depth (ζ)", legend = :bottomright)
        n_time = length(time)
        for i in 0:n_lines-1
            idx = Int(1 + i * (n_time - 1) / (n_lines - 1))
            plot!(T_air .* θ[:, idx] .- 273.15, ζ, label = "t=$(round(time[idx], digits=3)) s")
        end
        if display_flag 
            display(p1)
        end
        if !isnothing(path_to_save)
            savefig(p1, path_to_save)
        end
    end
end

if plot_heatmap_flag || plot_lines_flag
    time, θ = time_evolution_for_loop()
    if plot_lines_flag
        plot_lines(time, θ)
    end
    if plot_heatmap_flag
        plot_heatmap(time, θ)
    end
else
    r = @benchmark time_evolution_for_loop() samples = 3 evals = 1
    println("Benchmark for time_evolution_for_loop():")
    display(r)
    r = @benchmark time_evolution_no_for_loop() samples = 3 evals = 1
    println("Benchmark for time_evolution_no_for_loop():")
    display(r)
end
