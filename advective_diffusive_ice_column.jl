using Plots
using BenchmarkTools

# =========================
# Physics constants
# =========================
const Pe = 5.0
const γ = -0.35
const β = 0.5
const Ω = 0.0
const initial_θ = 0.5

# =========================
# Numerical parameters
# =========================
const n = 100
const Δτ = 0.00001
const tsteps = 100000
const plot_every = 200
const spacing = "even"

# =========================
# Plot switches
# =========================
const plot_heatmap = false
const plot_lines = true
const n_lines = 10 # number of line snapshots

# =========================
# Grid
# =========================
const ξ = LinRange(0, 1, n)
const ω = Pe .* ξ

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
# Time evolution
# =========================
function time_evolution()

    # =========================
    # Initialize state
    # =========================
    θ_before = fill(initial_θ, n)
    θ_now = fill(initial_θ, n)

    # =========================
    # Storage
    # =========================
    if plot_heatmap || plot_lines
        nt_plot = Int(floor(tsteps / plot_every)) + 1
        Θ = zeros(n, nt_plot)
        time = zeros(nt_plot)

        Θ[:, 1] .= θ_now
        time[1] = 0.0
        counter = 2
    end

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
                h_2 = h[i-1]
                h_3 = h[i]
                diffusion = 2*(h_2*θ_before[i+1] - (h_3+h_2)*θ_before[i] + h_3*θ_before[i-1]) / (h_3*h_2*(h_3+h_2))
                advection = ω[i] * (θ_before[i+1] - θ_before[i-1]) / (h_3 + h_2)
                θ_now[i] = θ_before[i] + Δτ*(diffusion - advection + Ω)
            end

            # ========================
            # Integrate base boundary point - 3 point forward first derivative
            # =======================
            θ_now[1] = (γ - b_2*θ_now[2] - b_3*θ_now[3]) / b_1 # 3 point forward first derivative

            # ========================
            # Integrate surface boundary point - backward first derivative
            # =======================
            θ_now[end] = (β*θ_now[end-1] + 1.0) / (1.0 + β / h[end])

            # ========================
            # Store results for plotting
            # =======================
            if plot_heatmap || plot_lines
                if mod(t, plot_every) == 0
                    Θ[:, counter] .= θ_now
                    time[counter] = t * Δτ
                    counter += 1
                end
            end

            # ========================
            # Update state for next time step
            # =======================
            θ_now, θ_before = θ_before, θ_now

        end

    end

    # =========================
    # Heatmap
    # =========================
    if plot_heatmap
        p1 = heatmap(time, ζ, Θ, xlabel = "Time", ylabel = "Depth (ζ)", colorbar_title = "Temperature", colormap = :thermal)
        display(p1)
    end

    # =========================
    # Line plot (optional)
    # =========================
    if plot_lines
        idx = round.(Int, range(1, size(Θ,2), length=n_lines))
        p2 = plot(xlabel = "Temperature", ylabel = "Depth (ζ)", legend = false)
        for (k, j) in enumerate(idx)
            gray = 0.85 * (1 - (k-1)/(n_lines-1))
            plot!(p2, Θ[:, j], ζ, color = RGB(gray, gray, gray), lw = 1.5)
        end
        display(p2)
    end

end

if plot_heatmap || plot_lines
    time_evolution()
else
    r = @benchmark time_evolution() samples = 3 evals = 1
    display(r)
end
