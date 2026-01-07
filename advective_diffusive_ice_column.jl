using Plots

# =========================
# Physics constants
# =========================
Pe = 5.0
γ = 0.35
β = 0.5
Ω = 0.0
initial_θ = 0.5

# =========================
# Numerical parameters
# =========================
n = 100
Δτ = 0.00001
tsteps = 100000
plot_every = 20
spacing = "even"

# =========================
# Plot switches
# =========================
plot_lines = true          
n_lines = 10 # number of line snapshots

# =========================
# Initialize state
# =========================
θ_before = fill(initial_θ, n)
θ_now = fill(initial_θ, n)

# =========================
# Grid
# =========================
ξ = LinRange(0, 1, n)
ω = Pe .* ξ
h = 1/(n-1)

# =========================
# Time evolution
# =========================
function time_evolution()

    # =========================
    # Storage
    # =========================
    nt_plot = Int(floor(tsteps / plot_every)) + 1
    Θ = zeros(n, nt_plot)
    time = zeros(nt_plot)

    Θ[:, 1] .= θ_now
    time[1] = 0.0
    counter = 2

    # =========================
    # Time loop
    # =========================
    for t in 1:tsteps

        diffusion = @. (-θ_before[1:end-4] + 16*θ_before[2:end-3] -30*θ_before[3:end-2] + 16*θ_before[4:end-1] -θ_before[5:end]) / (12*h^2)

        advection = @. ω[3:end-2] * (θ_before[4:end-1] - θ_before[2:end-3]) / (2*h)

        θ_now[3:end-2] = @. θ_before[3:end-2] + Δτ*(diffusion + advection + Ω)

        θ_now[2] = θ_before[2] + Δτ*(((θ_before[3] - 2*θ_before[2] + θ_before[1]) / h^2) + ω[2]*(θ_before[3]-θ_before[2])/(2*h) + Ω)
        θ_now[end-1] = θ_before[end-1] + Δτ*(((θ_before[end] - 2*θ_before[end-1] + θ_before[end-2]) / h^2) + ω[end-1]*(θ_before[end]-θ_before[end-2])/(2*h) + Ω)

        θ_now[1] = (2*h*γ + 4*θ_now[2] - θ_now[3]) / 3
        θ_now[end] = (1 - (β/(2*h))*(θ_now[end-2] - 4*θ_now[end-1])) / (1 + (3*β)/(2*h))

        if mod(t, plot_every) == 0
            Θ[:, counter] .= θ_now
            time[counter] = t * Δτ
            counter += 1
        end

        θ_before .= θ_now
    end

    # =========================
    # Heatmap
    # =========================
    p1 = heatmap(
        time, ζ, Θ,
        xlabel = "Time",
        ylabel = "Depth (ζ)",
        colorbar_title = "Temperature",
        colormap = :thermal,
        right_margin = 10Plots.mm
    )

    display(p1)

    # =========================
    # Line plot (optional)
    # =========================
    if plot_lines
        idx = round.(Int, range(1, size(Θ,2), length=n_lines))
        p2 = plot(
            xlabel = "Temperature",
            ylabel = "Depth (ζ)",
            legend = false
        )

        for (k, j) in enumerate(idx)
            gray = 0.85 * (1 - (k-1)/(n_lines-1))
            plot!(
                p2,
                Θ[:, j],
                ζ,
                color = RGB(gray, gray, gray),
                lw = 1.5
            )
        end

        display(p2)
    end
end

time_evolution()
