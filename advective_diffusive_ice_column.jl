using Plots

# =========================
# Physics constants
# =========================
Pe = 5.0
γ = 0.35
β = 0.5
Ω = 0.0
initial_θ = 1.0

# =========================
# Numerical parameters
# =========================
n = 100
Δτ = 1e-5
tsteps = 100_000
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
ω = - Pe .* ξ

function setup_grid(spacing; spacing_order=2.0, spacing_factor=2.0)
    spacing == "even" && return ξ
    spacing == "polynomial" && return ξ .^ spacing_order
    spacing == "exponential" &&
        return (exp.(spacing_factor .* ξ) .- 1) ./ (exp(spacing_factor) - 1)
    error("Unknown spacing")
end

ζ = setup_grid(spacing)
h = diff(ζ)

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
    # Coefficients
    # =========================
    h_1, h_2, h_3, h_4 = h[1:end-3], h[2:end-2], h[3:end-1], h[4:end]
    H_2 = h_1 .+ h_2 .+ h_3 .+ h_4

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

        diffusion = @. a_1*θ_before[1:end-4] + a_2*θ_before[2:end-3] +
                        a_3*θ_before[3:end-2] + a_4*θ_before[4:end-1] +
                        a_5*θ_before[5:end]

        advection = @. ω[3:end-2] *
                        (θ_before[4:end-1] - θ_before[2:end-3]) /
                        (h_3 + h_2)

        θ_now[3:end-2] = @. θ_before[3:end-2] +
                            Δτ*(diffusion - advection + Ω)

        θ_now[2] = θ_before[2] + Δτ*(
            2*(h_2[1]*θ_before[3] - (h_3[1]+h_2[1])*θ_before[2] + h_3[1]*θ_before[1]) /
            (h_3[1]*h_2[1]*(h_3[1]+h_2[1])) -
            ω[2]*(θ_before[3]-θ_before[2])/(h_3[1]+h_2[1]) + Ω
        )

        θ_now[end-1] = θ_before[end-1] + Δτ*(
            2*(h_2[end]*θ_before[end] - (h_3[end]+h_2[end])*θ_before[end-1] + h_3[end]*θ_before[end-2]) /
            (h_3[end]*h_2[end]*(h_3[end]+h_2[end])) -
            ω[end-1]*(θ_before[end]-θ_before[end-2])/(h_3[end]+h_2[end]) + Ω
        )

        θ_now[1] = (γ - b_2*θ_now[2] - b_3*θ_now[3]) / b_1
        θ_now[end] = (β*θ_now[end-1] + h[end]) / (β + h[end])

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
                @.(223.15*Θ[:, j]-273.15),
                ζ,
                color = RGB(gray, gray, gray),
                lw = 1.5
            )
        end

        display(p2)
    end
end

time_evolution()
