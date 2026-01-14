using ITensors
using ITensorMPS
using QuanticsTCI
using TensorCrossInterpolation
using Plots

ITensors.disable_warn_order()

"""
    tci(f, x, sites; tol=1e-14, f_type=Float64)

Construct an MPS using quantics tensor cross interpolation (QTT-TCI) representing the function `f` sampled on the grid `x`.
"""
function tci(f::Function, x::AbstractArray{<:Number}, sites::Vector{ITensors.Index{Int64}}; tol::Float64=1e-14, f_type::DataType=Float64)
    qtt, _, _ = quanticscrossinterpolate(f_type, f, x; tolerance=tol)
    return MPS(qtt.tci, sites=sites)
end

"""
    mps_to_list_inefficient(mps, N, n)

Fully contract an MPS and extract all coefficients as a dense vector. Used only for plotting and diagnostics.
"""
function mps_to_list_inefficient(mps, N, n)
    res = Vector{Float64}(undef, n)
    contracted = ITensorMPS.contract(mps)
    for i in 0:n-1
        binary = reverse(digits(i; base=2, pad=N) .+ 1)
        res[i+1] = contracted[binary...]
    end
    return res
end

"""
    get_triple_delta_op_list(sites)

Return a list of rank-3 delta tensors δ(s,s',s'') for converting an MPS into a diagonal MPO.
"""
function get_triple_delta_op_list(sites)
    return [delta(s, s', s'') for s in sites]
end

"""
    mps_to_mpo(mps, sites)

Convert a diagonal MPS into an MPO using triple-delta tensors.
"""
function mps_to_mpo(mps, sites)
    mpo = MPO(length(mps))
    deltas = get_triple_delta_op_list(sites)
    for i in eachindex(mps)
        mpo[i] = mps[i] * deltas[i]
    end
    setprime!(mpo, 0; :plev => 2)
    return mpo
end

"""
    sr1(N; pbc=false)

Construct the right-shift operator as an OpSum.
"""
function sr1(N; pbc=false)
    opsum = OpSum()
    for n in 1:N
        tmp = OpSum()
        tmp += "S+", n
        for i in n+1:N
            tmp *= "S-", i
        end
        opsum += tmp
    end
    return opsum
end

"""
    sl1(N; pbc=false)

Construct the left-shift operator as an OpSum.
"""
function sl1(N; pbc=false)
    opsum = OpSum()
    for n in 1:N
        tmp = OpSum()
        tmp += "S-", n
        for i in n+1:N
            tmp *= "S+", i
        end
        opsum += tmp
    end
    return opsum
end

"""
    get_diffusion_mpo(N, sites, Δx_inv)

Return the three-point symmetric second-derivative MPO.
"""
function get_diffusion_mpo(N, sites, Δx_inv; pbc=false)
    opsum = OpSum()
    opsum += -2, "Id", 1
    opsum += sr1(N; pbc=pbc) + sl1(N; pbc=pbc)
    return Δx_inv^2 * MPO(opsum, sites)
end

"""
    get_two_point_symmetric_first_derivative_mpo(N, sites, Δx_inv)

Return the two-point symmetric first-derivative MPO.
"""
function get_two_point_symmetric_first_derivative_mpo(N, sites, Δx_inv; pbc=false)
    opsum = sr1(N; pbc=pbc) - sl1(N; pbc=pbc)
    return 0.5 * Δx_inv * MPO(opsum, sites)
end

"""
    get_advection_mpo(N, sites, Δx_inv, Pe, n, ξ)

Return the advection MPO.
"""
function get_advection_mpo(N, sites, Δx_inv, Pe, n, ξ; cutoff=0.0)
    d_dx = get_two_point_symmetric_first_derivative_mpo(N, sites, Δx_inv)
    ω_mps = -Pe * tci(i -> ξ[Int(i)], 1:n, sites)
    ω_mpo = mps_to_mpo(ω_mps, sites)
    return apply(ω_mpo, d_dx; cutoff=cutoff)
end

"""
    get_time_step_mpo(N, sites, Δx_inv, Pe, n, Ω, Δτ, ξ)

Return the explicit Euler time-step MPO.
"""
function get_time_step_mpo(N, sites, Δx_inv, Pe, n, Ω, Δτ, ξ; cutoff=0.0)
    Id = MPO(sites, "Id")
    diffusion = Δτ * get_diffusion_mpo(N, sites, Δx_inv)
    advection = -Δτ * get_advection_mpo(N, sites, Δx_inv, Pe, n, ξ)
    strain = Δτ * Ω * Id
    return add(Id, diffusion, advection, strain; cutoff=cutoff)
end

"""
    get_desired_boundary_values(mps, mps_2, mps_3, mps_penultimate, Δx, γ, β)

Compute Robin-type boundary values using interior stencil points.
"""
function get_desired_boundary_values(mps, mps_2, mps_3, mps_penultimate, Δx, γ, β)
    val_2 = inner(mps, mps_2)
    val_3 = inner(mps, mps_3)
    val_penultimate = inner(mps, mps_penultimate)
    desired_base = (-2 * Δx * γ + 4 * val_2 - val_3) / 3
    desired_surface = (Δx + β * val_penultimate) / (Δx + β)
    return desired_base, desired_surface
end

"""
    set_boundary_conditions(mps, desired_base, desired_surface, base_mps, surface_mps)

Impose boundary values by projection onto endpoint basis states.
"""
function set_boundary_conditions(mps, desired_base, desired_surface, base_mps, surface_mps; cutoff=0.0)
    current_base = inner(mps, base_mps)
    current_surface = inner(mps, surface_mps)
    return add(mps, (-current_base + desired_base) * base_mps, (-current_surface + desired_surface) * surface_mps; cutoff=cutoff)
end

"""
    time_evolution(...)

Evolve the MPS forward in time and plot temperature profiles.
"""
function time_evolution(tsteps, mps, time_step_mpo, mps_2, mps_3, mps_penultimate, Δx, γ, β, base_mps, surface_mps, T_air, ξ, N, n, plot_every; cutoff = 0.0)
    
    p = plot(xlabel="Temperature (°C)", ylabel="Depth (ξ)", legend=false)
    plot!(T_air .* mps_to_list_inefficient(mps, N, n) .- 273.15, ξ, color = "black", alpha = 1/tsteps)
    
    for t in 1:tsteps
        
        mps = apply(time_step_mpo, mps; cutoff=cutoff)
        desired_base, desired_surface = get_desired_boundary_values(mps, mps_2, mps_3, mps_penultimate, Δx, γ, β)
        mps = set_boundary_conditions(mps, desired_base, desired_surface, base_mps, surface_mps; cutoff=cutoff)
        
        if t % plot_every == 0
            println("\r$t: max χ = $(maximum(linkdims(mps)))")
            plot!(T_air .* mps_to_list_inefficient(mps, N, n) .- 273.15, ξ, color = "black", alpha = t/tsteps)
        end

    end

    display(p)
end

"""
    get_initial_mps(initial_θ, n, sites)

Return a constant initial temperature profile as an MPS.
"""
function get_initial_mps(initial_θ, n, sites)
    return tci(_ -> initial_θ, 1:n, sites)
end

"""
    simulate()

Run the full advection–diffusion simulation.
"""
function simulate()

    tsteps = 1000
    plot_every = 100
    Δτ = 1e-3
    N = 4
    n = 2^N
    ξ = LinRange(0, 1, n)
    Pe = 5.0
    Δx = ξ[2] - ξ[1]
    Δx_inv = inv(Δx)
    γ = -0.35
    β = 0.5
    Ω = 0.0
    initial_θ = 1.0
    T_air = 223.15
    cutoff = 1e-14

    sites = siteinds("S=1/2", N)
    mps = get_initial_mps(initial_θ, n, sites)
    time_step_mpo = get_time_step_mpo(N, sites, Δx_inv, Pe, n, Ω, Δτ, ξ)
    
    pt(i) = reverse(digits(i - 1; base=2, pad=N) .+ 1)
    
    base_mps = productMPS(sites, pt(1))
    mps_2 = productMPS(sites, pt(2))
    mps_3 = productMPS(sites, pt(3))
    mps_penultimate = productMPS(sites, pt(n - 1))
    surface_mps = productMPS(sites, pt(n))
    
    @time time_evolution(tsteps, mps, time_step_mpo, mps_2, mps_3, mps_penultimate, Δx, γ, β, base_mps, surface_mps, T_air, ξ, N, n, plot_every; cutoff = cutoff)
    
end

simulate()
