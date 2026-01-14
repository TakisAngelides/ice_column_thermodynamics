using ITensors
using ITensorMPS
using QuanticsTCI
using TensorCrossInterpolation
using Plots
using BenchmarkTools
ITensors.disable_warn_order()

function tci(f::Function, x::AbstractArray{<:Number}, sites::Vector{ITensors.Index{Int64}}; tol::Float64 = 1e-14, f_type::DataType = Float64)

    qtt, ranks, errors = quanticscrossinterpolate(f_type, f, x; tolerance=tol) # right most site is least significant bit for x encoding
    mps = MPS(qtt.tci, sites = sites)
    return mps

end

function mps_to_list_inefficient(mps, N, n)

    res = []

    contracted_mps = ITensorMPS.contract(mps) # contract the bond indices

    for i in 0:n-1

        binary = reverse(digits(i; base=2, pad=N).+1) # right most site is least significant bit

        push!(res, contracted_mps[binary...])

    end

    return res

end

function get_triple_delta_op_list(sites)

    res = []

    for i in eachindex(sites)

        s = sites[i]
        d = delta(s, s', s'')
        push!(res, d)

    end

    return res

end

function boundary_conditions_MPO_unstable(sites, N, left_boundary_value, right_boundary_value)

    # For non-normalized states this is a numerically unstable method

    # The bond dimension 2 matrices of the MPS for the GHZ state |0...0> + |1...1> are A_sigma=1 = [[1 0], [0 0]], A_sigma=2 = [[0 0], [0 1]]
    # Our boundary conditions MPO needs to be Id - |0...0><0...0| - |1...1><1...1| so we will put a delta function_sigma,sigma' on the GHZ MPS
    # to convert it to an MPO

    # TODO: fix the description to include the new inputs that can specify the values on the boundaries beyond 0

    links = [Index(2, "Link,l=$kk") for kk in 1:N-1]
    mpo = MPO(sites) # This will store the MPO |0...0><0...0| + |1...1><1...1|

    for (site_idx, site) in enumerate(sites)

        if site_idx == 1

            s1, s2, l1 = prime(site), dag(site), links[1]
            mpo[1] = ITensor(l1, s1, s2)
            # A_sigma=1 = [1 0]
            mpo[site_idx][l1 => 1, s1 => 1, s2 => 1] = 1.0
            # A_sigma=2 = [0 1]
            mpo[site_idx][l1 => 2, s1 => 2, s2 => 2] = 1.0

        elseif site_idx == N

            s1, s2, l1 = prime(site), dag(site), links[N-1]
            mpo[N] = ITensor(l1, s1, s2)
            # A_sigma=1 = [1 0]
            mpo[site_idx][l1 => 1, s1 => 1, s2 => 1] = (1.0-left_boundary_value)
            # A_sigma=2 = [0 1]
            mpo[site_idx][l1 => 2, s1 => 2, s2 => 2] = (1.0-right_boundary_value)
            
        else

            s1, s2, l1, l2 = prime(site), dag(site), links[site_idx-1], links[site_idx]
            mpo[site_idx] = ITensor(l1, l2, s1, s2)
            # A_sigma=1 = [[1 0], [0 0]]
            mpo[site_idx][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1.0
            # A_sigma=2 = [[0 0], [0 1]]
            mpo[site_idx][l1 => 2, l2 => 2, s1 => 2, s2 => 2] = 1.0

        end

    end

    id_mpo = MPO(sites, "Id")

    final_mpo = id_mpo - mpo

    return final_mpo

end

function boundary_conditions_MPO_alternative_unstable(sites, N, left_boundary_value, right_boundary_value)

    # For non-normalized states this is a numerically unstable method

    # This can be used to specify which points need to be fixed to a given input value (would need a minor modification though to specify those points as inputs)

    mps_left_boundary = MPS(sites, fill("0", N))
    mps_right_boundary = MPS(sites, fill("1", N))

    mpo_left_boundary = outer(mps_left_boundary', mps_left_boundary)
    mpo_right_boundary = outer(mps_right_boundary', mps_right_boundary)

    return MPO(sites, "Id") - (1-left_boundary_value)*mpo_left_boundary - (1-right_boundary_value)*mpo_right_boundary

end

function mps_to_mpo(mps, sites)

    mpo = MPO(length(mps))
    triple_delta_op_list = get_triple_delta_op_list(sites) 

    for (i, A) in enumerate(mps)

        d = triple_delta_op_list[i]
        mpo[i] = A*d

    end

    setprime!(mpo, 0; :plev => 2)

    return mpo

end

function sr1(N; pbc = false)

    opsum = OpSum()

    for n in 1:N
        opsum_tmp = OpSum()
        opsum_tmp += "S+",n
        for i in n+1:N
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S-",1
        for i in 2:N
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp
    end

    return opsum

end

function sl1(N; pbc = false)

    opsum = OpSum()

    for n in 1:N           
        opsum_tmp = OpSum()
        opsum_tmp += "S-",n
        for i in n+1:N
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp            
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S+",1
        for i in 2:N
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp
    end

    return opsum

end

function get_three_point_symmetric_second_derivative_mpo(N, sites, Δx_inv; pbc = false)

    opsum = OpSum()
    opsum += -2,"Id",1  
    opsum += sr1(N; pbc = pbc) + sl1(N; pbc = pbc)
    return Δx_inv^2 * MPO(opsum, sites)

end

function get_diffusion_mpo(N, sites, Δx_inv; pbc = false)

    return get_three_point_symmetric_second_derivative_mpo(N, sites, Δx_inv; pbc = pbc)

end

function get_two_point_symmetric_first_derivative_mpo(N, sites, Δx_inv; pbc = false)
    
    opsum = sr1(N; pbc = pbc) - sl1(N; pbc = pbc)
    return 0.5*Δx_inv * MPO(opsum, sites)

end

function get_advection_mpo(N, sites, Δx_inv, Pe, n, ξ; pbc = false, cutoff = 0.0)

    mpo1 = get_two_point_symmetric_first_derivative_mpo(N, sites, Δx_inv; pbc = pbc)
    ω_mps = -Pe*tci(i -> ξ[Int(i)], 1:n, sites)
    mpo2 = mps_to_mpo(ω_mps, sites)
    return apply(mpo2, mpo1; cutoff = cutoff)

end

function set_boundary_conditions_way_1(mps, desired_base_value, desired_surface_value, base_mps, surface_mps; cutoff = 0.0)

    current_base_value = inner(mps, base_mps)
    current_surface_value = inner(mps, surface_mps)

    return add(mps, - (current_base_value - desired_base_value)*base_mps, - (current_surface_value - desired_surface_value)*surface_mps; cutoff = cutoff)

end

function set_boundary_conditions_way_2_unstable(mps, N, sites, desired_base_value, desired_surface_value; cutoff = 0.0)

    # For non-normalized mps states this is a numerically unstable method

    return apply(boundary_conditions_MPO_unstable(sites, N, desired_base_value, desired_surface_value), mps; cutoff = cutoff)

end

function set_boundary_conditions(mps, desired_base_value, desired_surface_value, base_mps, surface_mps; cutoff = 0.0)

    current_base_value = inner(mps, base_mps)
    current_surface_value = inner(mps, surface_mps)

    return add(mps, (-current_base_value + desired_base_value)*base_mps, (-current_surface_value + desired_surface_value)*surface_mps; cutoff = cutoff)

end

function get_time_step_mpo(N, sites, Δx_inv, Pe, n, Ω, Δτ, ξ; cutoff = 0.0)

    identity_mpo = MPO(sites, "Id")
    diffusion = Δτ*get_diffusion_mpo(N, sites, Δx_inv)
    advection = -Δτ*get_advection_mpo(N, sites, Δx_inv, Pe, n, ξ)
    strain = Δτ*Ω*identity_mpo

    return add(identity_mpo, diffusion, advection, strain; cutoff = cutoff)
    
end

function get_desired_boundary_values(mps, mps_2, mps_3, mps_penultimate, Δx, γ, β)

    val_2 = inner(mps, mps_2)
    val_3 = inner(mps, mps_3)
    val_penultimate = inner(mps, mps_penultimate)
    desired_base_value = (-2*Δx*γ + 4*val_2 - val_3)/3
    desired_surface_value = (Δx + β*val_penultimate)/(Δx + β)

    return desired_base_value, desired_surface_value

end

function time_evolution(tsteps, mps, time_step_mpo, mps_2, mps_3, mps_penultimate, Δx, γ, β, base_mps, surface_mps, T_air, ξ, N, n, plot_every)

    p1 = plot(xlabel = "Temperature (°C)", ylabel = "Depth (ξ)", legend = false)
    plot!(T_air .* mps_to_list_inefficient(mps, N, n) .- 273.15, ξ, color = :black)

    for t in 1:tsteps

        mps = apply(time_step_mpo, mps; cutoff = 0.0)
        desired_base_value, desired_surface_value = get_desired_boundary_values(mps, mps_2, mps_3, mps_penultimate, Δx, γ, β)
        mps = set_boundary_conditions(mps, desired_base_value, desired_surface_value, base_mps, surface_mps; cutoff = 0.0)
                
        if mod(t, plot_every) == 0

            println("\r $t: $(maximum(linkdims(mps)))")
            plot!((T_air .* mps_to_list_inefficient(mps, N, n)) .- 273.15, ξ, color = :black)

        end

    end

    display(p1)

end

function get_initial_mps(initial_θ, n, sites)

    return tci(i -> initial_θ, 1:n, sites)

end

function simulate()

    tsteps = 1000
    plot_every = 100
    Δτ = 1e-3
    N = 4
    n = 2^N
    ξ = LinRange(0, 1, n)
    Pe = 5.0
    Δx = ξ[2]-ξ[1]
    Δx_inv = Δx^(-1)
    γ = -0.35
    β = 0.5
    Ω = 0.0
    initial_θ = 1.0
    T_air = 223.15
    
    sites = siteinds("S=1/2", N)
    mps = get_initial_mps(initial_θ, n, sites)

    time_step_mpo = get_time_step_mpo(N, sites, Δx_inv, Pe, n, Ω, Δτ, ξ)

    pt_1 = reverse(digits(1-1; base = 2, pad = N).+1)
    pt_2 = reverse(digits(2-1; base = 2, pad = N).+1)
    pt_3 = reverse(digits(3-1; base = 2, pad = N).+1)
    pt_penultimate = reverse(digits(n-1-1; base = 2, pad = N).+1)
    pt_n = reverse(digits(n-1; base = 2, pad = N).+1)

    base_mps = productMPS(sites, pt_1)
    mps_2 = productMPS(sites, pt_2)
    mps_3 = productMPS(sites, pt_3)
    mps_penultimate = productMPS(sites, pt_penultimate)
    surface_mps = productMPS(sites, pt_n)

    time_evolution(tsteps, mps, time_step_mpo, mps_2, mps_3, mps_penultimate, Δx, γ, β, base_mps, surface_mps, T_air, ξ, N, n, plot_every)

end

# -------------------------------------------

# Parameters used in the tests below

# N = 8
# n = 2^N
# ξ = LinRange(0, 1, n)
# Pe = 5.0
# ω = - Pe .* ξ
# sites = siteinds("S=1/2", N)
# Δx = ξ[2]-ξ[1]
# Δx_inv = Δx^(-1)
# γ = -0.35
# β = 0.5

# -------------------------------------------

# Getting the tci of a vector and extracting the coefficients of certain basis states from the resulting mps

# ξ_mps = tci(i -> ξ[Int(i)], 1:n, sites) 

# mps1 = productMPS(sites, ["0", "0", "0", "1"])
# mpsN = productMPS(sites, ["1", "0", "0", "0"])

# println(inner(ξ_mps, mps1))
# println(inner(ξ_mps, mpsN))

# -------------------------------------------

# Testing new way to get the boundary mpo

# left_boundary_value, right_boundary_value = 1.3, 2.5
# @btime boundary_mpo_1 = get_boundary_conditions_MPO_unstable(sites, N, left_boundary_value, right_boundary_value) 
# @btime boundary_mpo_2 = get_boundary_conditions_MPO_alternative_unstable(sites, N, left_boundary_value, right_boundary_value)
# println(isapprox(boundary_mpo_1, boundary_mpo_2))

# -------------------------------------------

# Setting a desired value for the boundaries, testing two ways to do it

# mps = tci(i -> 1, 1:n, sites) 

# desired_left_boundary = 2.3
# desired_right_boundary = 1.5

# boundary_mpo = get_boundary_conditions_MPO_unstable(sites, N, desired_left_boundary, desired_right_boundary)

# mps1 = productMPS(sites, ["0", "0", "0", "0"])
# mpsN = productMPS(sites, ["1", "1", "1", "1"])

# function way_1()

#     current_left_boundary = inner(mps, mps1)
#     current_right_boundary = inner(mps, mpsN)

#     mps_boundaries = (current_left_boundary - desired_left_boundary)*mps1 + (current_right_boundary - desired_right_boundary)*mpsN

#     mps_new = mps - mps_boundaries

# end

# function way_2()

#     mps_new = apply(boundary_mpo, mps)

# end

# mps_new_1 = way_1()
# mps_new_2 = way_2() # this is faster with less malloc

# mps_list = mps_to_list_inefficient(mps, N, n)
# mps_new_1_list = mps_to_list_inefficient(mps_new_1, N, n)
# mps_new_2_list = mps_to_list_inefficient(mps_new_2, N, n)

# plot(mps_list)
# scatter!(mps_new_1_list, marker = :x)
# scatter!(mps_new_2_list, marker = :+)

# -------------------------------------------

# mps to mpo with triple delta function, multiplying ξ as an mpo to an mps

# mps0 = tci(i -> 1, 1:n, sites)

# mps = tci(i -> ξ[Int(i)], 1:n, sites)

# mpo = mps_to_mpo_old(mps, sites)

# mps = apply(mpo, mps0)

# mps_list = mps_to_list_inefficient(mps, N, n)

# plot(ξ, mps_list)

# -------------------------------------------

# Difffusion: 3 point symmetric second derivative mpo applied to the sin fn with fixed boundary values set to 0 by a boundary condition mpo

# x = LinRange(-π, π, n)
# Δx = x[2]-x[1]
# Δx_inv = Δx^(-1)

# mps = tci(i -> sin(i), x, sites)
# mps_list = mps_to_list_inefficient(mps, N, n)

# mpo = apply(get_boundary_conditions_MPO_unstable(sites, N, 0.0, 0.0), get_three_point_symmetric_second_derivative_mpo(N, sites, Δx_inv))
# mps_new = apply(mpo, mps)

# mps_new_list = mps_to_list_inefficient(mps_new, N, n)

# plot(x, mps_list)
# plot!(x, mps_new_list)

# -------------------------------------------

# Advection: 2 point symmetric first derivative multiplied by the ω mpo

# advection_mpo = get_advection_mpo(N, sites, Δx_inv, ω, n)

# -------------------------------------------

# Computing the values needed for the boundary conditions

# mps = tci(i -> sin(i), ξ, sites) 
# mps = apply(get_diffusion_mpo(N, sites, Δx_inv), mps)
# mps_list = mps_to_list_inefficient(mps, N, n)

# pt_1 = fill(1, N)
# pt_2 = reverse(digits(2-1; base = 2, pad = N).+1)
# pt_3 = reverse(digits(3-1; base = 2, pad = N).+1)
# pt_penultimate = reverse(digits(n-1-1; base = 2, pad = N).+1)
# pt_n = fill(2, N)

# base_mps = productMPS(sites, pt_1)
# mps_2 = productMPS(sites, pt_2)
# mps_3 = productMPS(sites, pt_3)
# mps_penultimate = productMPS(sites, pt_penultimate)
# surface_mps = productMPS(sites, pt_n)

# val_2 = inner(mps, mps_2)
# val_3 = inner(mps, mps_3)
# val_penultimate = inner(mps, mps_penultimate)
# desired_base_value = (2*Δx/3)*(γ + (2/Δx)*val_2 - (1/2*Δx)*val_3)
# desired_surface_value = (Δx + β*val_penultimate)/(Δx + β)

# mps_way_1 = set_boundary_conditions_way_1(mps, desired_base_value, desired_surface_value, base_mps, surface_mps)
# mps_way_2 = set_boundary_conditions_way_2_unstable(mps, N, sites, desired_base_value, desired_surface_value)

# mps_way_1_list = mps_to_list_inefficient(mps_way_1, N, n)
# mps_way_2_list = mps_to_list_inefficient(mps_way_2, N, n)

# println(mps_way_1_list[1], " ", desired_base_value)
# println(mps_way_1_list[end], " ", desired_surface_value)
# println()
# println(mps_way_2_list[1], " ", desired_base_value)
# println(mps_way_2_list[end], " ", desired_surface_value)

# plot(ξ, mps_list)
# plot!(ξ, mps_way_1_list)
# plot!(ξ, mps_way_2_list)

# -------------------------------------------

# Testing the boundary_conditions_MPO_alternative_unstable function part by part
# These methods boundary_conditions_MPO are unstable for non-normalized states but one can keep the state normalized
# and keep track of the normalization factors at each time step to recover the correct state when plotting or using for other post-simulation analysis
# TODO: think about this: This might be a good idea to do regardless, to keep numerics stable!

# mps = tci(i -> sin(i), ξ, sites) 
# mps = apply(get_diffusion_mpo(N, sites, Δx_inv), mps)

# pt_1 = fill(1, N)
# pt_2 = reverse(digits(2-1; base = 2, pad = N).+1)
# pt_3 = reverse(digits(3-1; base = 2, pad = N).+1)
# pt_penultimate = reverse(digits(n-1-1; base = 2, pad = N).+1)
# pt_n = fill(2, N)

# base_mps = productMPS(sites, pt_1)
# mps_2 = productMPS(sites, pt_2)
# mps_3 = productMPS(sites, pt_3)
# mps_penultimate = productMPS(sites, pt_penultimate)
# surface_mps = productMPS(sites, pt_n)

# val_2 = inner(mps, mps_2)
# val_3 = inner(mps, mps_3)
# val_penultimate = inner(mps, mps_penultimate)
# desired_base_value = (2*Δx/3)*(γ + (2/Δx)*val_2 - (1/2*Δx)*val_3)
# desired_surface_value = (Δx + β*val_penultimate)/(Δx + β)

# mps_left_boundary = base_mps
# mps_right_boundary = surface_mps

# mpo_left_boundary = outer(mps_left_boundary', mps_left_boundary)
# mpo_right_boundary = outer(mps_right_boundary', mps_right_boundary)

# a = MPO(sites, "Id") 
# b = - (1-desired_base_value)*mpo_left_boundary 
# c = - (1-desired_surface_value)*mpo_right_boundary

# # Below it shows why the boundary_condition_MPO methods can be numerically unstable for non-normalized mps states
# println(inner(mps', a, mps))
# println(inner(mps', b, mps))
# println(inner(mps', c, mps))

# -------------------------------------------

# Final test

simulate()

# -------------------------------------------

