
# # Step 0: Setup the environment
# If you haven't finished with this step, please do it.

# 1. Download and Install VScode and Julia
# (Windows) A suggestion is that Do Not Install in path C:

# Download  and Install Julia and configure the environment variable

# Download and Install VScode

# After installing VScode and Julia，install VScode extention "julia"

# Configure the extention of julia

# 2. Run 

# set_up_environment.jl

# Ctrl+A, Ctrl+Enter

# 3. Compiled the packages

# Ctrl+shift+B time consuming. Wait

# Step 1: Loading the packages. Do not need to change

using DifferentialEquations
using LinearAlgebra, DiffEqSensitivity, Optim
using Flux: flatten, params
using DiffEqFlux, Flux
using Plots
using Flux:train!
using GalacticOptim
using Optim
using Turing, Distributions
#using MCMCChains, StatsPlots
using CSV, DataFrames
using SymbolicRegression
using Random
Random.seed!(14);


# Step 2: Preprocessing the data.

# Study [Introduction · DataFrames.jl](https://dataframes.juliadata.org/stable/)

# [Tutorial · Plots](https://docs.juliaplots.org/latest/tutorial/)

# [DifferentialEquations.jl: Scientific Machine Learning (SciML) Enabled Simulation and Estimation · DifferentialEquations.jl](https://diffeq.sciml.ai/stable/)

# Here I generate data. 这里我是生成数据，你们需要使用实际数据。


# Solve a differential equation.
function SIR(du,u,p,t)
    β, γ, α, k = p
   S, I = u
   du[1] = - β*S*exp(-α*I)*I^k
   du[2] = β*S*exp(-α*I)*I^k - γ*I
end
u_0 = [10, 1]
p_data = [0.3,0.1,0.01,1.5]
tspan_data = (0.0, 3.0)
prob_data = ODEProblem(SIR,u_0,tspan_data,p_data)
data_solve = solve(prob_data, Tsit5(),abstol=1e-12, reltol=1e-12, saveat = 0.1)
data = Array(data_solve)
tspan_predict = (0.0, 4.0)
prob_predict = ODEProblem(SIR,u_0,tspan_predict,p_data)
test_data = solve(prob_predict, Tsit5(),abstol=1e-12, reltol=1e-12, saveat = 0.1)

# plot  the solution
scatter(data_solve.t, data[1,:],label = "Train S")
scatter!(data_solve.t, data[2,:],label = "Train I")
tspan_predict = (0.0, 4.0)
prob_predict = ODEProblem(SIR,u_0,tspan_predict,p_data)
test_data = solve(prob_predict, Tsit5(),abstol=1e-12, reltol=1e-12, saveat = 0.1)
plot!(test_data, label=["Test S" "Test I"])

# 
# Step 3: Set up the universal differential equation model.

# Study: [DiffEqFlux.jl: Generalized Physics-Informed and Scientific Machine Learning (SciML) · DiffEqFlux.jl](https://diffeqflux.sciml.ai/dev/)
# 
ann_node = FastChain(FastDense(1, 32, tanh),FastDense(32, 1))
p = Float64.(initial_params(ann_node))
p_know = 0.1
function SIR_nn(du,u,p,t)
   γ = p_know
   S, I = u
   du[1] =  - S*ann_node(I,p)[1]
   du[2] = S*ann_node(I,p)[1] - γ*I
   [du[1],du[2]]
end
prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, p)


# Step 4: Train the universal differential equation model.

function train(θ)
    Array(concrete_solve(prob_nn, Vern7(), u_0, θ, saveat = 0.1,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
 end
 function loss(θ)
    pred = train(θ)
    sum(abs2, (data .- pred)), pred # + 1e-5*sum(sum.(abs, params(ann)))
 end
 #println(loss(p))
 const losses = []
 callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
 end
 res1_node = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
 res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 300)


# 
# Step 5: Plot  the traning results
# 
scatter(data_solve.t, data[1,:],label = "Train S")
scatter!(data_solve.t, data[2,:],label = "Train I")
plot!(test_data, label=["Test S" "Test I"])
prob_nn2 = ODEProblem(SIR_nn, u_0, tspan_predict, res2_node.minimizer)
s_nn = solve(prob_nn2, Tsit5(), saveat = 0.1)
plot!(s_nn,label=["Learned S" "Learned I"])