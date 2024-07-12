using SARPOMDP
using POMDPs
using POMDPTools
using Test
using StaticArrays
using LinearAlgebra

#New Tests
# rewarddist = [-3.08638     1.04508  -38.9812     6.39193    7.2648     5.96755     9.32665   -9.62812   -0.114036    7.38693      3.39033   -5.17863  -12.7841;
# -8.50139     2.3827   -30.2106   -74.7224   -33.9783    -3.63283    -4.73628   -6.19297   -4.34958    -6.13309    -36.2926    -7.35857    0.417866;
# -12.0669      7.54123  -22.8483   -47.2838   -53.8302   -25.5759    -36.2189    -4.93866   -4.9971    -12.1572     -15.8788   -23.9603   -15.3152;
# -11.2335     -5.6023   -32.8484   -58.316    -35.6942   -40.4131    -80.1317     4.50919    0.302756   -0.238148     7.76839    2.78539   39.5031;
# -7.1815     -5.4408   -26.9287   -61.4094   -50.8607   -36.6915    -17.6218    -7.06897    0.190177   -0.0721576    8.61714   41.2753    69.0911;
# 2.89205   -14.3239   -87.9894   -64.7747   -68.2573   -45.2064    -62.6445   -59.5357   -32.3136    -52.6505       7.37878   28.6342    31.5646;
# -0.741237  -15.9554   -83.2767   -69.0195   -82.2122   -45.17      -21.2148    -8.11823    8.68415    16.4957       8.32323   16.4972    14.3504;
# 6.44794     7.12914  -88.2391   -68.5625   -74.8771   -21.2487    -11.021     -2.84843   -5.2219      1.83158     13.7386    -4.35878   17.0571;
# 11.0371      2.88455  -59.5524   -35.7124   -35.061     -9.27868    -8.9189    -8.82431  -51.8993      9.63887    -13.3222   -21.0979   -14.339;
# 7.90618     3.18679  -61.3164   -70.7954   -35.6381    -5.88295   -51.0393   -31.984    -49.4399    -25.144      -10.1865   -33.8935   -23.4304;
# 4.06703     9.92574   -9.96883  -48.9633   -55.4547   -29.8576    -37.7918   -49.4194   -25.8577    -34.64        15.5699     6.30979    8.75206;
# 6.93365     2.50252    9.63002    5.05564   -1.67295  -46.427     -69.802    -58.4468   -48.2396     -9.09721     20.3898    11.296      1.68226;
# 6.69843     0.88624    6.50904    7.60138  -15.8097   -55.7776    -39.8913   -56.2164     4.2347      2.45662      4.0834     4.77346    0.373309;
# 4.5434      1.84961    5.05996    1.71024  -16.2119   -70.8986    -68.3217   -42.1496    13.7424     14.7261       1.78606    8.92938    0.35768;
# 5.93137     2.38837    5.00692    2.17936   -6.58787  -48.8138    -27.0167   -10.6387     1.24938    21.9765       4.26369    6.6729     2.1039;
# 6.35598     1.425      2.92712    4.96801   13.0207    -0.589068  -15.8313    10.7642    16.1614     15.3144       3.59158    7.8918     9.1199]
locs = [[1,4],[4,1],[3,3]]
vals = [2.0,2.0,1.0]

rewarddist = zeros(5,5)
for (i,l) in enumerate(locs)
    rewarddist[l...] = vals[i]
end


#rewarddist = smallreward
#rewarddist = hallway
#rewarddist = load("rewardmat.jld2","rewarddist")
#rewarddist = rewarddist .+ abs(minimum(rewarddist)) .+ 0.01
rewarddist = abs.(rewarddist)
mapsize = reverse(size(rewarddist)) #(13,16)
maxbatt = Int(norm(mapsize,1)*2)
target = [4,4]
sinit = SAR_State([1,1], target, maxbatt)#rand(initialstate(msim))

pomdp = SAR_POMDP(sinit, 
                    map_size=mapsize, 
                    rewarddist=rewarddist, 
                    maxbatt=maxbatt)
pomdp2 = SAR_POMDP()

@testset "Indices" begin
    x = 0
    for state in ordered_states(pomdp)
        x+=1 
        @test x == stateindex(pomdp,state)
    end
    x = 0
    for action in ordered_actions(pomdp)
        x+=1 
        @test x == actionindex(pomdp,action)
    end
    x = 0
    for observation in ordered_observations(pomdp)
        x+=1 
        @test x == obsindex(pomdp,observation)
    end    
end

@testset "Reward" begin
    term_plus_one = 8
    for (i,l) in enumerate(locs)
        @test reward(pomdp,SAR_State(l,target,99),:stay,SAR_State(l,target,99)) == vals[i]
    end
    @test reward(pomdp,SAR_State(target,target,99),:stay,SAR_State(target,target,99)) == pomdp.r_find
    @test reward(pomdp,SAR_State(target,target,term_plus_one),:stay,SAR_State(target,target,term_plus_one)) == pomdp.r_find
end

@testset "Terminal" begin
    term_plus_one = 8
    @test isterminal(pomdp,SAR_State([-1,-1],[-1,-1],-1)) == true
    @test isterminal(pomdp,SAR_State(target,target,term_plus_one)) == false
    if pomdp.terminate_on_find == true
        @test isterminal(pomdp,support(transition(pomdp,SAR_State(target,target,term_plus_one-1),:up))[1]) == true
        @test isterminal(pomdp,support(transition(pomdp,SAR_State(target,target,term_plus_one),:up))[1]) == true
    end
    if pomdp.auto_home == true
        @show support(transition(pomdp,SAR_State(target+SVector{2}([1,0]),target,term_plus_one-1),:up))[1]
        @test isterminal(pomdp,support(transition(pomdp,SAR_State(target+SVector{2}([1,0]),target,term_plus_one-1),:up))[1]) == true
    end
end

@testset "No Autohome and No Terminate on Find" begin
    term_plus_one = 8
    rewarddist2 = abs.(rewarddist)
    mapsize2 = reverse(size(rewarddist2)) #(13,16)
    maxbatt2 = Int(norm(mapsize2,1)*2)
    target2 = [4,4]
    sinit2 = SAR_State([1,1], target2, maxbatt2)#rand(initialstate(msim))
    pomdp2 = SAR_POMDP(sinit2, 
                    map_size=mapsize2, 
                    rewarddist=rewarddist2, 
                    maxbatt=maxbatt2,auto_home=false,terminate_on_find=false)
    if pomdp2.terminate_on_find == false && pomdp2.auto_home == false
        @test isterminal(pomdp2,support(transition(pomdp2,SAR_State(target2,target2,term_plus_one-1),:up))[1]) == false
        @test isterminal(pomdp2,support(transition(pomdp2,SAR_State(pomdp2.robot_init+SVector{2}([1,1]),target2,term_plus_one-1),:up))[1]) == false
        @test isterminal(pomdp2,support(transition(pomdp2,SAR_State(pomdp2.robot_init+SVector{2}([1,0]),target2,term_plus_one-1),:left))[1]) == true
        @test isterminal(pomdp2,support(transition(pomdp2,SAR_State(pomdp2.robot_init+SVector{2}([0,1]),target2,term_plus_one-1),:down))[1]) == true
    end
    @test reward(pomdp2,SAR_State(target,target,term_plus_one-1),:stay) == -1e10
end

@testset "Initial State Tests" begin
    init_ro = [1,1]
    target = [4,4]
    maxbatt = 20
    rew_locs = SVector{2,Int}.([[1,4],[4,1],[3,3]])
    rew_vals = [2.0,2.0,1.0]
    r_find = 1000.0
    map_size=(5,5)
    auto_home=true
    terminate_on_find=true
    is = POMDPTools.Uniform(SAR_State(init_ro, SVector(x, y), maxbatt) for x in 1:map_size[1], y in 1:map_size[2])

    t_pomdp = SAR_POMDP(
    init_ro=init_ro,
    target = target,
    maxbatt = maxbatt,
    rew_locs = rew_locs,
    rew_vals = rew_vals,
    r_find = r_find,
    map_size = map_size,
    auto_home = auto_home,
    terminate_on_find = terminate_on_find,
    initial_state_dist=is)

    @test length(support(is)) == length(support(initialstate(t_pomdp)))
    @test all([state ∈ support(initialstate(t_pomdp)) for state in support(is)])
    @test [pdf(is,state) for state in support(is)] == [pdf(initialstate(t_pomdp),state) for state in support(initialstate(t_pomdp))]

    up = DiscreteUpdater(t_pomdp)

    @test initialize_belief(up,is) == initialize_belief(up,initialstate(t_pomdp))

    init_ro = [1,1]
    target = [5,5]
    maxbatt = 15
    rew_locs = SVector{2,Int}.([[1,4],[4,1],[3,3]])
    rew_vals = [3.0,3.0,3.0]
    r_find = 100.0
    map_size=(5,5)
    auto_home=true
    terminate_on_find=true
    is = POMDPTools.Uniform(SAR_State(init_ro, SVector(x, y), maxbatt) for x in 1:map_size[1], y in 1:map_size[2])

    t_pomdp = SAR_POMDP(
    init_ro=init_ro,
    target = target,
    maxbatt = maxbatt,
    rew_locs = rew_locs,
    rew_vals = rew_vals,
    r_find = r_find,
    map_size = map_size,
    auto_home = auto_home,
    terminate_on_find = terminate_on_find,
    initial_state_dist=is)

    @test length(support(is)) == length(support(initialstate(t_pomdp)))
    @test all([state ∈ support(initialstate(t_pomdp)) for state in support(is)])
    @test [pdf(is,state) for state in support(is)] == [pdf(initialstate(t_pomdp),state) for state in support(initialstate(t_pomdp))]

    up = DiscreteUpdater(t_pomdp)

    @test initialize_belief(up,is) == initialize_belief(up,initialstate(t_pomdp))

    init_ro = [1,1]
    target = [5,5]
    maxbatt = 15
    rew_locs = SVector{2,Int}.([[1,4],[4,1],[3,3]])
    rew_vals = [3.0,3.0,3.0]
    r_find = 100.0
    map_size=(5,5)
    auto_home=true
    terminate_on_find=true
    @show [SAR_State(init_ro, SVector(x, y), maxbatt) for x in 1:map_size[1], y in 1:map_size[2]]
    istates = support(initialstate(t_pomdp))
    is = SparseCat(istates,[0.5,0.5,zeros(23)...])

    t_pomdp = SAR_POMDP(
    init_ro=init_ro,
    target = target,
    maxbatt = maxbatt,
    rew_locs = rew_locs,
    rew_vals = rew_vals,
    r_find = r_find,
    map_size = map_size,
    auto_home = auto_home,
    terminate_on_find = terminate_on_find,
    initial_state_dist=is)

    @test length(support(is)) == length(support(initialstate(t_pomdp)))
    @test all([state ∈ support(initialstate(t_pomdp)) for state in support(is)])
    @test [pdf(is,state) for state in support(is)] == [pdf(initialstate(t_pomdp),state) for state in support(initialstate(t_pomdp))]

    up = DiscreteUpdater(t_pomdp)

    @test initialize_belief(up,is) == initialize_belief(up,initialstate(t_pomdp))
end

# @testset "Consistency Tests" begin
#     @test has_consistent_distributions(pomdp)
# end

# rewarddist = [3.0 3.0;
#               3.0 3.0]
# running_cost = -1.0
# ns = length(rewarddist)
# mapsize = (2,2)
# sinit = SAR_State(SVector{2}([1,1]),SVector{2}([3,3]),trues(prod(mapsize)))

# # m = RewardPOMDP(sinit, size=mapsize, rewarddist=rewarddist)
# s = sinit
# # sp = RewardState([2,1],[3,3],trues(prod(mapsize)))
# a = :right
# function getind(pos)
#     correct_ind = reverse(pos)
#     xind = m.size[2]+1 - correct_ind[1]
#     inds = [xind, correct_ind[2]] 
#     return inds
# end

# @testset "reward" begin
#     running_cost = -1.0
#     @test reward(m, s, a, sp) == running_cost + rewarddist[getind(sp.robot)...]
# end

# #sts = unique(getfield.(states(m), :robot))[1:end-1]
# acs = (:right, :up, :left, :down, :right)
# acs_len = length(acs)
# rtotal = 0.0
# r = 0.0
# @testset "rtotal" begin
#     function rtotaltest(rtotal,s,acs)
#         for i ∈ 1:acs_len
#             a = acs[i]
#             sp = rand(transition(m, s, a))
#             r = reward(m, s, a, sp)
#             rtotal += r
#             #println(reward(m, s, a, sp), ", a:", a)
#             println("s: ", s.robot, " -- ", s.visited, " | sp: ", sp.robot, " -- ", sp.visited, " | a: ", a, " | r: ", r)
#             s = sp
#         end
#         return rtotal
#     end
#     @test rtotaltest(rtotal,s,acs) == sum(rewarddist) + ns*running_cost - rewarddist[getind(sinit.robot)...]
# end