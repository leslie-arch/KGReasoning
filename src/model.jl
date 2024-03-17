#!/usr/bin/env julia

module KGModel

#using Base: offset_if_vec
#using Zygote: AbstractFFTs
using Base: batch_size_err_str, substrides
using SplitApplyCombine

export Identity, OffsetIntersection, CenterIntersection, BetaIntersection,
    BetaProjection, Regularizer, KGReasoning, KGRConfig, train_step, build_KGReasoning

using Flux;
using Zygote;
#using Statistics;
using Distributions;
using Distances;
#using Divergences;
using KLDivergences;
using MLUtils;

include("utils.jl")

function Identity(x)
    return x;
end
#=
function normDims(x, p::Int = 2; dims = 1)
    sum(abs.(x) .^ p; dims).^(1 / p)
end
=#
struct OffsetIntersection
    #dim::Int
    layer1::Flux.Dense
    layer2::Flux.Dense
end

OffsetIntersection(dim::Int) = OffsetIntersection(Flux.Dense(dim => dim, init = Flux.glorot_uniform),
                                                  Flux.Dense(dim => dim, init = Flux.glorot_uniform))

#Function-like Object
function (m::OffsetIntersection)(embeddings)
    @show "OffsetIntersection size embeddings $(size(embeddings))"
    layer1_act = Flux.relu(m.layer1(embeddings))
    layer1_mean = mean(layer1_act, dims=ndims(layer1_act))
    layer1_out = dropdims(layer1_mean, dims=ndims(layer1_mean))
    gate = Flux.sigmoid(m.layer2(layer1_out))

    offset = minimum(embeddings, dims=ndims(embeddings))
    offset_drop = dropdims(offset, dims=ndims(embeddings))

    return offset_drop .* gate
end

Flux.@functor OffsetIntersection

struct CenterIntersection
    #dim::Int
    layer1::Flux.Dense
    layer2::Flux.Dense
end

CenterIntersection(dim::Int) = CenterIntersection(Flux.Dense(dim => dim, init = Flux.glorot_uniform),
                                                  Flux.Dense(dim => dim, init = Flux.glorot_uniform))

function (m::CenterIntersection)(embeddings)
    layer1_act = Flux.relu(m.layer1(embeddings)) # ( dim, num_conj)
    layer2_out = m.layer2(layer1_act)
    attention = Flux.softmax(layer2_out, dims=ndims(layer2_out)) # (dim, num_conj, )
    println("CenterIntersection: size attention $(size(attention)) size embedding $(size(embeddings))")

    att_embeddings = attention .* embeddings
    embedding = sum(att_embeddings, dims=ndims(att_embeddings))

    return dropdims(embedding, dims=ndims(embedding))
end

Flux.@functor CenterIntersection

struct BetaIntersection
    layer1::Flux.Dense
    layer2::Flux.Dense
end

BetaIntersection(dim::Int) = BetaIntersection(Flux.Dense(2 * dim => 2 * dim, init = Flux.glorot_uniform),
                                              Flux.Dense(2 * dim => dim, init = Flux.glorot_uniform))

function (m::BetaIntersection)(alpha_embeddings, beta_embeddings)
    #println("BetaIntersection: size alpha $(size(alpha_embeddings)), beta $(size(beta_embeddings))")
    all_embeddings = cat(alpha_embeddings, beta_embeddings, dims = 1)
    #println("BetaIntersection: size all $(size(all_embeddings))")
    layer1_relu = Flux.relu(m.layer1(all_embeddings))
    layer2_out = m.layer2(layer1_relu)
    attention = Flux.softmax(layer2_out, dims=ndims(layer2_out))

    #println("BetaIntersection: size attention $(size(attention))")
    alpha_embedding = dropdims(sum(attention .* alpha_embeddings, dims=ndims(alpha_embeddings)),
                               dims=ndims(alpha_embeddings))
    beta_embedding = dropdims(sum(attention .* beta_embeddings, dims=ndims(beta_embeddings)),
                              dims=ndims(beta_embeddings))

    #println("BetaIntersection: size alpha $(size(alpha_embedding)) beta $(size(beta_embedding))")
    return alpha_embedding, beta_embedding
end

Flux.@functor BetaIntersection

struct Regularizer
    base_add::AbstractFloat
    min_val::AbstractFloat
    max_val::AbstractFloat
end

function (m::Regularizer)(entity_embedding)
    #println("Regularizer: embedding size [$(size(entity_embedding))]")
    #println("Regularizer: $(m.base_add) $(m.min_val) $(m.max_val)")
    #println("Regularizer: min max $(minimum(entity_embedding)) $(maximum(entity_embedding))")
    #clamp is for vector, add [.] to broadcast
    embedding = clamp.(entity_embedding .+ m.base_add, m.min_val, m.max_val)
    #println("Regularizer: clamp min max $(minimum(embedding)) $(maximum(embedding))")
    return embedding
end

#Flux.@functor Regularizer

struct BetaProjection
    layers::Dict{Symbol, Flux.Dense}
    projection_regularizer
end

function Base.setproperty!(m::BetaProjection, property::Symbol, value)
    if property == :layers || property == :projection_regularizer
        setfield(m, property, value)
    else
        getfield(m, :layers)[property] = value
    end
end

function Base.getproperty(m::BetaProjection, property::Symbol)
    if property == :layers || property == :projection_regularizer
        return getfield(m, property)
    else
        return getfield(m, :layers)[property]
    end
end

function Base.propertynames(m::BetaProjection, private = false)
    return keys(getproperty(m, :layers))
end

function BetaProjection(entity_dim, relation_dim, hidden_dim, num_layers, projection_regularizer, init=Flux.glorot_uniform)
    layers = Dict{Symbol, Flux.Dense}()
    layers[:layer1] = Flux.f64(Flux.Dense((entity_dim + relation_dim) => hidden_dim, init = init))
    layers[:layer0]  = Flux.f64(Flux.Dense(hidden_dim => entity_dim, init = init))
    for nl in range(2, num_layers)
        layers[Symbol("layer$(nl)")] = Flux.f64(Flux.Dense(hidden_dim => hidden_dim, init = init))
    end

    @eval Flux.trainable(m::BetaProjection) = (values(m.layers))

    return BetaProjection(layers, projection_regularizer)
end

function (m::BetaProjection)(e_embedding, r_embedding)
    x = cat(e_embedding, r_embedding, dims=1)

    for n in range(1,length(m.layers) - 1)
        x = Flux.relu(getproperty(m, Symbol("layer$(n)"))(x))
    end
    x = getproperty(m, :layer0)(x)
    x = m.projection_regularizer(x)

    return x
end

Flux.@functor BetaProjection

Base.@kwdef struct KGRConfig
    nentity::Int
    nrelation::Int
    geo::String
    use_cuda::Bool

    batch_entity_range #TODO type and initialize

    gamma::AbstractFloat
    epsilon::AbstractFloat
    hidden_dim::Int
    entity_dim::Int
    relation_dim::Int
    query_name_dict::Dict{Tuple, String}

    ######################################
    box_activation::Union{Function, Missing} = missing
    box_center::Union{Float64, Missing} = missing

    beta_hidden_dim::Int
    beta_num_layers::Int
    beta_entity_regularizer::Union{Regularizer, Missing} = missing
    beta_projection_regularizer::Union{Regularizer, Missing} = missing
end

function KGRConfig(nentity, nrelation, hidden_dim, gamma, query_name_dict, geo,
                   box_mode=nothing, beta_mode=nothing,  cuda = false, batch = 1)
    epsilon = 2.0
    entity_dim = hidden_dim
    relation_dim = hidden_dim
    batch_entity_range = repeat(convert.(Float32, range(0, nentity - 1)), 1, batch)

    local box_activation = missing
    activation, box_center = box_mode
    local beta_hidden_dim, beta_num_layers = (0, 0)
    beta_entity_regularizer, beta_projection_regularizer = (missing, missing)

    println("KGRConfig: activation $(activation)")
    if geo == "box"
        if activation == nothing
            box_activation = Identity;
        elseif activation == "relu"
            box_activation = Flux.relu;
        elseif activation == "softplus"
            box_activation = Flux.softplus;
        end
    elseif geo == "beta"
        beta_hidden_dim, beta_num_layers = beta_mode
        #println("KGRConfig: beta_mode $(beta_mode) beta_hidden: $(beta_hidden_dim) num_layers: $(beta_num_layers)")
        # make sure the parameters of beta embeddings are positive
        beta_entity_regularizer = Regularizer(1, 0.05, 1e9)
        # make sure the parameters of beta embeddings after relation projection are positive
        beta_projection_regularizer = Regularizer(1, 0.05, 1e9)
        #println("KGRConfig: regularizer: $(beta_entity_regularizer)")
        #println("KGRConfig: regularizer: $(beta_projection_regularizer)")
    end
    #println("box_mode: activation: $(activation) - $(box_activation) center: $(box_center)")
    return KGRConfig(nentity, nrelation, geo, cuda,
                     batch_entity_range,
                     gamma, epsilon,
                     hidden_dim, entity_dim, relation_dim,
                     query_name_dict,
                     box_activation, box_center,
                     beta_hidden_dim, beta_num_layers,
                     beta_entity_regularizer,
                     beta_projection_regularizer)
end

Base.@kwdef struct KGReasoning
    gamma::Vector{Float64}

    embedding_range::Vector{Float64}

    entity_embedding::Matrix{Float64}
    relation_embedding::Matrix{Float64}
    offset_embedding::Union{Missing, Matrix{Float64}} = missing

    center_net::Union{BetaIntersection, CenterIntersection, Missing} = missing
    offset_net::Union{OffsetIntersection, Missing} = missing
    projection_net::Union{BetaProjection, Missing} = missing
end

Flux.@functor KGReasoning

function build_KGReasoning(conf, init = Flux.glorot_uniform)
    gamma = [conf.gamma]
    embedding_range = (gamma .+ conf.epsilon) ./ conf.hidden_dim
    #println("embedding_range: $(embedding_range) type: $(typeof(embedding_range)) gamma $(gamma) $(typeof(gamma))")
    local entity_embedding
    if conf.geo == "box"
        entity_embedding = init(conf.entity_dim, conf.nentity) # centor for entities
        #entity_embedding = init(conf.nentity, conf.entity_dim)
    elseif conf.geo == "vec"
        entity_embedding = init(conf.entity_dim, conf.nentity)
        #entity_embedding = init(conf.nentity, conf.entity_dim)
    elseif conf.geo == "beta"
        #entity_embedding = init(conf.nentity, conf.entity_dim * 2)
        entity_embedding = init(conf.entity_dim * 2, conf.nentity)
    end

    relation_embedding = init(conf.relation_dim, conf.nrelation)

    local offset_embedding, center_net, offset_net, projection_net = Vector{Missing}(undef, 4)
    if conf.geo == "box"
        offset_embedding = init(conf.entity_dim, conf.nrelation)
        println("offset_embedding: $(typeof(offset_embedding))")

        center_net = CenterIntersection(conf.entity_dim)
        offset_net = OffsetIntersection(conf.entity_dim)
    elseif conf.geo == "vec"
        center_net = CenterIntersection(conf.entity_dim)
    elseif conf.geo == "beta"
        center_net = BetaIntersection(conf.entity_dim)
        projection_net = BetaProjection(conf.entity_dim * 2,
                                        conf.relation_dim,
                                        conf.beta_hidden_dim,
                                        conf.beta_num_layers,
                                        conf.beta_projection_regularizer)
    end

    model =  KGReasoning(gamma, embedding_range,
                         entity_embedding, relation_embedding, offset_embedding,
                         center_net, offset_net, projection_net);
    return model
end

function (m::KGReasoning)(conf::KGRConfig, positive_sample, negative_sample, subsampling_weight,
                          batch_distribution_args...)

    println("KGReasoning: geo $(conf.geo)")
    if conf.geo == "beta"
        all_idxs, all_alpha_embeddings, all_beta_embeddings,
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = batch_distribution_args

        return forward_beta(m, conf, positive_sample, negative_sample, subsampling_weight,
                            all_idxs, all_alpha_embeddings, all_beta_embeddings,
                            all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings)
    elseif conf.geo == "box"
        all_idxs, all_center_embeddings, all_offset_embeddings,
        all_union_idxs, all_union_center_embeddings, all_union_offset_embeddings = batch_distribution_args

        return forward_box(m, conf, positive_sample, negative_sample, subsampling_weight,
                           all_idxs, all_center_embeddings, all_offset_embeddings,
                           all_union_idxs, all_union_center_embeddings, all_union_offset_embeddings)
    elseif conf.geo == "vec"
        all_idxs, all_center_embeddings, all_union_idxs, all_union_center_embeddings = batch_distribution_args

        return forward_vec(m, conf, positive_sample, negative_sample, subsampling_weight,
                           all_idxs, all_center_embeddings, all_union_idxs, all_union_center_embeddings)
    end
end

####
# embed a batch of queries with same structure using Query2box
# queries: a flattened batch of queries
####
function embed_query_box(m::KGReasoning, conf::KGRConfig, queries, query_structure, idx)

    all_relation_flag = true
     # whether the current query tree has mfferged to one branch and only need to do relation traversal,
     # e.g., path queries or conjunctive queries after the intersection
    for r in last(query_structure)
        if !(r in ["r", "n"])
            all_relation_flag = false
            break
        end
    end

    if all_relation_flag
        if query_structure[1] == "e"
            embedding = selectdim(m.entity_embedding,
                                  ndims(m.entity_embedding),
                                  queries[idx, :] .+ 1)
            #embedding = torch.index_select(m.entity_embedding, dim=0, index=queries[:, idx])
            offset_embedding = zeros(size(embedding))
            if conf.use_cuda
                offset_embedding = zeros(size(embedding)) .|> gpu
            end
            idx += 1
        else
            embedding, offset_embedding, idx = embed_query_box(m, conf, queries, query_structure[1], idx)
        end

        for i in range(1, length(last(query_structure)))
            if last(query_structure)[i] == "n"
                @assert false "box cannot handle queries with negation"
            else
                r_embedding = selectdim(m.relation_embedding,
                                        ndims(m.relation_embedding),
                                        queries[idx, :] .+ 1)
                #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                r_offset_embedding = selectdim(m.offset_embedding,
                                               ndims(m.offset_embedding),
                                               queries[idx, :] .+ 1)
                #r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                embedding .+= r_embedding
                offset_embedding .+= conf.box_activation(r_offset_embedding)
            end
            idx += 1
        end
    else
        embedding_list = []
        offset_embedding_list = []
        for i in range(1, length(query_structure))
            embedding, offset_embedding, idx = embed_query_box(m, conf, queries, query_structure[i], idx)
            push!(embedding_list, embedding)
            push!(offset_embedding_list, offset_embedding)
        end
        embedding = m.center_net(stack(embedding_list))
        offset_embedding = m.offset_net(stack(offset_embedding_list))
    end
    return embedding, offset_embedding, idx
end

#=
Iterative embed a batch of queries with same structure using GQE
queries: a flattened batch of queries
=#
function embed_query_vec(m::KGReasoning, queries, query_structure, idx)

    all_relation_flag = true
    # whether the current query tree has merged to one branch and only need to do relation traversal,
    # e.g., path queries or conjunctive queries after the intersection
    for ele in last(query_structure)
        if !(ele in ["r", "n"])
            all_relation_flag = false
            break
        end
    end
    if all_relation_flag
        if query_structure[1] == "e"
            embedding = selectdim(m.entity_embedding,
                                  ndims(m.entity_embedding),
                                  queries[idx, :] .+ 1)
            #embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
            idx += 1
        else
            embedding, idx = embed_query_vec(m, queries, query_structure[1], idx)
        end

        for i in range(1, length(last(query_structure)))
            if last(query_structure)[i] == "n"
                @assert false  "vec cannot handle queries with negation"
            else
                r_embedding = selectdim(m.relation_embedding,
                                        ndims(m.relation_embedding),
                                        queries[idx, :] .+ 1)
                #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                embedding .+= r_embedding
            end
            idx += 1
        end
    else
        embedding_list = []
        for i in range(1, length(query_structure))
            embedding, idx = embed_query_vec(m, queries, query_structure[i], idx)
            push!(embedding_list, embedding)
        end
        embedding = m.center_net(stack(embedding_list))
    end
    return embedding, idx
end

#=
Iterative embed a batch of queries with same structure using BetaE
queries: a flattened batch of queries
* all the queries have the same structure(entitie and relation indexes)
=#
function embed_query_beta(m::KGReasoning, conf::KGRConfig, queries, query_structure, idx)

    all_relation_flag = true
    # whether the current query tree has merged to one branch and only need to do relation traversal,
    # e.g., path queries or conjunctive queries after the intersection
    for ele in last(query_structure)
        if !(ele in ["r", "n"])
            all_relation_flag = false
            break
        end
    end
    #println("embed_query_beta: $(query_structure) last $(last(query_structure)) all_relation: $(all_relation_flag)")
    #println("embed_query_beta: queries $(queries)")
    #println("embed_query_beta: typeof queries $(typeof(queries)) idx $(idx)")
    #================
    ('e',('r',)): '1p',
    ('e', ('r', 'r')): '2p',
    ('e', ('r', 'r', 'r')): '3p',
    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
    =================#
    if all_relation_flag
        # the first element is whether an "e"[entity] or ("e" , ...)[sub question]
        if query_structure[1] == "e"
            embeddings = copy(selectdim(m.entity_embedding,
                                        ndims(m.entity_embedding),
                                        queries[idx, :] .+ 1))
            embeddings = conf.beta_entity_regularizer(embeddings)
            idx += 1
        else
            alpha_embedding, beta_embedding, idx = embed_query_beta(m, conf, queries, query_structure[1], idx)
            embeddings = cat(alpha_embedding, beta_embedding, dims=1)
        end

        # there is only one entity in a query or an subquery
        # the last(query_structure) is one or more ralations
        relations = last(query_structure)
        for i in range(1, length(relations))
            if relations[i] == "n"
                @assert all(x -> x == -2, queries[idx, :])
                embeddings = 1 ./ embeddings
            else
                r_embeddings = copy(selectdim(m.relation_embedding,
                                              ndims(m.relation_embedding),
                                              queries[idx, :] .+ 1))
                embeddings = m.projection_net(embeddings, r_embeddings)
            end
            idx += 1
        end
        ###############################TODO####################################
        alpha_embedding, beta_embedding = chunk(embeddings, 2, dims=1)
    else
        #=========================
        (('e', ('r',)), ('e', ('r',))): '2i',
        (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
        (('e', ('r', 'r')), ('e', ('r',))): 'pi',
        (('e', ('r',)), ('e', ('r', 'n'))): '2in',
        (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
        (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
        (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
        (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
        ==========================#
        alpha_embedding_list = []
        beta_embedding_list = []
        for i in range(1, length(query_structure))
            alpha_embedding, beta_embedding, idx = embed_query_beta(m, conf, queries, query_structure[i], idx)
            push!(alpha_embedding_list, alpha_embedding)
            push!(beta_embedding_list, beta_embedding)
        end
        alpha_embedding, beta_embedding = m.center_net(stack(alpha_embedding_list), stack(beta_embedding_list))
    end

    return alpha_embedding, beta_embedding, idx
end

#============================================
    transform 2u queries to two 1p queries
    transform up queries to two 2p queries
============================================#
function transform_union_query(conf::KGRConfig, queries, query_structure)

    if conf.query_name_dict[query_structure] == "2u-DNF"
        #=
        (('e', ('r',)), ('e', ('r',)), ('u',)) => (('e', ('r',)), ('e', ('r',)))
        =#
        queries = queries[1:(size(queries, 1) - 1, :)] # remove union -1
    elseif conf.query_name_dict[query_structure] == "up-DNF"
        #=
        ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)) => ((e, (r, r)), (e, (r, r)))
        =#
        queries = cat(cat(queries[1:2, :], queries[6:6, :], dims=1), cat(queries[3:4, :], queries[6:6, :], dims=1), dims=1)
    end
    queries = reshape(queries, :, size(queries)[1]*2)
    return queries
end

function transform_union_structure(conf::KGRConfig, query_structure)
    if conf.query_name_dict[query_structure] == "2u-DNF"
        return ("e", ("r",))
    elseif conf.query_name_dict[query_structure] == "up-DNF"
        return ("e", ("r", "r"))
    end
end

function logit_beta(m::KGReasoning, entity_embedding, query_dist)
    ##########################TODO#######################################
    alpha_embedding, beta_embedding = chunk(entity_embedding, 2, dims = 1)
    entity_dist = Distributions.Beta.(alpha_embedding, beta_embedding)
    #println("logit_beta: alpha_embedding $(size(alpha_embedding))")
    #println("logit_beta: beta_embedding $(size(beta_embedding))")
    #println("logit_beta: beta_embedding $(size(entity_dist))")
    kld = KLDivergences.KL.(entity_dist, query_dist)
    #println("logit_beta: kld $(size(kld)))")
    logit = m.gamma .- norm_pd(KLDivergences.KL.(entity_dist, query_dist), 1, dims=1)
    return logit
end

function forward_beta(m::KGReasoning, conf::KGRConfig,
                      positive_sample, negative_sample, subsampling_weight,
                      all_idxs, all_alpha_embeddings, all_beta_embeddings,
                      all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings)

    if length(all_alpha_embeddings) > 0
        #println("forward_beta: all_x_embedding: $(size(all_alpha_embeddings)) -- $(size(all_beta_embeddings))")
        all_alpha_embeddings = reduce(hcat, all_alpha_embeddings)
        all_alpha_embeddings= Flux.unsqueeze(all_alpha_embeddings, ndims(all_alpha_embeddings))

        all_beta_embeddings = reduce(hcat, all_beta_embeddings)
        all_beta_embeddings= Flux.unsqueeze(all_beta_embeddings, ndims(all_beta_embeddings))
        #println("forward_beta: all_x_embedding: $(size(all_alpha_embeddings)) -- $(size(all_beta_embeddings))")

        all_dists = Distributions.Beta.(all_alpha_embeddings, all_beta_embeddings)
    end

    if length(all_union_alpha_embeddings) > 0
        #all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
        #all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
        all_union_alpha_embeddings = reduce(hcat, all_union_alpha_embeddings)
        all_union_alpha_embeddings = Flux.unsqueeze(all_union_alpha_embeddings, ndims(all_union_alpha_embeddings))

        all_union_beta_embeddings = reduce(hcat, all_union_beta_embeddings)
        all_union_beta_embeddings = Flux.unsqueeze(all_union_beta_embeddings, ndims(all_union_beta_embeddings))

        all_union_alpha_embeddings = reshape(all_union_alpha_embeddings, :, 1, 2,
                                             div(size(all_union_alpha_embeddings, ndims(all_union_alpha_embeddings)), 2))

        all_union_beta_embeddings = reshape(all_union_beta_embeddings, :, 1, 2,
                                            div(size(all_union_beta_embeddings, ndims(all_union_beta_embeddings)), 2))

        #all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)
        #println("forward_beta: union_x_embedding: $(size(all_union_alpha_embeddings)) -- $(size(all_union_beta_embeddings))")
        all_union_dists = Distributions.Beta.(all_union_alpha_embeddings, all_union_beta_embeddings)
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[[all_idxs; all_union_idxs]]
    end

    #println("forward_beta: m.entity_embedding size  $(size(m.entity_embedding))")
    local positive_logit, positive_alpha_logit, positive_union_logit
    if typeof(positive_sample) != typeof(nothing)
        if length(all_alpha_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
            #println("forward_beta: positive_sample_regular idxs: $(all_idxs)")
            #println("forward_beta: positive_sample_regular positive_sample_regular $(positive_sample_regular)")
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_regular .+ 1);
            entity_embedding_unsqueeze = Flux.unsqueeze(entity_embedding_select, ndims(entity_embedding_select))
            #println("entity_embedding: $(size(entity_embedding_select)) - $(size(entity_embedding_unsqueeze))")

            positive_embedding = conf.beta_entity_regularizer(entity_embedding_unsqueeze)
            positive_alpha_logit = logit_beta(m, positive_embedding, all_dists)
        else
            positive_alpha_logit = []
        end
        #println("forward_beta: size positive_logit $(size(positive_alpha_logit))")

        if length(all_union_alpha_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch

            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_union .+ 1);
            entity_embedding_select_unsqueeze = Flux.unsqueeze(entity_embedding_select,
                                                               dims=ndims(entity_embedding_select));
            positive_embedding = conf.beta_entity_regularizer(entity_embedding_select_unsqueeze)
            positive_union_logit = logit_beta(m, positive_embedding, all_union_dists)
            #println("forward_beta: size positive_union_logit $(size(positive_union_logit))")
            positive_union_logit = maximum(positive_union_logit, dims=1)[1]
        else
            positive_union_logit = []
        end
        #println("forward_beta: size positive_union_logit $(size(positive_union_logit))")

        if size(positive_alpha_logit) == size(positive_union_logit)
            positive_logit = cat(positive_alpha_logit, positive_union_logit, dims=ndims(positive_logit))
        elseif length(positive_alpha_logit) > 0
            positive_logit = positive_alpha_logit
        elseif length(positive_union_logit) > 0
            positive_logit = positive_union_logit
        end
    else
        positive_logit = nothing
    end

    local negative_logit, negative_alpha_logit, negative_union_logit
    if typeof(negative_sample) != typeof(nothing)
        if length(all_alpha_embeddings) > 0
            #println("forward_beta: size all_idxs $(all_idxs)\n size negative_sample $(size(negative_sample))")
            negative_sample_regular = negative_sample[:, all_idxs]
            #println("forward_beta: size negative_sample_regular $(size(negative_sample_regular))")
            negative_size, batch_size = size(negative_sample_regular)

            negative_regular_reshape = reshape(negative_sample_regular, :)
            #println("forward_beta: max min negative sample : $(maximum(negative_regular_reshape)) $(minimum(negative_regular_reshape))")
            negative_regular_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                negative_regular_reshape .+ 1)
            #println("forward_beta: negative regular size $(size(negative_regular_select))")
            negative_regular_reshape = reshape(negative_regular_select, :, negative_size, batch_size)
            #println("forward_beta: negative regular reshape size $(size(negative_regular_reshape))")
            negative_embedding = conf.beta_entity_regularizer(negative_regular_reshape)

            negative_alpha_logit = logit_beta(m, negative_embedding, all_dists)
        else
            negative_alpha_logit = []
        end
        #println("forward_beta: size negative_alpha_logit $(size(negative_alpha_logit))")

        if length(all_union_alpha_embeddings) > 0
            negative_sample_union = negative_sample[:, all_union_idxs]
            negative_size, batch_size = size(negative_sample_union)

            negative_union_reshape = reshape(negative_sample_union, :)
            #println("forward_beta: max min negative sample : $(maximum(negative_union_reshape)) $(minimum(negative_union_reshape))")
            negative_union_select = selectdim(m.entity_embedding,
                                              ndims(m.entity_embedding),
                                              negative_union_reshape .+ 1)
            #println("forward_beta: negative union size $(size(negative_union_select))")
            negative_union_reshape = reshape(negative_union_select, :, negative_size, batch_size)
            #println("forward_beta: negative union reshape size $(size(negative_union_reshape))")
            negative_embedding = conf.beta_entity_regularizer(negative_union_reshape)
            negative_union_logit = logit_beta(m, negative_embedding, all_union_dists)
            #println("forward_beta: negative union logit size $(size(negative_union_logit))")
            negative_union_logit = maximum(negative_union_logit, dims=1)[1]
        else
            negative_union_logit = []
        end
        #println("forward_beta: size negative_union_logit $(size(negative_union_logit))")

        if size(negative_alpha_logit) == size(negative_union_logit)
            negative_logit = cat(negative_alpha_logit, negative_union_logit, dims=ndims(negative_logit))
        elseif length(negative_alpha_logit) > 0
            negative_logit = negative_alpha_logit
        elseif length(negative_union_logit) > 0
            negative_logit = negative_union_logit
        end
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, [all_idxs, all_union_idxs]
end

function logit_box(m::KGReasoning, conf::KGRConfig, entity_embedding, query_center_embedding, query_offset_embedding)

    #println("logit_box: embedding size entity $(size(entity_embedding)) query_center $(size(query_center_embedding)) query_offset $(size(query_offset_embedding))")
    delta = abs.(entity_embedding .- query_center_embedding)
    distance_out = Flux.relu(delta .- query_offset_embedding)
    #println("logit_box: size delta $(size(delta)) distance_out $(size(distance_out))")
    distance_in = min.(delta, query_offset_embedding)

    norm_distance_out = norm_pd(distance_out, 1, dims=1)
    norm_distance_in = norm_pd(distance_in, 1, dims=1)
    #println("logit_box:  size distance $(size(norm_distance_out)) $(size(norm_distance_in))")
    #logit = m.gamma .- dropdims(norm_distance_out, dims=1) .- conf.box_center .* dropdims(norm_distance_in, dims=1)
    logit = m.gamma .- norm_distance_out .- conf.box_center .* norm_distance_in
    return logit
end

function forward_box(m::KGReasoning, conf::KGRConfig, positive_sample, negative_sample, subsampling_weight,
                     all_idxs, all_center_embeddings, all_offset_embeddings,
                     all_union_idxs, all_union_center_embeddings, all_union_offset_embeddings)

    if length(all_center_embeddings) > 0 && length(all_offset_embeddings) > 0
        all_center_embeddings_cat = reduce(hcat, all_center_embeddings)
        #println("forward_box: size all_center_embeddings $(size(all_center_embeddings)) $(size(all_center_embeddings_cat))")
        all_center_embeddings = unsqueeze(all_center_embeddings_cat,
                                          dims = ndims(all_center_embeddings_cat))

        all_offset_embeddings_cat = reduce(hcat, all_offset_embeddings)
        #println("forward_box: size all_offset_embeddings $(size(all_offset_embeddings)) $(size(all_offset_embeddings_cat))")
        all_offset_embeddings = unsqueeze(all_offset_embeddings_cat,
                                          dims = ndims(all_offset_embeddings_cat))
        #all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
    end

    if length(all_union_center_embeddings) > 0 && length(all_union_offset_embeddings) > 0
        #all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
        #all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
        all_union_center_embeddings_cat = reduce(hcat, all_union_center_embeddings)
        #println("forward_box: size all_union_center_embeddings $(size(all_union_center_embeddings)) $(size(all_union_center_embeddings_cat))")
        all_union_center_embeddings = unsqueeze(all_union_center_embeddings_cat,
                                                dims = ndims(all_union_center_embeddings_cat))

        all_union_offset_embeddings_cat = reduce(hcat, all_union_offset_embeddings)
        #println("forward_box: size all_union_offset_embeddings $(size(all_union_offset_embeddings)) $(size(all_union_offset_embeddings_cat))")
        all_union_offset_embeddings = unsqueeze(all_union_offset_embeddings_cat,
                                                dims = ndims(all_union_offset_embeddings_cat))
        #println("forward_box: size all_union_embeddings $(size(all_union_center_embeddings)) $(size(all_union_offset_embeddings))")
        #all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
        #all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)
        all_union_center_embeddings = reshape(all_union_center_embeddings,
                                              :, 1, 2, div(last(size(all_union_center_embeddings)), 2))
        all_union_offset_embeddings = reshape(all_union_offset_embeddings,
                                              :, 1, 2, div(last(size(all_union_offset_embeddings)), 2))
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[[all_idxs; all_union_idxs]]
    end

    if typeof(positive_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs]
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_regular .+ 1)

            positive_embedding = unsqueeze(entity_embedding_select, ndims(entity_embedding_select))
            #println("forward_box: size  embedding entity $(size(entity_embedding_select)) positive $(size(positive_embedding))")
            positive_logit = logit_box(m, conf,
                                       positive_embedding,
                                       all_center_embeddings,
                                       all_offset_embeddings)
            #println("forward_box: size positive logit $(size(positive_logit))")
        else
            #positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = [] .|> gpu
        end

        if length(all_union_center_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs]
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_union .+ 1)
            entity_embedding_select_unquezze = unsqueeze(entity_embedding_select, ndims(entity_embedding_select))
            positive_embedding = unsqueeze(entity_embedding_select_unquezze, ndims(entity_embedding_select_unquezze))

            positive_union_logit = logit_box(m, conf, positive_embedding,
                                             all_union_center_embeddings,
                                             all_union_offset_embeddings)
            positive_union_logit = maximum(positive_union_logit, dims=ndims(positive_union_logit))[1]
        else
            #positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_union_logit = [] .|> gpu
        end

        #println("forward_box: positive_logit $(size(positive_logit)) positive_union_logit $(size(positive_union_logit))")
        if size(positive_logit) == size(positive_union_logit)
            negative_logit = cat(positive_logit, positive_union_logit, dims=ndims(positive_logit))
        elseif length(positive_logit) > 0
            negative_logit = positive_logit
        elseif length(positive_union_logit) > 0
            negative_logit = positive_union_logit
        end
    else
        positive_logit = nothing
    end

    if typeof(negative_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            negative_sample_regular = negative_sample[:, all_idxs]
            negative_size, batch_size = size(negative_sample_regular)

            negative_sample_reshape = reshape(negative_sample_regular, :)
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                negative_sample_reshape .+ 1)
            negative_embedding = reshape(entity_embedding_select, :, negative_size, batch_size)
            negative_logit = logit_box(m, conf,
                                       negative_embedding,
                                       all_center_embeddings,
                                       all_offset_embeddings)
            #println("forward_box: size negative_logit $size(negative_logit)")
        else
            #negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = [] .|> gpu
        end

        if length(all_union_center_embeddings) > 0
            negative_sample_union = negative_sample[:, all_union_idxs]
            negative_size, batch_size = size(negative_sample_union)

            negative_union_reshape = reshape(negative_sample_union, :)
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                negative_union_reshape .+ 1)
            negative_embedding = reshape(entity_embedding_select, :, negative_size, 1, batch_size)
            negative_union_logit = logit_box(m, conf,
                                             negative_embedding,
                                             all_union_center_embeddings,
                                             all_union_offset_embeddings)
            #println("forward_box: size negative_union_logit $size(negative_union_logit)")
            negative_union_logit = maximum(negative_union_logit, dims=ndims(negative_union_logit))[1]
        else
            #negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_union_logit = [] .|> gpu
        end

        if size(negative_logit) == size(negative_union_logit)
            negative_logit = cat(negative_logit, negative_union_logit, dims=ndims(negative_logit))
        elseif length(negative_logit) > 0
            negative_logit = negative_logit
        elseif length(negative_union_logit) > 0
            negative_logit = negative_union_logit
        end
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, [all_idxs; all_union_idxs]
end

function logit_vec(m::KGReasoning, entity_embedding, query_embedding)
    distance = entity_embedding .- query_embedding
    println("logit_vec: size distance $(size(distance))")
    logit = m.gamma .- norm_pd(distance, 1, dims=1)
    return logit
end

function forward_vec(m::KGReasoning, conf::KGRConfig, positive_sample, negative_sample, subsampling_weight,
                     all_idxs, all_center_embeddings, all_union_idxs, all_union_center_embeddings)

    if length(all_center_embeddings) > 0
        all_center_embeddings_cat = reduce(hcat, all_center_embeddings)
        all_center_embeddings = unsqueeze(all_center_embeddings_cat,
                                          ndims(all_center_embeddings_cat))
    end

    if length(all_union_center_embeddings) > 0
        all_union_center_embeddings_cat = reduce(hcat, all_union_center_embeddings)

        all_union_center_embeddings_unsqueeze = unsqueeze(all_union_center_embeddings_cat,
                                                          ndims(all_union_center_embeddings_cat))
        #all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
        all_union_center_embeddings = reshape(all_union_center_embeddings_unsqueeze,
                                              :, 1, 2, div(last(size(all_union_center_embeddings)), 2))
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[[all_idxs; all_union_idxs]]
    end

    if typeof(positive_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs]
            positive_embedding_select = selectdim(m.entity_embedding,
                                                  ndims(m.entity_embedding),
                                                  positive_sample_regular .+ 1)
            positive_embedding = unsqueeze(positive_embedding_select, ndims(positive_embedding_select))
            positive_logit = logit_vec(m, positive_embedding, all_center_embeddings)
        else
            positive_logit = [] .|> gpu
        end

        if length(all_union_center_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs]
            positive_embedding_select = selectdim(m.entity_embedding,
                                                  ndims(m.entity_embedding),
                                                  positive_sample_regular.+ 1)
            positive_embedding_unsqueeze = unsqueeze(positive_embedding_select, ndims(positive_embedding_select))
            positive_embedding = unsqueeze(positive_embedding_unsqueeze, ndims(positive_embedding_unsqueeze))

            positive_union_logit = logit_vec(m, positive_embedding, all_union_center_embeddings)
            positive_union_logit = maximum(positive_union_logit, dims=ndims(positive_union_logit))[1]
        else
            positive_union_logit = [] .|> gpu
        end

        if size(positive_logit) == size(positive_union_logit)
            negative_logit = cat(positive_logit, positive_union_logit, dims=ndims(positive_logit))
        elseif length(positive_logit) > 0
            negative_logit = positive_logit
        elseif length(positive_union_logit) > 0
            negative_logit = positive_union_logit
        end
        #positive_logit = reduce([positive_logit, positive_union_logit]) do x, y
        #                     cat(x, y, dims=ndims(x))
        #                 end
    else
        positive_logit = nothing
    end

    if typeof(negative_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            negative_regular = negative_sample[:, all_idxs]
            negative_size, batch_size = size(negative_regular)

            negative_regular_reshape = reshape(negative_regular, :)
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                negative_regular_reshape .+ 1)
            negative_embedding = reshape(entity_embedding_select, :, negative_size, batch_size)
            negative_logit = logit_vec(m, negative_embedding, all_center_embeddings)
        else
            negative_logit = [] .|> gpu
        end

        if length(all_union_center_embeddings) > 0
            negative_union = negative_sample[:, all_union_idxs]
            negative_size, batch_size = size(negative_union)

            negative_union_reshape = reshape(negative_union, :)
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                negative_union_reshape .+ 1)
            negative_embedding = reshape(entity_embedding_select, :, negative_size, 1, batch_size)
            negative_union_logit = logit_vec(m, negative_embedding, all_union_center_embeddings)
            negative_union_logit = maximum(negative_union_logit, dim=1)[1]
        else
            negative_union_logit = [] .|> gpu
        end

        if size(positive_logit) == size(positive_union_logit)
            negative_logit = cat(positive_logit, positive_union_logit, dims=ndims(positive_logit))
        elseif length(positive_logit) > 0
            negative_logit = positive_logit
        elseif length(positive_union_logit) > 0
            negative_logit = positive_union_logit
        end
        #negative_logit = reduce([negative_logit, negative_union_logit]) do x, y
        #    cat(x, y, dims=ndims(x))
        #end
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, [all_idxs; all_union_idxs]
end

function loss(positive_logit, negative_logit, subsampling_weight)

    #println("train_step: positive_logit $(positive_logit)")
    #println("train_step: negative_logit $(negative_logit)")
    #println("train_step: subsampling_weight $(subsampling_weight)")
    negative_logsigmoid = Flux.logsigmoid(-negative_logit)
    negative_score = mean(negative_logsigmoid, dims=ndims(negative_logsigmoid) - 1)
    negative_score = dropdims(negative_score, dims = ndims(negative_score) - 1)

    positive_logsigmoid =Flux.logsigmoid(positive_logit)
    positive_score = dropdims(positive_logsigmoid, dims=ndims(positive_logsigmoid) - 1)

    positive_sample_loss = - sum(subsampling_weight * positive_score)
    negative_sample_loss = - sum(subsampling_weight * negative_score)

    positive_sample_loss /= sum(subsampling_weight)
    negative_sample_loss /= sum(subsampling_weight)

    return loss = (positive_sample_loss + negative_sample_loss)/2
end

function train_step(model::KGReasoning, conf::KGRConfig, opt_state, data, args, step)

    positive_sample, negative_sample, subsampling_weight, queries, query_structures = data
    negative_sample = combinedims(negative_sample)

    batch_queries_dict = Dict{Any, Union{Vector{Vector{Int}}, Matrix{Int}}}()
    batch_idxs_dict = Dict{Any, Vector{Int}}()
    for (i, query) in enumerate(queries) # group queries with same structure
        push!(get!(batch_queries_dict, query_structures[i], Vector{Vector{Int}}()), query)
        push!(get!(batch_idxs_dict, query_structures[i], []), i)
    end

    for query_structure in keys(batch_queries_dict)
        batch_queries_dict[query_structure] = combinedims(batch_queries_dict[query_structure])
    end
    #@info "train_step: queries_dict $(batch_queries_dict)"
    #@info "train_step: idxs_dict $(batch_idxs_dict)"

    if args["cuda"]
        positive_sample = positive_sample |> gpu
        negative_sample = negative_sample |> gpu
        subsampling_weight = subsampling_weight |> gpu
    end

    local positive_sample_loss, negative_sample_loss, loss
    local positive_logit, negative_logit, subsampling_weight
    if conf.geo == "beta"
        all_idxs = Array{Int, 1}()
        all_alpha_embeddings = Array{Matrix{Float64}, 1}()
        all_beta_embeddings = Array{Matrix{Float64}, 1}()

        all_union_idxs = Array{Int, 1}()
        all_union_alpha_embeddings = Array{Matrix{Float64}, 1}()
        all_union_beta_embeddings = Array{Matrix{Float64}, 1}()

        #println("train_step: typeof positive_sample: $(typeof(positive_sample))")
        #println("train_step: typeof negative_sample: $(typeof(negative_sample))")

        for query_structure in keys(batch_queries_dict)
            #println("train_step: query structure $(query_structure)------------------------------------------ooo")
            if occursin('u', conf.query_name_dict[query_structure]) && occursin("DNF", conf.query_name_dict[query_structure])
                alpha_embeddings, beta_embeddings, _ = embed_query_beta(model, conf,
                                                                        transform_union_query(conf,
                                                                                              batch_queries_dict[query_structure],
                                                                                              query_structure),
                                                                        transform_union_structure(conf, query_structure),
                                                                        1)
                append!(all_union_idxs, batch_idxs_dict[query_structure])
                #all_union_idxs.extend(batch_idxs_dict[query_structure])
                push!(all_union_alpha_embeddings, alpha_embeddings)
                push!(all_union_beta_embeddings, beta_embeddings)
            else
                alpha_embeddings, beta_embeddings, _ = embed_query_beta(model, conf,
                                                                        batch_queries_dict[query_structure],
                                                                        query_structure,
                                                                        1)
                #println("train_step:\n \
                #         $(typeof(alpha_embeddings)) $(size(alpha_embeddings))\n \
                #         $(typeof(beta_embeddings)) $(size(beta_embeddings))")
                append!(all_idxs, batch_idxs_dict[query_structure])
                #all_idxs.extend(batch_idxs_dict[query_structure])
                push!(all_alpha_embeddings, alpha_embeddings)
                push!(all_beta_embeddings, beta_embeddings)
            end
        end

        positive_logit, negative_logit,
        subsampling_weight, _ = model(conf, positive_sample, negative_sample, subsampling_weight,
                                      all_idxs, all_alpha_embeddings, all_beta_embeddings,
                                      all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings)

    elseif conf.geo == "box"
        all_idxs = Array{Int, 1}()
        all_center_embeddings = Array{Matrix{Float64}, 1}()
        all_offset_embeddings = Array{Matrix{Float64}, 1}()

        all_union_idxs = Array{Int, 1}()
        all_union_center_embeddings = Array{Matrix{Float64}, 1}()
        all_union_offset_embeddings = Array{Matrix{Float64}, 1}()

        for query_structure in keys(batch_queries_dict)
            println("train_step: query_structure $(query_structure) $(conf.query_name_dict[query_structure])")
            if occursin('u', conf.query_name_dict[query_structure])
                center_embedding, offset_embedding, _ = embed_query_box(model, conf,
                                                                        transform_union_query(conf,
                                                                                              batch_queries_dict[query_structure],
                                                                                              query_structure),
                                                                        transform_union_structure(conf, query_structure),
                                                                        1)
                println("train_step: size query_box u embedding $(size(center_embedding)) $(size(offset_embedding))")
                println("train_step: size query_box u embedding $(typeof(center_embedding)) $(typeof(offset_embedding))")

                append!(all_union_idxs, batch_idxs_dict[query_structure])
                push!(all_union_center_embeddings, center_embedding)
                push!(all_union_offset_embeddings, offset_embedding)
            else
                center_embedding, offset_embedding, _ = embed_query_box(model, conf,
                                                                        batch_queries_dict[query_structure],
                                                                        query_structure,
                                                                        1)
                println("train_step: size query_box embedding $(size(center_embedding)) $(size(offset_embedding))")
                println("train_step: size query_box embedding $(typeof(center_embedding)) $(typeof(offset_embedding))")

                append!(all_idxs, batch_idxs_dict[query_structure])
                push!(all_center_embeddings, center_embedding)
                push!(all_offset_embeddings, offset_embedding)
            end
        end

        positive_logit, negative_logit,
        subsampling_weight, _ = model(conf, positive_sample, negative_sample, subsampling_weight,
                                      all_idxs, all_center_embeddings, all_offset_embeddings,
                                      all_union_idxs, all_union_center_embeddings, all_union_offset_embeddings)
    elseif conf.geo == "vec"
        all_idxs = Array{Int, 1}()
        all_center_embeddings = Array{Matrix{Float64}, 1}()

        all_union_idxs = Array{Int, 1}()
        all_union_center_embeddings = Array{Matrix{Float64}, 1}()

        for query_structure in keys(batch_queries_dict)
            if occursin('u', conf.query_name_dict[query_structure])
                center_embedding, _ = embed_query_vec(model, transform_union_query(conf,
                                                                                   batch_queries_dict[query_structure],
                                                                                   query_structure),
                                                      transform_union_structure(conf, query_structure),
                                                      1)

                append!(all_union_idxs, batch_idxs_dict[query_structure])
                push!(all_union_center_embeddings, center_embedding)
            else
                center_embedding, _ = embed_query_vec(model, batch_queries_dict[query_structure], query_structure, 1)

                append!(all_idxs, batch_idxs_dict[query_structure])
                push!(all_center_embeddings, center_embedding)
            end
        end

        positive_logit, negative_logit,
        subsampling_weight, _ = model(conf, positive_sample, negative_sample, subsampling_weight,
                                      all_idxs, all_center_embeddings,
                                      all_union_idxs, all_union_center_embeddings)
    end

    println("train_step: logit size $(size(positive_logit)) $(size(negative_logit)) $(size(subsampling_weight))")
    opt_grads = Flux.gradient(model, positive_logit, negative_logit,
                              subsampling_weight) do m, x, y, z
                                  positive_logsigmoid =Flux.logsigmoid(x)
                                  positive_score = dropdims(positive_logsigmoid, dims=ndims(positive_logsigmoid) - 1)

                                  negative_logsigmoid = Flux.logsigmoid(-y)
                                  negative_score = mean(negative_logsigmoid, dims=ndims(negative_logsigmoid) - 1)
                                  negative_score = dropdims(negative_score, dims = ndims(negative_score) - 1)

                                  positive_sample_loss = - sum(z .* positive_score)
                                  negative_sample_loss = - sum(z .* negative_score)

                                  positive_sample_loss /= sum(z)
                                  negative_sample_loss /= sum(z)

                                  loss = (positive_sample_loss + negative_sample_loss)/2
                              end
    r_state, r_model = Flux.update!(opt_state, model, opt_grads[1])

    log = Dict(
        "positive_sample_loss" => positive_sample_loss,
        "negative_sample_loss" => negative_sample_loss,
        "loss" => loss,
    )
    return r_state, r_model, log
end

function test_step(model, easy_answers, hard_answers, args, test_dataloader,
                   query_name_dict, save_result=False, save_str="", save_empty=False)
    step = 0
    total_steps = length(test_dataloader)
    #logs = collections.defaultdict(list)
    logs = Dict()

    #with torch.no_grad():
    for (negative_sample, queries, queries_unflatten, query_structures) in tqdm(test_dataloader)
        batch_queries_dict = Dict() #collections.defaultdict(list)
        batch_idxs_dict = Dict() #collections.defaultdict(list)
        for (i, query) in enumerate(queries)
            push!(batch_queries_dict[query_structures[i]], query)
            push!(batch_idxs_dict[query_structures[i]], i)
        end

        for query_structure in batch_queries_dict
            if args["cuda"]
                batch_queries_dict[query_structure] = Int64.(batch_queries_dict[query_structure]) .|> gpu
            else
                batch_queries_dict[query_structure] = Int64.(batch_queries_dict[query_structure])
            end
        end

        if args["cuda"]
            negative_sample = negative_sample .|> gpu
        end

        _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
        queries_unflatten = [queries_unflatten[i] for i in idxs]
        query_structures = [query_structures[i] for i in idxs]
        argsort = sortperm(negative_logit, dim=ndims(negative_logit)-1, rev=true)
        ranking = Float32.(copy(argsort))
        if length(argsort) == args["test_batch_size"] # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
            #ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
            ranking = getindex(model.batch_entity_range, argsort)
        else # otherwise, create a new torch Tensor for batch_entity_range
            if args["cuda"]
                #ranking = ranking.scatter_(1,
                #                           argsort,
                #                           torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                #                                                                              1).cuda()
                #                           ) # achieve the ranking of all entities
                target = repeat(Float32.(collect(1:model.nentity)), 1, size(argsort, ndims(argsort)))
                ranking = getindex(argsort, target) |> gpu
            else
                #ranking = ranking.scatter_(1,
                #                           argsort,
                #                           torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                #                                                                              1)
                #                           ) # achieve the ranking of all entities
                target = repeat(Float32.(collect(1:model.nentity)), 1, size(argsort, ndims(argsort)))
                ranking = getindex(argsort, target)
            end
        end

        for (idx, (i, query, query_structure)) in enumerate(zip(argsort[:, ndims(argsort)], queries_unflatten, query_structures))
            hard_answer = hard_answers[query]
            easy_answer = easy_answers[query]
            num_hard = length(hard_answer)
            num_easy = length(easy_answer)
            @assert length(hard_answer.intersection(easy_answer)) == 0
            cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
            cur_ranking, indices = sortperm(cur_ranking)
            masks = indices >= num_easy
            if args["cuda"]
                answer_list = Float32.(collect(1:(num_hard + num_easy))) .|> gpu
            else
                answer_list = Float32.(collect(1:(num_hard + num_easy)))
            end
            cur_ranking = cur_ranking .- answer_list + 1 # filtered setting
            cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

            mrr = collect(mean(1 ./ cur_ranking))
            h1 = collect(Float32.(mean((cur_ranking <= 1))))
            h3 = collect(Float32.(mean((cur_ranking <= 3))))
            h10 = collect(Float32.(mean((cur_ranking <= 10))))

            push!(get!(logs, query_structure, Dict()), Dict(
                "MRR"=> mrr,
                "HITS1"=> h1,
                "HITS3"=> h3,
                "HITS10"=> h10,
                "num_hard_answer"=> num_hard
            ))
        end

        if step % args.test_log_steps == 0
            @info("Evaluating the model... ($step/$total_steps)")
        end
        step += 1
    end

    #metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    metrics = Dict()
    for query_structure in logs
        for metric in keys(logs[query_structure][0])
            if metric in ["num_hard_answer"]
                continue
            end
            metrics[:query_structure][:metric] = sum([log[metric] for log in logs[query_structure]])/length(logs[query_structure])
        end
        metrics[:query_structure]["num_queries"] = length(logs[query_structure])
    end

    return metrics
end

end #end module
