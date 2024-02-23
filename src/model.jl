#!/home/leslie/julia/1.10.0/bin/julia

module KGModel

using Base: offset_if_vec
using Zygote: AbstractFFTs
export Identity, normDims, BoxOffsetIntersection, CenterIntersection, BetaIntersection,
    BetaProjection, Regularizer, KGReasoning, KGRConfig, train_step

using Flux;
using Zygote;
using Statistics;
using Distributions;
using MLUtils;

include("utils.jl")

function Identity(x)
    return x;
end

function normDims(itr, p::Real=2; dim)
    sum(itr .^ p; dims=dim).^(1 / p)
end

struct BoxOffsetIntersection
    #dim::Int
    layer1::Flux.Dense
    layer2::Flux.Dense
end

function BoxOffsetIntersection(dim::Int)
    layer1 = Flux.Dense(dim => dim);
    layer2 = Flux.Dense(dim => dim);

    return BoxOffsetIntersection(layer1, layer2);
end

#Function-like Object
function (m::BoxOffsetIntersection)(embeddings)
    @show embeddings
    layer1_act = Flux.relu(m.layer1(embeddings))
    @show layer1_act
    layer1_mean = mean(layer1_act, dims=length(size(layer1_act)))
    @show layer1_mean
    gate = Flux.sigmoid(m.layer2(layer1_mean))
    @show gate
    offset = minimum(embeddings, dims=length(size(layer1_act)))

    return offset .* gate
end

Flux.@functor BoxOffsetIntersection

struct CenterIntersection
    #dim::Int
    layer1::Flux.Dense
    layer2::Flux.Dense
end

function CenterIntersection(dim::Int)
    layer1 = Flux.Dense(dim => dim)
    layer2 = Flux.Dense(dim => dim)

    #Flux.Dense is initialized by  xavier defaultly
    return CenterIntersection(layer1, layer2)
end

function (m::CenterIntersection)(embeddings)
    layer1_act = Flux.relu(m.layer1(embeddings)) # ( dim, num_conj)
    attention = Flux.softmax(m.layer2(layer1_act), dims=length(size(layer1_act))) # (dim, num_conj, )
    embedding = sum(attention * embeddings, dims=length(size(layer1_act)))

    return embedding
end

Flux.@functor CenterIntersection

struct BetaIntersection
    #dim::Int
    layer1::Flux.Dense
    layer2::Flux.Dense
end

function BetaIntersection(dim::Int)
    layer1 = Flux.Dense(2 * dim, 2 * dim)
    layer2 = Flux.Dense(2 * dim, dim)

    return BetaIntersection(layer1, layer2)
end

function (m::BetaIntersection)(alpha_embeddings, beta_embeddings)
    all_embeddings = cat(length(size(alpha_embeddings)), alpha_embeddings, beta_embeddings)
    layer1_act = Flux.relu(m.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
    attention = Flux.softmax(m.layer2(layer1_act), dims=length(size(alpha_embeddings))) # (num_conj, batch_size, dim)

    alpha_embedding = sum(attention * alpha_embeddings, dims=length(size(alpha_embeddings)))
    beta_embedding = sum(attention * beta_embeddings, dims=length(size(alpha_embeddings)))

    return alpha_embedding, beta_embedding
end

Flux.@functor BetaIntersection

struct Regularizer
    base_add::AbstractFloat
    min_val::AbstractFloat
    max_val::AbstractFloat
end

function (m::Regularizer)(entity_embedding)
    return clamp(entity_embedding + m.base_add, m.min_val, m.max_val)
end

Flux.@functor Regularizer

struct BetaProjection
    #entity_dim::Int
    #relation_dim::Int
    #hidden_dim::Int
    #num_layers::Int

    layers::Dict{Symbol, Flux.Dense}
    projection_regularizer
end

function Base.setproperty!(m::BetaProjection, property::Symbol, value)
    getfield(m, :layers)[property] = value
end

function Base.getproperty(m::BetaProjection, property::Symbol, value)
    return getfield(m, :layers)[property]
end

function Base.propertynames(m::BetaProjection, private = false)
    return keys(getproperty(m, :layers))
end

function BetaProjection(entity_dim, relation_dim, hidden_dim, num_layers, projection_regularizer)
    layer1 = Flux.Dense((entity_dim + relation_dim) => hidden_dim) # 1st layer
    layer0 = Flux.Dense(hidden_dim => entity_dim) # final layer

    layers = Dict{Symbol, Flux.Dense}()
    layers[:layer1] = Flux.Dense((entity_dim + relation_dim) => hidden_dim) # 1st layer
    layers[:layer0]  = Flux.Dense(hidden_dim => entity_dim) # final layer
    for nl in range(2, num_layers)
        layers[Symbol("layer$(nl)")] = Flux.Dense(hidden_dim, hidden_dim)
    end

    return BetaProjection(layers, projection_regularizer)
end

function (m::BetaProjection)(e_embedding, r_embedding)
    x = cat(1, e_embedding, r_embedding)
    for nl in range(1, m.num_layers)
        x = Flux.relu(getproperty(m, Symbol("layer$(nl)")(x)))
    end
    x = getproperty(m, :layer0)(x)
    x = m.projection_regularizer(x)

    return x
end

Flux.@functor BetaProjection

Base.@kwdef mutable struct KGRConfig
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
    local beta_hidden_dim, beta_num_layers
    beta_entity_regularizer, beta_projection_regularizer = (missing, missing)

    if geo == "box"
        if activation == "nothing"
            box_activation = Identity;
        elseif activation == "relu"
            box_activation = Flux.relu;
        elseif activation == "softplus"
            box_activation = Flux.softplus;
        end
    elseif geo == "beta"
        beta_hidden_dim, beta_num_layers = beta_mode
        println("KGRConfig: beta_mode $(beta_mode) beta_hidden: $(beta_hidden_dim) num_layers: $(beta_num_layers)")
        # make sure the parameters of beta embeddings are positive
        beta_entity_regularizer = Regularizer(1, 0.05, 1e9)
        # make sure the parameters of beta embeddings after relation projection are positive
        beta_projection_regularizer = Regularizer(1, 0.05, 1e9)
        println("KGRConfig: regularizer: $(beta_entity_regularizer)")
        println("KGRConfig: regularizer: $(beta_projection_regularizer)")
    end
    println("box_mode: activation: $(activation) - $(box_activation) center: $(box_center)")
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

Base.@kwdef mutable struct KGReasoning
    gamma::Float64 # parame

    embedding_range # parame

    entity_embedding # nn.Parameter
    relation_embedding
    offset_embedding

    center_net::Union{BetaIntersection, CenterIntersection, Missing} = missing
    offset_net::Union{BoxOffsetIntersection, Missing} = missing
    projection_net::Union{BetaProjection, Missing} = missing
end

Flux.@functor KGReasoning

function KGReasoning(conf, init = Flux.glorot_uniform)
    gamma = conf.gamma
    embedding_range = (gamma .+ conf.epsilon) / conf.hidden_dim
    println("embedding_range: $(embedding_range) type: $(typeof(embedding_range))")
    local entity_embedding
    if conf.geo == "box"
        entity_embedding = init(conf.nentity, conf.entity_dim) # centor for entities
    elseif conf.geo == "vec"
        entity_embedding = init(conf.nentity, conf.entity_dim)
    elseif conf.geo == "beta"
        entity_embedding = init(conf.nentity, conf.entity_dim * 2)
    end
    #nn.init.uniform_(
    #    tensor=self.entity_embedding,
    #    ###########################TODO##################################
    #    a = -embedding_range,
    #    b = embedding_range
    #)
    #relation_embedding = Flux.params(zeros(nrelation, relation_dim))
    relation_embedding = init(conf.nrelation, conf.relation_dim)
    println("relation_embedding: $(typeof(relation_embedding))")
    #nn.init.uniform_(
    #    tensor=relation_embedding,
    #    a = -embedding_range,
    #    b = embedding_range
    #)

    local offset_embedding, center_net, offset_net, projection_net = Vector{Missing}(undef, 4)
    if conf.geo == "box"
        offset_embedding = init(conf.nrelation, conf.entity_dim)
        println("offset_embedding: $(typeof(offset_embedding))")
        #self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
        #nn.init.uniform_(
        #    tensor=self.offset_embedding,
        #    a=0.,
        #    b=self.embedding_range.item()
        #)
        center_net = CenterIntersection(conf.entity_dim)
        offset_net = BoxOffsetIntersection(conf.entity_dim)
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

    if conf.geo == "box"
        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding,
                                                m.offset_embedding, m.center_net, m.offset_net)
    elseif conf.geo == "vec"
        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding, m.center_net)
    elseif conf.geo == "beta"
        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding,
                                                m.center_net, m.projection_net)
    end
    return model
end
#
#function set_trainable(m::KGReasoning, conf::KGRConfig)
#    if conf.geo == "box"
#        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding,
#                                                m.offset_embedding, m.center_net, m.offset_net)
#    elseif conf.geo == "vec"
#        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding, m.center_net)
#    elseif conf.geo == "beta"
#        @eval Flux.trainable(m::KGReasoning) = (m.gamma, m.embedding_range, m.entity_embedding, m.relation_embedding,
#                                                m.center_net, m.projection_net)
#    end
#end

function forward(m::KGReasoning,conf::KGRConfig,
                 positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    if conf.geo == "box"
        return forward_box(m, conf, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    elseif conf.geo == "vec"
        return forward_vec(m, conf, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    elseif conf.geo == "beta"
        return forward_beta(m, conf, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    end
end

function (m::KGReasoning)(conf::KGRConfig,
                          positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    forward(m, conf, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
end

####
# embed a batch of queries with same structure using Query2box
# queries: a flattened batch of queries
####
function embed_query_box(m::KGReasoning, queries, query_structure, idx)
    #@printf (queries)
    #@printf (query_structure)
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
        if query_structure[0] == "e"
            embedding = m.entity_embedding[:, queries[:, idx]]
            #embedding = torch.index_select(m.entity_embedding, dim=0, index=queries[:, idx])
            offset_embedding = zeros(size(embedding))
            if m.use_cuda
                offset_embedding = zeros(size(embedding)) .|> gpu
            end
            idx += 1
        else
            embedding, offset_embedding, idx = embed_query_box(m, queries, query_structure[0], idx)
        end

        for i in range(1, length(last(query_structure)))
            if last(query_structure)[i] == "n"
                @assert false "box cannot handle queries with negation"
            else
                r_embedding = m.ralation_embedding[:, queries[:, idx]]
                #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                r_offset_embedding = offset_bedding[:, queries[:, idx]]
                #r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                embedding += r_embedding
                offset_embedding += m.func(r_offset_embedding)
            end
            idx += 1
        end
    else
        embedding_list = []
        offset_embedding_list = []
        for i in range(1, length(query_structure))
            embedding, offset_embedding, idx = embed_query_box(m, queries, query_structure[i], idx)
            push!(embedding_list, embedding)
            push!(offset_embedding_list, offset_embedding)
        end
        embedding = m.center_net(vcat(embedding_list))
        offset_embedding = m.offset_net(vcat(offset_embedding_list))
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
            embedding = m.entity_embedding[:,queries[:, idx]]
            #embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
            idx += 1
        else
            embedding, idx = embed_query_vec(m, queries, query_structure[0], idx)
        end

        for i in range(length(last(query_structure)))
            if last(query_structure)[i] == "n"
                @assert false  "vec cannot handle queries with negation"
            else
                r_embedding = m.relation_embedding[:, queries[:, idx]]
                #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                embedding += r_embedding
            end
            idx += 1
        end
    else
        embedding_list = []
        for i in range(1, length(query_structure))
            embedding, idx = embed_query_vec(m, queries, query_structure[i], idx)
            push!(dembedding_list, embedding)
        end
        embedding = m.center_net(vcat(embedding_list))
    end
    return embedding, idx
end

#=
Iterative embed a batch of queries with same structure using BetaE
queries: a flattened batch of queries
=#
function embed_query_beta(m::KGReasoning, conf::KGRConfig, queries, query_structure, idx)
    #=
    Iterative embed a batch of queries with same structure using BetaE
    queries: a flattened batch of queries

    * all the queries have the same structure(entitie and relation indexes)
    =#
    println("embed_query_beta: $(query_structure) idx: $(idx)")
    all_relation_flag = true
    # whether the current query tree has merged to one branch and only need to do relation traversal,
    # e.g., path queries or conjunctive queries after the intersection
    for ele in last(query_structure)
        if !(ele in ["r", "n"])
            all_relation_flag = false
            break
        end
    end
    println("embed_query_beta: last $(last(query_structure))\n $(size(m.entity_embedding))")
    println("embed_query_beta: queries: $(queries) \n entity embedding ndims: $(ndims(m.entity_embedding))")
    if all_relation_flag
        if query_structure[1] == "e"
            embedding = conf.beta_entity_regularizer(selectdim(m.entity_embedding, dims=ndims(m.entity_embedding),
                                                               queries[:, idx]))
            #embedding = m.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
            idx += 1
        else
            alpha_embedding, beta_embedding, idx = embed_query_beta(m, conf, queries, query_structure[1], idx)
            embedding = cat(alpha_embedding, beta_embedding, dim=0)
        end
        for i in range(1, length(last(query_structure)))
            if last(query_structure)[i] == "n"
                @assert (queries[:, idx] == -2).all()
                embedding = 1 ./ embedding
            else
                r_embedding = conf.relation_embedding(queries[:, idx], :)
                #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                embedding = m.projection_net(embedding, r_embedding)
            end
            idx += 1
        end
        ###############################TODO####################################
        alpha_embedding, beta_embedding = chunk(embedding, 2, dim=ndims(embedding))
    else
        alpha_embedding_list = []
        beta_embedding_list = []
        for i in range(1, length(query_structure))
            alpha_embedding, beta_embedding, idx = embed_query_beta(m, conf, queries, query_structure[i], idx)
            push!(alpha_embedding_list, alpha_embedding)
            push!(beta_embedding_list, beta_embedding)
        end
        alpha_embedding, beta_embedding = m.center_net(cat(alpha_embedding_list), cat(beta_embedding_list))
    end
    return alpha_embedding, beta_embedding, idx
end

#============================================
    transform 2u queries to two 1p queries
    transform up queries to two 2p queries
============================================#
function transform_union_query(m::KGReasoning, queries, query_structure)

    if m.query_name_dict[query_structure] == "2u-DNF"
        queries = queries[:, 1:(size(queries, 2) - 1)] # remove union -1
    elseif m.query_name_dict[query_structure] == "up-DNF"
        queries = cat(cat(queries[:, 1:2], queries[:, 5:6], dims=1), cat(queries[:, 2:4], queries[:, 5:6], dims=1), dims=1)
    end
    queries = reshape(queries, :, size(queries)[1]*2)
    return queries
end

function transform_union_structure(m::KGReasoning, query_structure)
    if m.query_name_dict[query_structure] == "2u-DNF"
        return ("e", ("r",))
    elseif m.query_name_dict[query_structure] == "up-DNF"
        return ("e", ("r", "r"))
    end
end

function cal_logit_beta(m::KGReasoning, entity_embedding, query_dist)
    ##########################TODO#######################################
    alpha_embedding, beta_embedding = chunk(entity_embedding, 2)
    entity_dist = Distributions.Beta(alpha_embedding, beta_embedding)
    logit = m.gamma - normDims(Distributions.KLDivergence(entity_dist, query_dist), 1)
    return logit
end

function forward_beta(m::KGReasoning, conf::KGRConfig,
                      positive_sample, negative_sample, subsampling_weight,
                      batch_queries_dict, batch_idxs_dict)
    local all_idxs, all_alpha_embeddings, all_beta_embeddings,
    all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings

    for query_structure in keys(batch_queries_dict)
        println("forward_beta: query structure $(query_structure)")
        if occursin('u', conf.query_name_dict[query_structure]) && occursin("DNF", conf.query_name_dict[query_structure])
            alpha_embedding, beta_embedding, _ = embed_query_beta(m,
                                                                  transform_union_query(m, batch_queries_dict[query_structure],
                                                                                        query_structure),
                                                                  transform_union_structure(m, query_structure),
                                                                  0)
            push!(all_union_idxs, batch_idxs_dict[query_structure])
            #all_union_idxs.extend(batch_idxs_dict[query_structure])
            push!(all_union_alpha_embeddings, alpha_embedding)
            push!(all_union_beta_embeddings, beta_embedding)
        else
            alpha_embedding, beta_embedding, _ = embed_query_beta(m, batch_queries_dict[query_structure],
                                                                  query_structure,
                                                                  0)
            push!(all_idxs, batch_idxs_dict[query_structure])
            #all_idxs.extend(batch_idxs_dict[query_structure])
            push!(all_alpha_embeddings, alpha_embedding)
            push!(all_beta_embeddings, beta_embedding)
        end
    end

    if length(all_alpha_embeddings) > 0
        #all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
        all_alpha_embeddfings = reduce((x, y) -> cat(x, y, dims=ndims(x)), all_alpha_embeddings)
        all_beta_embeddings = reduce(all_beta_embeddings) do x, y
                                         cat(x, y, dims=ndims(x))
                                     end
        all_beta_embeddings= unsqueeze(all_beta_embeddings, dims = ndims(all_beta_embeddings))
        all_dists = Distributions.Beta(all_alpha_embeddings, all_beta_embeddings)
    end

    if length(all_union_alpha_embeddings) > 0
        #all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
        #all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
        #all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
        #all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
        #all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)
        all_union_alpha_embeddings = reduce(all_union_alpha_embeddings) do x, y
                                             cat(x, y, dim=ndims(x))
                                         end
        #all_union_alpha_embeddings = cat(all_union_alpha_embeddings, dims = ndims(all_union_alpha_embeddings) + 1)
        all_union_alpha_embeddings = unsqueeze(all_union_alpha_embeddings, dims = ndims(all_union_alpha_embeddings))
        all_union_beta_embeddings = reduce(all_union_beta_embeddings) do x, y
                                             cat(x, y, dim=ndims(x))
                                         end
        all_union_beta_embeddings = unsqueeze(all_union_beta_embeddings, dims = ndims(all_union_beta_embeddings))
        #################################################################################################
        #all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
        #all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
        all_union_alpha_embeddings = reshape(all_union_alpha_embeddings, :, 1, 2,
                                             div(size(all_union_alpha_embeddings, ndims(all_union_alpha_embedding)), 2))
        all_union_beta_embeddings = reshape(all_union_beta_embeddings, :, 1, 2,
                                             div(size(all_union_beta_embeddings, ndims(all_union_beta_embedding)), 2))

        all_union_dists = Distributions.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]
    end

    if typeof(positive_sample) != typeof(Nothing)
        if length(all_alpha_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_regular);
            positive_embedding = conf.entity_regularizer(unsqueeze(entity_embedding_select,
                                                                   ndims(entity_embedding_select)))
            positive_logit = cal_logit_beta(m, positive_embedding, all_dists)
        else
            positive_logit = [] .|> Flux.get_device()
        end

        if length(all_union_alpha_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs] # positive samples for union queries in this batch

            entity_embedding_select = selectdim(m.entity_embedding,
                                                ndims(m.entity_embedding),
                                                positive_sample_union);
            entity_embedding_select_unsqueeze = unsqueeze(entity_embedding_select,
                                                        dims=ndims(entity_embedding_select) - 1);
            positive_embedding = conf.entity_regularizer(entity_embedding_select_unsqueeze)
            positive_union_logit = cal_logit_beta(m, positive_embedding, all_union_dists)
            positive_union_logit = max(positive_union_logit, dim=1)[0]
        else
            positive_union_logit = [] .|> Flux.get_device()
        end
        positive_logit = cat(positive_logit, positive_union_logit, dims=ndims(positive_logit))
    else
        positive_logit = nothing
    end

    if typeof(negative_sample) != typeof(nothing)
        if length(all_alpha_embeddings) > 0
            negative_sample_regular = negative_sample[all_idxs]
            batch_size, negative_size = negative_sample_regular.shape
            #negative_embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
            negative_embedding = m.entity_regularizer(reshape(reshape(selectdim(m.entity_embedding, ndims(m.entity_embedding), negative_sample_regular), :), :, negative_size, batch_size))
            negative_logit = cal_logit_beta(m, negative_embedding, all_dists)
        else
            ########################## TODO ##################################
            #negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = [] .|> Flux.get_device()
        end

        if length(all_union_alpha_embeddings) > 0
            negative_sample_union = negative_sample[all_union_idxs]
            batch_size, negative_size = size(negative_sample_union)
            negative_embedding = conf.entity_regularizer(reshape(reshape(selectdim(m.entity_embedding, 0, negative_sample_union), :), (:, negative_size, 1, batch_size)))
            negative_union_logit = cal_logit_beta(m, negative_embedding, all_union_dists)
            negative_union_logit = max(negative_union_logit, dim=2)[0]
        else
            ######################### TODO  ###################################
            negative_union_logit = [] .|> Flux.get_device()
        end

        negative_logit = cat(negative_logit, negative_union_logit, dim=ndims(negative_logit))
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
end

function cal_logit_box(m::KGReasoning, entity_embedding, query_center_embedding, query_offset_embedding)
    delta = abs(entity_embedding - query_center_embedding)
    distance_out = Flux.relu(delta - query_offset_embedding)
    distance_in = min(delta, query_offset_embedding)
    logit = m.gamma - normDims(distance_out, 1; dims=0) - m.cen * normDims(distance_in, 1, dims=0)
    return logit
end

function forward_box(m::KGReasoning, conf::KGRConfig, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
    all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
    for query_structure in batch_queries_dict
        if "u" in m.query_name_dict[query_structure]
            center_embedding, offset_embedding, _ = \
                embed_query_box(m, transform_union_query(batch_queries_dict[query_structure],
                                                         query_structure),
                                transform_union_structure(m, query_structure),
                                0)
            push!(all_union_center_embeddings, center_embedding)
            push!(all_union_offset_embeddings, offset_embedding)
            push!(all_union_idxs, batch_idxs_dict[query_structure])
        else
            center_embedding, offset_embedding, _ = embed_query_box(m, batch_queries_dict[query_structure],
                                                                    query_structure,
                                                                    0)
            push!(all_center_embeddings, center_embedding)
            push!(all_offset_embeddings, offset_embedding)
            push!(all_idxs, batch_idxs_dict[query_structure])
        end
    end

    if length(all_center_embeddings) > 0 && length(all_offset_embeddings) > 0
        all_center_embeddings_cat = reduce(all_center_embeddings) do x, y
                                          cat(x, y, dims=ndims(x))
                                    end
        all_center_embeddings_cat_unsqueeze = unsqueeze(all_center_embeddings_cat,
                                          dims = ndims(all_center_embeddings_cat) - 1)

        all_offset_embeddings_cat = reduce(all_offset_embeddings) do x, y
                                          cat(x, y, dims=ndims(x))
                                    end
        all_offset_embeddings_cat_unsqueeze = unsqueeze(all_offset_embeddings_cat,
                                                        dims = ndims(all_offset_embeddings_cat) - 1)
        #all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
    end

    if length(all_union_center_embeddings) > 0 && length(all_union_offset_embeddings) > 0
        #all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
        #all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
        all_union_center_embeddings_cat = reduce(all_union_center_embeddings) do x, y
                                              cat(x, y, dims=ndims(x))
                                          end
        all_union_center_embeddings_cat_unsqueeze = unsqueeze(all_union_center_embeddings_cat,
                                                              dims = ndims(all_union_center_embeddings_cat) - 1)
        all_union_offset_embeddings_cat = reduce(all_union_offset_embeddings) do x, y
                                              cat(x, y, dims=ndims(x))
                                          end
        all_union_offset_embeddings_cat_unsqueeze = unsqueeze(all_union_offset_embeddings_cat,
                                                              dims = ndims(all_offset_embeddings_cat) - 1)
        #all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
        #all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)
        all_union_center_embeddings = reshape(all_union_center_embeddings_cat_unsqueeze,
                                              :, 1, 2, div(ndims(all_union_center_embeddings_cat_unsqueeze), 2))
        all_union_offset_embeddings = reshape(all_union_offset_embeddings_cat_unsqueeze,
                                              :, 1, 2, div(ndims(all_union_offset_embeddings_cat_unsqueeze), 2))
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]
    end

    if typeof(positive_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs]
            entity_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), positive_sample_regular)
            positive_embedding = unsqueeze(entity_embedding_select, ndims(entity_embedding_select) - 1)
            positive_logit = cal_logit_box(m, positive_embedding, all_center_embeddings, all_offset_embeddings)
        else
            #positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = [] .|> Flux.get_device()
        end

        if length(all_union_center_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs]
            entity_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), positive_sample_union)
            entity_embedding_select_unquezze = unsqueeze(entity_embedding_select, ndims(entity_embedding_select) - 1)
            positive_embedding = unsqueeze(entity_embedding_select_unquezze, ndims(entity_embedding_select_unquezze) - 1)
            positive_union_logit = cal_logit_box(m, positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
            positive_union_logit = max(positive_union_logit, dims=ndims(positive_union_logit) - 1)[1]
        else
            #positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_union_logit = [] .|> Flux.get_device()
        end
        positive_logit = reduce([positive_logit, positive_union_logit]) do x, y
                              cat(x, y, dim=ndims(x))
                         end
    else
        positive_logit = nothing
    end

    if typeof(negative_sample) != typeof(nothing)
        if len(all_center_embeddings) > 0
            negative_sample_regular = negative_sample[all_idxs]
            batch_size, negative_size = size(negative_sample_regular)
            entity_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), reshape(negative_sample_regular, :))
            negative_embedding = reshape(entity_embedding_select, :, negative_size, batch_size)
            negative_logit = cal_logit_box(m, negative_embedding, all_center_embeddings, all_offset_embeddings)
        else
            #negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = [] .|> Flux.get_device()
        end

        if length(all_union_center_embeddings) > 0
            negative_sample_union = negative_sample[all_union_idxs]
            batch_size, negative_size = size(negative_sample_union)
            entity_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), reshape(negative_sample_union, :))
            negative_embedding = reshape(entity_embedding_select, :, negative_size, 1, batch_size)
            negative_union_logit = cal_logit_box(m, negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
            negative_union_logit = max(negative_union_logit, dims=ndims(negative_union_logit) - 1)[1]
        else
            #negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_union_logit = [] .|> Flux.get_device()
        end
        negative_logit = reduce([negative_logit, negative_union_logit]) do x, y
                              cat(x, y, dim=ndims(x))
                         end
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
end

function cal_logit_vec(m::KGReasoning, entity_embedding, query_embedding)
    distance = entity_embedding - query_embedding
    logit = m.gamma - normDims(distance, 1, dim=2)
    return logit
end

function forward_vec(m::KGReasoning, conf::KGRConfig, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
    all_center_embeddings, all_idxs = [], []
    all_union_center_embeddings, all_union_idxs = [], []
    for query_structure in batch_queries_dict
        if "u" in m.query_name_dict[query_structure]
            center_embedding, _ = embed_query_vec(m, transform_union_query(m, batch_queries_dict[query_structure],
                                                                           query_structure),
                                                  transform_union_structure(query_structure), 0)
            push!(all_union_center_embeddings, center_embedding)
            append!(all_union_idxs, batch_idxs_dict[query_structure])
        else
            center_embedding, _ = embed_query_vec(m, batch_queries_dict[query_structure], query_structure, 0)
            push!(all_center_embeddings, center_embedding)
            append!(all_idxs, batch_idxs_dict[query_structure])
        end
    end

    if length(all_center_embeddings) > 0
        all_center_embeddings_cat = reduce(all_center_embeddings) do x, y
                                        cat(x, y, dims = ndims(x))
                                    end
        all_center_embeddings = unsqueeze(all_center_embeddings_cat, ndims(all_center_embeddings_cat) - 1)
    end

    if length(all_union_center_embeddings) > 0
        all_union_center_embeddings_cat = reduce(all_union_center_embeddings) do x, y
                                              cat(x, y, dims = ndims(x))
                                          end
        all_union_center_embeddings_unsqueeze = unsqueeze(all_union_center_embeddings_cat, ndims(all_union_center_embeddings_cat) - 1)
        #all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
        all_union_center_embeddings = reshape(all_union_center_embeddings_unsqueeze,
                                              :, 1, 2, div(size(all_union_center_embeddings,
                                                                ndims(all_union_center_embeddings)),
                                                           2))
    end

    if typeof(subsampling_weight) != typeof(nothing)
        subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]
    end

    if typeof(positive_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            positive_sample_regular = positive_sample[all_idxs]
            positive_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), positive_sample_regular)
            positive_embedding = unsqueeze(positive_embedding_select, ndims(positive_embedding_select) - 1)
            positive_logit = cal_logit_vec(m, positive_embedding, all_center_embeddings)
        else
            positive_logit = [] .|> Flux.get_device()
        end

        if length(all_union_center_embeddings) > 0
            positive_sample_union = positive_sample[all_union_idxs]
            positive_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding), positive_sample_regular)
            positive_embedding_unsqueeze = unsqueeze(positive_embedding_select, ndims(positive_embedding_select) - 1)
            positive_embedding = unsqueeze(positive_embedding_unsqueeze, ndims(positive_embedding_unsqueeze) - 1)
            positive_union_logit = cal_logit_vec(m, positive_embedding, all_union_center_embeddings)
            positive_union_logit = max(positive_union_logit, dims=ndims(positive_union_logit) - 1)[1]
        else
            positive_union_logit = [] .|> Flux.get_device()
        end
        positive_logit = reduce([positive_logit, positive_union_logit]) do x, y
                             cat(x, y, dims=ndims(x))
                         end
    else
        positive_logit = nothing
    end

    if typeof(negative_sample) != typeof(nothing)
        if length(all_center_embeddings) > 0
            negative_sample_regular = negative_sample[all_idxs]
            batch_size, negative_size = size(negative_sample_regular)
            entity_embedding_select = selectdim(m.entity_embedding, 0, reshape(negative_sample_regular, :))
            negative_embedding = reshape(entity_embedding_select, :, negative_size, batch_size)
            negative_logit = cal_logit_vec(m, negative_embedding, all_center_embeddings)
        else
            negative_logit = [] .|> Flux.get_device()
        end

        if length(all_union_center_embeddings) > 0
            negative_sample_union = negative_sample[all_union_idxs]
            batch_size, negative_size = size(negative_sample_union)
            entity_embedding_select = selectdim(m.entity_embedding, ndims(m.entity_embedding, reshape(negative_sample_union, :)))
            negative_embedding = reshape(entity_embedding_select, :, negtive_size, 1, batch_size)
            negative_union_logit = cal_logit_vec(m, negative_embedding, all_union_center_embeddings)
            negative_union_logit = max(negative_union_logit, dim=ndims(negative_union_logit) -1)[0]
        else
            negative_union_logit = [] .|> Flux.get_device()
        end

        negative_logit = reduce([negative_logit, negative_union_logit]) do x, y
                             cat(x, y, dim=ndims(x))
                         end
    else
        negative_logit = nothing
    end

    return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
end
#=================================================================================
function mean_loss(y_bar)
    negative_logsigmoid = Flux.logsigmoid(-y_bar[:negative_logit])
    negative_score = mean.(negative_logsigmoid, dims=ndims(negative_logsigmoid))
    positive_logsigmoid =Flux.logsigmoid(-y_bar[:positive_logit])
    positive_score = squeeze(positive_logsigmoid, dim=ndims(positive_logsigmoid))
    positive_sample_loss = - sum(y_bar[:subsampling_weight] * positive_score)
    negative_sample_loss = - sum(y_bar[:subsampling_weight] * negative_score)
    positive_sample_loss /= sum(y_bar[:subsampling_weight])
    negative_sample_loss /= sum(y_bar[:subsampling_weight])

    loss = (positive_sample_loss + negative_sample_loss)/2
end
===============================================================================#

function loss(model, data)
    ########################################################################################################
    #model.train() # set model as train mode
    #optimizer.zero_grad() # clear grad, set to zero
    positive_sample, negative_sample, subsampling_weight, querie, query_structures = data

    #batch_queries_dict = collections.defaultdict(list)
    #batch_idxs_dict = collections.defaultdict(list)
    batch_queries_dict = Dict{Any, Any}()
    batch_idxs_dict = Dict{Any, Any}()
    for (i, query) in enumerate(querie) # group queries with same structure
        push!(get!(batch_queries_dict, query_structures[i], []), query)
        push!(get!(batch_idxs_dict, query_structures[i], []), i)
    end

    for query_structure in batch_queries_dict
        if args["cuda"]
            batch_queries_dict[query_structure] = Int64.(batch_queries_dict[query_structure]) .|> gpu
        else
            batch_queries_dict[query_structure] = Int64.(batch_queries_dict[query_structure])
        end
    end

    if args["cuda"]
        positive_sample = positive_sample |> gpu
        negative_sample = negative_sample |> gpu
        subsampling_weight = subsampling_weight |> gpu
    end

    opt_grads = Flux.gradient(model) do m
        positive_logit, negative_logit,
        subsampling_weight, _ = model(positive_sample, negative_sample,
                                      subsampling_weight, batch_queries_dict, batch_idxs_dict)
        negative_logsigmoid = Flux.logsigmoid(negative_logit)
        negative_score = mean.(negative_logsigmoid, dims=ndims(negative_logsigmoid))
        positive_logsigmoid =Flux.logsigmoid(positive_logit)
        positive_score = squeeze(positive_logsigmoid, dim=ndims(positive_logsigmoid))
        positive_sample_loss = - sum(subsampling_weight * positive_score)
        negative_sample_loss = - sum(subsampling_weight * negative_score)
        positive_sample_loss /= sum(subsampling_weight)
        negative_sample_loss /= sum(subsampling_weight)

        loss = (positive_sample_loss + negative_sample_loss)/2
    end
    return 0
end

# @staticmethod
function train_step(model::KGReasoning, conf::KGRConfig, opt_state, data, args, step)
    #opti_stat = Flux.setup(model, optimizer)

    #println("train_step data :: $(data)")

    #Flux.train!(loss, model, data, opt_state)# do model, data

    ########################################################################################################
    #model.train() # set model as train mode
    #optimizer.zero_grad() # clear grad, set to zero
    positive_sample, negative_sample, subsampling_weight, queries, query_structures = data
    println("train_data: $(positive_sample), $(negative_sample), $(subsampling_weight), $(queries), $(query_structures)")
    #batch_queries_dict = collections.defaultdict(list)
    #batch_idxs_dict = collections.defaultdict(list)
    batch_queries_dict = Dict{Any, Any}()
    batch_idxs_dict = Dict{Any, Any}()
    for (i, query) in enumerate(queries) # group queries with same structure
        push!(get!(batch_queries_dict, query_structures[i], []), query)
        push!(get!(batch_idxs_dict, query_structures[i], []), i)
    end
    @info "train_step: $(batch_queries_dict)"
    @info "train_step: $(batch_idxs_dict)"

    for query_structure in keys(batch_queries_dict)
        if args["cuda"]
            batch_queries_dict[query_structure] = Vector{Int64}.(batch_queries_dict[query_structure]) .|> gpu
        else
            batch_queries_dict[query_structure] = Vector{Int64}.(batch_queries_dict[query_structure])
        end
    end

    if args["cuda"]
        positive_sample = positive_sample |> gpu
        negative_sample = negative_sample |> gpu
        subsampling_weight = subsampling_weight |> gpu
    end

    println("train_step positive_sample: $(positive_sample)")
    println("train_step negative_sample: $(negative_sample)")
    println("train_step subsampling_weight: $subsampling_weight")

    local positive_sample_loss, negative_sample_loss, loss
    opt_grads = Flux.gradient(model) do m
        positive_logit, negative_logit,
        subsampling_weight, _ = m(conf, positive_sample, negative_sample,
                                  subsampling_weight, batch_queries_dict, batch_idxs_dict)
        negative_logsigmoid = Flux.logsigmoid(negative_logit)
        negative_score = mean.(negative_logsigmoid, dims=ndims(negative_logsigmoid))
        positive_logsigmoid =Flux.logsigmoid(positive_logit)
        positive_score = squeeze(positive_logsigmoid, dim=ndims(positive_logsigmoid))
        positive_sample_loss = - sum(subsampling_weight * positive_score)
        negative_sample_loss = - sum(subsampling_weight * negative_score)
        positive_sample_loss /= sum(subsampling_weight)
        negative_sample_loss /= sum(subsampling_weight)

        loss = (positive_sample_loss + negative_sample_loss)/2
    end
    #end
    #=========================================================================
    negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
    positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
    positive_sample_loss = - (subsampling_weight * positive_score).sum()
    negative_sample_loss = - (subsampling_weight * negative_score).sum()
    positive_sample_loss /= subsampling_weight.sum()
    negative_sample_loss /= subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss)/2
    loss.backward()
    optimizer.step()
    ==========================================================================#
    log = Dict{
        "positive_sample_loss": positive_sample_loss.item(),
        "negative_sample_loss": negative_sample_loss.item(),
        "loss": loss.item(),
    }
    return log
end

#@staticmethod
function test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False)
#    model.eval()

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
