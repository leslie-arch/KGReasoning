#!/home/leslie/julia/1.10.0/bin/julia

module KGDataset

using MLUtils
using Pickle

include("utils.jl")

export numobs, getobs, getindex, TrainDataset, TestDataset, load_data, iterate

abstract type Dataset end
struct TrainDataset <: Dataset
    queries::Vector{Any}
    answer::Dict{Any, Any}
    nentity::Int
    nrelation::Int
    negative_sample_size::Int
    count::Dict{Tuple, Int}
end
#=
function collate_fn(data::TrainDataset)
positive_sample = cat([_[0] for _ in data], dim=0)
negative_sample = stack([_[1] for _ in data], dim=0)
subsample_weight = cat([_[2] for _ in data], dim=0)
query = [_[3] for _ in data]
query_structure = [_[4] for _ in data]
return positive_sample, negative_sample, subsample_weight, query, query_structure
end
=#
function count_frequency(queries, answer, start=4)
    count = Dict{Tuple, Int}()
    for (query, _) in queries
        #println("$(query) -- $(length(answer[query]))")
        count[query] = start + length(answer[query])
    end
    return count
end

function TrainDataset(queries, answers, nentity, nrelation, negtaive_sample_size)
    count = count_frequency(queries, answers);
    return TrainDataset(queries, answers, nentity, nrelation, negtaive_sample_size, count)
end

#Authors of custom data containers should implement Base.length for their type instead of numobs.
#numobs should only be implemented for types where there is a difference between numobs and Base.length
#(such as multi-dimensional arrays).

#function numobs(data::TrainDataset)
#    return length(data.queries)
#end

function Base.length(data::TrainDataset)
    return length(data.queries)
end

function Base.getindex(data::TrainDataset, idx)
    if isa(idx, Int) idx = [idx] end
    println("Base.getindex idx: $(idx)")

    query_name_vec = data.queries[idx]
    query = first.(data.queries[idx])
    query_structure = last.(data.queries[idx])
    @info "Base.getindex: [$(idx)] -> $(query_name_vec)"
    @info "query: $(query)"
    @info "query_structure: $(query_structure)"

    tail = map(s->rand(collect(data.answer[s])), query)
    subsampling_weight = map(q->data.count[q], query)
    subsampling_weight = sqrt.(1 ./ subsampling_weight)

    local positive_sample, negative_sample

    negative_sample = []
    map(query) do q
        negative_sample_item = []
        partial_negative_sample_size = 0
        partial_negative_sample_list = []
        while partial_negative_sample_size < data.negative_sample_size
            partial_negative_sample_list = rand(1:data.nentity, data.negative_sample_size*2)
            # check whether the items in ar1 belong to ar2, return a vector
            # has the same length with ar1, filled with true or false
            #mask = np.in1d(negative_sample, data.answer[query],
            #               assume_unique=true, invert=true)

            avail_index = indexin(partial_negative_sample_list, collect(data.answer[q]))
            mask = falses(data.negative_sample_size * 2)
            map(enumerate(avail_index)) do (x, y)
                if y != nothing
                    #println("getobs set mask at $x  value: $(negative_sample[x])")
                    mask[x] = true
                end
            end
            reverse!(mask)

            partial_negative_sample_list = partial_negative_sample_list[mask]
            append!(negative_sample_item, partial_negative_sample_list)
            partial_negative_sample_size += length(partial_negative_sample_list)
        end

        #negative_sample = stack((negative_sample, negative_sample_list[1:data.negative_sample_size]))
        push!(negative_sample, negative_sample_item[1:data.negative_sample_size])
        #negative_sample = negative_sample # original: torch.from_numpy
    end
    positive_sample = convert.(Float64, tail)
    @info "tail $(tail)"
    @info "positive_sample: $(positive_sample)"
    return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure
end

#Authors of custom data containers should implement Base.getindex for their type instead of getobs.
#getobs should only be implemented for types where there is a difference between getobs and Base.getindex
#(such as multi-dimensional arrays).

#function getobs(data::TrainDataset, idx...)
#    println("getobs idx: $(idx)")
#    return map(x -> getindex(data, x), idx);
#end

#=
struct SingleDirectionalOneShotIterator
data_loader::DataLoader
end


function iterate(iter::SingleDirectionalOneShotIterator)
state = 1
if numobs(iter.data_loader.data) <= 0
return nothing
end

return ( getobs(iter.data_loader.data, state), state + 1 )
end

function iterate(iter::SingleDirectionalOneShotIterator, state = 1)
if length(iter.data_loader.data.queries) < state
return nothing
end

return ( getobs(iter.data_loader.data, state), state + 1 )
end
=#

struct TestDataset <: Dataset
    queries::Vector{Any}
    answer::Dict{Any, Any}
    nentity::Int
    nrelation::Int
    negative_sample_size::Int
    count::Dict{Tuple, Int}
end

function numobs(data::TestDataset)
    return length(data.queries)
end

function getobs(data::TestDataset, idx)
    query = data.queries[idx][0]
    query_structure = data.queries[idx][1]
    tail = rand(data.answer[query])
    subsampling_weight = data.count[query]
    subsampling_weight = sqrt.(1 / [subsampling_weight])
    negative_sample_list = []
    negative_sample_size = 0
    while negative_sample_size < m.negative_sample_size
        negative_sample = rand(1:data.nentity, data.negative_sample_size*2)
        avail_index = indexin(negative_sample, collect(data.answer[query]))
        mask = falses(data.negative_sample_size * 2)
        map(enumerate(avail_index)) do (x, y)
            if y != nothing
                println("getobs set mask at $x int  $(Int(x)) max $(length(negative_sample))")
                println("value: $(negative_sample[x])")
                mask[x] = true
            end
        end
        reverse!(mask)

        negative_sample = negative_sample[mask]
        append!(negative_sample_list, negative_sample)
        negative_sample_size += length(negative_sample)
    end
    negative_sample = stack(negative_sample_list)[1:data.negative_sample_size]
    negative_sample = negative_sample # original: torch.from_numpy
    positive_sample = convert.(Float64, [tail])

    return positive_sample, negative_sample, subsampling_weight, flatten(query), (query_structure)
end

function load_data(args, queries_dict)
    @info "loading data...."
    data_path = args["data_path"];
    f_train_queries = "train-queries.pkl";
    f_train_answers = "train-answers.pkl";
    f_valid_queries = "valid-queries.pkl";
    f_valid_hard_answers = "valid-hard-answers.pkl";
    f_valid_easy_answers = "valid-easy-answers.pkl";
    f_test_queries = "test-queries.pkl";
    f_test_hard_answers = "test-hard-answers.pkl";
    f_test_easy_answers = "test-easy-answers.pkl";

    train_queries = Pickle.load(open(joinpath(data_path, f_train_queries)));
    train_answers = Pickle.load(open(joinpath(data_path, f_train_answers)));
    valid_queries = Pickle.load(open(joinpath(data_path, f_valid_queries)));
    valid_hard_answers = Pickle.load(open(joinpath(data_path, f_valid_hard_answers)));
    valid_easy_answers = Pickle.load(open(joinpath(data_path, f_valid_easy_answers)));
    test_queries = Pickle.load(open(joinpath(data_path, f_test_queries)));
    test_hard_answers = Pickle.load(open(joinpath(data_path, f_test_hard_answers)));
    test_easy_answers = Pickle.load(open(joinpath(data_path, f_test_easy_answers)));

    valid_tasks = split(args["tasks"], ".")
    all_tasks = keys(queries_dict)

    # remove tasks not in args.tasks
    if (valid_tasks !=  all_tasks)
        for name in all_tasks
            if 'u' in name
                name, evaluate_union = split(name, "-")
            else
                evaluate_union = args["evaluate_union"]
            end

            if !(name in valid_tasks) || evaluate_union != args["evaluate_union"]
                query_structure = queries_dict[eval(if !('u' in name) name else join([name, evaluate_union], "-") end)]
                #println("load_data: deleteing structure...:\n $(query_structure)")
                if haskey(train_queries, query_structure)
                    delete!(train_queries, query_structure);
                end
                if haskey(valid_queries, query_structure)
                delete!(valid_queries, query_structure);
                end
                if haskey(test_queries, query_structure)
                    delete!(test_queries, query_structure)
                end
            end
        end
    end
    @info "load data....Done"
    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers
end

end # end module
