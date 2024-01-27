#!/home/leslie/julia/1.10.0/bin/julia

module KGDataset

import MLUtils: DataLoader
import Pickle

export numobs, getobs, TrainDataset, TestDataset, SingleDirectionalOneShotIterator, load_data

abstract type Dataset end

struct TrainDataset <: Dataset
    queries::Vector{Any}
    answer::Dict{Any, Any}
    nentity::Int
    nrelation::Int
    negative_sample_size::Int
end

function numobs(data::TrainDataset)
    return length(data.queries)
end

function getobs(data::TrainDataset, idx)
    return data.queries[idx]
end

struct TestDataset <: Dataset
    queries::Vector{Any}
    nentity::Int
    nrelation::Int
end

function numobs(data::TestDataset)
    return length(data.queries)
end

function getobs(data::TestDataset, idx)
    return data.queries[idx]
end

struct SingleDirectionalOneShotIterator
    data_loader::DataLoader
end

function Base.iterate(iter::SingleDirectionalOneShotIterator, state = 1)
    if length(iter.data_loader.data.queries) < state
        return nothing
    end

    return ( iter.data_loader.data.queries[state], state + 1 )
end

function load_data(args, tasks, all_tasks, query_dict)
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

    # remove tasks not in args.tasks
    for name in all_tasks
        if 'u' in name
            name, evaluate_union = split(name, "-")
        else
            evaluate_union = args["evaluate_union"]
        end
        if !(name in tasks) || evaluate_union != args["evaluate_union"]
            query_structure = query_dict[eval(if !('u' in name) name else join([name, evaluate_union], "-") end)]
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
    @info "load data....Done"
    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers
end

function flatten_query(queries)
    all_queries = []
    for query_structure in keys(queries)
        list_queries = collect(queries[query_structure])
        ttt = [(query, query_structure) for query in list_queries]
        #println("query_structure key: $(query_structure) queries length: $(length(list_queries)) tuple length: $(length(ttt))");
        append!(all_queries, [(query, query_structure) for query in list_queries])
    end
    return all_queries
end


function iterate(iter::SingleDirectionalOneShotIterator)
    return ( iter.data_loader.data.queries[1], 1 )
end

function length(iter::SingleDirectionalOneShotIterator)
    return length(iter.data_loader.data.queries);
end

function next(iter::SingleDirectionalOneShotIterator, state)
    if length(iter.data_loader.data.queries) < state
        return nothing
    end

    return ( iter.data_loader.data.queries[state], state + 1)
end

function isdone(iter::SingleDirectionalOneShotIterator, state)
    return length(iter.data_loader.data.queries) < state
end

end # end module
