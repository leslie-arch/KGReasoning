#!/usr/bin/env julia

using MLUtils
using Random
using ArgParse
using LoggingExtras, TensorBoardLogger
using Dates
using Lux
using LuxCUDA
using Optimisers
using CUDA
using JLD2
using ComponentArrays

include("dataloader.jl")
include("model.jl")
include("utils.jl")

using .KGDataset
using .KGModel

f_dir = "dataset";
f_model = "FB15k-betae";

query_name_dict = Dict{Tuple, String}(("e",("r",))=> "1p",
                                      ("e", ("r", "r"))=> "2p",
                                      ("e", ("r", "r", "r"))=> "3p",
                                      (("e", ("r",)), ("e", ("r",)))=> "2i",
                                      (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))=> "3i",
                                      ((("e", ("r",)), ("e", ("r",))), ("r",))=> "ip",
                                      (("e", ("r", "r")), ("e", ("r",)))=> "pi",
                                      (("e", ("r",)), ("e", ("r", "n")))=> "2in",
                                      (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n")))=> "3in",
                                      ((("e", ("r",)), ("e", ("r", "n"))), ("r",))=> "inp",
                                      (("e", ("r", "r")), ("e", ("r", "n")))=> "pin",
                                      (("e", ("r", "r", "n")), ("e", ("r",)))=> "pni",
                                      (("e", ("r",)), ("e", ("r",)), ("u",))=> "2u-DNF",
                                      ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))=> "up-DNF",
                                      ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",))=> "2u-DM",
                                      ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r"))=> "up-DM"
                                      );
name_query_dict = Dict{String, Tuple}((y => x) for (x, y) in query_name_dict);
all_tasks = collect(keys(name_query_dict));

function parse_cmdargs(args::Vector{String})
    s = ArgParseSettings(
        description = "Training and Testing Knowledge Graph Embedding Models",
        usage = "julia --project=[/path/to/project] src/$(@__FILE__) [<args>] [-h | --help]"
    )

    @add_arg_table s begin
        "--cuda"
        action= :store_true
        help="use GPU"
        "--train"
        action= :store_true
        help="do train"
        "--valid"
        action= :store_true
        help="do valid"
        "--test"
        action= :store_true
        help="do test"
        "--data_path"
        arg_type=String
        default= nothing
        help="KG data path"
        "-n", "--negative_sample_size"
        default=128
        arg_type=Int
        help="negative entities sampled per query"
        "-d", "--hidden_dim"
        default=500
        arg_type=Int
        help="embedding dimension"
        "-g", "--gamma"
        default=12.0
        arg_type=Float64
        help="margin in the loss"
        "-b", "--batch_size"
        default=1024
        arg_type=Int
        help="batch size of queries"
        "--test_batch_size"
        default=1
        arg_type=Int
        help="valid/test batch size"
        "--learning_rate"
        default=0.0001
        arg_type=Float64
        "--cpu"
        default=10
        arg_type=Int
        help="used to speed up torch.dataloader"
        "--save_path"
        default="."
        arg_type=String
        help="no need to set manually, will configure automatically"
        "--max_steps"
        default=100000
        arg_type=Int
        help="maximum iterations to train"
        "--warm_up_steps"
        default=nothing
        arg_type=Int
        help="no need to set manually, will configure automatically"
        "--save_checkpoint_steps"
        default=500
        arg_type=Int
        help="save checkpoints every xx steps"
        "--valid_steps"
        default=100
        arg_type=Int
        help="evaluate validation queries every xx steps"
        "--log_steps"
        default=100
        arg_type=Int
        help="train log every xx steps"
        "--test_log_steps"
        default=100
        arg_type=Int
        help="valid/test log every xx steps"
        "--nentity"
        arg_type=Int
        default=0
        help="DO NOT MANUALLY SET"
        "--nrelation"
        arg_type=Int
        default=0
        help="DO NOT MANUALLY SET"
        "--geo"
        default="vec"
        arg_type=String
        help="the reasoning model, vec for GQE, box for Query2box, beta for BetaE"
        "--print_on_screen"
        action= :store_false
        "--tasks"
        default="1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"
        arg_type=String
        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task"
        "--seed"
        default=1024
        arg_type=Int
        help="random seed"
        "--beta_mode"
        default="(1600,2)"
        arg_type=String
        help="(hidden_dim,num_layer) for BetaE relational projection"
        "--box_mode"
        default="(nothing,0.02)"
        arg_type=String
        help="(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist"
        "--prefix"
        default=nothing
        arg_type=String
        help="prefix of the log path"
        "--checkpoint_path"
        default=nothing
        arg_type=String
        help="path for loading the checkpoints"
        "--evaluate_union"
        default="DNF"
        arg_type=String
        help="the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\"s laws (DM)"
    end

    return parse_args(args, s)
end

#="""
Write logs to console and log file
"""=#
function set_logger(args)

    if args["train"] == true
        log_file = joinpath(args["save_path"], "train.log")
    else
        log_file = joinpath(args["save_path"], "test.log")
    end

    log_io = open(log_file, "w");
    datefmt=DateFormat("YY-mm-dd HH:MM:SS");

    timestamp_logger(logger) = TransformerLogger(logger) do log
        merge(log, (; message = "$(Dates.format(now(), datefmt)) $(log.message)"))
    end

    file_logger = timestamp_logger(FileLogger(log_file));
    global_logger(file_logger)

    if args["print_on_screen"]
        time_loger = timestamp_logger(ConsoleLogger(stdout, Logging.Info));

        tl = TeeLogger(file_logger, time_loger);
        global_logger(tl)
    end
end

function log_metrics(mode, step, metrics)
    for m in metrics
        @info "$mode $(m.first) at step $(step): $(m.second)"
    end
end

function evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, tblogger)
    average_metrics = Dict{Float}()
    all_metrics = Dict{Float}()

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics
        log_metrics(mode * " " * query_name_dict[query_structure], step, metrics[query_structure])

        for metric in metrics[query_structure]
            with_logger(tblogger) do
                @info join([mode, query_name_dict[query_structure], metric], "_") metrics[query_structure][metric] = step
            end
            all_metrics[join([query_name_dict[query_structure], metric], "_")] = metrics[query_structure][metric]
            if metric != "num_queries"
                average_metrics[metric] += metrics[query_structure][metric]
            end
        end
        num_queries += metrics[query_structure]["num_queries"]
        num_query_structures += 1
    end

    for metric in average_metrics
        average_metrics[metric] /= num_query_structures
        with_logger(tblogger) do
            @info join([mode, "average", metric], "_") average_metrics[metric] = step
        end
        all_metrics[join(["average", metric], "_")] = average_metrics[metric]
    end

    log_metrics("$mode average", step, average_metrics)
    return all_metrics
end

function main(cmd_args)
    #global train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers
    args = parse_cmdargs(cmd_args)

    rng = Random.default_rng()
    Random.seed!(rng, args["seed"])
    tasks = split(args["tasks"], ".")
    for task in tasks
        if 'n' in task && args["geo"] in ["box", "vec"]
            @assert false "Q2B and GQE cannot handle queries with negation"
        end
    end
    if args["evaluate_union"] == "DM"
        @assert args["geo"] == "beta" "only BetaE supports modeling union using De Morgan's Laws"
    end

    cur_time = format_time()
    if args["prefix"] == nothing
        prefix = "stage_out"
    else
        prefix = args["prefix"]
    end

    args["save_path"] = joinpath(prefix, last(split(args["data_path"], "/")), args["tasks"], args["geo"])
    if args["geo"] == "box"
        save_str = "g-$(args["gamma"])-mode-$(args["box_mode"])"
    elseif args["geo"] == "vec"
        save_str = "g-$(args["gamma"])"
    elseif args["geo"] == "beta"
        save_str = "g-$(args["gamma"])-mode-$(args["beta_mode"])"
    end

    if args["checkpoint_path"] != nothing
        args["save_path"] = args["checkpoint_path"]
    else
        args["save_path"] = joinpath(args["save_path"], save_str, cur_time)
    end

    mkpath(args["save_path"])

    set_logger(args)
    @info ("overwritting saving path: $(args["save_path"])")
    if ! args["train"] # if not training, then create tensorboard files in some tmp location
        tblogger = TBLogger("./$(prefix)/unused-tb", tb_overwrite)
    else
        tblogger = TBLogger(args["save_path"], tb_overwrite)
    end

    nentity, nrelation = open(joinpath(args["data_path"], "stats.txt")) do f
        entrel = readlines(f)
        nentity = parse(Int, last(split(entrel[1], " ")))
        nrelation = parse(Int, last(split(entrel[2], " ")))

        (nentity, nrelation)
    end

    args["nentity"] = nentity
    args["nrelation"] = nrelation

    @info(repeat("-----------------------", 2))
    @info("Geo: $(args["geo"])")
    @info("Data Path: $(args["data_path"])")
    @info "Use CUDA: $(args["cuda"])"
    @info("nentity: $(nentity)")
    @info("nrelation: $(nrelation)")
    @info("max steps: $(args["max_steps"])")
    @info("Evaluate unoins using: $(args["evaluate_union"])")
    @info("tasks = $(args["tasks"])")
    @info("batch_size = $(args["batch_size"])")
    @info("hidden_dim = $(args["hidden_dim"])")
    @info("gamma = $(args["gamma"])")
    @info("box_mode = $(eval_tuple(args["box_mode"]))")
    @info("beta_mode: $(eval_tuple(args["beta_mode"]))")

    use_cuda = false
    @info ("main: CUDA functional $(CUDA.functional())")
    if args["cuda"] && LuxCUDA.functional()
        @info "main: configed to use CUDA."
        use_cuda = true
    end

    model = KGReasoning(nentity,
                        nrelation,
                        args["hidden_dim"],
                        args["gamma"],
                        query_name_dict,
                        args["geo"],
                        eval_tuple(args["box_mode"]),
                        eval_tuple(args["beta_mode"]),
                        use_cuda)
    model_debug = Lux.Experimental.@debug_mode model
    if use_cuda
        model = model |> gpu
    end
    if args["train"]
        ps, st = Lux.setup(rng, model)
        #ps = ps |> ComponentArray
        warn_up_steps = floor(args["max_steps"] / 2)
    end

    local init_step, checkpoint, step, current_learning_rate, warn_up_steps
    current_learning_rate = args["learning_rate"]
    if args["checkpoint_path"] != nothing
        @info("Loading checkpoint $(args["checkpoint_path"])...")
        trained_state = JLD2.load(joinPath(args["checkpoint_path"], "checkpoint"))

        init_step = trained_state["step"]
        ps_trained = checkpoint["trained_ps"]
        #model.load_state_dict(checkpoint["model_state_dict"])

        if args["train"]
            current_learning_rate = trained_state["current_learning_rate"]
            warn_up_steps = trained_state["warn_up_steps"]
            ps = ps_trained
            #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        end
    else
        @info("Ramdomly Initializing $(args["geo"]) Model...")
        init_step = 0
    end

    # setup Optimisers
    opt_rule = Optimisers.ADAM(current_learning_rate)
    opt_state = Optimisers.setup(opt_rule, ps)

    @info ("main: ps keys $(keys(ps))")
    @time begin
        train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers,
        test_queries, test_hard_answers, test_easy_answers = KGDataset.load_data(args, name_query_dict)
    end

    local train_path_dataloader, train_other_dataloader
    if args["train"]
        @info("Training ...")
        train_path_queries = Dict{Any, Set}()
        train_other_queries = Dict{Any, Set}()
        query_path_list = ["1p", "2p", "3p"]
        for query_structure in keys(train_queries)
            if query_name_dict[query_structure] in query_path_list
                train_path_queries[query_structure] = train_queries[query_structure]
            else
                train_other_queries[query_structure] = train_queries[query_structure]
            end
        end

        train_path_queries = flatten_query(train_path_queries)
        train_path_dataset = KGDataset.TrainDataset(train_path_queries, train_answers, nentity, nrelation,
                                                    args["negative_sample_size"])

        if use_cuda
            train_path_queries = train_path_queries |> gpu
        end
        train_path_dataloader = MLUtils.DataLoader(train_path_dataset, batchsize = args["batch_size"], shuffle = false)

        if length(train_other_queries) > 0
            train_other_queries = flatten_query(train_other_queries)
            train_other_dataset = KGDataset.TrainDataset(train_other_queries, train_answers, nentity, nrelation,
                                                         args["negative_sample_size"])
            if use_cuda
                train_other_dataset = train_other_dataset |> gpu
            end
            train_other_dataloader = MLUtils.DataLoader(train_other_dataset, batchsize=args["batch_size"], shuffle=true)
        else
            train_other_dataloader = nothing
        end
    end

    if args["valid"]
        @info("Validation required...")
        valid_all_queries = flatten_query(valid_queries)
        valid_dataloader = KGDataset.DataLoader(KGDataset.TestDataset(valid_all_queries, nentity, nrelation),
                                                batchsize=args["test_batch_size"]);
    end

    if args["test"]
        @info("Test requried...")
        test_queries = flatten_query(test_queries)
        test_dataloader = KGDataset.DataLoader(KGDataset.TestDataset(test_queries, nentity, nrelation),
                                               batchsize=args["test_batch_size"]);
    end

    step = init_step
    if args["train"]
        @info("Training init_step: $(init_step)...")
        training_logs = []
        # #Training Loop
        local path_data, path_next;
        local other_data, other_next;
        path_data, path_next = iterate(train_path_dataloader)
        other_data, other_next = iterate(train_other_dataloader)
        for step in range(init_step, args["max_steps"])
            @info("Training step: .......................$(step)")
            if step == 2 * floor(args["max_steps"] / 3)
                args["valid_steps"] *= 4
            end

            opt_state, ps, log = KGModel.train_step(model, path_data, ps, st, opt_state)
            # data for next step
            path_data, path_next = iterate(train_path_dataloader, path_next)
            for metric in log
                with_logger(tblogger) do
                    @info "Training: path_" * "$(metric)" log[metric] = step
                end
            end

            if train_other_dataloader != nothing
                opt_state, ps, log = KGModel.train_step(model, other_data, ps, st, opt_state)
                other_data, other_next = iterate(train_other_dataloader, other_next)
                for metric in log
                    with_logger(tblogger) do
                        @info "Training: other_" * "$(metric)" log[metric] = step
                    end
                end
                opt_state, ps, log = KGModel.train_step(model, path_data, ps, st, opt_state)
                path_data, path_next = iterate(train_path_dataloader, other_next)
            end

            push!(training_logs, log)

            if step >= warn_up_steps
                current_learning_rate = current_learning_rate / 5
                @info("Training Step: Change learning_rate to $(current_learning_rate) at step $(step)")

                opt_state = Optimisers.setup(Optimisers.Adam(current_learning_rate), ps)
                warn_up_steps = warn_up_steps * 1.5
            end

            if step % args["save_checkpoint_steps"] == 0
                save_variable_list = (
                    "step"=> step,
                    "current_learning_rate" => current_learning_rate,
                    "warm_up_steps" => warn_up_steps
                )
                println("main: save model at step $(step) to $(joinpath(args["save_path"], "checkpoint-$(step).jld2"))")
                jldsave(joinpath(args["save_path"], "checkpoint-$(step).jld2"),
                        opt = opt_state,
                        #model_state = Lux.state(model), opt = opt_state,
                        variables = save_variable_list,
                        params = args,
                        ps = ps)
            end

            if step > 0 && step % args["valid_steps"] == 0 && step > 0
                if args["valid"]
                    @info("Evaluating on Valid Dataset...")
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args,
                                                 valid_dataloader, query_name_dict, "Valid", step, tblogger)
                end

                if args["test"]
                    @info("Evaluating on Test Dataset...")
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args,
                                                test_dataloader, query_name_dict, "Test", step, tblogger)
                end
            end

            if step > 0 && step % args["log_steps"] == 0
                metrics = Dict()
                #println("main: $(training_logs)")
                if(length(training_logs) > 0)
                    for metric in keys(training_logs[1])
                        metrics[metric] = sum([log[metric] for log in training_logs])/length(training_logs)
                    end
                end

                log_metrics("Training average", step, metrics)
                training_logs = []
            end
            #=
            save_variable_list = (
                "step" => step,
                "current_learning_rate" => current_learning_rate,
                "warm_up_steps" => warn_up_steps
            )
            JLD2.save(joinpath(args["save_path"], "checkpoint-$(step).jld2"),
                      model_state_dict = Flux.state(model), opt = opt_state,
                      variables = save_variable_list, params = args)
            =#
        end
        @info("Training finished!!")
    end

    #    if args["test"]
    #        @info("Evaluating on Test Dataset...")
    #        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, "Test", step, tblogger)
    #    end
    #
end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDA.@allowscalar(true)
    args = Vector{String}(["--train", "--data_path", "dataset/FB15k-betae", "--cuda",
                           "-n", "128", "-b", "64", "-d", "128", "-g", "24","--learning_rate",
                           "0.0001", "--max_steps", "4501",
                           "--cpu", "1", "--geo", "beta", "--valid_steps", "150"])

    main(ARGS)

end
