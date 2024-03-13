def get_model(model_name, args):
    name = model_name.lower()
    if name == "incsr":
        from methods.incsr import Learner
    elif name == "eval":
        print("eval mode")
        from methods.eval import Learner
    else:
        assert 0

    return Learner(args)
