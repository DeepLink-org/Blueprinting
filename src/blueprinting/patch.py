from fire.core import _IsFlag


def _ParseKeywordArgs(args, fn_spec):
    """Parses the supplied arguments for keyword arguments.

    Given a list of arguments, finds occurrences of --name value, and uses 'name'
    as the keyword and 'value' as the value. Constructs and returns a dictionary
    of these keyword arguments, and returns a list of the remaining arguments.

    Only if fn_keywords is None, this only finds argument names used by the
    function, specified through fn_args.

    This returns the values of the args as strings. They are later processed by
    _ParseArgs, which converts them to the appropriate type.

    Args:
      args: A list of arguments.
      fn_spec: The inspectutils.FullArgSpec describing the given callable.
    Returns:
      kwargs: A dictionary mapping keywords to values.
      remaining_kwargs: A list of the unused kwargs from the original args.
      remaining_args: A list of the unused arguments from the original args.
    Raises:
      FireError: If a single-character flag is passed that could refer to multiple
          possible args.
    """
    kwargs = {}
    remaining_kwargs = []
    remaining_args = []
    fn_keywords = fn_spec.varkw
    fn_args = fn_spec.args + fn_spec.kwonlyargs
    # print(dir(fn_spec), fn_spec.defaults, fn_spec.varkw, fn_args)
    if not args:
        return kwargs, remaining_kwargs, remaining_args

    defaults = dict(zip(reversed(fn_args), reversed(fn_spec.defaults)))
    skip_argument = False

    for index, argument in enumerate(args):
        if skip_argument:
            skip_argument = False
            continue

        if _IsFlag(argument):
            # This is a named argument. We get its value from this arg or the next.

            # Terminology:
            # argument: A full token from the command line, e.g. '--alpha=10'
            # stripped_argument: An argument without leading hyphens.
            # key: The contents of the stripped argument up to the first equal sign.
            # "shortcut flag": refers to an argument where the key is just the first
            #   letter of a longer keyword.
            # keyword: The Python function argument being set by this argument.
            # value: The unparsed value for that Python function argument.
            contains_equals = "=" in argument
            stripped_argument = argument.lstrip("-")
            if contains_equals:
                key, value = stripped_argument.split("=", 1)
            else:
                key = stripped_argument

            key = key.replace("-", "_")
            is_bool_syntax = not contains_equals and (index + 1 == len(args) or _IsFlag(args[index + 1]))

            # Determine the keyword.
            keyword = ""  # Indicates no valid keyword has been found yet.
            if key in fn_args or (is_bool_syntax and key.startswith("no") and key[2:] in fn_args) or fn_keywords:
                keyword = key
            elif len(key) == 1:
                # This may be a shortcut flag.
                matching_fn_args = [arg for arg in fn_args if arg[0] == key]
                if len(matching_fn_args) == 1:
                    keyword = matching_fn_args[0]
                elif len(matching_fn_args) > 1:
                    raise FireError(
                        f"The argument '{argument}' is ambiguous as it could "
                        f"refer to any of the following arguments: {matching_fn_args}"
                    )

            # Determine the value.
            if not keyword:
                got_argument = False
            elif contains_equals:
                # Already got the value above.
                got_argument = True
            elif is_bool_syntax:
                # There's no next arg or the next arg is a Flag, so we consider this
                # flag to be a boolean.
                got_argument = True
                if keyword in fn_args:
                    value = "True"
                elif keyword.startswith("no"):
                    keyword = keyword[2:]
                    value = "False"
                else:
                    value = "True"
            else:
                # The assert should pass. Otherwise either contains_equals or
                # is_bool_syntax would have been True.
                assert index + 1 < len(args)
                value = args[index + 1]
                got_argument = True

            # In order for us to consume the argument as a keyword arg, we either:
            # Need to be explicitly expecting the keyword, or we need to be
            # accepting **kwargs.
            skip_argument = not contains_equals and not is_bool_syntax
            if got_argument:
                if keyword not in kwargs:
                    kwargs[keyword] = defaults.get(keyword)
                if isinstance(kwargs[keyword], list):
                    kwargs[keyword].append(value)
                else:
                    kwargs[keyword] = value
            else:
                remaining_kwargs.append(argument)
                if skip_argument:
                    remaining_kwargs.append(args[index + 1])
        else:  # not _IsFlag(argument)
            remaining_args.append(argument)

    return kwargs, remaining_kwargs, remaining_args
