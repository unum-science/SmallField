# Options affecting formatting.
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True


# Comments are left exactly as written. With markup enabled, cmake-format treats a comment as prose
# and reflows it to the column limit, which collapses an aligned option table into a paragraph and
# joins the lines of an example block together. Same failure as clang-format re-splitting a string
# literal: the tool cannot tell authored layout from accidental line breaks, so it does not get to
# guess.
with section("markup"):
    enable_markup = False

# Without these the project's own commands are parsed as a flat argument list, and every
# argument lands on its own line rather than grouping under the keyword it belongs to.
with section("parse"):
    additional_commands = {
        "define_smallfield_target": {
            "pargs": 1,
            "kwargs": {"SOURCES": "*", "LINK": "*", "DEFINES": "*"},
        },
        "define_smallfield_test": {
            "pargs": 2,
            "flags": ["MIXED"],
            "kwargs": {"LINK": "*"},
        },
        "define_rdkit_library": {"pargs": 2},
    }
