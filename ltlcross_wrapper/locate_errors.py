# Copyright (c) 2019 Fanda Blahoudek
"""This file includes functions that help with looking for
errors detected by ltlcross.
"""
import re
import subprocess


def bogus_to_lcr(form):
    """Converts a formula as it is printed in ``_bogus.ltl`` file
    (uses ``--relabel=abc``) to use ``pnn`` AP names.
    """
    args = ['-r0', '--relabel=pnn', '-f', form]
    return subprocess.check_output(["ltlfilt"] + args, universal_newlines=True).strip()


def parse_check_log(log_f):
    """Parses a given log file and locates cases where
    sanity checks found some error.

    Returns:
    bugs: a dict: ``form_id``->``list of error lines``
    bogus_forms: a dict: ``form_id``->``form``
    tools: a dict: ``tool_id``->``command``
    """
    log = open(log_f, 'r')
    bugs = {}
    bogus_forms = {}

    formula = re.compile(".*ltl:(\d+): (.*)$")
    empty_line = re.compile('^\s$')
    problem = re.compile("error: .* nonempty")

    for line in log:
        m_form = formula.match(line)
        if m_form:
            form = m_form
            f_bugs = []
        m_empty = empty_line.match(line)
        if m_empty:
            if len(f_bugs) > 0:
                form_id = int(form.group(1)) - 1
                bugs[form_id] = f_bugs
                bogus_forms[form_id] = form.group(2)
        m_prob = problem.match(line)
        if m_prob:
            f_bugs.append(m_prob.group(0))
    log.close()
    tools = parse_log_tools(log_f)
    return bugs, bogus_forms, tools


def find_log_for(tool_code, form_id, log_f):
    """Returns an array of lines from log for
    given tool code (P1,N3,...) and form_id. The
    form_id is taken from runner - thus we search for
    formula number ``form_id+1``
    """
    log = open(log_f, 'r')
    current_f = -1
    formula = re.compile(".*ltl:(\d+): (.*)$")
    tool = re.compile(".*\[([PN]\d+)\]: (.*)$")
    gather = re.compile("Performing sanity checks and gathering statistics")
    output = []
    for line in log:
        m_form = formula.match(line)
        if m_form:
            current_f = int(m_form.group(1))
            curr_tool = ""
        if current_f < form_id + 1:
            continue
        if current_f > form_id + 1:
            break
        m_tool = tool.match(line)
        if m_tool:
            curr_tool = m_tool.group(1)
        if gather.match(line):
            curr_tool = "end"
        if curr_tool == tool_code:
            output.append(line.strip())
    log.close()
    return output


def hunt_error_types(log_f):
    log = open(log_f, 'r')
    errors = {}
    err_forms = {}

    formula = re.compile(".*ltl:(\d+): (.*)$")
    empty_line = re.compile("^\s$")
    tool = re.compile(".*\[([PN]\d+)\]: (.*)$")
    problem = re.compile("error: .*")
    nonempty = re.compile("error: (.*) is nonempty")

    for line in log:
        m_form = formula.match(line)
        if m_form:
            form = m_form
            f_bugs = {}
        m_tool = tool.match(line)
        if m_tool:
            tid = m_tool.group(1)
        m_empty = empty_line.match(line)
        if m_empty:
            if len(f_bugs) > 0:
                form_id = int(form.group(1)) - 1
                errors[form_id] = f_bugs
                err_forms[form_id] = form.group(2)
        m_prob = problem.match(line)
        if m_prob:
            prob = m_prob.group(0)
            m_bug = nonempty.match(line)
            if m_bug:
                prob = "nonempty"
                tid = m_bug.group(1)
            if prob not in f_bugs:
                f_bugs[prob] = []
            f_bugs[prob].append(tid)
    log.close()
    tools = parse_log_tools(log_f)
    return errors, err_forms, tools


def parse_log_tools(log_f):
    log = open(log_f, 'r')
    tools = {}
    tool = re.compile(".*\[(P\d+)\]: (.*)$")
    empty_line = re.compile("^\s$")
    for line in log:
        m_tool = tool.match(line)
        m_empty = empty_line.match(line)
        if m_empty:
            break
        if m_tool:
            tid = m_tool.group(1)
            tcmd = m_tool.group(2)
            tools[tid] = tcmd
    log.close()
    return tools
