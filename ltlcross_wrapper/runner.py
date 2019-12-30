# -*- coding: utf-8 -*-
# Copyright (c) 2019 Fanda Blahoudek

import subprocess
import os.path
from datetime import datetime


class LtlcrossRunner(object):
    """A class for running Spot's `ltlcross`.

    Parameters
    ----------
    tools : a dict (String -> String)
        The records in the dict of the form ``name : ltlcross_cmd``
    >>> tools = {"LTL3HOA"    : "ltl3hoa -d -x -i -p 2 -f %f > %O",
    >>>          "SPOT":   : "ltl2tgba"
    >>>         }

    formulas : String or list of Strings
        paths to files with formulas to be fed to `ltlcross`
    res_filename : String
        filename to store the ltlcross`s results
        `ltlcross_results.csv` by default
    log_filename : String
        filename used to store log of ltlcross
        `{basename}.log` by default where basename is
        the res_filename without extension
    """

    def __init__(self,
                 tools,
                 formulas,
                 res_filename='ltlcross_results.csv',
                 log_filename=None,
                 ):
        assert isinstance(tools, dict)
        self.tools = tools

        # formula file(s)
        if isinstance(formulas, str):
            self.f_files = [formulas]
        else:
            self.f_files = formulas
        for f in self.f_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f)

        self.res_file = res_filename

        if log_filename is None:
            self.log_file = self.res_file[:-3] + 'log'
        else:
            self.log_file = log_filename

    def create_args(self,
                    automata=True,
                    check=False,
                    timeout='120',
                    res_file=None,
                    save_bogus=True,
                    tool_subset=None,
                    forms=True,
                    escape_tools=False):
        """Creates args that are passed to run_ltlcross
        """
        if res_file is None:
            res_file = self.res_file
        if tool_subset is None:
            tool_subset = self.tools.keys()

        ### Prepare ltlcross command ###
        tools_strs = ["{" + name + "}" + cmd for (name, cmd) in self.tools.items() if name in tool_subset]
        if escape_tools:
            tools_strs = ["'{}'".format(t_str) for t_str in tools_strs]
        args = tools_strs
        if forms:
            args += ' '.join(['-F ' + F for F in self.f_files]).split()
        if timeout:
            args.append('--timeout=' + timeout)
        if automata:
            args.append('--automata')
        if save_bogus:
            args.append('--save-bogus={}_bogus.ltl'.format(res_file[:-4]))
        if not check:
            args.append('--no-checks')
        # else:
        #    args.append('--reference={ref_Spot}ltl2tgba -H %f')
        args.append('--products=0')
        args.append('--csv=' + res_file)
        return args

    def ltlcross_cmd(self,
                     args=None,
                     automata=True,
                     check=False,
                     timeout='300',
                     log_file=None,
                     res_file=None,
                     save_bogus=True,
                     tool_subset=None,
                     forms=True,
                     lcr='ltlcross'):
        """Returns ltlcross command for the parameters.
        """
        if log_file is None:
            log_file = self.log_file
        if res_file is None:
            res_file = self.res_file
        if tool_subset is None:
            tool_subset = self.tools.keys()
        if args is None:
            args = self.create_args(automata=automata,
                                    check=check,
                                    timeout=timeout,
                                    res_file=res_file,
                                    save_bogus=save_bogus,
                                    tool_subset=tool_subset,
                                    forms=forms,
                                    escape_tools=True)
        return ' '.join([lcr] + args)

    def run_ltlcross(self,
                     args=None,
                     automata=True,
                     check=False,
                     timeout='120',
                     log_file=None,
                     res_file=None,
                     save_bogus=True,
                     tool_subset=None,
                     lcr='ltlcross'):
        """Removes any older version of ``self.res_file`` and runs `ltlcross`
        on all tools.

        Parameters
        ----------
        args : a list of ltlcross arguments that can be used for subprocess
        tool_subset : a list of names from self.tools
        """
        if log_file is None:
            log_file = self.log_file
        if res_file is None:
            res_file = self.res_file
        if tool_subset is None:
            tool_subset = self.tools.keys()
        if args is None:
            args = self.create_args(automata=automata,
                                    check=check,
                                    timeout=timeout,
                                    res_file=res_file,
                                    save_bogus=save_bogus,
                                    tool_subset=tool_subset)

        # Delete ltlcross result and lof files
        subprocess.call(["rm", "-f", res_file, log_file])

        ## Run ltlcross ##
        log = open(log_file, 'w')
        cmd = self.ltlcross_cmd(args, lcr=lcr)
        print(cmd, file=log)
        print(datetime.now().strftime('[%d.%m.%Y %T]'), file=log)
        print('=====================', file=log, flush=True)
        self.returncode = subprocess.call([lcr] + args, stderr=subprocess.STDOUT, stdout=log)
        log.writelines([str(self.returncode) + '\n'])
        log.close()


def param_runner(name, tools, data_dir='data_param'):
    r = LtlcrossRunner(tools,
                       res_filename=f'{data_dir}/{name}.csv',
                       formulas=f'formulae/{name}.ltl')
    return r
