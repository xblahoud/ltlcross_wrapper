# Copyright (c) 2019 Fanda Blahoudek

import os.path
import math

import pandas as pd
import matplotlib.pyplot
import seaborn
import spot

from ltlcross_wrapper.locate_errors import bogus_to_lcr

def pretty_print(form):
    """Runs Spot to format formulas nicer."""
    return spot.formula(form).to_str()


def hoa_to_spot(hoa):
    return spot.automaton(hoa + "\n")


def remove_tools(source, output, tool_set):
    """Remove data for tool from ltlcross resutls

    Parameters
    ==========
    `source`   : str, filename that contains the source data
    `output`   : str, filename where to write data, will be overwritten
    `tool_set` : list, names of tool for which data will be removed
    """
    a = ResAnalyzer(source)
    for tool in a.tools:
        if tool not in a.tools:
            raise ValueError(f"{tool} is already not in the data (file {source})")

    data = pd.read_csv(source)
    data = data.loc[~data["tool"].isin(tool_set)]
    data.to_csv(output, index=False)


def merge_resfiles(base_results, new_results, output, tool_set=None):
    """Merge two files with ltlcross results.

    The 2 files have to contain data for disjoint toolnames. This restriction
    can be relaxed by specifting `tool_set` such that it does not contain any
    tool from `base_results`.

    Parameters
    ==========
    `base_results`  : str, filename that contains the base source data
    `new_results`  : str, filename that contains the added source data
    `output`   : str, filename where to write data, will be overwritten
    `tool_set` : list of tools from `new_results`.
        Only tools in `tool_set` will be added to those in `base_results` if specified
    """
    tools1 = ResAnalyzer(base_results).tools
    tools2 = ResAnalyzer(new_results).tools

    # is tool_set valid
    if tool_set is not None:
        for t in tool_set:
            if t not in tools2:
                raise ValueError(f"{t} not a toolname in {new_results}")
    else:
        tool_set = tools2

    # Disjoint tools?
    for t in tool_set:
        if t in tools1:
            raise ValueError(f"{t} already in {base_results}")

    base = pd.read_csv(base_results)
    add = pd.read_csv(new_results)
    add = add.loc[add["tool"].isin(tool_set)]
    pd.concat([base,add]).to_csv(output, index=False)


def update_resfile(base_results, new_results, output, tool_set=None, add_new_tools=True):
    """Update ltlcross results in file base by values from new.

    By default, values for tools in `new_results` overwrite the values in
    `base_results`, and tools not previously in `base_results` are added to
    the result, unless specifying `add_new_tools=False`.

    `tool_set` controls which data from `new_results` should be used.

    Parameters
    ==========
    `base_results` : str, filename that contains the base source data
    `new_results`  : str, filename that contains the new source data
    `output`       : str, filename where to write updated data, will be overwritten
    `tool_set`     : list of tools from `new_results`.
        Only tools in `tool_set` will be updated/added in/to those in `base_results`
        By default use all tools from `new_results`.
    `add_new_tools` : Bool, if `False`, only update values for tools already in
        `base_results`. Do not add new tools.
    """
    tools1 = ResAnalyzer(base_results).tools
    tools2 = ResAnalyzer(new_results).tools

    # is tool_set valid
    if tool_set is not None:
        for t in tool_set:
            if t not in tools2:
                raise ValueError(f"{t} not a toolname in {new_results}")
    else:
        tool_set = tools2

    shared_tools = [t for t in tool_set if t in tools1]
    if not add_new_tools:
        tool_set = shared_tools

    base = pd.read_csv(base_results)
    base = base.loc[~base["tool"].isin(shared_tools)]
    add = pd.read_csv(new_results)
    add = add.loc[add["tool"].isin(tool_set)]

    pd.concat([base, add]).to_csv(output, index=False)


def gather_cumulative(benchmarks, transpose=True, highlight=True, **kwargs):
    """Display cumulative numbers for multiple benchmarks.

    For each benchmark, highlight the best

    `benchmarks` : dict (name : ResAnalyzer)
    `transpose` : bool, swap tool_set & benchmark names
                  by default, tool_set are rows, benchmark cols
    `highlight` : bool, if True, highlight the best in each benchmark
                  `True` by default
    `kwargs` : are passed to ResAnalyzer.cumulative()
    """
    data = pd.DataFrame()
    for (name, b) in benchmarks.items():
        tmp = pd.DataFrame(b.cumulative(highlight=False, **kwargs))
        tmp.columns = [name]
        data = data.append(tmp.transpose())
    if transpose:
        if highlight:
            return data.transpose().style.apply(highlight_min, axis=0)
        return data.transpose()
    else:
        if highlight:
            return data.style.apply(highlight_min, axis=1)
        return data


def gather_mins(benchmarks, transpose=True, highlight=True, **kwargs):
    """Display numbers of minimal automata in multiple benchmarks.

    Show for how many formulas each tool produces automaton that has
    the smallest number of states. The minimum ranges over `tool_set`.
    The number in min hits shows how many times the same size as the
    smallest automaton was achieved. The number in unique min hits counts
    only cases where the given tool is the only tool with such a small
    automaton.

    `benchmarks` : dict (name : ResAnalyzer)
    `transpose` : bool, swap tool_set & benchmark names
                  by default, tool_set are rows, benchmark cols
    `highlight` : bool, if True, highlight the best in each benchmark
                  `True` by default
    `kwargs` : are passed to ResAnalyzer.min_counts()
    """
    data = pd.DataFrame()
    for (name, b) in benchmarks.items():
        tmp = b.min_counts(**kwargs)
        tmp.columns = pd.MultiIndex.from_tuples([(name, c) for c in tmp.columns])
        data = data.append(tmp.transpose(), sort=False).fillna(0)
    if transpose:
        if highlight:
            return data.transpose().style.apply(highlight_max, axis=0)
        return data.transpose()
    if highlight:
        return data.style.apply(highlight_max, axis=1)
    return data


def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: lightgreen' if v else '' for v in is_min]


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]


class ResAnalyzer:
    """Analyze `.csv` files with results of ltlcross.

    Parameters
    ----------
    res_filename : String
        filename to store the ltlcross`s results
    cols : list of Strings, default ``['states','edges','transitions','acc']``
        names of ltlcross's statistics columns to be recorded
    tool_set : default tool_set for which you want to display values
    """

    def __init__(self,
                 res_filename,
                 cols=None,
                 tool_set=None,
                 ):
        self.res_file = res_filename

        self.tools = None
        self.cols = ['states', 'edges', 'transitions', 'acc'] if cols is None else cols

        # Main DataFrame with values
        self.values = None

        # Helper DataFrames
        self.automata = None
        self.exit_status = None
        self.form = None
        self.incorrect = None

        # Store precomputed minimums among tools
        self.mins = []

        # Highlighting defaults
        self.light_highlight_color = "#E0FFE0"
        self.highlight_color = "lightgreen"

        self.parse_results()

        self.tool_set = self.tools if tool_set is None else tool_set

    def parse_results(self):
        """Parse the ``self.res_file`` and sets the values, automata, and
        form.
        """
        if not os.path.isfile(self.res_file):
            raise FileNotFoundError(self.res_file)
        res = pd.read_csv(self.res_file)

        # Add incorrect columns to track flawed automata
        if not 'incorrect' in res.columns:
            res['incorrect'] = False

        # Removes unnecessary parenthesis from formulas
        res.formula = res['formula'].map(pretty_print)

        form = pd.DataFrame(res.formula.drop_duplicates())
        form['form_id'] = range(len(form))
        form.index = form.form_id

        res = form.merge(res)
        self.form = form.set_index(['form_id', 'formula'])

        # Shape the table & parse tools
        table = res.set_index(['form_id', 'formula', 'tool'])
        table = table.unstack(2)
        table.axes[1].set_names(['column', 'tool'], inplace=True)

        # Get the list of tools from data
        self.tools = list(table.columns.levels[1])

        # Create separate tables for automata
        automata = None
        if 'automaton' in table.columns.levels[0]:
            automata = table[['automaton']]

            # Removes formula column from the index
            automata.index = automata.index.levels[0]

            # Removes `automata` from column names -- flatten the index
            automata.columns = automata.columns.levels[1]
        self.automata = automata

        # Store incorrect and exit_status information separately
        self.incorrect = table[['incorrect']]
        self.incorrect.columns = self.incorrect.columns.droplevel()
        self.exit_status = table[['exit_status']]
        self.exit_status.columns = self.exit_status.columns.droplevel()

        # stores the followed columns only
        values = table[self.cols]
        self.values = values.sort_index(axis=1, level=['column', 'tool'])

    def aut_for_id(self, form_id, tool):
        """For given formula id and tool it returns the corresponding
        non-deterministic automaton as a Spot's object.

        Parameters
        ----------
        form_id : int
            id of formula to use
        tool : String
            name of the tool to use to produce the automaton
        """
        if self.automata is None:
            raise AssertionError("No results parsed yet")
        if tool not in self.tools:
            raise ValueError(tool)
        return hoa_to_spot(self.automata.loc[form_id, tool])

    def compute_sbacc(self, col='states'):
        """Convert automata to state-based BA and check store
        values for this converted automata.
        """

        def get_sbacc(aut):
            if isinstance(aut, float) and math.isnan(aut):
                return None
            a = spot.automata(aut + '\n')
            aut = next(a)
            aut = spot.sbacc(aut)
            if col == 'states':
                return aut.num_states()
            if col == 'acc':
                return aut.num_sets()

        df = self.automata.copy()

        # Recreate the same index as for other cols
        n_i = [(l, self.form_of_id(l, False)) for l in df.index]
        df.index = pd.MultiIndex.from_tuples(n_i)
        df.index.names = ['form_id', 'formula']
        # Recreate the same columns hierarchy
        df = df.T
        df['column'] = 'sb_{}'.format(col)
        self.cols.append('sb_{}'.format(col))
        df = df.set_index(['column'], append=True)
        df = df.T.swaplevel(axis=1)

        # Compute the requested values and add them to others
        df = df.applymap(get_sbacc)
        self.values = self.values.join(df)

    def compute_best(self, tool_set=None, new_col_name="Minimum"):
        """Computes minimum values over tools in `tool_set` for all
        formulas and stores them in column `new_col_name`.

        Parameters
        ----------
        tool_set : list of Strings
            column names that are used to compute the min over
            all tools by default
        new_col_name : String
            name of column used to store the computed values
        """
        if tool_set is None:
            tool_set = self.tools

        self.mins.append(new_col_name)
        for col in self.cols:
            self.values[col, new_col_name] = self.values[col][tool_set].min(axis=1)

        # Check if at least one tool finshed ok
        def check_status(x):
            for tool in tool_set:
                if x[tool] == "ok":
                    return "ok"
            return pd.np.nan
        self.exit_status[new_col_name] = self.exit_status.apply(check_status, 1)

        self.values.sort_index(axis=1, level=0, inplace=True)

    def cumulative(self, tool_set=None, col="states", highlight=True):
        """Returns table with cumulative numbers of states.

         For each tool, sums the values of `col` over formulas
         with no timeout.

         If `tool_set` is given, use only tools within this
         subset. Only formulas with timeouts within the subset
         are removed. The sum of values for `tool_set` where
         all formulas with some timeout are removed run
         ```
         self.cumulative().loc[tool_set]
         ```

        Parameters
        ---------
        col : String or list of Strings
            One or more columns (``states`` default).

        tool_set : list, contains tools from `self.tools`
            Restrict the output to given subset of tools
        highlight : Bool (default `True`)
            Highlight the minimal value for each metric
        """
        if tool_set is None:
            tool_set = self.tool_set
        data = self.values.loc[:, (col, tool_set)].dropna().sum()

        # Format as DataFrame, remove unnecessary index labels
        df = pd.DataFrame(data)
        df = df.unstack(level=0)
        df.columns = df.columns.droplevel()
        df.columns.name = ""
        if highlight:
            df = df.style.apply(self._highlight_min, axis=0)
        return df

    def smaller_than(self, t1, t2, col='states', **kwargs):
        """Returns a dataframe with results where ``col`` for ``tool1``
        has strictly smaller value than ``col`` for ``tool2``.

        Parameters
        ----------
        t1 : String
            name of tool for comparison (the better one)
            must be among self.tools
        t2 : String
            name of tool for comparison (the worse one)
            must be among self.tools
        col : String, default ``'states'``
            name of column use for comparison.

        **kwargs can take following keys:
            reverse : Boolean, default ``False``
                if ``True``, it switches ``t1`` and ``t2``
            restrict_cols : Boolean, default ``True``
                if ``True``, the returned DataFrame contains only the compared
                property columns
            restrict_tools : Boolean, default ``True``
                if ``True``, the returned DataFrame contains only the compared
                tools
        """
        return self.better_than(t1, t2, compare_on=[col], include_fails=False, **kwargs)

    def better_than(self, t1, t2,
                    reverse=False,
                    **kwargs
                    ):
        """Compares ``t1`` against ``t2`` lexicographicaly
        on cols from ``compare_on`` and returns DataFrame with
        results where ``t1`` is better than ``t2``.

        Parameters
        ----------
        t1 : String
            name of tool for comparison (the better one)
            must be among tools
        t2 : String
            name of tool for comparison (the worse one)
            must be among tools

        kwargs can contain:
            compare_on : list of Strings, default (['states','acc','transitions'])
                list of columns on which we want the comparison (in order)
            reverse : Boolean, default ``False``
                if ``True``, it switches ``t1`` and ``t2``
            include_fails : Boolean, default ``True``
                if ``True``, include formulae where t2 fails and t1 does not
                fail
            restrict_cols : Boolean, default ``True``
                if ``True``, the returned DataFrame contains only the compared
                property columns
            restrict_tools : Boolean, default ``True``
                if ``True``, the returned DataFrame contains only the compared
                tools
        """
        if t1 not in list(self.tools) + self.mins:
            raise ValueError(t1)
        if t2 not in list(self.tools) + self.mins:
            raise ValueError(t2)

        compare_on = kwargs.get("compare_on", ['states', 'acc', 'transitions'])
        include_fails  = kwargs.get("include_fails", True)
        restrict_cols  = kwargs.get("restrict_cols", True)
        restrict_tools = kwargs.get("restrict_tools", True)

        if reverse:
            t1, t2 = t2, t1
        v = self.values
        t1_ok = self.exit_status[t1] == 'ok'
        if include_fails:
            t2_ok = self.exit_status[t2] == 'ok'
            # non-fail beats fail
            c = v[t1_ok & ~t2_ok]
            # We work on non-failures only from now on
            eq = t1_ok & t2_ok
        else:
            c = pd.DataFrame()
            eq = t1_ok
        for prop in compare_on:
            # For each prop we add t1 < t2
            better = v[prop][t1] < v[prop][t2]
            # but only from those which were equivalent so far
            equiv_and_better = v.loc[better & eq]
            c = c.append(equiv_and_better)
            # And now choose those equivalent also on prop to eq
            eq = eq & (v[prop][t1] == v[prop][t2])

        # format the output
        idx = pd.IndexSlice
        tools = [t1, t2] if restrict_tools else slice(None)
        compare_on = compare_on if restrict_cols else slice(None)
        return c.loc[:, idx[compare_on, tools]]

    def form_of_id(self, form_id, spot_obj=True):
        """For given form_id returns the formula

        Parameters
        ----------
        form_id : int
            id of formula to return
        spot_obj : Bool
            If ``True``, returns Spot formula object (uses Latex to
            print the formula in Jupyter notebooks)
        """
        f = self.values.index[form_id][1]
        if spot_obj:
            return spot.formula(f)
        return f

    def id_of_form(self, f, convert=False):
        """Returns id of a given formula. If ``convert`` is ``True``
        it also calls ``bogus_to_lcr`` first.
        """
        if convert:
            f = bogus_to_lcr(f)
        ni = self.values.index.droplevel(0)
        return ni.get_loc(f)

    def mark_incorrect(self, form_id, tool, output_file=None, input_file=None):
        """Marks automaton given by the formula id and tool as flawed
        and writes it into the .csv file
        """
        if tool not in self.tools:
            raise ValueError(tool)
        # Put changes into the .csv file
        if output_file is None:
            output_file = self.res_file
        if input_file is None:
            input_file = self.res_file
        csv = pd.read_csv(input_file)
        if not 'incorrect' in csv.columns:
            csv['incorrect'] = False
        cond = (csv['formula'].map(pretty_print) ==
                pretty_print(self.form_of_id(form_id, False))) & \
               (csv.tool == tool)
        csv.loc[cond, 'incorrect'] = True
        csv.to_csv(output_file, index=False)

        # Mark the information into self.incorrect
        self.incorrect.loc[self.index_for(form_id)][tool] = True

    def na_incorrect(self):
        """Change values for flawed automata (marked as incorrect
        by `self.mark_incorrect`) to N/A. This causes
        that the touched formulae will be removed from cumulative
        etc. if computed again. To reverse this information you
        have to parse the results again.

        It also sets ``exit_status`` to ``incorrect``
        """
        self.values = self.values[~self.incorrect]
        self.exit_status[self.incorrect] = 'incorrect'

    def index_for(self, form_id):
        return form_id, self.form_of_id(form_id, False)

    def _get_error_count(self, err_type='timeout', drop_zeros=True):
        """Returns a Series with total number of er_type errors for
        each tool.

        Parameters
        ----------
        err_type : String one of `timeout`, `parse error`,
                                 `incorrect`, `crash`, or
                                 'no output'
                  Type of error we seek
        drop_zeros : Boolean (default True)
                    If true, rows with zeros are removed
        """
        valid_errors = ['timeout', 'parse error',
                        'incorrect', 'crash',
                        'no output']
        if err_type not in valid_errors:
            raise ValueError(f"Invalid err_type. The value must be in:\n"
                             f"\t{valid_errors}\n"
                             f"\t{err_type} given")

        if err_type == 'crash':
            c1 = self.exit_status == 'exit code'
            c2 = self.exit_status == 'signal'
            res = (c1 | c2).sum()
        else:
            res = (self.exit_status == err_type).sum()
        if drop_zeros:
            return res.iloc[res.to_numpy().nonzero()]
        return res

    def get_error_counts(self, drop_zeros=True, error_types=None):
        """Return DataFrame with numbers of errors of all types
        for each tool.

        Parameters
        ----------
        drop_zeros : Boolean (default True)
            If true, show only tools with some errors
        error_types : Iterable of Strings, can contain only following (default all):
            * `timeout`,
            * `parse error`,
            * `incorrect`,
            * `crash`,
            * 'no output'
        """
        if error_types is None:
            error_types = ['timeout', 'parse error', 'incorrect', 'crash', 'no output']

        data = {}
        for t in error_types:
            data[t] = self._get_error_count(t, drop_zeros)
        res = pd.DataFrame(data)
        return res.fillna(0, downcast="infer")

    def cross_compare(self,
                      tool_set=None,
                      total=True,
                      highlight=True,
                      **kwargs):
                      #include_other=True):
        """Create a "league" table.

        For each pair of tools (`t1`,`t2`) from `tool_set` compute
        in how many cases `t1` produced better automaton than `t2`.
        Being better is based on metrics in `compare_on` (in order).
        The number of victories of `t1` against `t2` is stored at
        row `t1` and column `t2`.

        Parameters:
        ===========
        tool_set : list, contains tools from `self.tools`
            Restrict the output to given subset of tools
        total : Bool
            include the sum of victories of each tool as the last
            column (called `V`)
        highlight : Bool (default `True`)
            color proportional amount of each cell by its value

        kwargs can contain:
        compare_on : list of metrics to decide victories
            default: ['states', 'acc', 'transitions']
            if t1["states"] == t1["states"], the apply `acc`, ...
        include_fails : Boolean, default `True`
            if `True`, count formulas where `t2` fails and `t1`
            does not as victories of `t1` (and do not consider
            if `False`)
        """
        def count_better(tool1, tool2):
            if tool1 == tool2:
                return float('nan')
            #try:
            return len(self.better_than(tool1, tool2, **kwargs))
            #except ValueError as e:
            #    if include_other:
            #        return float('nan')
            #    else:
            #        raise e

        if tool_set is None:
            tool_set = self.tool_set
        c = pd.DataFrame(index=tool_set, columns=tool_set).fillna(0)
        for tool in tool_set:
            c[tool] = pd.DataFrame(c[tool]).apply(lambda x: count_better(x.name, tool), 1)
        if total:
            c['V'] = c.sum(axis=1)
        if highlight:
            c = c.style.bar(color=self.light_highlight_color, vmin=0)
        return c

    def min_counts(self, tool_set=None,
                   unique_only="both",
                   restrict_tools=True,
                   col='states',
                   min_name='min(count)'):
        """Compute number of cases where each tool produces the minimum automaton.

        Parameters
        ==========
         * `tool_set`    : tools to check for values, self.tool_set by default
         * `unique_only` : `bool` or "both". If `True`, count only unique hits
                           of the minimum value (such no other tool reached it)
                           If `"both"`, return values both for unique and
                           non-unique min hits (default).
         * `restrict_tools` : `bool`, default `True`. If `False`, consider also
                              tools not in `tool_set` for computation of the min
                              values.
         * `col` : name of column to use, "states" by default
         * `min_name` : `str`, default "min(count)"
                        column name used to store the minimum values for each formula
        """
        if unique_only == "both":
            unique = self.min_counts(tool_set=tool_set, unique_only=True,
                                     restrict_tools=restrict_tools, col=col,
                                     min_name=min_name)
            shared = self.min_counts(tool_set=tool_set , unique_only=False,
                                     restrict_tools=restrict_tools, col=col,
                                     min_name=min_name)
            return pd.merge(unique, shared, how="outer", left_index=True, right_index=True)

        if not isinstance(unique_only, bool):
            raise ValueError(f'unique_only has to be "both" or `bool`. Given {unique_only}')

        if tool_set is None:
            tool_set = self.tool_set
        else:
            tool_set = [t for t in tool_set if
                        t in self.tools or
                        t in self.mins]

        # Compute the minimum over considered tools
        min_tools = tool_set if restrict_tools else self.tools
        self.compute_best(tool_set=min_tools, new_col_name=min_name)
        vals = self.values.loc(axis=1)[col]
        df = vals.loc(axis=1)[tool_set + [min_name]]

        def is_min(x):
            return x[x == x[min_name]]

        # min_hits computes for each formula, how many tools
        # hits the minimum. NOte that the virtual tool that
        # always returns the best result is included in the count.
        min_hits = df.apply(is_min, axis=1).count(axis=1)

        selected_cases = (df[min_hits == 2]) if unique_only else df
        selected_cases = selected_cases.index

        # Compute how many times each tool matches the minimum value
        min_counts = df.loc[selected_cases].apply(is_min, axis=1).count()

        # Format the output DataFrame
        res = pd.DataFrame(min_counts[min_counts.index != min_name])
        res.index.name = "tool"
        res.columns = ["unique min hits"] if unique_only else ["min hits"]
        return res

    def get_plot_data(self, tool1, tool2,
                      include_equal=False,
                      col="states",
                      add_count=True,
                      **kwargs):
        """Prepare data to create scatter plot for 2 tools.

        `tool1`, `tool2` : tools to plot
        Optional arguments
        ==================
        `include_equal` : Bool
            if `False` (default) do not include formulas with the same
            values for both tools
        `col` : String
            name of ltlcross metric to plot, `states` by default
        `add_count` : Bool
            if `True` (default), group by values of `tool1` and `tool2`
            and add column `count` with number of occurrences of these
            values.
        """
        vals = self.values.loc[:, col]

        if include_equal:
            to_plot = vals.loc[:, [tool1, tool2]]
        else:
            nonequal = vals[tool1] != vals[tool2]
            to_plot = vals.loc()[nonequal, [tool1, tool2]]

        if add_count:
            to_plot['count'] = 1
            to_plot.dropna(inplace=True)
            to_plot = to_plot.groupby([tool1, tool2]).count().reset_index()
        return to_plot

    def seaborn_scatter_plot(self, tool1, tool2, log=False, **kwargs):
        """Return (and show) non-interactive scatter plot
        comparing 2 tools rendered using seaborn.

        Needs matplotlib and seaborn

        Always return the `bokeh.plotting.Figure` instance with the
        plot. This can be used to further tune the plot.

         `tool1` (axis `x`) and `tool2` (axis `y`)

        Possible kwargs
        ===============
        `show` : Bool, indicates, whether or not show the plot (in Jupyter)

        `col` : String
            name of ltlcross metric to plot, `states` by default
        `merge_same` : Bool
            if `True` (default), merge same instances and add colorbar
            for count, see `add_count` of `self.get_plot_data`.
        `include_equal` : Bool
            if `False` (default) do not include formulas with the same
            values for both tools

        And we have 4 arguments that control the appearance of the plot
        `palette` : color palette to use if `merge_same` is `True`
            default : `viridis`
        `alpha` : alpha of marks
            default `1` if `merge_same` and `.3` otherwise

        All remaining kwargs are supplied to `seaborn.relplot`
        """
        # Get the arguments
        merge_same = kwargs.pop("merge_same", True)
        alpha = kwargs.pop("alpha", .8) if merge_same else kwargs.pop("alpha", .3)
        marker_size = kwargs.pop("marker_size", 10)
        include_equal = kwargs.pop("include_equal", True)
        col = kwargs.pop("col", "states")
        palette = kwargs.pop("palette","viridis")

        # Prepare the data
        data = self.get_plot_data(tool1, tool2, add_count=merge_same, include_equal=include_equal, col=col)

        # Create the basic plot object
        seaborn.set_context("notebook", rc={"ms": 130})
        if merge_same:
            p = seaborn.relplot(x=tool1, y=tool2, data=data, size="count",alpha=alpha, hue="count", palette=palette, **kwargs)
        else:
            p = seaborn.relplot(x=tool1, y=tool2, data=data,alpha=alpha,**kwargs)
        p.fig.suptitle(f"Numbers of {col}")
        if log:
            p.ax.loglog()

        # Add line
        m = self.get_plot_data(tool1,tool2, add_count=False, include_equal=include_equal).max().max()
        matplotlib.pyplot.plot([0, m], [0, m], color="gray")
        return p


    def bokeh_scatter_plot(self, tool1, tool2, **kwargs):
        """Return (and show) an interactive scatter plot comparing
         2 tools rendered in bokeh library.

        Needs bokeh and colorcet libraries.

        Always return the `bokeh.plotting.Figure` instance with the
        plot. This can be used to further tune the plot.

         `tool1` (axis `x`) and `tool2` (axis `y`)
         `show` : Bool
            if `True` (default), show the plot in Jupyter notebook

        Possible kwargs
        ===============
        `show` : Bool, indicates, whether or not show the plot (in Jupyter)

        `col` : String
            name of ltlcross metric to plot, `states` by default
        `merge_same` : Bool
            if `True` (default), merge same instances and add colorbar
            for count, see `add_count` of `self.get_plot_data`.
        `include_equal` : Bool
            if `False` (default) do not include formulas with the same
            values for both tools

        And we have 4 arguments that control the appearance of the plot
        `palette` : color palette to use if `merge_same` is `True`
            default : `bwy` from `colorcet`
        `marker_color` : color to use if `merge_same` is `False`
            default : "navy"
        `alpha` : alpha of marks
            default `1` if `merge_same` and `.3` otherwise
        `marker_size` : int
            default `10`

        All remaining kwargs are supplied to `bokeh.plotting.scatter`
        """
        from bokeh.models import ColumnDataSource, CustomJS, ColorBar, TapTool, HoverTool, Slope
        from bokeh.transform import linear_cmap
        import bokeh.plotting as bplt

        # Get the arguments
        merge_same = kwargs.pop("merge_same", True)
        alpha = kwargs.pop("alpha", 1) if merge_same else kwargs.pop("alpha", .3)
        marker_size = kwargs.pop("marker_size", 10)
        show = kwargs.pop("show", True)
        include_equal = kwargs.pop("include_equal", False)
        col = kwargs.pop("col", "states")
        # Import colorcet for palette
        if merge_same:
            import colorcet as cc
            palette = kwargs.pop("palette", cc.bgy)

        # Make the graph render in notebooks
        if show:
            bplt.output_notebook()

        # Create the basic plot object
        p = bplt.figure(title=f"Numbers of {col}")
        p.xaxis.axis_label = f"{tool1}"
        p.yaxis.axis_label = f"{tool2}"

        # Prepare the data
        data = self.get_plot_data(tool1, tool2, add_count=merge_same, include_equal=include_equal, col=col)
        if not merge_same:
            # We want to have the form_id and formula fields available for tooltip
            data = data.reset_index()
        source = ColumnDataSource(data)

        # Tooltips
        tooltips = [
            (tool1, f"@{{{tool1}}}"),
            (tool2, f"@{{{tool2}}}"),
        ]

        if merge_same:
            # Map count of cases to color
            mapper = linear_cmap(palette=palette, field_name="count", low=1, high=data["count"].max())
            color = mapper

            # Add count to tooltip
            tooltips.append(("count", "@count"))

            # Print command to display selected formulas
            callback = CustomJS(args=dict(source=source), code=f"""
                // Select the data
                var inds = source.selected.indices;
                var data = source.data;
                var x = data['{tool1}'][inds];
                var y = data['{tool2}'][inds];

                // Create the two commands
                var fst_row = "data = a.get_plot_data('{tool1}','{tool2}',add_count=False)";
                var snd_row = "data[(data['{tool1}'] == " + x + ") & (data['{tool2}'] == " + y + ")]";

                // Instructions
                var instructions = "Use the following code to list the formulas.\\n";
                instructions += "Replace `a` with the ResAnalyzer` object:\\n\\n"
                alert(instructions + fst_row + "\\n" + snd_row);
                """)
        else:
            color = kwargs.pop("marker_color", "navy")
            tooltips.append(("formula id", "@form_id"))

            # Print formula on selection (currently only works for 1)
            callback = CustomJS(args=dict(source=source), code=f"""
                // Select the data
                var inds = source.selected.indices;
                var data = source.data;

                // Print formulas ids
                var output = data['form_id'][inds[0]];
                for (var i = 1; i < inds.length; i++) {{
                    var f = data['form_id'][inds[i]];
                    output += ', ' + f;
                }}
                output += '\\n'

                // Print formulas (1 per line)
                for (var i = 0; i < inds.length; i++) {{
                    var f = data['formula'][inds[i]];
                    output += f + '\\n';
                }}
                alert(output);
                """)

        # Plot data and add `y=x`
        slope = Slope(gradient=1, y_intercept=0, line_color="orange", line_width=2, line_dash="dashed")
        p.add_layout(slope)
        p.scatter(x=tool1, y=tool2, source=source, color=color, alpha=alpha, size=marker_size, **kwargs)

        # Add the hoover & selecting tool
        p.add_tools(TapTool(callback=callback))
        p.add_tools(HoverTool(tooltips=tooltips, mode="mouse"))

        if merge_same:
            color_bar = ColorBar(color_mapper=mapper['transform'], width=16, location=(0, 0))
            p.add_layout(color_bar, 'right')

        if show:
            bplt.show(p)
        return p

    def _highlight_min(self, s):
        is_min = s == s.min()
        return [f'background-color: {self.highlight_color}' if v else '' for v in is_min]