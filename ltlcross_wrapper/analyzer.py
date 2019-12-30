# Copyright (c) 2019 Fanda Blahoudek

import os.path
import math

import pandas as pd
import spot

from ltlcross_runner.locate_errors import bogus_to_lcr


def pretty_print(form):
    """Runs Spot to format formulas nicer."""
    return spot.formula(form).to_str()


def hoa_to_spot(hoa):
    return spot.automaton(hoa + "\n")


class ResAnalyzer:
    """Analyze `.csv` files with results of ltlcross

    Parameters
    ----------
    res_filename : String
        filename to store the ltlcross`s results
    cols : list of Strings, default ``['states','edges','transitions','acc']``
        names of ltlcross's statistics columns to be recorded
    """

    def __init__(self,
                 res_filename,
                 cols=None,
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
        self.values.sort_index(axis=1, level=0, inplace=True)

    def cumulative(self, tool_set=None, col="states"):
        """Returns table with cumulative numbers of states.

         For each tool, sums the values of `col` over formulas
         with no timeout.

         If `tool_subset` is given, use only tools within this
         subset. Only formulas with timeouts within the subset
         are removed. The sum of values for `tool_subset` where
         all formulas with some timeout are removed run
         ```
         self.cumulative().loc[tool_subset]
         ```

        Parameters
        ---------
        col : String or list of Strings
            One or more columns (``states`` default).

        tool_set : list, contains tools from `self.tools`
            Restrict the output to given subset of tools
        """
        if tool_set is None:
            tool_set = self.tools
        data = self.values.loc[:, (col, tool_set)].dropna().sum()

        # Format as DataFrame, remove unnecessary index labels
        df = pd.DataFrame(data)
        df = df.unstack(level=0)
        df.columns = df.columns.droplevel()
        df.columns.name = ""
        return df

    def smaller_than(self, t1, t2, col='states', **kwargs):
        """Returns a dataframe with results where ``col`` for ``tool1``
        has strictly smaller value than ``col`` for ``tool2``.

        Parameters
        ----------
        t1 : String
            name of tool for comparison (the better one)
            must be among tools
        t2 : String
            name of tool for comparison (the worse one)
            must be among tools
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

    def get_error_count(self, err_type='timeout', drop_zeros=True):
        """Returns a Series with total number of er_type errors for
        each tool.

        Parameters
        ----------
        err_type : String one of `timeout`, `parse error`,
                                 `incorrect`, `crash`, or
                                 'no output'
                  Type of error we seek
        drop_zeros: Boolean (default True)
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

    def cross_compare(self,
                      tool_set=None,
                      total=True,
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
            tool_set = self.tools
        c = pd.DataFrame(index=tool_set, columns=tool_set).fillna(0)
        for tool in tool_set:
            c[tool] = pd.DataFrame(c[tool]).apply(lambda x: count_better(x.name, tool), 1)
        if total:
            c['V'] = c.sum(axis=1)
        return c

    def min_counts(self, tool_set=None, restrict_tools=False, unique_only=False, col='states', min_name='min(count)'):
        if tool_set is None:
            tool_set = list(self.tools)
        else:
            tool_set = [t for t in tool_set if
                        t in self.tools or
                        t in self.mins]
        min_tools = tool_set if restrict_tools else list(self.tools)
        self.compute_best(tool_set=min_tools, new_col_name=min_name)
        s = self.values.loc(axis=1)[col]
        df = s.loc(axis=1)[tool_set + [min_name]]
        is_min = lambda x: x[x == x[min_name]]
        best_t_count = df.apply(is_min, axis=1).count(axis=1)
        choose = (df[best_t_count == 2]) if unique_only else df
        choose = choose.index
        min_counts = df.loc[choose].apply(is_min, axis=1).count()
        return pd.DataFrame(min_counts[min_counts.index != min_name])

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

    def bokeh_scatter_plot(self, tool1, tool2, **kwargs):
        """Return (and show) a scatter plot for 2 tools rendered in bokeh library.

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