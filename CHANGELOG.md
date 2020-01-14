# Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
 * `Modulizer`
    - Each process created by `Modulizer` sets an unique value to environment variable
        `LCW_TMP` that can be used in names of temporary files to prevent data races.
    - `Modulizer.delete_intermediate()` removes the directory with partial results
 * `ResAnalyzer`
    - `cumulative` and `cross_compare` add visual information about the results by
        highlighting the cells. For `cumulative`, the best value (tool) is highlighted
        in each column. In `cross_compare`, cells contain a background bar that is
        proportional to the cell's value within the given column.
    - `ResAnalyzer()` now takes argument `tool_set` which is the default set of tools you
        for which results are shown.
    - `get_error_counts` shows numbers of errors found in the benchmark by categories.
    - `min_counts` can return counts of both unique and non-unique min-hits with `unique_only="both"`.
 * Updating files with results (all can be supplied `tool_set`)
    - `remove_tools(resfile, tools, output)` removes lines for given `tools` from `resfile`
        and stores the reduced file in `output`.
    - `update_resfile(base, new, output)` updates `base` resfile with values from `new`
        resfile and stores the updated results in `output`.
    - `merge_resfiles(base, new, output)` merges 2 resfiles into `output`.
 * `gather_cumulative` and `gather_mins` show tables for multiple benchmarks.

### Changed
 - `parse_results` is now called from constructor of `ResAnalyzer`.

## [0.6.1] - 2019-12-30
### Added
 - Class `Modulizer` splits a big `ltlcross` task into smaller ones, execute the small
    tasks in parallel, and merge the intermediate results into one final `.csv`, `.log`,
    and `_bogus.ltl` files.
 - Class `ResAnalyzer` parses the results of `ltlcross`, and implements several functions
    to analyze and visualize the results, mainly in Jupyter notebooks. 
    The main features are:
    * Tables with cumulative sums for metrics like states, transitions, acc, over all
        formulas in the benchmark are generated using `ResAnalyzer.cumulative()`.
    * Tables that show how many times `tool1` delivers better result than `tool2`, for
        each pair of tools are generated using `ResAnalyzer.cross_compare()`.
    * Cases (with concrete formulas) where `tool1` is better than `tool2` can be 
        displayed using `ResAnalyzer.better_than`
    * Portfolio approach (running multiple tools/configurations in parallel and choosing
        the best) can be simulated using `ResAnalyzer.compute_best`
    * Automaton produced by `tool` for formula indexed by `i` in the benchmark can be
        shown by `ResAnalyzer.aut_for_id(i, tool)`.
    * Interactive scatter plots rendered by [bokeh](https://bokeh.org/) are created using
        `ResAnalyzer.bokeh_scatter_plot(tool1, tool2)`.

[Unreleased]: https://github.com/xblahoud/ltlcross_wrapper/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/xblahoud/ltlcross_wrapper/tags/v0.6.1
